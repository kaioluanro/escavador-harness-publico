"""C1 — segmentação da ingestão: de linhas classificadas a :class:`Documento`.

Recebe a saída do passo B1 (:func:`harness.ingest.classificar.classificar_arquivo`)
e agrupa as linhas lógicas em **documentos**: cada documento começa num
``ENUNCIADO`` em negrito e vai até antes do próximo ``ENUNCIADO`` (ou o fim da
edição), anexando ao corpo as linhas intermediárias (inclusive as de tabela).

Decisões aplicadas (``knowledge/regras_diario.md`` §1, ``comms/DECISOES.md``
D011–D013 e D018):

- **Ordem de leitura por coluna (C1/C2).** Nas páginas de duas colunas a leitura
  é coluna esquerda (topo→base) e depois direita (topo→base); linhas que
  atravessam o gutter separam faixas horizontais. Duas rotas:
  ``metodo_ordem="propria"`` (heurística própria de gutter + faixas, padrão por
  ter medido melhor) e ``metodo_ordem="column_boxes"``
  (``pymupdf4llm.helpers.multi_column.column_boxes``, D014). A rota por
  ``column_boxes`` cai para a própria se o utilitário não estiver disponível.
- **ID não é fronteira (D011).** O ``ID:`` de 18 dígitos logo acima do enunciado,
  na mesma coluna, é apenas **metadado**: vira ``id_dom`` do documento que o
  enunciado abre.
- **Seção/órgão alimenta ``orgao``.** Siglas ``¨…¨`` (exceto marcadores de
  subseção ``DIRETA``/``INDIRETA``/``LICITAÇÃO``) atualizam o ``orgao`` corrente,
  herdado pelos documentos seguintes.
- **Anti-fantasma A1 antes de classificar/segmentar (D018).** A extração usa
  :mod:`harness.ingest.extrair`/``fantasma.spans_visiveis``: o texto fantasma sai
  **antes** de B1/C1. O masthead/rodapé restante é decidido só por banda vertical
  + fonte estrita (:func:`classificar._masthead`) — nunca por substring; corpo
  como ``impressora``/``Laserjet``/dia da semana no meio da linha NÃO é removido
  (R014/R014b).
- **Cláusula primária D026.** Bloco sem enunciado no topo de coluna/página vai
  para o documento cuja **frase** ele continua (linha inicia em minúscula/hífen e
  o candidato da mesma coluna está com frase aberta); se indecidível, usa o
  documento anterior na ordem de leitura. Fronteiras incertas recebem
  ``continuacao_ambigua``.
- **Tabela pertence ao corpo (regras §1.4).** As linhas de tabela entram em
  ``texto``; a extração estrutural por ``find_tables()`` é o passo E (fora deste).
- **Rede de segurança de tamanho (regras §1.8):** documentos acima de
  :data:`LIMITE_CHARS_ALERTA` recebem a flag ``documento_gigante_a_verificar``
  (F1/F4, D012), sem quebrar o documento.

Limitações registradas:

- ``pagina_fim`` é a página da última linha do corpo antes do próximo enunciado
  na ordem de leitura. Casos de transbordamento de coluna com artigos
  entrelaçados (ex.: ``4196 doc6``) e cabeçalhos órfãos no topo de página podem
  divergir do ouro; esses documentos recebem flag de incerteza.
- Com ``enriquecer=True`` (padrão), :func:`segmentar_arquivo` roda também
  D-meta-1/D-ent-1 (+NER, :mod:`harness.ingest.pipeline`); ``enriquecer=False``
  devolve só a segmentação (usado em testes/diagnóstico).
- O contrato :class:`harness.contracts.Documento` proíbe campos extras; as
  flags/proveniência viajam em :class:`DocumentoSegmentado` e num *sidecar*
  JSONL opcional, sem alterar o contrato.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:  # PyMuPDF >= 1.24 expõe ``pymupdf``; versões antigas só ``fitz``.
    import pymupdf
except ImportError:  # pragma: no cover - compatibilidade com versões antigas
    import fitz as pymupdf  # type: ignore[no-redef]

from harness.contracts import Documento, LinhaLogica
from harness.ingest.classificar import (
    ALTURA_PAGINA_PADRAO,
    LinhaClassificada,
    _masthead as _masthead_b1,
    classificar_linhas,
    normalizar,
    token_canonico,
)
from harness.ingest.extrair import alturas_do_pdf, extrair_linhas_visiveis

__all__ = [
    "GAP_ID_MAX_PT",
    "LIMITE_CHARS_ALERTA",
    "METODOS_ORDEM",
    "SIGLAS_IGNORADAS",
    "DocumentoSegmentado",
    "MarcadorFronteira",
    "documentos",
    "edicao_do_conteudo",
    "edicao_do_nome",
    "escrever_proveniencia",
    "escrever_snapshot",
    "ler_snapshot",
    "listar_pdfs",
    "main",
    "ordenar_leitura",
    "segmentar",
    "segmentar_arquivo",
    "segmentar_ordenadas",
]

#: Métodos de ordem de leitura aceitos.
METODOS_ORDEM: tuple[str, ...] = ("propria", "column_boxes", "nao")

#: Distância vertical máxima (pt) entre o ``ID:`` e o enunciado que ele abre.
GAP_ID_MAX_PT: float = 40.0

#: Limite (pt) de ``x`` que separa as colunas esquerda/direita do DOM.
X_CORTE_PT: float = 300.0

#: Faixa ``x`` que caracteriza uma linha que atravessa o gutter (largura total):
#: começa bem à esquerda e termina bem à direita.
X_FULL_ESQ_PT: float = 200.0
X_FULL_DIR_PT: float = 400.0

#: Tolerância (pt) para separar faixas horizontais nas linhas ``F``.
TOL_BANDA_PT: float = 0.01

#: Siglas que marcam subseção (não são ``orgao`` do documento).
SIGLAS_IGNORADAS: frozenset[str] = frozenset({"DIRETA", "INDIRETA", "LICITACAO", "LICITAÇÃO"})

#: Alerta de fusão indevida (D012): não quebra o documento, só sinaliza.
LIMITE_CHARS_ALERTA: int = 50_000

_RE_ID18 = re.compile(r"\d{18}")
_RE_NOME_EDICAO = re.compile(r"DOM(?P<numero>\d[\d.]*)-(?P<data>\d{8})", re.IGNORECASE)
#: Final de frase: usado para decidir/flagged continuação entre páginas.
_RE_FIM_FRASE = re.compile(r"[.!?:;\"”\)\]]\s*$")

#: Rótulos que delimitam o BLOCO DE EXPEDIENTE (p.1–3): lista de autoridades/
#: secretários/prefeito. Regras §1.6: masthead/expediente não entram em documento.
_ROTULOS_EXPEDIENTE_BLOCO: tuple[str, ...] = (
    "PREFEITURA MUNICIPAL DE TERESINA",
    "GERENCIA DE IMPRENSA OFICIAL",
    "SECRETARIA MUNICIPAL DE ADMINISTRACAO",
    "ORGAO DE COMUNICACAO OFICIAL",
    "CADERNO UNICO",
    "IMPRESSO NA PRODATER",
    "ASSINATURA DIGITAL",
    "PREFEITO MUNICIPAL DE TERESINA",
)
#: Fonte/tamanho típicos do expediente (distintos do corpo, 8 pt).
_TAMANHO_EXPEDIENTE_BLOCO: float = 7.5

#: Cargos de autoridade que denunciam o expediente (categoria geral; o layout
#: muda a cada ano, os cargos não). Comparados sem acento/hífen.
_CARGOS_EXPEDIENTE: frozenset[str] = frozenset(
    {
        "PREFEITO",
        "PREFEITA",
        "VICEPREFEITO",
        "VICEPREFEITA",
        "VICE",
        "SECRETARIO",
        "SECRETARIA",
        "SUBSECRETARIO",
        "SUBSECRETARIA",
        "PROCURADOR",
        "PROCURADORA",
        "PRESIDENTE",
        "VICEPRESIDENTE",
        "SUPERINTENDENTE",
        "GERENTE",
        "DIRETOR",
        "DIRETORA",
        "DIRETORPRESIDENTE",
        "COORDENADOR",
        "COORDENADORA",
        "CHEFE",
        "ASSESSOR",
        "ASSESSORA",
        "CONTROLADOR",
        "CONTROLADORA",
        "OUVIDOR",
        "OUVIDORA",
        "CORREGEDOR",
        "CORREGEDORA",
        "COMANDANTE",
        "DELEGADO",
        "DELEGADA",
    }
)

#: Rótulos de crédito/expediente (gráfica, diagramação, tiragem, órgão oficial).
#: Casados por INÍCIO de linha — corpo com ``impressora``/``laser`` no meio não
#: casa (R014).
_ROTULOS_CREDITO_EXPEDIENTE: tuple[str, ...] = (
    "GERENTE DE IMPRENSA",
    "DIAGRAMADOR",
    "IMPRESSO",
    "PRODATER",
    "LASER",
    "ASSINATURA DIGITAL",
    "ASSINADO DE FORMA DIGITAL",
    "TIRAGEM",
    "EXEMPLARES",
    "ESTA EDICAO E COMPOSTA",
    "PELO SISTEMA",
    "ORGAO DE COMUNICACAO OFICIAL",
    "ORGAO DESTINADO A PUBLICACAO",
    "CADERNO UNICO",
    "DIVISAO DE EDICAO",
    "RUA FIRMINO PIRES",
    "DIARIO OFICIAL DO MUNICIPIO",
    "PREFEITURA MUNICIPAL DE",
    "SECRETARIA MUNICIPAL DE ADMINISTRACAO",
    "SECRETARIA MUNICIPAL DE COMUNICACAO",
)

#: Conectores que podem aparecer dentro de um nome de pessoa (expediente).
_CONECTORES_EXPEDIENTE: frozenset[str] = frozenset(
    {"DE", "DA", "DO", "DAS", "DOS", "E", "D"}
)

#: Marcas de corpo de ato: uma linha que as contenha NÃO é expediente, mesmo que
#: cite um cargo (``O PREFEITO MUNICIPAL DE TERESINA ... resolve``).
_RE_MARCA_CORPO_EXPEDIENTE = re.compile(
    r"\b(NO USO|RESOLVE|CONSIDERANDO|DECRETA|REVOGA|PARAGRAFO|CLAUSULA|"
    r"ARTIGO|ART\.|FICA|CONFORME|VIGOR)\b"
)

#: Páginas em que o bloco de expediente pode aparecer (front matter). Outras
#: edições trazem o expediente na p.2 e continuam na p.3; documentos reais não
#: são afetados porque o bloco exige densidade de cargos/créditos.
MAX_PAGINA_EXPEDIENTE: int = 3

#: Número mínimo de linhas de expediente para o bloco ser reconhecido (evita
#: confundir o bloco de assinatura de um ato — ~3–6 linhas — com expediente).
MIN_LINHAS_EXPEDIENTE: int = 8

#: Nº mínimo de cargos DISTINTOS para o bloco ser expediente. O expediente real
#: lista muitas autoridades; blocos de assinatura/mesa têm poucos cargos.
MIN_CARGOS_EXPEDIENTE: int = 5

#: Linhas não-expediente toleradas dentro do bloco (colunas/tabelas do layout).
MAX_GAP_EXPEDIENTE: int = 2

#: Margem (pt) somada às bordas verticais do bloco.
MARGEM_EXPEDIENTE_PT: float = 3.0


def _palavra_expediente(texto: str) -> str:
    """Só as letras do token, sem acento/hífen, em maiúsculas."""
    return re.sub(r"[^A-Z]", "", normalizar(texto))


def _tem_credito_expediente(texto: str) -> bool:
    """True se a linha COMEÇA por um rótulo de crédito/expediente."""
    return any(texto.startswith(rotulo) for rotulo in _ROTULOS_CREDITO_EXPEDIENTE)


def _cargos_da_linha(classe: LinhaClassificada) -> set[str]:
    """Cargos de autoridade presentes na linha (sem acento/hífen)."""
    return {
        palavra
        for palavra in (
            _palavra_expediente(token) for token in normalizar(classe.linha.texto).split()
        )
        if palavra in _CARGOS_EXPEDIENTE
    }


def _e_nome_proprio_linha(texto: str) -> bool:
    """True se a linha inteira é um nome de pessoa (2–7 tokens capitalizados).

    Aceita iniciais pontuadas e conectores; rejeita linha com dígito, cargo,
    verbo ou pontuação de frase. É o sinal de "lista de autoridades" que, somado
    à densidade de cargos, caracteriza o expediente.
    """
    limpo = (texto or "").strip()
    if not limpo or len(limpo) > 60 or re.search(r"\d", limpo):
        return False
    if re.sub(r"[A-Za-zÀ-ÿ'’.\s-]", "", limpo):
        return False
    tokens = [token for token in re.split(r"\s+", limpo) if token]
    if not 2 <= len(tokens) <= 7:
        return False
    nucleos = 0
    for token in tokens:
        base = _palavra_expediente(token)
        if not base:
            continue
        if base in _CARGOS_EXPEDIENTE:
            return False
        if base in _CONECTORES_EXPEDIENTE:
            continue
        if not token[0].isupper():
            return False
        if len(base) == 1 and not token.endswith("."):
            return False
        nucleos += 1
    return nucleos >= 2


def _e_linha_repetida(texto: str) -> bool:
    """True se a linha repete o mesmo token 3+ vezes (camadas sobrepostas).

    Ex.: ``Prefeitura Prefeitura Prefeitura`` / ``Municipal Municipal Municipal``
    — ruído de masthead que NÃO deve servir de ponte entre o bloco de assinatura
    de um ato e o expediente.
    """
    palavras = normalizar(texto).split()
    if len(palavras) < 3:
        return False
    contagem: dict[str, int] = {}
    for palavra in palavras:
        contagem[palavra] = contagem.get(palavra, 0) + 1
        if contagem[palavra] >= 3:
            return True
    return False


def _e_linha_expediente(classe: LinhaClassificada) -> bool:
    """True para uma linha candidata a expediente (crédito, cargo ou nome só)."""
    if classe.categoria in (
        "ID",
        "ENUNCIADO",
        "SECAO_ORGAO",
        "CABECALHO_TABELA",
        "CABECALHO_ANEXO",
    ):
        return False
    texto = normalizar(classe.linha.texto)
    if not texto:
        return False
    if _e_linha_repetida(texto):
        return False
    if _tem_credito_expediente(texto):
        return True
    if _RE_MARCA_CORPO_EXPEDIENTE.search(texto):
        return False
    if len(texto.split()) <= 14 and _cargos_da_linha(classe):
        return True
    return _e_nome_proprio_linha(classe.linha.texto)


def _run_expediente_valido(run: Sequence[LinhaClassificada]) -> bool:
    """True se o agrupamento é o bloco de expediente (densidade + cargos).

    Exige muitas linhas candidatas E vários cargos DISTINTOS. O crédito de
    gráfica/diagramação é um sinal adicional, mas não basta sozinho: blocos de
    assinatura de um ato (poucos cargos) não podem ser removidos.
    """
    expediente = [classe for classe in run if _e_linha_expediente(classe)]
    if len(expediente) < MIN_LINHAS_EXPEDIENTE:
        return False
    cargos: set[str] = set()
    for classe in expediente:
        cargos |= _cargos_da_linha(classe)
    return len(cargos) >= MIN_CARGOS_EXPEDIENTE


def _registrar_regiao_expediente(
    run: Sequence[LinhaClassificada],
    regioes: dict[int, tuple[float, float]],
    pagina: int,
) -> None:
    """Acrescenta/expande a faixa ``(y0, y1)`` do bloco, se ele for válido."""
    if not _run_expediente_valido(run):
        return
    ys = [classe.linha.y for classe in run]
    y0 = min(ys) - MARGEM_EXPEDIENTE_PT
    y1 = max(ys) + MARGEM_EXPEDIENTE_PT
    atual = regioes.get(pagina)
    if atual is None:
        regioes[pagina] = (y0, y1)
    else:
        regioes[pagina] = (min(atual[0], y0), max(atual[1], y1))


def _regioes_expediente(
    ordenadas: Sequence[LinhaClassificada],
) -> dict[int, tuple[float, float]]:
    """Faixa ``(y0, y1)`` do bloco de expediente por página (regras §1.6).

    Duas rotas complementares:

    1. **Rótulo + fonte pequena** (layout de 2026): ``y0`` = menor ``y`` de uma
       linha com rótulo inequívoco; ``y1`` = ``y`` do primeiro ENUNCIADO abaixo.
    2. **Conteúdo + estrutura** (qualquer layout, T025): agrupa linhas candidatas
       a expediente (crédito de gráfica/diagramação, cargo de autoridade, nome
       próprio isolado) em blocos verticais das páginas :data:`MAX_PAGINA_EXPEDIENTE`,
       sem ENUNCIADO dentro, e exige densidade de cargos/créditos. Não depende de
       coordenadas fixas nem dos rótulos de um ano.
    """
    marcadores: dict[int, list[float]] = defaultdict(list)
    enunciados: dict[int, list[float]] = defaultdict(list)
    for classe in ordenadas:
        linha = classe.linha
        if classe.categoria == "ENUNCIADO":
            enunciados[linha.pagina].append(linha.y)
            continue
        if not linha.spans or linha.spans[0].tamanho > _TAMANHO_EXPEDIENTE_BLOCO:
            continue
        texto = normalizar(linha.texto)
        if any(texto.startswith(rotulo) for rotulo in _ROTULOS_EXPEDIENTE_BLOCO):
            marcadores[linha.pagina].append(linha.y)

    regioes: dict[int, tuple[float, float]] = {}
    for pagina, ys in marcadores.items():
        y0 = min(ys) - 3.0
        abaixo = [y for y in enunciados.get(pagina, []) if y > y0 + 3.0]
        y1 = min(abaixo) if abaixo else float("inf")
        regioes[pagina] = (y0, y1)

    # Rota geral (T025): blocos por conteúdo/estrutura.
    por_pagina: dict[int, list[LinhaClassificada]] = defaultdict(list)
    for classe in ordenadas:
        if classe.linha.pagina <= MAX_PAGINA_EXPEDIENTE:
            por_pagina[classe.linha.pagina].append(classe)

    for pagina, itens in por_pagina.items():
        itens.sort(key=lambda c: (c.linha.y, c.linha.x0))
        run: list[LinhaClassificada] = []
        gaps = 0
        for classe in itens:
            if classe.categoria == "ENUNCIADO":
                _registrar_regiao_expediente(run, regioes, pagina)
                run, gaps = [], 0
                continue
            if _e_linha_expediente(classe):
                run.append(classe)
                gaps = 0
            elif run:
                gaps += 1
                if gaps > MAX_GAP_EXPEDIENTE:
                    _registrar_regiao_expediente(run, regioes, pagina)
                    run, gaps = [], 0
                else:
                    run.append(classe)
        _registrar_regiao_expediente(run, regioes, pagina)

    return regioes
#: Fim de frase "fechado" (``.``/``!``/``?``): a próxima linha abre oração nova.
_RE_TERMINAL = re.compile(r"[.!?]\s*$")


def _frase_aberta(texto: str) -> bool:
    """True se a linha termina em frase aberta (não fecha em ``.!?``)."""
    return bool((texto or "").rstrip()) and not _RE_TERMINAL.search((texto or "").rstrip())


# --------------------------------------------------------------------------- #
# Estruturas de resultado                                                     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MarcadorFronteira:
    """Proveniência de uma fronteira de documento (início ou fim)."""

    tipo: str
    regra: str
    motivo: str
    evidencias: Mapping[str, object] = field(default_factory=dict)

    def para_dict(self) -> dict[str, object]:
        """Versão serializável para o *sidecar* de proveniência."""
        return {
            "tipo": self.tipo,
            "regra": self.regra,
            "motivo": self.motivo,
            "evidencias": dict(self.evidencias),
        }


@dataclass(frozen=True)
class DocumentoSegmentado:
    """Um :class:`Documento` mais flags de incerteza e proveniência das fronteiras.

    O contrato ``Documento`` não carrega campos extras; este invólucro mantém o
    snapshot JSONL compatível com o gate de extração e expõe a instrumentação
    exigida por I004 (''qual regra decidiu cada fronteira'').
    """

    documento: Documento
    flags: tuple[str, ...] = ()
    proveniencia: tuple[MarcadorFronteira, ...] = ()

    @property
    def doc_id(self) -> str:
        return self.documento.doc_id

    @property
    def id_dom(self) -> str | None:
        return self.documento.id_dom

    def para_dict(self) -> dict[str, object]:
        """Projeção serializável (documento + flags + proveniência)."""
        return {
            "doc_id": self.documento.doc_id,
            "arquivo": self.documento.arquivo,
            "id_dom": self.documento.id_dom,
            "flags": list(self.flags),
            "proveniencia": [m.para_dict() for m in self.proveniencia],
        }


# --------------------------------------------------------------------------- #
# Ordem de leitura por coluna                                                 #
# --------------------------------------------------------------------------- #


def _coluna(linha: LinhaLogica) -> str:
    """Coluna aproximada da linha: ``"L"``, ``"R"`` ou ``"F"`` (atravessa o gutter)."""
    if linha.x0 < X_FULL_ESQ_PT and linha.x1 > X_FULL_DIR_PT:
        return "F"
    centro = (linha.x0 + linha.x1) / 2.0
    return "L" if centro < X_CORTE_PT else "R"


def _ordem_propria(linhas: Sequence[LinhaLogica]) -> list[LinhaLogica]:
    """Ordem de leitura por coluna (heurística própria, C2)."""
    por_pagina: dict[int, list[LinhaLogica]] = defaultdict(list)
    for linha in linhas:
        por_pagina[linha.pagina].append(linha)

    ordenadas: list[LinhaLogica] = []
    for pagina in sorted(por_pagina):
        da_pagina = por_pagina[pagina]
        ys_full = sorted(l.y for l in da_pagina if _coluna(l) == "F")

        def chave(linha: LinhaLogica) -> tuple[float, int, float, float]:
            faixa = sum(1 for y in ys_full if y < linha.y - TOL_BANDA_PT)
            coluna = _coluna(linha)
            if coluna == "F":
                return (faixa, 0, linha.y, linha.x0)
            return (faixa, 1 if coluna == "L" else 2, linha.y, linha.x0)

        ordenadas.extend(sorted(da_pagina, key=chave))
    return ordenadas


def _indice_retangulo(linha: LinhaLogica, retangulos: Sequence[object]) -> int:
    """Índice do retângulo de ``column_boxes`` com maior sobreposição com a linha."""
    melhor = -1
    melhor_area = -1.0
    for indice, retangulo in enumerate(retangulos):
        x0 = max(linha.x0, float(retangulo.x0))  # type: ignore[attr-defined]
        y0 = max(linha.y - 8.0, float(retangulo.y0))  # type: ignore[attr-defined]
        x1 = min(linha.x1, float(retangulo.x1))  # type: ignore[attr-defined]
        y1 = min(linha.y, float(retangulo.y1))  # type: ignore[attr-defined]
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if area > melhor_area:
            melhor_area = area
            melhor = indice
    return melhor


def _ordem_column_boxes(
    linhas: Sequence[LinhaLogica], caminho: str | Path
) -> list[LinhaLogica]:
    """Ordem de leitura usando ``pymupdf4llm.helpers.multi_column.column_boxes``.

    Levanta :class:`ImportError` se o utilitário não estiver instalado; o chamador
    cai para :func:`_ordem_propria`.
    """
    from pymupdf4llm.helpers.multi_column import column_boxes  # noqa: PLC0415

    paginas = sorted({linha.pagina for linha in linhas})
    retangulos: dict[int, list[object]] = {}
    with pymupdf.open(str(caminho)) as doc:
        for pagina in paginas:
            try:
                retangulos[pagina] = list(column_boxes(doc[pagina - 1]))
            except Exception:  # noqa: BLE001 - layout inesperado: usa ordem vazia
                retangulos[pagina] = []

    return sorted(
        linhas,
        key=lambda linha: (
            linha.pagina,
            _indice_retangulo(linha, retangulos.get(linha.pagina, [])),
            linha.y,
            linha.x0,
        ),
    )


def _ordenar_linhas(
    linhas: Sequence[LinhaLogica], *, metodo: str, caminho: str | Path | None
) -> list[LinhaLogica]:
    """Aplica o método de ordem requisitado, com queda para a heurística própria."""
    if metodo not in METODOS_ORDEM:
        raise ValueError(f"metodo_ordem inválido: {metodo!r} (use: {', '.join(METODOS_ORDEM)})")
    if metodo == "nao":
        return sorted(linhas, key=lambda l: (l.pagina, l.y, l.x0))
    if metodo == "column_boxes":
        if caminho is None:
            raise ValueError("metodo_ordem='column_boxes' exige o caminho do PDF")
        try:
            return _ordem_column_boxes(linhas, caminho)
        except ImportError:
            pass  # pymupdf4llm ausente: cai para a heurística própria
    return _ordem_propria(linhas)


def ordenar_leitura(
    classificadas: Sequence[LinhaClassificada],
    *,
    metodo: str = "propria",
    caminho: str | Path | None = None,
) -> list[LinhaClassificada]:
    """Reordena as classificações pela ordem de leitura por coluna escolhida."""
    por_linha = {id(item.linha): item for item in classificadas}
    linhas = [item.linha for item in classificadas]
    ordenadas = _ordenar_linhas(linhas, metodo=metodo, caminho=caminho)
    return [por_linha[id(linha)] for linha in ordenadas]


# --------------------------------------------------------------------------- #
# Identificação e órgão                                                       #
# --------------------------------------------------------------------------- #


def _digitos_id(texto: str) -> str | None:
    """Extrai o ID de 18 dígitos de uma linha ``ID: …``."""
    achado = _RE_ID18.search(texto)
    return achado.group(0) if achado else None


def _id_do_enunciado(
    enunciado: LinhaClassificada,
    identificadores: Sequence[LinhaClassificada],
    usados: set[str],
) -> tuple[str | None, float | None]:
    """``ID:`` logo acima do enunciado, na mesma coluna (D011: metadado, não fronteira)."""
    alvo = enunciado.linha
    coluna = _coluna(alvo)
    melhor: LinhaClassificada | None = None
    melhor_gap: float | None = None
    for ident in identificadores:
        if ident.linha.pagina != alvo.pagina or _coluna(ident.linha) != coluna:
            continue
        gap = alvo.y - ident.linha.y
        if gap <= 0 or gap > GAP_ID_MAX_PT:
            continue
        if melhor_gap is None or gap < melhor_gap:
            melhor = ident
            melhor_gap = gap
    if melhor is None:
        return None, None
    valor = _digitos_id(melhor.linha.texto)
    if valor is None or valor in usados:
        return None, None
    usados.add(valor)
    return valor, round(melhor_gap, 1)


def _sigla_orgao(classe: LinhaClassificada) -> str | None:
    """Sigla ``¨…¨`` que atualiza o ``orgao`` (ignora marcadores de subseção)."""
    if classe.proveniencia.regra != "orgao_sigla":
        return None
    valor = classe.linha.texto.strip().strip("¨").strip()
    if not valor or normalizar(valor) in SIGLAS_IGNORADAS:
        return None
    return valor


#: Sigla no próprio cabeçalho de seção: ``… Recursos Humanos – SEMA``.
_RE_SIGLA_NO_CABECALHO = re.compile(r"[–—-]\s*([A-ZÀ-Þ][A-ZÀ-Þ0-9]{1,9})\s*$")


def _orgao_do_cabecalho(classe: LinhaClassificada) -> str | None:
    """Órgão que um cabeçalho de seção/órgão alimenta (D037, T024 item 1).

    - sigla ``¨…¨`` → a sigla (ignora marcadores de subseção);
    - cabeçalho por extenso (BoldItal) → a **sigla** se houver no próprio
      cabeçalho (``… – SEMA``); senão o **nome** do cabeçalho normalizado.
    """
    sigla = _sigla_orgao(classe)
    if sigla:
        return sigla
    if classe.proveniencia.regra != "secao_orgao_boldital":
        return None
    texto = re.sub(r"\s+", " ", classe.linha.texto).strip().rstrip(".,;:")
    achado = _RE_SIGLA_NO_CABECALHO.search(texto)
    if achado:
        return achado.group(1)
    return texto or None


# --------------------------------------------------------------------------- #
# Texto / título                                                             #
# --------------------------------------------------------------------------- #


#: Início de oração (``. Palavra``) — corta corpo que veio fundido no título (H3).
_RE_INICIO_ORACAO = re.compile(r"[.!?]\s+(?=[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ])")
#: Hífen de quebra de linha no fim do enunciado (``... MARIA JO-``).
_RE_HIFEN_FINAL = re.compile(r"[-–—]\s*$")
#: Hífen de quebra seguido de espaço entre letras (``PROCES- SO`` → ``PROCESSO``).
_RE_HIFEN_ESPACO_LETRA = re.compile(r"(?<=[A-Za-zÀ-ÿ])[-–—]\s+(?=[A-Za-zÀ-ÿ])")
#: Hífen de quebra seguido de espaço entre dígitos (``2022- 03`` → ``2022-03``).
_RE_HIFEN_ESPACO_DIGITO = re.compile(r"(?<=\d)[-–—]\s+(?=\d)")


def _limpar_enunciado(texto: str) -> str:
    """Normaliza o título: colapsa espaços, junta hífen de quebra e corta corpo.

    O ouro guarda só o título; a fusão H3 pode anexar o início do corpo na mesma
    linha lógica. Cortar no início de uma oração (``. Palavra``) aproxima o campo
    do ouro e evita que o pareamento do gate confunda dois documentos irmãos
    (ex.: dois ``EXTRATO ... AUTO DE INFRAÇÃO Nº 0X`` na mesma página). O hífen de
    quebra seguido de espaço é limpo (``PROCES- SO`` → ``PROCESSO``; ``2022- 03``
    → ``2022-03``) — T024 item 5.
    """
    texto = re.sub(r"\s+", " ", texto).strip()
    texto = _RE_HIFEN_ESPACO_LETRA.sub("", texto)
    texto = _RE_HIFEN_ESPACO_DIGITO.sub("-", texto)
    texto = _RE_HIFEN_FINAL.sub("", texto).strip()
    corte = _RE_INICIO_ORACAO.search(texto)
    if corte:
        texto = texto[: corte.start() + 1].strip()
    return texto


def _enunciado_de(classe: LinhaClassificada) -> str:
    """Título efetivo do documento.

    Usa o título fundido (H4/H3) quando houver; senão, o **run inicial de spans
    em negrito** da linha (evita engolir o início do corpo na mesma linha, H2).
    """
    if classe.texto_fundido:
        return _limpar_enunciado(classe.texto_fundido)
    partes: list[str] = []
    for span in classe.linha.spans:
        if not span.negrito:
            break
        partes.append(span.texto)
    titulo = _limpar_enunciado("".join(partes))
    return titulo or _limpar_enunciado(classe.linha.texto)


def _resto_apos_negrito(classe: LinhaClassificada) -> str:
    """Texto regular que segue o título na MESMA linha lógica (H2)."""
    indice: int | None = None
    viu_negrito = False
    for posicao, span in enumerate(classe.linha.spans):
        if span.negrito:
            viu_negrito = True
        elif viu_negrito:
            indice = posicao
            break
    if indice is None:
        return ""
    texto = "".join(span.texto for span in classe.linha.spans[indice:])
    return re.sub(r"\s+", " ", texto).strip()


def _e_ruido_masthead(
    linha: LinhaLogica, *, altura_pagina: float = ALTURA_PAGINA_PADRAO
) -> tuple[bool, str]:
    """Guard A3 ancorado: só masthead/rodapé fora das bandas/fonte de expediente.

    Reusa :func:`harness.ingest.classificar._masthead`, que decide por **banda
    vertical** (topo/base da página) + fonte, e por rótulo de expediente — nunca
    por substring em qualquer posição. Isso evita apagar corpo legítimo como
    ``impressora`` (casa ``IMPRESSO``), ``Laserjet`` (casa ``LASER``) ou um dia da
    semana no meio de uma linha (R014).
    """
    regra = _masthead_b1(linha, altura_pagina=altura_pagina)
    if regra:
        return True, regra
    return False, ""


def _valor_tipo(enunciado: str) -> str:
    """``tipo`` canônico do ato; a extração fina de número/ano é o passo D."""
    token = token_canonico(enunciado)
    if token:
        return token
    primeira = re.split(r"\s+", enunciado.strip())[0] if enunciado.strip() else ""
    return primeira.strip(".,;:").upper()


# --------------------------------------------------------------------------- #
# Segmentação                                                                 #
# --------------------------------------------------------------------------- #


def _proveniencia_inicio(
    classe: LinhaClassificada, id_dom: str | None, gap_id: float | None, coluna: str
) -> MarcadorFronteira:
    evidencias: dict[str, object] = {
        "pagina": classe.linha.pagina,
        "y": classe.linha.y,
        "x0": classe.linha.x0,
        "coluna": coluna,
        "categoria_b1": classe.categoria,
        "regra_b1": classe.proveniencia.regra,
        "id_dom": id_dom,
    }
    if gap_id is not None:
        evidencias["gap_id"] = gap_id
    return MarcadorFronteira(
        tipo="inicio",
        regra="enunciado_b1",
        motivo=classe.proveniencia.motivo,
        evidencias=evidencias,
    )


def _proveniencia_fim(
    regra: str,
    motivo: str,
    *,
    pagina: int,
    y: float | None = None,
    proximo: str | None = None,
) -> MarcadorFronteira:
    evidencias: dict[str, object] = {"pagina": pagina}
    if y is not None:
        evidencias["y"] = y
    if proximo is not None:
        evidencias["proximo_enunciado"] = proximo
    return MarcadorFronteira(tipo="fim", regra=regra, motivo=motivo, evidencias=evidencias)


def segmentar_ordenadas(
    ordenadas: Sequence[LinhaClassificada],
    *,
    arquivo: str,
    edicao_numero: str,
    edicao_data: date | str,
    ano_edicao: int | None = None,
    alturas_pagina: Mapping[int, float] | None = None,
) -> list[DocumentoSegmentado]:
    """Segmenta uma sequência de linhas classificadas **já em ordem de leitura**.

    Regras principais: documento inicia em ``ENUNCIADO``; termina na linha
    anterior ao próximo ``ENUNCIADO``; ``ID:`` logo acima (mesma coluna) vira
    ``id_dom``; ``SECAO_ORGAO`` atualiza o ``orgao`` herdado; masthead/rodapé e
    datelines ficam fora do corpo (guard A3 ancorado por banda/fonte).
    ``alturas_pagina`` mapeia página → altura (pontos) para ancorar as bandas.
    """
    identificadores = [item for item in ordenadas if item.categoria == "ID"]
    alturas = alturas_pagina or {}
    # Primeira linha de corpo de cada (página, coluna): usada para detectar
    # continuação em minúscula/hífen na página seguinte (undershoot ambíguo).
    primeira_coluna: dict[tuple[int, str], str] = {}
    ignorar = {
        "ID",
        "MASTHEAD_RODAPE",
        "CABECALHO_TABELA",
        "CABECALHO_ANEXO",
        "SECAO_ORGAO",
        "ENUNCIADO",
    }
    for item in ordenadas:
        if item.categoria in ignorar:
            continue
        chave = (item.linha.pagina, _coluna(item.linha))
        primeira_coluna.setdefault(chave, item.linha.texto.strip())
    regioes_expediente = _regioes_expediente(ordenadas)
    usados: set[str] = set()
    documentos: list[DocumentoSegmentado] = []

    # Acumulador do documento aberto.
    atual: dict[str, object] | None = None
    orgao_atual: str | None = None
    seq = 0
    # D026: estado da atribuição de blocos de topo de coluna/página.
    # ``claim`` marca (página, coluna) já "reivindicada" por um ENUNCIADO;
    # ``alvo_bloco`` fixa o dono de um bloco já atribuído; ``aberto`` guarda o
    # documento cuja última linha está em cada (página, coluna).
    claim: set[tuple[int, str]] = set()
    alvo_bloco: dict[tuple[int, str], dict[str, object]] = {}
    aberto: dict[tuple[int, str], dict[str, object]] = {}

    # Acumuladores fechados, materializados só no fim: a cláusula primária D026
    # pode rotear linhas para um documento já fechado (R022, bloqueador 1).
    fechados: list[tuple[dict[str, object], LinhaClassificada | None, str, str]] = []

    def finalizar(proximo: LinhaClassificada | None, regra_fim: str, motivo_fim: str) -> None:
        nonlocal atual
        if atual is None:
            return
        fechados.append((atual, proximo, regra_fim, motivo_fim))
        atual = None

    def materializar(
        atual: dict[str, object],
        proximo: LinhaClassificada | None,
        regra_fim: str,
        motivo_fim: str,
    ) -> None:
        corpo: list[str] = atual["corpo"]  # type: ignore[assignment]
        pagina_ini = atual["pagina_inicio"]  # type: ignore[assignment]
        pagina_fim = atual["pagina_fim"]  # type: ignore[assignment]
        flags_set: set[str] = atual["flags"]  # type: ignore[assignment]
        if atual["ruido_masthead"]:  # type: ignore[index]
            flags_set.add("masthead_fantasma_no_corpo")
        if atual["fronteira_suspeita"]:  # type: ignore[index]
            flags_set.add("possivel_fronteira_perdida")
        if atual["continuacao_ambigua"]:  # type: ignore[index]
            flags_set.add("continuacao_ambigua")
        texto = "\n".join(parte for parte in corpo if parte).strip()
        if not texto:
            flags_set.add("corpo_ausente")
        if len(texto) > LIMITE_CHARS_ALERTA:
            flags_set.add("documento_gigante_a_verificar")
        if atual["id_dom"] is None:
            flags_set.add("id_dom_ausente")

        id_dom = atual["id_dom"]  # type: ignore[assignment]
        if id_dom:
            doc_id = f"{Path(arquivo).stem}#{int(str(id_dom)[10:14])}"
        else:
            doc_id = f"{Path(arquivo).stem}#{atual['seq']}"

        documento = Documento(
            doc_id=doc_id,
            arquivo=arquivo,
            edicao_numero=edicao_numero,
            edicao_data=edicao_data,
            pagina_inicio=pagina_ini,  # type: ignore[arg-type]
            pagina_fim=pagina_fim,  # type: ignore[arg-type]
            enunciado=atual["enunciado"],  # type: ignore[arg-type]
            tipo=atual["tipo"],  # type: ignore[arg-type]
            numero=None,
            ano=None,
            id_dom=id_dom,  # type: ignore[arg-type]
            orgao=atual["orgao"],  # type: ignore[arg-type]
            texto=texto,
            tabelas=[],
            entidades=[],
        )
        prox_enunciado = None
        if proximo is not None and proximo.categoria == "ENUNCIADO":
            prox_enunciado = _enunciado_de(proximo)
        if atual.get("estendido_d026"):
            # D026 estendeu um documento já fechado: o corpo passa do ENUNCIADO
            # que o fechara; o fim é a última linha da continuação (R023).
            regra_fim = "continuacao_d026"
            motivo_fim = (
                "fim estendido pela cláusula primária D026: bloco de topo de "
                "coluna/página continua a frase aberta deste documento"
            )
        marcador_fim = _proveniencia_fim(
            regra_fim,
            motivo_fim,
            pagina=pagina_fim,  # type: ignore[arg-type]
            y=atual.get("y_fim"),  # type: ignore[arg-type]
            proximo=prox_enunciado,
        )
        documentos.append(
            DocumentoSegmentado(
                documento=documento,
                flags=tuple(sorted(flags_set)),
                proveniencia=(
                    atual["proveniencia_inicio"],  # type: ignore[arg-type]
                    marcador_fim,
                ),
            )
        )

    for classe in ordenadas:
        if classe.categoria == "SECAO_ORGAO":
            orgao = _orgao_do_cabecalho(classe)
            if orgao:
                orgao_atual = orgao
            continue

        if classe.categoria == "ENUNCIADO":
            # Enunciado da OUTRA coluna pode fechar cedo um documento cuja frase
            # continua na coluna seguinte: é ambiguidade real → flag (R014).
            if (
                atual is not None
                and atual["coluna_inicio"] is not None
                and _coluna(classe.linha) != atual["coluna_inicio"]
                and not atual["fim_frase_inicio_col"]
            ):
                atual["continuacao_ambigua"] = True
            # Undershoot de mesma página: o documento termina "completo", mas a
            # página seguinte abre na mesma coluna com minúscula/hífen (provável
            # continuação) → incerteza (R014).
            if (
                atual is not None
                and atual["y_fim"] is not None
                and atual["fim_frase"]
                and atual["pagina_fim"] == classe.linha.pagina
            ):
                seguinte = primeira_coluna.get(
                    (atual["pagina_fim"] + 1, atual["coluna_inicio"])  # type: ignore[operator]
                )
                if seguinte and (seguinte[:1].islower() or seguinte[:1] in "-–—"):
                    atual["continuacao_ambigua"] = True
            # Fecha o documento anterior no próximo enunciado.
            if atual is not None:
                finalizar(
                    classe,
                    "proximo_enunciado",
                    "fronteira fechada pelo ENUNCIADO seguinte na ordem de leitura",
                )
            seq += 1
            id_dom, gap_id = _id_do_enunciado(classe, identificadores, usados)
            atual = {
                "seq": seq,
                "pagina_inicio": classe.linha.pagina,
                "pagina_fim": classe.linha.pagina,
                "enunciado": _enunciado_de(classe),
                "tipo": _valor_tipo(_enunciado_de(classe)),
                "id_dom": id_dom,
                "orgao": orgao_atual,
                "corpo": [],
                "flags": set(),
                "ruido_masthead": False,
                "fronteira_suspeita": False,
                "continuacao_ambigua": False,
                "proveniencia_inicio": _proveniencia_inicio(
                    classe, id_dom, gap_id, _coluna(classe.linha)
                ),
                "y_fim": None,
                "fim_frase": True,
                "coluna_fim": _coluna(classe.linha),
                "coluna_inicio": _coluna(classe.linha),
                "fim_frase_inicio_col": True,
            }
            claim.add((classe.linha.pagina, _coluna(classe.linha)))
            resto = _resto_apos_negrito(classe)
            if resto:
                atual["corpo"].append(resto)  # type: ignore[union-attr]
                atual["fim_frase"] = bool(_RE_FIM_FRASE.search(resto))
                atual["fim_frase_inicio_col"] = atual["fim_frase"]
                aberto[(classe.linha.pagina, _coluna(classe.linha))] = atual
            continue

        if atual is None:
            continue
        if classe.categoria in (
            "ID",
            "MASTHEAD_RODAPE",
            "CABECALHO_TABELA",
            "CABECALHO_ANEXO",
        ):
            continue

        texto = classe.linha.texto
        regiao = regioes_expediente.get(classe.linha.pagina)
        if regiao is not None and regiao[0] <= classe.linha.y < regiao[1]:
            # Expediente da p.2: não pertence a nenhum documento (§1.6).
            continue
        ruido, _motivo = _e_ruido_masthead(
            classe.linha,
            altura_pagina=alturas.get(classe.linha.pagina, ALTURA_PAGINA_PADRAO),
        )
        if ruido:
            atual["ruido_masthead"] = True  # type: ignore[index]
            continue
        # Possível fronteira perdida: negrito com token canônico classificado
        # como OUTRO (réguas do B1) dentro do corpo.
        if (
            classe.categoria == "OUTRO"
            and classe.linha.negrito_inicio
            and classe.proveniencia.regra == "token_generico_sem_reforco"
        ):
            atual["fronteira_suspeita"] = True  # type: ignore[index]
        coluna_linha = _coluna(classe.linha)
        chave_coluna = (classe.linha.pagina, coluna_linha)
        # D026 — cláusula PRIMÁRIA: bloco sem enunciado no topo de coluna/página
        # vai para o documento cuja FRASE ele continua (a linha começa em
        # minúscula/hífen e o candidato da mesma coluna, página anterior, está com
        # frase aberta). FALLBACK: documento anterior na ordem de leitura (atual).
        alvo = atual
        if chave_coluna not in claim:
            roteado = alvo_bloco.get(chave_coluna)
            if roteado is not None:
                alvo = roteado
            else:
                candidato = aberto.get((classe.linha.pagina - 1, coluna_linha))
                limpo = texto.strip()
                inicia_continuacao = bool(limpo) and (
                    limpo[0].islower() or limpo[0] in "-–—"
                )
                ultimo = (
                    candidato["corpo"][-1]
                    if candidato is not None and candidato["corpo"]
                    else ""
                )
                ultimo_atual = atual["corpo"][-1] if atual["corpo"] else ""
                if (
                    candidato is not None
                    and candidato is not atual
                    and inicia_continuacao
                    and not _frase_aberta(str(ultimo_atual))
                    and _frase_aberta(str(ultimo))
                ):
                    alvo = candidato
                    alvo_bloco[chave_coluna] = candidato
                    candidato["continuacao_ambigua"] = True  # type: ignore[index]
                    # O candidato já estava fechado: o fim dele passa a ser
                    # desta continuação, não do ENUNCIADO que o fechou (R023).
                    candidato["estendido_d026"] = True  # type: ignore[index]

        if classe.linha.pagina > alvo["pagina_fim"]:  # type: ignore[operator]
            # Transbordou de página: a atribuição do bloco de topo é ambígua —
            # ou a frase anterior não fechou (pode pertencer a outro artigo), ou
            # fechou e a nova linha abre outra oração (pode ser documento novo)
            # → flag de incerteza (R014).
            anterior = alvo["corpo"][-1] if alvo["corpo"] else ""  # type: ignore[index]
            primeira = texto.strip()[:1]
            if (anterior and not _RE_FIM_FRASE.search(str(anterior))) or (
                primeira.isalpha() and primeira.isupper()
            ):
                alvo["continuacao_ambigua"] = True  # type: ignore[index]
            alvo["pagina_fim"] = classe.linha.pagina
        alvo["corpo"].append(texto)  # type: ignore[union-attr]
        alvo["fim_frase"] = bool(_RE_FIM_FRASE.search(texto))
        alvo["coluna_fim"] = coluna_linha
        if coluna_linha == alvo["coluna_inicio"]:
            alvo["fim_frase_inicio_col"] = alvo["fim_frase"]
        alvo["y_fim"] = classe.linha.y
        aberto[chave_coluna] = alvo

    if atual is not None:
        finalizar(None, "fim_edicao", "sem ENUNCIADO seguinte: corpo vai até o fim da edição")
    for acumulador, proximo, regra_fim, motivo_fim in fechados:
        materializar(acumulador, proximo, regra_fim, motivo_fim)

    return documentos


# --------------------------------------------------------------------------- #
# Entrada por arquivo / diretório                                             #
# --------------------------------------------------------------------------- #


def edicao_do_nome(caminho: str | Path) -> tuple[str, str | None]:
    """Extrai ``(edicao_numero, edicao_data)`` do nome ``DOM<num>-<ddmmaaaa>…``.

    ``edicao_numero`` sai na forma canônica sem separador de milhar (``4196``) e
    ``edicao_data`` em ``ddmmaaaa`` (aceita pelo contrato) ou ``None``.
    """
    nome = Path(caminho).name
    achado = _RE_NOME_EDICAO.search(nome)
    if not achado:
        return "", None
    numero = re.sub(r"\D", "", achado.group("numero")).lstrip("0") or "0"
    return numero, achado.group("data")


def edicao_do_conteudo(caminho: str | Path) -> tuple[str | None, date | None]:
    """Extrai ``(edicao_numero, edicao_data)`` do masthead da página 1 (T027).

    Cobre ``Ano AAAA - Nº NNNN - DD de MÊS de AAAA`` e ``Anexo ao DOM nº N.NNN``
    (com a data por extenso em outra linha do topo). Usa a extração com o filtro
    anti-fantasma; devolve ``(None, None)`` para PDF sem páginas.
    """
    # Import tardio: parser puro de masthead (sem dependência de volta ao ingest).
    from harness.tools import masthead as _masthead  # noqa: PLC0415

    caminho = Path(caminho)
    alturas = alturas_do_pdf(caminho)
    if not alturas:
        return None, None
    linhas = extrair_linhas_visiveis(caminho, 1, 1)
    return _masthead.extrair_edicao_de_linhas(linhas, altura_pagina=alturas.get(1))


def segmentar_arquivo(
    caminho: str | Path,
    *,
    metodo_ordem: str = "propria",
    arquivo: str | None = None,
    edicao_numero: str | None = None,
    edicao_data: date | str | None = None,
    ano_edicao: int | None = None,
    aplicar_fantasma: bool = True,
    enriquecer: bool = True,
) -> list[DocumentoSegmentado]:
    """Segmenta um PDF: extrai (com A1), classifica (B1), ordena (C1) e agrupa.

    O filtro anti-fantasma (:mod:`harness.ingest.fantasma`) é aplicado **antes** de
    classificar/segmentar (T020/D018); ``aplicar_fantasma=False`` serve só para
    diagnóstico/paridade. Com ``enriquecer=True`` (padrão) roda também o pipeline
    D-meta-1/D-ent-1 (+NER) de :mod:`harness.ingest.pipeline`.
    """
    caminho = Path(caminho)
    numero_padrao, data_padrao = edicao_do_nome(caminho)
    numero = edicao_numero if edicao_numero is not None else numero_padrao
    data = edicao_data if edicao_data is not None else data_padrao
    if data is None or not numero:
        # Nome sem ``DOM<num>-<ddmmaaaa>`` (ex.: "ANEXO AO DOM3159-ASSINADO.pdf"):
        # lê número/data do masthead da página 1 (T027).
        numero_conteudo, data_conteudo = edicao_do_conteudo(caminho)
        numero = numero or numero_conteudo or ""
        data = data if data is not None else data_conteudo
    if data is None:
        raise ValueError(f"não foi possível obter a data da edição de {caminho.name!r}")
    if ano_edicao is None and isinstance(data, str) and len(data) == 8:
        ano_edicao = int(data[4:])
    elif ano_edicao is None and isinstance(data, date):
        ano_edicao = data.year

    alturas = alturas_do_pdf(caminho)
    linhas = extrair_linhas_visiveis(
        caminho, 1, len(alturas), aplicar_fantasma=aplicar_fantasma
    )
    classificadas = classificar_linhas(linhas)
    ordenadas = ordenar_leitura(classificadas, metodo=metodo_ordem, caminho=caminho)
    itens = segmentar_ordenadas(
        ordenadas,
        arquivo=arquivo if arquivo is not None else str(caminho),
        edicao_numero=numero,
        edicao_data=data,
        ano_edicao=ano_edicao,
        alturas_pagina=alturas,
    )
    if not enriquecer:
        return itens
    # Import tardio: ``pipeline`` importa ``segmentar`` no topo (evita ciclo).
    from harness.ingest import pipeline as _pipeline  # noqa: PLC0415

    return [_pipeline.enriquecer_segmentado(item) for item in itens]


def listar_pdfs(entrada: str | Path) -> list[Path]:
    """Resolve a entrada para uma lista ordenada de PDFs (arquivo ou diretório)."""
    caminho = Path(entrada)
    if caminho.is_dir():
        return sorted(p for p in caminho.iterdir() if p.suffix.lower() == ".pdf")
    if caminho.is_file():
        return [caminho]
    raise FileNotFoundError(f"entrada não encontrada: {caminho}")


def segmentar(
    entrada: str | Path,
    *,
    metodo_ordem: str = "propria",
    enriquecer: bool = True,
) -> list[DocumentoSegmentado]:
    """Segmenta um PDF ou todos os PDFs de um diretório (ordem alfabética)."""
    resultado: list[DocumentoSegmentado] = []
    for pdf in listar_pdfs(entrada):
        resultado.extend(
            segmentar_arquivo(pdf, metodo_ordem=metodo_ordem, enriquecer=enriquecer)
        )
    return resultado


def documentos(segmentados: Iterable[DocumentoSegmentado]) -> list[Documento]:
    """Projeta a lista de :class:`DocumentoSegmentado` nos contratos ``Documento``."""
    return [item.documento for item in segmentados]


# --------------------------------------------------------------------------- #
# Snapshot JSONL                                                              #
# --------------------------------------------------------------------------- #


def escrever_snapshot(segmentados: Iterable[DocumentoSegmentado], caminho: str | Path) -> Path:
    """Grava um JSONL de :class:`Documento` (formato consumido pelo gate)."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as arquivo:
        for item in segmentados:
            arquivo.write(item.documento.model_dump_json() + "\n")
    return caminho


def escrever_proveniencia(
    segmentados: Iterable[DocumentoSegmentado], caminho: str | Path
) -> Path:
    """Grava o *sidecar* de flags/proveniência (uma linha por documento)."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as arquivo:
        for item in segmentados:
            arquivo.write(json.dumps(item.para_dict(), ensure_ascii=False) + "\n")
    return caminho


def ler_snapshot(caminho: str | Path) -> list[Documento]:
    """Lê um JSONL de ``Documento`` (autochecagem e testes)."""
    documentos_lidos: list[Documento] = []
    with Path(caminho).open(encoding="utf-8") as arquivo:
        for numero, linha in enumerate(arquivo, start=1):
            linha = linha.strip()
            if not linha:
                continue
            try:
                documentos_lidos.append(Documento.model_validate_json(linha))
            except Exception as exc:  # noqa: BLE001 - contexto de linha no erro
                raise ValueError(f"{caminho}:{numero}: Documento inválido: {exc}") from exc
    return documentos_lidos


# --------------------------------------------------------------------------- #
# CLI autônoma (o subcomando ``harness ingest`` reusa a mesma implementação)  #
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m harness.ingest <pdf|dir> --saida snapshot.jsonl``.

    É a mesma rotina que o subcomando ``harness ingest`` deve chamar; fica aqui
    para permitir rodar a ingestão C1 sem depender da CLI raiz.
    """
    parser = argparse.ArgumentParser(
        prog="python -m harness.ingest",
        description="Ingestão C1: segmenta diários em documentos (snapshot JSONL).",
    )
    parser.add_argument(
        "entrada",
        type=Path,
        help="PDF do diário ou diretório com os PDFs (ex.: diarios/).",
    )
    parser.add_argument(
        "--saida",
        "-s",
        type=Path,
        required=True,
        help="Snapshot JSONL de Documento (ex.: dados/snapshots/lote.jsonl).",
    )
    parser.add_argument(
        "--ordem",
        choices=METODOS_ORDEM,
        default="propria",
        help="Ordem de leitura (padrão: propria).",
    )
    parser.add_argument(
        "--proveniencia",
        type=Path,
        default=None,
        help="Sidecar JSONL opcional com flags/proveniência.",
    )
    args = parser.parse_args(argv)

    try:
        segmentados = segmentar(args.entrada, metodo_ordem=args.ordem)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
        return 2

    caminho = escrever_snapshot(segmentados, args.saida)
    if args.proveniencia is not None:
        escrever_proveniencia(segmentados, args.proveniencia)
    print(f"ingest: {len(segmentados)} documento(s) -> {caminho}")
    por_arquivo: dict[str, int] = defaultdict(int)
    for item in segmentados:
        por_arquivo[item.documento.arquivo] += 1
    for arquivo, quantidade in sorted(por_arquivo.items()):
        print(f"  {Path(arquivo).name}: {quantidade}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

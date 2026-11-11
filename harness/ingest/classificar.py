"""B1 — classificação das linhas em negrito (primeiro passo da ingestão).

Recebe as **linhas lógicas** de :mod:`harness.tools.inspect` (``extrair_linhas_logicas``)
e diz, para cada linha em negrito, se ela é um **enunciado** (início de documento),
um cabeçalho de **seção/órgão**, um cabeçalho de **tabela**, **masthead/rodapé**,
um **ID** do diário ou **outro** (corpo em negrito, fragmento de título, fantasma).

A árvore de decisão é determinística e **explicável**: cada linha classificada
carrega uma :class:`Proveniencia` com a regra que decidiu e as evidências usadas
(fonte, tamanho, ``y``/``x``, token canônico, reforço de ID, flag de fantasma).

Regras seguidas (``knowledge/regras_diario.md`` §1 e ``dados/catalogo_padroes_v1.md`` §1):

1. O enunciado é negrito no **início** da linha lógica (aqui só classificamos
   linhas com ``negrito_inicio`` True).
2. Lista canônica fechada no código, com os compostos (``TERMO ADITIVO``,
   ``RESOLUÇÃO ADMINISTRATIVA``, ``AVISO DE CREDENCIAMENTO``) e **títulos com
   ordinal** (``PRIMEIRO TERMO ADITIVO``); casamento do token canônico por
   **prefixo de palavra** (aceita flexão ``-S``/``-ES``), insensível a caixa/acento.
3. Tokens **genéricos** (``RELATÓRIO``/``PROCESSO``/``ATO``/``DECISÃO``/``TERMO``)
   **não decidem sozinhos**: exigem reforço (composto canônico na própria linha ou
   um ``ID:`` na mesma coluna, logo acima). Isso evita que ``RELATÓRIO DE ENTREGAS``
   ou ``PROCESSO SELETIVO`` em Bold 8 virem documento.
4. ``ID:`` não é fronteira (D011); aqui ele é apenas **reforço**/verificação.
5. Cabeçalho de tabela/seção/masthead nunca vira enunciado.
6. **Título duplicado fantasma** (ex.: ``198 p4 y=770.2``, com ``TERMO ADITIVO``
   repetido no fim da linha) é detectado e sinalizado; o texto é cortado na primeira
   cópia para casar com o enunciado anotado.

Limitações registradas:

- O contrato :class:`harness.contracts.Span` não expõe os ``flags`` crus (bit 2 =
  serifado); usamos o nome da fonte (``Times``/``Helvetica``/``BoldItal``) e o
  tamanho, como no catálogo §1.
- A fusão de fragmentos de título (H4) usa um limiar fixo calibrado nos 3 PDFs
  (lacuna de fragmento ≤ 9,0 pt; gutter de coluna ≥ 10,0 pt). Está isolado em
  :data:`LIMIAR_FUSAO_PT` para recalibração futura.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Sequence

try:  # PyMuPDF >= 1.24 expõe ``pymupdf``; versões antigas só ``fitz``.
    import pymupdf
except ImportError:  # pragma: no cover - compatibilidade com versões antigas
    import fitz as pymupdf  # type: ignore[no-redef]

from harness.contracts import LinhaLogica
from harness.tools.inspect import extrair_linhas_logicas

__all__ = [
    "Categoria",
    "COMPOSTOS_REFORCO",
    "GAP_ID_MAX_PT",
    "LIMIAR_FUSAO_PT",
    "LinhaClassificada",
    "ORDINAIS",
    "Proveniencia",
    "TOKENS_CANONICOS",
    "TOKENS_GENERICOS",
    "classificar_arquivo",
    "classificar_linha",
    "classificar_linhas",
    "detectar_titulo_duplicado",
    "e_id",
    "enunciados",
    "normalizar",
    "textos_repetidos",
    "token_canonico",
]

#: Classificação de uma linha em negrito.
Categoria = Literal[
    "ENUNCIADO",
    "SECAO_ORGAO",
    "CABECALHO_TABELA",
    "CABECALHO_ANEXO",
    "MASTHEAD_RODAPE",
    "ID",
    "OUTRO",
]

#: Lista canônica de tokens que podem abrir documento (sem acento, maiúsculas).
TOKENS_CANONICOS: tuple[str, ...] = (
    "LEI",
    "DECRETO",
    "PORTARIA",
    "PLANO",
    "TERMO",
    "CONTRATO",
    "EXTRATO",
    "EDITAL",
    "ERRATA",
    "AVISO",
    "ATA",
    "ATO",
    "APOSTILAMENTO",
    "ADITIVO",
    "NOTIFICACAO",
    "SUSPENSAO",
    "RESOLUCAO",
    "RETIFICACAO",
    "REGULAMENTO",
    "RATIFICACAO",
    "REPUBLICACAO",
    "INEXIGIBILIDADE",
    "DISPENSA",
    "PROCESSO",
    "DECISAO",
    "JUSTIFICATIVA",
    "RELATORIO",
    "APOSTILA",
    "COMUNICADO",
    "ACORDAO",
    "INSTRUCAO",
)

#: Tokens canônicos que também são rótulos comuns de tabela/corpo. Pela regra do
#: usuário §1.3 / D020 eles **valem sozinhos** como ENUNCIADO; o que os afasta de
#: documento é CONTEXTO (cabeçalho repetido entre páginas, região/fonte de
#: tabela), nunca a ausência de ``ID:``.
TOKENS_GENERICOS: frozenset[str] = frozenset(
    {"RELATORIO", "PROCESSO", "ATO", "DECISAO", "TERMO"}
)

#: Compostos que reforçam um token genérico (o composto em si é forte).
COMPOSTOS_REFORCO: tuple[tuple[str, ...], ...] = (
    ("TERMO", "ADITIVO"),
    ("TERMO", "DE", "AJUSTE"),
    ("ATO", "DE", "DISPENSA"),
    ("DECISAO", "ADMINISTRATIVA"),
)

#: Ordinais por extenso que podem anteceder o token canônico (``PRIMEIRO TERMO ADITIVO``).
ORDINAIS: frozenset[str] = frozenset(
    {
        "PRIMEIRO",
        "SEGUNDO",
        "TERCEIRO",
        "QUARTO",
        "QUINTO",
        "SEXTO",
        "SETIMO",
        "OITAVO",
        "NONO",
        "DECIMO",
        "VIGESIMO",
        "TRIGESIMO",
        "QUADRAGESIMO",
        "QUINQUAGESIMO",
        "CENTESIMO",
    }
)

#: Lacuna máxima (pt) entre fragmentos do MESMO título na mesma linha visual.
#: Medido nos 3 PDFs: fragmentos H4 chegam a 8,1 pt; o gutter de coluna é ≥ 10,0 pt.
LIMIAR_FUSAO_PT: float = 9.0

#: Tolerância (pt) de ``y`` para considerar duas linhas na mesma linha visual.
TOL_Y_FUSAO_PT: float = 1.5

#: Distância vertical máxima (pt) para a continuação de título na linha de baixo (H3).
#: A altura de linha medida nos 3 PDFs é ≈ 9,4–9,8 pt.
LIMIAR_CONTINUACAO_PT: float = 15.0

#: Distância vertical máxima (pt) entre um ``ID:`` e o enunciado reforçado.
GAP_ID_MAX_PT: float = 35.0

#: Tolerância (pt) ao testar se duas faixas de ``x`` pertencem à mesma coluna.
TOL_X_PT: float = 1.0

#: Contexto de anexo: cabeçalho em negrito repetido em >= N páginas distintas,
#: com >= N tokens, é cabeçalho de anexo/tabela — não abre documento (D020).
MIN_PAGINAS_REPETIDO: int = 3
MIN_TOKENS_REPETIDO: int = 2

#: Frações da altura da página que definem as bandas de masthead (topo) e de
#: rodapé/expediente (base). Substituem o ``y < 60`` calibrado no dev.
FRACAO_BANDA_TOPO: float = 0.10
FRACAO_BANDA_BASE: float = 0.21
#: Altura assumida quando o chamador não informa a geometria (A4 do DOM).
ALTURA_PAGINA_PADRAO: float = 841.89
#: Tamanho máximo (pt) da fonte de expediente (Times Bold ~6–7 pt); evita ancorar
#: rodapé só por substring de título.
TAMANHO_EXPEDIENTE_PT: float = 7.0

#: Limiares relativos ao tamanho da fonte (anti-overfit): para 8 pt recaem nos
#: valores já calibrados no dev (9,0 / 15,0 / 1,5), mas escalam com a fonte.
FATOR_LIMIAR_FUSAO: float = 1.10
FATOR_LIMIAR_CONTINUACAO: float = 1.80
FATOR_TOL_Y: float = 0.18

_RE_ID = re.compile(r"^ID:\s*\d{18}$")
_RE_SIGLA = re.compile(r"^¨.*¨$")
_RE_DIA_SEMANA = re.compile(
    r"\b(SEGUNDA|TERCA|QUARTA|QUINTA|SEXTA|SABADO|DOMINGO)-FEIRA\b"
)
_RE_ANO_MASTHEAD = re.compile(r"^ANO\s+\d{4}\b")
_RE_SO_DIGITOS = re.compile(r"^\d+$")
#: Rótulos inequívocos de expediente/rodapé (sem acento). Note que NÃO usamos
#: ``DIARIO OFICIAL`` (genérico, casa seções legítimas como "Diário Oficial da
#: Câmara"); usamos a forma do expediente ``DIARIO OFICIAL DO MUNICIPIO``.
_ROTULOS_EXPEDIENTE: tuple[str, ...] = (
    "IMPRESSO",
    "PRODATER",
    "LASER",
    "ASSINATURA DIGITAL",
    "ORGAO DE COMUNICACAO OFICIAL",
    "CADERNO UNICO",
    "GERENCIA DE IMPRENSA OFICIAL",
    # forma do expediente (com `` - TERESINA``): não casar a citação de corpo
    # ``Diário Oficial do Município, DOM - Teresina ...`` (4197#23).
    "DIARIO OFICIAL DO MUNICIPIO - TERESINA",
)
#: Rótulos que só contam quando a linha é EXATAMENTE o expediente (não corpo que
#: começa com o mesmo texto, ex.: ``Secretaria Municipal de Administração e
#: Recursos Humanos. O SECRETÁRIO ...``).
_ROTULOS_EXATOS: frozenset[str] = frozenset({"SECRETARIA MUNICIPAL DE ADMINISTRACAO"})
#: Título no topo da página (``DOM - Teresina ...``).
_RE_DOM = "DOM - TERESINA"

#: Famílias sem serifa conhecidas (cabeçalho de tabela/diagramação), ampliadas
#: além de Helvetica/Arial para não depender do dev.
_FAMILIAS_SANS: tuple[str, ...] = (
    "HELVETICA",
    "ARIAL",
    "CALIBRI",
    "OPENSANS",
    "DEJAVUSANS",
    "LIBERATIONSANS",
    "ROBOTO",
    "SEGOE",
    "TAHOMA",
    "VERDANA",
    "NOTOSANS",
    "SANS",
)

#: Conectores que denunciam título quebrado na linha de baixo (regras §1.1).
_CONECTORES_FIM: frozenset[str] = frozenset(
    {
        "A",
        "AO",
        "AOS",
        "AS",
        "COM",
        "DA",
        "DAS",
        "DE",
        "DO",
        "DOS",
        "E",
        "EM",
        "NA",
        "NAS",
        "NO",
        "NOS",
        "O",
        "OS",
        "PARA",
        "POR",
        "SEM",
        "SOB",
    }
)


# --------------------------------------------------------------------------- #
# Normalização e tokens                                                       #
# --------------------------------------------------------------------------- #


def _sem_acento(texto: str) -> str:
    decomposto = unicodedata.normalize("NFKD", texto or "")
    return "".join(c for c in decomposto if not unicodedata.combining(c))


def normalizar(texto: str) -> str:
    """Maiúsculas, sem acento e com espaços colapsados (base do casamento)."""
    return re.sub(r"\s+", " ", _sem_acento(texto or "").upper()).strip()


def _palavra(texto: str) -> str:
    """Só as letras da palavra (remove pontuação/números), em maiúsculas."""
    return re.sub(r"[^A-Z]", "", normalizar(texto))


def _e_ordinal(token: str) -> bool:
    """True para ordinal por extenso (``PRIMEIRO``) ou numérico (``1º``/``1.º``).

    O ordinal numérico exige o indicador ``º``/``°`` (que ``normalizar`` vira
    ``O``): ``1º``, ``1.º``, ``10º``. Um número solto (``10``) **não** é ordinal
    (antes ``\\d{1,3}[º°O.]*`` aceitava qualquer número).
    """
    bruto = normalizar(token)
    if re.fullmatch(r"\d{1,3}\s*\.?\s*[Oº°]", bruto):
        return True
    return _palavra(bruto) in ORDINAIS


def _casa_token(palavra: str, token: str) -> bool:
    """Casamento por prefixo de palavra (aceita a flexão simples ``-S``/``-ES``).

    Aceita o token canônico **colado a um órgão por barra** (``RESOLUÇÃO/CMAS``,
    ``PORTARIA/SEMEC``): compara só a parte antes da barra (T024 item 3).
    """
    limpa = _palavra(palavra.split("/")[0])
    return limpa == token or limpa in (f"{token}S", f"{token}ES")


def token_canonico(texto: str) -> str | None:
    """Token canônico que abre a linha (por prefixo de palavra) ou ``None``.

    Também aceita **ordinal + token** (``PRIMEIRO TERMO ADITIVO`` → ``''TERMO''``)
    e ordinal numérico (``1º TERMO ADITIVO``).
    """
    tokens = normalizar(texto).split()
    if not tokens:
        return None
    for token in TOKENS_CANONICOS:
        if _casa_token(tokens[0], token):
            return token
    if _e_ordinal(tokens[0]):
        for bruto in tokens[1:4]:
            for token in TOKENS_CANONICOS:
                if _casa_token(bruto, token):
                    return token
    return None


def _tem_composto(tokens: Sequence[str]) -> tuple[str, ...] | None:
    """Composto de reforço no início da linha (``TERMO ADITIVO``), se houver.

    Pula ordinais iniciais (``PRIMEIRO``/``QUARTO``/``DÉCIMO QUARTO``) para que
    ``PRIMEIRO TERMO ADITIVO`` também seja reconhecido como composto.
    """
    inicio = 0
    if tokens and _e_ordinal(tokens[0]):
        inicio = 1
        if inicio < len(tokens) and _e_ordinal(tokens[inicio]):
            inicio += 1
    palavras = [_palavra(token) for token in tokens[inicio:]]
    for composto in COMPOSTOS_REFORCO:
        if tuple(palavras[: len(composto)]) == composto:
            return composto
    return None


def e_id(texto: str) -> bool:
    """True se o texto é um ``ID:`` de 18 dígitos (linha própria)."""
    return bool(_RE_ID.match(texto.strip()))


def detectar_titulo_duplicado(texto: str) -> tuple[bool, str]:
    """Detecta título duplicado colado (fantasma) e devolve ``(flag, texto_cortado)``.

    Caso real: ``198 p4 y=770.2`` →
    ``TERMO ADITIVO Nº 003/2025 - GAB/PREFEITO TERMO ADITIVO``.
    O corte remove a cópia final, deixando o título canônico para o casamento.
    """
    tokens = texto.split()
    n = len(tokens)
    for k in range(2, min(6, n // 2) + 1):
        if tokens[:k] == tokens[-k:]:
            return True, " ".join(tokens[: n - k]).strip()
    metade = n // 2
    if n >= 4 and n % 2 == 0 and tokens[:metade] == tokens[metade:]:
        return True, " ".join(tokens[:metade]).strip()
    return False, texto


# --------------------------------------------------------------------------- #
# Estruturas de resultado                                                     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Proveniencia:
    """Qual regra decidiu a categoria e com que evidências."""

    regra: str
    motivo: str
    evidencias: Mapping[str, object] = field(default_factory=dict)

    def para_dict(self) -> dict[str, object]:
        """Versão serializável (``dict``) para logs/testes."""
        return {
            "regra": self.regra,
            "motivo": self.motivo,
            "evidencias": dict(self.evidencias),
        }


@dataclass(frozen=True)
class LinhaClassificada:
    """Uma linha lógica com categoria e proveniência."""

    linha: LinhaLogica
    categoria: Categoria
    proveniencia: Proveniencia
    texto_fundido: str | None = None
    duplicata: bool = False

    @property
    def enunciado(self) -> str:
        """Texto que representa o enunciado (título fundido quando houver)."""
        return self.texto_fundido or self.linha.texto

    def para_dict(self) -> dict[str, object]:
        """Projeção serializável da classificação (sem os spans)."""
        return {
            "pagina": self.linha.pagina,
            "y": self.linha.y,
            "x0": self.linha.x0,
            "x1": self.linha.x1,
            "texto": self.linha.texto,
            "categoria": self.categoria,
            "texto_fundido": self.texto_fundido,
            "duplicata": self.duplicata,
            "proveniencia": self.proveniencia.para_dict(),
        }


# --------------------------------------------------------------------------- #
# Evidências auxiliares                                                       #
# --------------------------------------------------------------------------- #


def _fonte_tamanho(linha: LinhaLogica) -> tuple[str, float]:
    """Fonte e tamanho do primeiro span (a assinatura que classifica a linha)."""
    if not linha.spans:
        return "", 0.0
    span = linha.spans[0]
    return span.fonte, float(span.tamanho)


def _e_fonte_sans(fonte: str) -> bool:
    """True se o nome da fonte pertence a família sem serifa (tabela/diagramação).

    Generaliza ``("Helvetica", "Arial")`` para outras famílias sans (Calibri,
    OpenSans, DejaVuSans, ...) — anti-overfit às 3 edições.
    """
    nome = normalizar(fonte).replace(" ", "")
    return any(familia in nome for familia in _FAMILIAS_SANS)


def _e_bold_fonte(fonte: str) -> bool:
    """True se o nome da fonte indica negrito (``Bold``/``Black``/...)."""
    nome = normalizar(fonte)
    return any(p in nome for p in ("BOLD", "BLACK", "SEMIBOLD", "HEAVY"))


def _e_bold_italico(fonte: str) -> bool:
    """True para itálico em negrito, aceitando variantes com espaços.

    ``TimesNewRomanPS-BoldItal`` e ``Times New Roman Bold Italic`` casam; antes
    só a substring ``BoldItal`` funcionava.
    """
    nome = normalizar(fonte)
    return _e_bold_fonte(fonte) and ("ITAL" in nome or "OBLIQUE" in nome)


def _limiar_fusao(tamanho: float) -> float:
    """Lacuna máxima de fusão de fragmentos, relativa ao tamanho da fonte."""
    return max(LIMIAR_FUSAO_PT, FATOR_LIMIAR_FUSAO * tamanho)


def _tol_y(tamanho: float) -> float:
    """Tolerância de ``y`` para a mesma linha visual, relativa à fonte."""
    return max(TOL_Y_FUSAO_PT, FATOR_TOL_Y * tamanho)


def _limiar_continuacao(tamanho: float) -> float:
    """Distância máxima da continuação vertical (H3), relativa à fonte."""
    return max(LIMIAR_CONTINUACAO_PT, FATOR_LIMIAR_CONTINUACAO * tamanho)


def _evidencias(linha: LinhaLogica, **extras: object) -> dict[str, object]:
    fonte, tamanho = _fonte_tamanho(linha)
    base: dict[str, object] = {
        "fonte": fonte,
        "tamanho": round(tamanho, 2),
        "y": linha.y,
        "x0": linha.x0,
        "x1": linha.x1,
    }
    base.update(extras)
    return base


def _enriquecer_duplicata(
    proveniencia: Proveniencia, duplicata: bool, texto: str
) -> Proveniencia:
    """Acrescenta à proveniência o corte do título duplicado fantasma, se houver."""
    if not duplicata:
        return proveniencia
    _, texto_limpo = detectar_titulo_duplicado(texto)
    evidencias = dict(proveniencia.evidencias)
    evidencias["duplicata"] = True
    evidencias["texto_sem_duplicata"] = texto_limpo
    return Proveniencia(proveniencia.regra, proveniencia.motivo, evidencias)


def _mesma_coluna(linha: LinhaLogica, alvo: LinhaLogica) -> bool:
    """True se as faixas de ``x`` se sobrepõem (tolerância :data:`TOL_X_PT`)."""
    return not (
        linha.x1 < alvo.x0 - TOL_X_PT or alvo.x1 < linha.x0 - TOL_X_PT
    )


def _termina_em_conector(texto: str) -> bool:
    """True se o texto termina em hífen ou conector (título quebrado na linha de baixo)."""
    ultimo = texto.rstrip()
    if not ultimo:
        return False
    if ultimo.endswith(("-", "–", "—")):
        return True
    return _palavra(ultimo.split()[-1]) in _CONECTORES_FIM


def _reforco_id(
    linha: LinhaLogica, ids: Sequence[LinhaLogica]
) -> tuple[LinhaLogica, float] | None:
    """``ID:`` logo acima, na mesma coluna (reforço de token genérico)."""
    candidatos = [
        identificador
        for identificador in ids
        if identificador.pagina == linha.pagina
        and 0 < linha.y - identificador.y <= GAP_ID_MAX_PT
        and _mesma_coluna(linha, identificador)
    ]
    if not candidatos:
        return None
    mais_proximo = min(candidatos, key=lambda identificador: linha.y - identificador.y)
    return mais_proximo, round(linha.y - mais_proximo.y, 1)


# --------------------------------------------------------------------------- #
# Árvore de decisão                                                           #
# --------------------------------------------------------------------------- #


def _masthead(
    linha: LinhaLogica, *, altura_pagina: float = ALTURA_PAGINA_PADRAO
) -> str | None:
    """Regra de masthead/rodapé ancorada em banda vertical + fonte (seção 1.2).

    - Topo: ``y`` na primeira :data:`FRACAO_BANDA_TOPO` da página, com as linhas
      fixas ``DOM - Teresina``, dia da semana ou nº de página.
    - Base: ``y`` na última :data:`FRACAO_BANDA_BASE`, com rótulo inequívoco de
      expediente.
    - Fonte de expediente (Times Bold de até :data:`TAMANHO_EXPEDIENTE_PT`) também
      ancora o rodapé, independentemente da banda (o expediente real fica em
      ~6–7 pt; o enunciado é 8 pt).

    Não usa ``search`` global de ``DIARIO OFICIAL``: isso classificava a seção
    legítima ``Diário Oficial da Câmara`` como masthead.
    """
    texto = normalizar(linha.texto)
    fonte, tamanho = _fonte_tamanho(linha)
    topo = linha.y <= FRACAO_BANDA_TOPO * altura_pagina
    base = linha.y >= (1.0 - FRACAO_BANDA_BASE) * altura_pagina

    if topo and (
        texto.startswith(_RE_DOM)
        or _RE_DIA_SEMANA.match(texto)
        or _RE_SO_DIGITOS.match(texto)
    ):
        return "masthead_topo"
    if _RE_ANO_MASTHEAD.match(texto):
        return "masthead_ano"

    # Rodapé/expediente: linha que COMEÇA por um rótulo inequívoco, OU linha
    # EXATA de expediente, OU rótulo em fonte de expediente (Times Bold ≤ ~7 pt).
    # Não basta a substring em qualquer posição: corpo legítimo como "na
    # publicação do Diário Oficial do Município" (4198#51) e "(quinta-feira) -
    # Semana Santa" (4196#83) não são removidos.
    rotulo_inicio = texto in _ROTULOS_EXATOS or any(
        texto.startswith(padrao) for padrao in _ROTULOS_EXPEDIENTE
    )
    fonte_expediente = _e_bold_fonte(fonte) and tamanho <= TAMANHO_EXPEDIENTE_PT
    if rotulo_inicio and (topo or base or fonte_expediente):
        return "masthead_rodape"
    if fonte_expediente and any(padrao in texto for padrao in _ROTULOS_EXPEDIENTE):
        return "masthead_rodape"
    return None


def _classificar_negrito(
    linha: LinhaLogica,
    texto: str,
    ids: Sequence[LinhaLogica],
    *,
    repetidos: frozenset[str] = frozenset(),
    altura_pagina: float = ALTURA_PAGINA_PADRAO,
) -> tuple[Categoria, Proveniencia, bool]:
    """Aplica a árvore de decisão a uma linha em negrito.

    ``texto`` é o texto efetivo (título fundido, quando houver); ``repetidos``
    são textos normalizados que aparecem em várias páginas (contexto de anexo).
    O retorno inclui a flag de duplicata fantasma detectada.
    """
    fonte, tamanho = _fonte_tamanho(linha)
    duplicata, texto_limpo = detectar_titulo_duplicado(texto)

    # 1. ID do diário (metadado, nunca fronteira — D011).
    if e_id(linha.texto):
        return (
            "ID",
            Proveniencia(
                "id_18_digitos",
                "texto casa 'ID: <18 dígitos>'; é metadado (D011), não documento",
                _evidencias(linha),
            ),
            duplicata,
        )

    # 2. Sigla de órgão ¨...¨ (rotacionada ou horizontal).
    if _RE_SIGLA.match(linha.texto.strip()):
        return (
            "SECAO_ORGAO",
            Proveniencia(
                "orgao_sigla",
                "texto entre '¨...¨' (sigla de órgão); alimenta orgao, não é documento",
                _evidencias(linha),
            ),
            duplicata,
        )

    # 3. Cabeçalho de seção/órgão por extenso (BoldItal 10/14/20). Vem ANTES do
    # masthead para que "Diário Oficial da Câmara" (BoldItal) não vire rodapé.
    if _e_bold_italico(fonte) and tamanho >= 9.5:
        return (
            "SECAO_ORGAO",
            Proveniencia(
                "secao_orgao_boldital",
                "fonte itálica em negrito ≥ 9,5 pt: cabeçalho de seção/órgão por extenso",
                _evidencias(linha),
            ),
            duplicata,
        )

    # 4. Masthead / rodapé / nº de página (banda vertical + fonte).
    regra_masthead = _masthead(linha, altura_pagina=altura_pagina)
    if regra_masthead:
        return (
            "MASTHEAD_RODAPE",
            Proveniencia(
                regra_masthead,
                "linha fixa de topo/expediente/rodapé; não participa da segmentação",
                _evidencias(linha, texto=linha.texto),
            ),
            duplicata,
        )

    # 5. Cabeçalho de tabela (fonte sem serifa fora da lista ou tamanho < 6 pt).
    if _e_fonte_sans(fonte) or tamanho < 6.0:
        return (
            "CABECALHO_TABELA",
            Proveniencia(
                "cabecalho_tabela_fonte_pequena",
                "fonte sem serifa ou tamanho < 6 pt: cabeçalho de tabela",
                _evidencias(linha),
            ),
            duplicata,
        )

    # 6. Enunciado por token canônico (com ordinal) na linha/fusão.
    token = token_canonico(texto_limpo)
    if token:
        tokens = normalizar(texto_limpo).split()
        composto = _tem_composto(tokens)
        if token not in TOKENS_GENERICOS:
            return (
                "ENUNCIADO",
                Proveniencia(
                    "enunciado_token_canonico",
                    f"linha começa pelo token canônico forte {token!r}",
                    _evidencias(linha, token=token, ordinal=_e_ordinal(tokens[0])),
                ),
                duplicata,
            )
        if composto:
            return (
                "ENUNCIADO",
                Proveniencia(
                    "enunciado_composto",
                    f"token genérico {token!r} reforçado pelo composto {' '.join(composto)!r}",
                    _evidencias(linha, token=token, composto=" ".join(composto)),
                ),
                duplicata,
            )
        # D020: genérico sozinho (ou com complemento) é ENUNCIADO. O que o afasta
        # de documento é CONTEXTO: cabeçalho repetido entre páginas (anexo) ou
        # fonte/região de tabela (tratada acima) — nunca a ausência de ID.
        chave = normalizar(texto_limpo)
        if chave in repetidos:
            return (
                "CABECALHO_ANEXO",
                Proveniencia(
                    "anexo_cabecalho_repetido",
                    (
                        f"token genérico {token!r} repetido em várias páginas: "
                        "cabeçalho de anexo/tabela, não documento (D020)"
                    ),
                    _evidencias(linha, token=token),
                ),
                duplicata,
            )
        evidencias: dict[str, object] = {"token": token}
        reforco = _reforco_id(linha, ids)
        if reforco:
            identificador, gap = reforco
            evidencias["id_acima"] = identificador.texto
            evidencias["gap_id"] = gap
        return (
            "ENUNCIADO",
            Proveniencia(
                "enunciado_generico",
                (
                    f"token genérico {token!r} abre documento sozinho (§1.3/D020); "
                    "ID é opcional e não decide a fronteira"
                ),
                _evidencias(linha, **evidencias),
            ),
            duplicata,
        )

    # 7. Negrito que não é nenhuma das categorias acima (corpo, fragmento, fantasma).
    return (
        "OUTRO",
        Proveniencia(
            "negrito_sem_regra",
            "negrito sem token canônico, sem ID/sigla/seção/tabela associados",
            _evidencias(linha),
        ),
        duplicata,
    )


# --------------------------------------------------------------------------- #
# Fusão de fragmentos e continuação de título                                 #
# --------------------------------------------------------------------------- #


def _achar_continuacao_vertical(
    linha: LinhaLogica,
    processados: Sequence[int],
    linhas: Sequence[LinhaLogica],
    lider_de: Mapping[int, int],
    texto_fundido: Mapping[int, str],
) -> int | None:
    """Índice do título (líder) cuja continuação é ``linha``, se houver (H3).

    A linha de baixo só continua um título quando o texto acumulado do líder
    termina em hífen/conector (``DO``/``DE``/``AO``...), está na mesma coluna e
    não passa de :data:`LIMIAR_CONTINUACAO_PT` pt abaixo.
    """
    for anterior in reversed(processados):
        vizinha = linhas[anterior]
        if vizinha.pagina != linha.pagina:
            break
        gap = linha.y - vizinha.y
        _, tamanho_vizinho = _fonte_tamanho(vizinha)
        if gap <= 0 or gap > _limiar_continuacao(tamanho_vizinho):
            continue
        if not _mesma_coluna(linha, vizinha):
            continue
        lider = lider_de.get(anterior, anterior)
        acumulado = texto_fundido.get(lider, vizinha.texto)
        if token_canonico(acumulado) is None:
            continue
        if _termina_em_conector(acumulado):
            return anterior
    return None


def _mapear_grupos(
    linhas: Sequence[LinhaLogica],
) -> tuple[dict[int, int], dict[int, str]]:
    """Agrupa fragmentos (H4) e continuações (H3) em títulos por líder.

    Retorna ``(lider_de, texto_fundido)``: ``lider_de[i]`` é o índice do início do
    grupo e ``texto_fundido[inicio]`` o título completo reconstruído.
    """
    indices = [i for i, linha in enumerate(linhas) if linha.negrito_inicio]
    ordenados = sorted(
        indices, key=lambda i: (linhas[i].pagina, linhas[i].y, linhas[i].x0)
    )
    lider_de: dict[int, int] = {}
    texto_fundido: dict[int, str] = {}
    processados: list[int] = []
    for indice in ordenados:
        linha = linhas[indice]
        lider = indice
        # H4: fragmento na MESMA linha visual (justificação partiu o título).
        if processados:
            anterior = processados[-1]
            vizinha = linhas[anterior]
            fonte_a, tamanho_a = _fonte_tamanho(vizinha)
            fonte, tamanho = _fonte_tamanho(linha)
            mesma_visual = (
                vizinha.pagina == linha.pagina
                and abs(vizinha.y - linha.y) <= _tol_y(tamanho)
                and fonte_a == fonte
                and abs(tamanho_a - tamanho) <= 0.5
                and vizinha.x1 <= linha.x0
                and (linha.x0 - vizinha.x1) <= _limiar_fusao(tamanho)
            )
            if mesma_visual:
                lider = lider_de.get(anterior, anterior)
        # H3: continuação do título na linha de baixo (hífen/conector).
        if lider == indice:
            anterior = _achar_continuacao_vertical(
                linha, processados, linhas, lider_de, texto_fundido
            )
            if anterior is not None:
                lider = lider_de.get(anterior, anterior)
        lider_de[indice] = lider
        if lider == indice:
            texto_fundido[indice] = linha.texto
        else:
            acumulado = f"{texto_fundido.get(lider, '')} {linha.texto}"
            texto_fundido[lider] = re.sub(r"\s+", " ", acumulado).strip()
        processados.append(indice)
    return lider_de, texto_fundido


# --------------------------------------------------------------------------- #
# API pública                                                                 #
# --------------------------------------------------------------------------- #


def textos_repetidos(linhas: Sequence[LinhaLogica]) -> frozenset[str]:
    """Textos normalizados que aparecem em >= :data:`MIN_PAGINAS_REPETIDO` páginas.

    Usados como CONTEXTO de anexo: um cabeçalho em negrito repetido entre páginas
    é cabeçalho de anexo/tabela, não abre documento (D020). Exige
    >= :data:`MIN_TOKENS_REPETIDO` tokens para não afetar tokens genéricos
    legítimos e repetidos (ex.: ``EXTRATO``).
    """
    paginas_por_texto: dict[str, set[int]] = defaultdict(set)
    for linha in linhas:
        if not linha.negrito_inicio:
            continue
        chave = normalizar(linha.texto)
        if len(chave.split()) >= MIN_TOKENS_REPETIDO:
            paginas_por_texto[chave].add(linha.pagina)
    return frozenset(
        chave
        for chave, paginas in paginas_por_texto.items()
        if len(paginas) >= MIN_PAGINAS_REPETIDO
    )


def classificar_linha(
    linha: LinhaLogica,
    *,
    ids: Sequence[LinhaLogica] = (),
    texto: str | None = None,
    altura_pagina: float = ALTURA_PAGINA_PADRAO,
) -> LinhaClassificada:
    """Classifica uma única linha lógica (sem contexto de fusão/repetição)."""
    if not linha.negrito_inicio:
        return LinhaClassificada(
            linha=linha,
            categoria="OUTRO",
            proveniencia=Proveniencia(
                "linha_nao_negrito",
                "a linha não começa em negrito; não pode abrir documento",
                _evidencias(linha),
            ),
        )
    categoria, proveniencia, duplicata = _classificar_negrito(
        linha, texto if texto is not None else linha.texto, ids,
        altura_pagina=altura_pagina,
    )
    proveniencia = _enriquecer_duplicata(
        proveniencia, duplicata, texto if texto is not None else linha.texto
    )
    return LinhaClassificada(
        linha=linha,
        categoria=categoria,
        proveniencia=proveniencia,
        duplicata=duplicata,
    )


def classificar_linhas(
    linhas: Sequence[LinhaLogica],
    *,
    alturas_pagina: Mapping[int, float] | None = None,
) -> list[LinhaClassificada]:
    """Classifica todas as linhas, com fusão de fragmentos e contexto de anexo.

    ``alturas_pagina`` mapeia página → altura em pontos (para ancorar as bandas de
    masthead/rodapé); quando ausente usa :data:`ALTURA_PAGINA_PADRAO`.
    A saída é alinhada 1:1 com a entrada (mesma ordem e tamanho).
    """
    identificadores = [linha for linha in linhas if e_id(linha.texto)]
    repetidos = textos_repetidos(linhas)
    alturas = alturas_pagina or {}
    lider_de, texto_fundido = _mapear_grupos(linhas)

    resultado: list[LinhaClassificada] = []
    for indice, linha in enumerate(linhas):
        if not linha.negrito_inicio:
            resultado.append(
                LinhaClassificada(
                    linha=linha,
                    categoria="OUTRO",
                    proveniencia=Proveniencia(
                        "linha_nao_negrito",
                        "a linha não começa em negrito; não pode abrir documento",
                        _evidencias(linha),
                    ),
                )
            )
            continue

        lider = lider_de.get(indice, indice)
        if lider != indice:
            _, tamanho_linha = _fonte_tamanho(linha)
            mesma_visual = abs(linhas[lider].y - linha.y) <= _tol_y(tamanho_linha)
            if mesma_visual:
                regra = "fragmento_titulo"
                motivo = (
                    "fragmento do título iniciado na mesma linha visual "
                    f"(p. {linhas[lider].pagina} y={linhas[lider].y:.1f})"
                )
            else:
                regra = "continuacao_titulo"
                motivo = (
                    "continuação do título iniciado na linha de cima "
                    f"(p. {linhas[lider].pagina} y={linhas[lider].y:.1f})"
                )
            resultado.append(
                LinhaClassificada(
                    linha=linha,
                    categoria="OUTRO",
                    proveniencia=Proveniencia(
                        regra,
                        motivo,
                        _evidencias(linha, lider_x0=linhas[lider].x0),
                    ),
                    texto_fundido=None,
                )
            )
            continue

        efetivo = texto_fundido.get(indice, linha.texto)
        categoria, proveniencia, duplicata = _classificar_negrito(
            linha,
            efetivo,
            identificadores,
            repetidos=repetidos,
            altura_pagina=alturas.get(linha.pagina, ALTURA_PAGINA_PADRAO),
        )
        proveniencia = _enriquecer_duplicata(proveniencia, duplicata, efetivo)
        resultado.append(
            LinhaClassificada(
                linha=linha,
                categoria=categoria,
                proveniencia=proveniencia,
                texto_fundido=efetivo if efetivo != linha.texto else None,
                duplicata=duplicata,
            )
        )
    return resultado


def classificar_arquivo(caminho: str | Path) -> list[LinhaClassificada]:
    """Classifica todas as páginas de um PDF (via ``extrair_linhas_logicas``).

    Informa a altura de cada página para ancorar as bandas de masthead/rodapé.
    """
    with pymupdf.open(str(caminho)) as documento:
        total = documento.page_count
        alturas = {n: float(documento[n - 1].rect.height) for n in range(1, total + 1)}
    linhas = extrair_linhas_logicas(caminho, 1, total)
    return classificar_linhas(linhas, alturas_pagina=alturas)


def enunciados(classificadas: Sequence[LinhaClassificada]) -> list[LinhaClassificada]:
    """Filtra apenas as linhas classificadas como ``ENUNCIADO``."""
    return [item for item in classificadas if item.categoria == "ENUNCIADO"]

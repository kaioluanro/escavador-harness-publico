"""D-meta-1 — metadados do ato: ``tipo``, ``numero``, ``ano`` e ``orgao``.

Passo seguinte ao C1 (:mod:`harness.ingest.segmentar`): recebe um
:class:`~harness.contracts.Documento` já segmentado (``enunciado``, ``texto``,
``orgao``, ``id_dom``, ``edicao_*``) e devolve uma **cópia** com os metadados
preenchidos. Nada é mutado no documento de entrada — a função é pura.

Política (``reports/ideias/I004-ingestao-sintese.md`` item (d) e D015):

- ``tipo`` = token canônico do enunciado, incluindo **compostos** e **ordinais**
  (``LEI COMPLEMENTAR``, ``DECRETO LEGISLATIVO``, ``TERMO ADITIVO``,
  ``ATO DE DISPENSA``, ``INSTRUÇÃO NORMATIVA``). ``REPUBLICAÇÃO DA PORTARIA``
  devolve o tipo do ato republicado (``PORTARIA``). A lista base é a de
  :data:`harness.ingest.classificar.TOKENS_CANONICOS` (regras §1.2/D007).
- ``numero`` = **número próprio do ato** quando presente no enunciado; senão o
  do **instrumento referido** (D024): ``TERMO ADITIVO Nº 05/2026 AO CONTRATO
  02/2023`` → ``05/2026``; ``QUARTO TERMO ADITIVO AO CONTRATO Nº 02/2023`` →
  ``02/2023``. Em ambos os casos é o primeiro valor após um marcador
  ``Nº``/``N.º``/``N°``/``nº``/``N``, aceitando sufixos (``034/2026/GAB/SEMEC``),
  espaços antes da barra (``11 /2026``) e pontos de milhar (``28.597``). Números
  no formato de **processo** (``00045.018411/2025-46``) **não** são o número do
  ato (D015: processos vão para entidades). Quando o enunciado foi truncado pelo
  segmentador, o início do ``texto`` (continuação do título) é consultado.
- ``ano`` = ano embutido no número do ato/instrumento referido (``043/2010`` →
  2010); senão uma DATA explícita no enunciado (``DE 11 DE FEVEREIRO DE 2026``);
  senão ``None`` — **nunca** o ano da edição nem o ano de um número que não seja
  o escolhido (D029). Números de atos apenas **citados** (ex.: ``ATA … EDITAL
  Nº 01/2026 E Nº 02/2026``) não são o número do ato.
- ``orgao`` = **SIGLA canônica ou ``None``** (D037 confirmado por D050):
  conferido contra o **código de secretaria** embutido no ``id_dom`` (§1.7:
  ``id_secretaria(5)``), usando a tabela :data:`ORGAOS_POR_ID_SECRETARIA`
  (prefixos documentados em ``dados/catalogo_padroes_v1.md``). Sem ``id_dom``
  (edições antigas) — ou com código desconhecido — o ``orgao`` herdado da seção
  pelo C1 é normalizado por :func:`sigla_canonica`: só sai SIGLA (marcador
  ``¨…¨``, sigla no fim do cabeçalho ou nome canônico de
  :data:`ORGAOS_POR_NOME`); sem resolução → ``None``. **Nunca** texto de
  cabeçalho, nome longo nem sufixo territorial (D050: o gate cobra sigla, então
  FN é melhor que FP).

Uso::

    uv run python -m harness.ingest.metadados dados/snapshots/dev.jsonl -o out.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

from harness.contracts import Documento
from harness.ingest.classificar import TOKENS_CANONICOS, normalizar

__all__ = [
    "ORGAOS_POR_ID_SECRETARIA",
    "ORGAOS_POR_NOME",
    "SIGLAS_SUBSECAO",
    "SUFIXOS_TERRITORIAIS",
    "TIPO_REPUBLICADO",
    "ano_do_documento",
    "ano_do_enunciado",
    "carregar_snapshot",
    "escrever_snapshot",
    "main",
    "numero_do_documento",
    "numero_do_enunciado",
    "orgao_do_documento",
    "preencher_metadados",
    "preencher_snapshot",
    "sigla_canonica",
    "tipo_canonico",
]

#: Compostos canônicos: ``(token base, sufixo)``. O sufixo é reconhecido somente
#: imediatamente após o token base (pulando ordinais iniciais), de modo que
#: ``EXTRATO DO TERMO ADITIVO`` continua ``EXTRATO``.
COMPOSTOS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("LEI", ("COMPLEMENTAR",)),
    ("DECRETO", ("LEGISLATIVO",)),
    ("TERMO", ("ADITIVO",)),
    ("ATO", ("DE", "DISPENSA")),
    ("INSTRUCAO", ("NORMATIVA",)),
)

#: Token ``REPUBLICACAO`` apenas republica o ato seguinte; o ``tipo`` é o do ato.
TIPO_REPUBLICADO: str = "REPUBLICACAO"

#: Órgãos por prefixo ``id_secretaria`` (5 primeiros dígitos do ``id_dom``).
#: Fonte: ``knowledge/regras_diario.md`` §1.7 e ``dados/catalogo_padroes_v1.md``
#: (prefixos observados). Códigos novos/desconhecidos caem no ``orgao`` do C1.
ORGAOS_POR_ID_SECRETARIA: dict[str, str] = {
    "00001": "CÂMARA",
    "00041": "IPMT",
    "00042": "SEMA",
    "00043": "SEMF",
    "00044": "SEMEC",
    "00045": "FMS",
    "00046": "SEMPLAN",
    "00048": "PODER",
    "00049": "SEMCASPI",
    "00055": "ARSETE",
    "00057": "SEMJUV",
    "00066": "FMC",
    "00070": "SDUSUL",
    "00077": "STRANS",
    "00078": "SEMEL",
    "00081": "ETURB",
    "00082": "SDULESTE",
}

#: ``id_secretaria`` dos comunicados "ineditoriais" (sem órgão próprio).
ID_SECRETARIA_INEDITORIAL: str = "00042"

_ORDINAIS: frozenset[str] = frozenset(
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

#: Marcador de número e o valor subsequente. Aceita ``Nº``/``N.º``/``N°``/``nº.``
#: e ``N`` solto, e captura o número com sufixos (``/GAB/SEMEC``), pontos de
#: milhar (``28.597``) e espaços antes da barra (``11 /2026``).
_RE_NUMERO = re.compile(
    r"\bN\s*[.º°o]{0,4}\s*"
    r"([0-9][0-9A-Za-z./\-]*(?:\s*/\s*[0-9A-Za-z][0-9A-Za-z./\-]*)*)",
    re.IGNORECASE,
)
#: Número no formato de processo/SEI (não é o número do ato; D015).
_RE_PROCESSO = re.compile(r"^\d{2,5}(?:\.\d+)+/\d{4}-\d{2}$")
#: Número de ano embutido em um identificador (``/2026``, ``-2026``).
_RE_ANO_NO_NUMERO = re.compile(r"[/-]((?:19|20)\d{2})\b")
#: DATA explícita por extenso no enunciado (``DE 11 DE FEVEREIRO DE 2026``).
_RE_DATA_EXTENSO = re.compile(r"\b\d{1,2}\s+DE\s+[A-Z]+\s+DE\s+((?:19|20)\d{2})\b")
#: DATA explícita numérica no enunciado (``11/02/2026``).
_RE_DATA_NUMERICA = re.compile(r"\b\d{1,2}/\d{1,2}/((?:19|20)\d{2})\b")
#: Número após referência a contrato/fomento sem marcador (``CONTRATO 22/2025``).
_RE_NUMERO_CONTRATO = re.compile(
    r"(?:CONTRATO|FOMENTO|COLABORA[CÇ][AÃ]O)\D{0,40}?(\d{1,6}/\d{4})"
)
#: Continuação do título no corpo: primeira linha começa por ``03/2022``.
_RE_NUMERO_INICIO = re.compile(r"^\s*(\d{1,6}/\d{4})\b")
#: Separa candidatos em uma lista de atos citados (``Nº 01/2026 E Nº 02/2026``).
_RE_CONECTOR_LISTA = re.compile(r"^\s*[EOU]\s*$", re.IGNORECASE)
#: Extensão (chars) do corpo varrida atrás do número de um AVISO DE LICITAÇÃO.
CHARS_CORPO_AVISO: int = 300
#: Finais que denunciam título truncado pelo segmentador (aguardando o número).
_FINAIS_TRUNCADOS: frozenset[str] = frozenset(
    {"N", "NO", "CONTRATO", "FOMENTO", "COLABORACAO", "CONTRATUAL", "ADITIVO"}
)


# --------------------------------------------------------------------------- #
# Normalização auxiliar                                                       #
# --------------------------------------------------------------------------- #


def _palavra(texto: str) -> str:
    """Somente as letras da palavra, sem acento e em maiúsculas."""
    return re.sub(r"[^A-Z]", "", normalizar(texto))


def _casa_token(palavra: str, token: str) -> bool:
    """Casamento por prefixo de palavra (aceita ``-S``/``-ES``), como no B1.

    Aceita o token colado a órgão por barra (``RESOLUÇÃO/CMAS``): compara só a
    parte antes da barra (T024 item 3).
    """
    limpa = _palavra(palavra.split("/")[0])
    return limpa == token or limpa in (f"{token}S", f"{token}ES")


def _e_ordinal(token: str) -> bool:
    """True para ordinal por extenso (``QUARTO``) ou numérico (``1º``)."""
    bruto = normalizar(token)
    if re.fullmatch(r"\d{1,3}\s*\.?\s*[Oº°]", bruto):
        return True
    return _palavra(bruto) in _ORDINAIS


def _indice_base(palavras: Sequence[str]) -> tuple[str, int] | None:
    """Token canônico base e seu índice no enunciado (pulando ordinais iniciais)."""
    for indice, palavra in enumerate(palavras):
        for token in TOKENS_CANONICOS:
            if _casa_token(palavra, token):
                return token, indice
    return None


# --------------------------------------------------------------------------- #
# Tipo                                                                        #
# --------------------------------------------------------------------------- #


def tipo_canonico(enunciado: str) -> str:
    """``tipo`` canônico do ato a partir do enunciado (D007/D020).

    Reconhece os compostos de :data:`COMPOSTOS` e o ato de uma
    ``REPUBLICAÇÃO``. Enunciado vazio/irreconhecível devolve a primeira palavra
    em maiúsculas (sem pontuação), como o C1 já fazia.
    """
    palavras = normalizar(enunciado).split()
    if not palavras:
        return ""
    achado = _indice_base(palavras)
    if achado is None:
        return _palavra(palavras[0]) or palavras[0].strip(".,;:")
    base, indice = achado

    if base == TIPO_REPUBLICADO:
        for palavra in palavras[indice + 1 : indice + 5]:
            for token in TOKENS_CANONICOS:
                if token != TIPO_REPUBLICADO and _casa_token(palavra, token):
                    return token

    for token, sufixo in COMPOSTOS:
        if token != base:
            continue
        seguinte = [
            _palavra(p) for p in palavras[indice + 1 : indice + 1 + len(sufixo)]
        ]
        if tuple(seguinte) == tuple(sufixo):
            return f"{base} {' '.join(sufixo)}"
    return base


# --------------------------------------------------------------------------- #
# Número e ano                                                                #
# --------------------------------------------------------------------------- #


def _candidatos(texto: str) -> list[str]:
    """Números marcados do texto, descartando atos apenas CITADOS (D029).

    ``Nº 01/2026 E Nº 02/2026`` é uma lista de atos citados (ex.: a ATA que cita
    os EDITAI 01/2026 e 02/2026); esses números não são o número do ato.
    """
    achados = list(_RE_NUMERO.finditer(texto or ""))
    citados: set[int] = set()
    for anterior, seguinte in zip(achados, achados[1:]):
        entre = (texto or "")[anterior.end() : seguinte.start()]
        if _RE_CONECTOR_LISTA.match(entre):
            citados.update({anterior.start(), seguinte.start()})
    valores: list[str] = []
    for achado in achados:
        if achado.start() in citados:
            continue
        valor = achado.group(1).strip().rstrip(".,;:")
        if valor:
            valores.append(valor)
    return valores


def _e_numero_de_processo(valor: str) -> bool:
    """True para identificador de processo/SEI (não é o número do ato)."""
    return bool(_RE_PROCESSO.match(valor.replace(" ", "")))


def _primeiro_numero(texto: str) -> str | None:
    """Primeiro número marcado que não é de processo."""
    for valor in _candidatos(texto):
        if not _e_numero_de_processo(valor):
            return valor
    return None


def numero_do_enunciado(enunciado: str) -> str | None:
    """Número do ato no enunciado (``None`` se só houver processo ou nada)."""
    return _primeiro_numero(enunciado)


#: Número do ato SEM o marcador ``Nº``: ``N/ANO`` ou milhar (``28.597``).
_RE_NUMERO_SEM_MARCADOR = re.compile(r"^\d{1,6}/\d{4}(?:-\d{2})?$|^\d{1,3}(?:\.\d{3})+$")


def _numero_apos_token(enunciado: str) -> str | None:
    """Número do ato logo após o token canônico, mesmo sem ``Nº`` (T024 item 2).

    ``PORTARIA 13/2023. A Secretária…`` → ``13/2023``; ``DECRETO 28.597`` →
    ``28.597``. Exige forma ``N/ANO`` ou milhar para não capturar o dia de uma
    data (``DECRETO DE 11 DE FEVEREIRO…``).
    """
    palavras = normalizar(enunciado).split()
    achado = _indice_base(palavras)
    if achado is None:
        return None
    _, indice = achado
    for bruto in palavras[indice + 1 : indice + 5]:
        valor = bruto.strip(".,;:")
        if _RE_NUMERO_SEM_MARCADOR.match(valor) and not _e_numero_de_processo(valor):
            return valor
    return None


def numero_do_documento(enunciado: str, texto: str = "", tipo: str = "") -> str | None:
    """Número do ato ou do instrumento referido (D024), tolerando enunciado truncado.

    O primeiro número marcado do enunciado é o **próprio** do ato quando aparece
    logo no título (``TERMO ADITIVO Nº 05/2026 AO CONTRATO 02/2023`` → ``05/2026``);
    quando o ato não tem número próprio, é o do **instrumento referido**
    (``QUARTO TERMO ADITIVO AO CONTRATO Nº 02/2023`` → ``02/2023``). Ordem de
    tentativa:

    1. primeiro número marcado do ``enunciado`` (exclui processo; D015);
    2. continuação do título na 1ª linha do ``texto`` (mesma regra);
    3. continuação sem marcador (``03/2022`` no início da 1ª linha) quando o
       enunciado termina numa referência a contrato/fomento;
    4. número de contrato/fomento sem marcador no próprio enunciado.
    """
    valor = _primeiro_numero(enunciado)
    if valor:
        return valor

    # T024 item 2: número do ato sem o marcador ``Nº`` (``PORTARIA 13/2023``).
    valor = _numero_apos_token(enunciado)
    if valor:
        return valor

    linhas = (texto or "").split("\n")
    primeira_linha = linhas[0] if linhas else ""
    valor = _primeiro_numero(primeira_linha)
    if valor:
        return valor

    normalizado = normalizar(enunciado)
    if re.search(r"CONTRATO|FOMENTO|COLABORA[CÇ][AÃ]O", normalizado):
        achado = _RE_NUMERO_INICIO.match(primeira_linha)
        if achado and not re.search(r"\d", enunciado):
            return achado.group(1)

    # Título quebrou de novo na 2ª linha do corpo (ex.: ``... EDITAL`` /
    # ``Nº 009/2025``): só aceita número com ano, para não capturar endereços.
    if len(linhas) > 1 and _titulo_pede_continuacao(enunciado, primeira_linha):
        valor = _primeiro_numero(linhas[1])
        if valor and _RE_ANO_NO_NUMERO.search(valor):
            return valor

    achado = _RE_NUMERO_CONTRATO.search(normalizado)
    if achado:
        return achado.group(1)

    # AVISO DE LICITAÇÃO truncado: o número do certame (``PREGÃO ... Nº 90019/2026``)
    # continua no corpo (D024: número do instrumento referido pelo ato). Só quando
    # o enunciado é um FRAGMENTO (sem fim de frase): um título completo como
    # ``AVISO DE LICITAÇÃO.`` não empresta o número do corpo.
    if "AVISO DE LICITACAO" in normalizado and not re.search(r"[.;]", enunciado):
        for achado in _RE_NUMERO.finditer((texto or "")[:CHARS_CORPO_AVISO]):
            valor = achado.group(1).strip().rstrip(".,;:")
            if valor and _RE_ANO_NO_NUMERO.search(valor) and not _e_numero_de_processo(valor):
                return valor
    return None


def _titulo_pede_continuacao(enunciado: str, primeira_linha: str) -> bool:
    """True se o título (enunciado + 1ª linha) terminou sem o número do ato."""
    palavras = normalizar(f"{enunciado} {primeira_linha}").split()
    if not palavras:
        return False
    ultima = _palavra(palavras[-1])
    if ultima in _FINAIS_TRUNCADOS:
        return True
    return any(_casa_token(palavras[-1], token) for token in TOKENS_CANONICOS)


def ano_do_enunciado(enunciado: str, numero: str | None = None) -> int | None:
    """Ano do ato (D029): do número escolhido ou de DATA explícita; senão ``None``.

    - ``043/2010`` → 2010 (ano embutido no número próprio/referido, D024);
    - se o número não traz ano, vale a DATA explícita do enunciado
      (``... DE 11 DE FEVEREIRO DE 2026`` → 2026);
    - **nunca** o ano da edição (já é ``edicao_data``) nem o ano de um número
      que não seja o escolhido (ex.: o contrato referido quando o ato tem número
      próprio sem ano → ``None``).
    """
    if numero:
        embutido = _RE_ANO_NO_NUMERO.search(numero)
        if embutido:
            return int(embutido.group(1))
    texto = normalizar(enunciado or "")
    for padrao in (_RE_DATA_EXTENSO, _RE_DATA_NUMERICA):
        achado = padrao.search(texto)
        if achado:
            return int(achado.group(1))
    return None


# --------------------------------------------------------------------------- #
# Órgão                                                                       #
# --------------------------------------------------------------------------- #


def orgao_do_documento(documento: Documento) -> str | None:
    """Órgão do documento, conferido pelo ``id_secretaria`` quando há ``id_dom``.

    Sem ``id_dom`` (edições antigas) mantém o ``orgao`` herdado da seção pelo C1.
    """
    id_dom = documento.id_dom
    if id_dom:
        codigo = id_dom[:5]
        if codigo == ID_SECRETARIA_INEDITORIAL and normalizar(documento.tipo) == "COMUNICADO":
            return "INEDITORIAL"
        orgao = ORGAOS_POR_ID_SECRETARIA.get(codigo)
        if orgao:
            return orgao
    return documento.orgao


# --------------------------------------------------------------------------- #
# Função principal / snapshot                                                 #
# --------------------------------------------------------------------------- #


def preencher_metadados(documento: Documento) -> Documento:
    """Devolve uma **cópia** do documento com ``tipo``/``numero``/``ano``/``orgao``.

    É pura: não altera ``documento``. ``numero``/``ano`` vêm do ato (enunciado +
    continuação no texto), conforme D024/D029; ``orgao`` é conferido pelo
    ``id_secretaria`` quando o ``id_dom`` existe (ver :func:`orgao_do_documento`).
    """
    tipo = tipo_canonico(documento.enunciado)
    numero = numero_do_documento(documento.enunciado, documento.texto, tipo)
    ano = ano_do_enunciado(documento.enunciado, numero)
    return documento.model_copy(
        update={
            "tipo": tipo,
            "numero": numero,
            "ano": ano,
            "orgao": orgao_do_documento(documento),
        }
    )


def preencher_snapshot(documentos: Iterable[Documento]) -> list[Documento]:
    """Aplica :func:`preencher_metadados` a uma sequência de documentos."""
    return [preencher_metadados(documento) for documento in documentos]


def carregar_snapshot(caminho: str | Path) -> list[Documento]:
    """Lê um JSONL de :class:`Documento` (mesmo formato do gate)."""
    documentos: list[Documento] = []
    with Path(caminho).open(encoding="utf-8") as arquivo:
        for numero, linha in enumerate(arquivo, start=1):
            linha = linha.strip()
            if not linha:
                continue
            try:
                documentos.append(Documento.model_validate_json(linha))
            except Exception as exc:  # noqa: BLE001 - contexto da linha no erro
                raise ValueError(f"{caminho}:{numero}: Documento inválido: {exc}") from exc
    return documentos


def escrever_snapshot(documentos: Iterable[Documento], caminho: str | Path) -> Path:
    """Grava um JSONL de :class:`Documento` (formato consumido pelo gate)."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as arquivo:
        for documento in documentos:
            arquivo.write(documento.model_dump_json() + "\n")
    return caminho


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m harness.ingest.metadados <snapshot.jsonl> -o <saida.jsonl>``."""
    parser = argparse.ArgumentParser(
        prog="python -m harness.ingest.metadados",
        description="D-meta-1: preenche tipo/numero/ano/orgao de um snapshot.",
    )
    parser.add_argument("entrada", type=Path, help="JSONL de Documento segmentado.")
    parser.add_argument(
        "--saida",
        "-o",
        type=Path,
        default=None,
        help="JSONL de saída (padrão: <entrada>.metadados.jsonl).",
    )
    args = parser.parse_args(argv)

    try:
        documentos = carregar_snapshot(args.entrada)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
        return 2

    saida = args.saida or args.entrada.with_suffix(".metadados.jsonl")
    preenchidos = preencher_snapshot(documentos)
    caminho = escrever_snapshot(preenchidos, saida)
    print(f"metadados: {len(preenchidos)} documento(s) -> {caminho}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

"""Leitura do masthead da página 1 para recuperar edição/data (T027).

Ferramenta de plataforma, **somente leitura**. Quando o nome do arquivo não traz
o padrão ``DOM<num>-<ddmmaaaa>`` (ex.: ``ANEXO AO DOM3159-ASSINADO.pdf``), a CLI
de ingestão consulta o CONTEÚDO da página 1 para obter o número da edição e a
data de publicação.

Formatos reconhecidos no masthead do DOM Teresina:

- ``Ano 2026 - Nº 4.196 - 11 de fevereiro de 2026`` (número + data);
- ``DOM - Teresina - Ano 2026 - nº 4.196`` (só número);
- ``Anexo ao DOM nº 3.159`` (só número; pode vir com data na mesma linha);
- ``Quarta-feira, 11 de fevereiro de 2026`` (só data, por extenso);
- ``11/02/2026`` (data numérica).

A extração é conservadora: só devolve o que casa um padrão inequívoco. O
chamador decide o que fazer quando falta número ou data (a CLI registra o erro
do arquivo e segue o lote). Este módulo é **puro** (não abre PDF): a leitura das
linhas fica na CLI, que usa a extração anti-fantasma da ingestão.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from typing import Iterable

from harness.contracts import LinhaLogica

__all__ = [
    "FRACAO_TOPO_MASTHEAD",
    "MESES",
    "data_da_edicao",
    "extrair_edicao_de_linhas",
    "extrair_edicao_do_texto",
    "numero_da_edicao",
]

#: Fração da altura da página que define a banda de topo (masthead). O masthead
#: real do DOM fica em ~0,16 da altura; 0,30 dá folga sem invadir o corpo.
FRACAO_TOPO_MASTHEAD: float = 0.30

#: Meses por extenso (sem acento, minúsculos) → número do mês.
MESES: dict[str, int] = {
    "janeiro": 1,
    "fevereiro": 2,
    "marco": 3,
    "abril": 4,
    "maio": 5,
    "junho": 6,
    "julho": 7,
    "agosto": 8,
    "setembro": 9,
    "outubro": 10,
    "novembro": 11,
    "dezembro": 12,
}

#: ``Ano 2026 - Nº 4.196`` (cobre também ``DOM - Teresina - Ano 2026 - nº 4.196``).
_RE_ANO_NUM = re.compile(r"\bano\s+\d{4}\s*[-–—]\s*n[º°o.]*\s*([\d.]+)", re.IGNORECASE)
#: ``Anexo ao DOM nº 3.159``.
_RE_ANEXO = re.compile(r"\banexo\s+ao\s+dom\s*n[º°o.]*\s*([\d.]+)", re.IGNORECASE)
#: ``11 de fevereiro de 2026``.
_RE_DATA_EXTENSO = re.compile(
    r"\b(\d{1,2})\s+de\s+([A-Za-zÀ-ÿ]+)\s+de\s+(\d{4})\b", re.IGNORECASE
)
#: ``11/02/2026``.
_RE_DATA_NUMERICA = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")


def _sem_acento(texto: str) -> str:
    """Remove acentos (NFKD) preservando o restante do texto."""
    decomposto = unicodedata.normalize("NFKD", texto or "")
    return "".join(c for c in decomposto if not unicodedata.combining(c))


def _normalizar(texto: str) -> str:
    """Sem acento e com espaços colapsados (base do casamento)."""
    return re.sub(r"\s+", " ", _sem_acento(texto or "")).strip()


def _normalizar_numero(valor: str) -> str:
    """Número da edição canônico: só dígitos, sem zeros à esquerda (``4.196`` → ``4196``)."""
    digitos = re.sub(r"\D", "", valor)
    return digitos.lstrip("0") or "0"


def numero_da_edicao(texto: str) -> str | None:
    """Número da edição no texto do masthead, ou ``None``.

    Reconhece ``Ano AAAA - Nº <num>`` e ``Anexo ao DOM nº <num>``; o número sai
    canônico (sem separador de milhar).
    """
    normalizado = _normalizar(texto)
    for padrao in (_RE_ANO_NUM, _RE_ANEXO):
        achado = padrao.search(normalizado)
        if achado:
            return _normalizar_numero(achado.group(1))
    return None


def data_da_edicao(texto: str) -> date | None:
    """Data de publicação no texto do masthead, ou ``None``.

    Aceita a forma por extenso (``11 de fevereiro de 2026``) e a numérica
    (``11/02/2026``). Datas impossíveis (ex.: 31 de fevereiro) devolvem ``None``.
    """
    normalizado = _normalizar(texto)
    achado = _RE_DATA_EXTENSO.search(normalizado)
    if achado:
        mes = MESES.get(achado.group(2).casefold())
        if mes:
            try:
                return date(int(achado.group(3)), mes, int(achado.group(1)))
            except ValueError:
                return None
    achado = _RE_DATA_NUMERICA.search(normalizado)
    if achado:
        try:
            return date(int(achado.group(3)), int(achado.group(2)), int(achado.group(1)))
        except ValueError:
            return None
    return None


def extrair_edicao_do_texto(texto: str) -> tuple[str | None, date | None]:
    """Extrai ``(numero, data)`` de um texto multilinha (masthead).

    Prioriza a linha que traz **número e data juntos** (``Ano ... Nº ... - DD de
    MÊS de AAAA``); senão, o primeiro número e a primeira data encontrados.
    """
    linhas = [linha for linha in (texto or "").splitlines() if linha.strip()]
    numero: str | None = None
    data: date | None = None
    for linha in linhas:
        achado_numero = numero_da_edicao(linha)
        achado_data = data_da_edicao(linha)
        if achado_numero and achado_data:
            return achado_numero, achado_data
        if numero is None and achado_numero:
            numero = achado_numero
        if data is None and achado_data:
            data = achado_data
    return numero, data


def extrair_edicao_de_linhas(
    linhas: Iterable[LinhaLogica],
    *,
    altura_pagina: float | None = None,
) -> tuple[str | None, date | None]:
    """Extrai ``(numero, data)`` das linhas da página 1.

    Considera as linhas da **banda de topo** (masthead) e qualquer linha que cite
    ``Anexo ao DOM`` (que pode ficar fora da banda). Sem ``altura_pagina``,
    considera todas as linhas.
    """
    topo: list[str] = []
    anexo: list[str] = []
    for linha in linhas:
        texto = linha.texto
        if _RE_ANEXO.search(_normalizar(texto)):
            anexo.append(texto)
        if altura_pagina is None or linha.y <= FRACAO_TOPO_MASTHEAD * altura_pagina:
            topo.append(texto)
    candidatas = topo + [texto for texto in anexo if texto not in topo]
    return extrair_edicao_do_texto("\n".join(candidatas))
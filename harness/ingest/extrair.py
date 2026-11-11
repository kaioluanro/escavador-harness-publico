"""Extração de linhas lógicas com o filtro anti-fantasma A1 aplicado (D018).

O PyMuPDF **não respeita clipping** na extração (§2.1): a diagramação reaproveita
edições antigas e o texto fantasma continua extraível. O módulo
:mod:`harness.ingest.fantasma` interpreta o content stream e decide, por span de
``get_texttrace()``, o que é visível. Aqui essa máscara é aplicada **antes** de
montar as linhas lógicas — antes de B1 (classificar) e C1 (segmentar), como manda
a T020/D018.

Estratégia: manter a extração clássica ``page.get_text("dict")`` (que preserva
espaços e a ordem dos spans) e **descartar os spans cujo gêmeo de
``get_texttrace()`` está invisível** (mesmo texto normalizado + sobreposição de
bbox). Assim não se perde a fidelidade de espaçamento do ``dict`` — o
``get_texttrace`` não emite glifos de espaço — e o fantasma sai do corpo.

Uso::

    from harness.ingest.extrair import extrair_linhas_visiveis
    linhas = extrair_linhas_visiveis("diarios/DOM4196-...pdf")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

try:  # PyMuPDF >= 1.24 expõe ``pymupdf``; versões antigas só ``fitz``.
    import pymupdf
except ImportError:  # pragma: no cover - compatibilidade com versões antigas
    import fitz as pymupdf  # type: ignore[no-redef]

from harness.contracts import LinhaLogica
from harness.ingest import fantasma
from harness.tools.inspect import SpanInfo, extrair_spans, montar_linhas_info

__all__ = [
    "alturas_do_pdf",
    "deduplicar_spans",
    "extrair_linhas_visiveis",
    "filtrar_spans_fantasma",
]

#: Sobreposição mínima (fração da menor área) para casar o span do ``dict`` com o
#: span invisível de ``get_texttrace``.
IOU_MINIMA: float = 0.5

#: Sobreposição mínima (fração da menor área) para tratar dois spans do ``dict``
#: como a MESMA superfície de texto desenhada mais de uma vez (camadas
#: sobrepostas do expediente) ou como letra capitular desmembrada.
IOU_DUPLICATA: float = 0.5

#: Vão horizontal (pt) máximo para ainda considerar dois spans sobrepostos; acima
#: disso são spans distintos na mesma linha e o par não é comparado.
VAO_DUPLICATA_PT: float = 2.0

#: Tolerância (pt) de baseline para agrupar spans candidatos a duplicata.
TOL_BASELINE_DUPLICATA_PT: float = 1.0

_ESPACOS = re.compile(r"\s+")


def _texto_do_span(span: Mapping[str, object]) -> str:
    """Reconstrói o texto bruto de um span de ``get_texttrace`` pelos chars."""
    partes: list[str] = []
    for char in span.get("chars") or ():  # type: ignore[union-attr]
        try:
            codigo = int(char[0])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= codigo <= 0x10FFFF:
            partes.append(chr(codigo))
    return "".join(partes)


def _chave_texto(texto: str) -> str:
    """Chave de casamento: sem espaços e em caixa baixa (o texttrace perde espaços)."""
    return _ESPACOS.sub("", texto or "").casefold()


def _bbox(span: Mapping[str, object]) -> tuple[float, float, float, float]:
    try:
        return tuple(float(v) for v in span["bbox"])  # type: ignore[union-attr,index,return-value]
    except (TypeError, ValueError, KeyError):
        return (0.0, 0.0, 0.0, 0.0)


def _sobreposicao(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Interseção sobre a menor área (0 se não houver)."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    menor = min(area_a, area_b)
    return inter / menor if menor > 0 else 0.0


def _spans_invisiveis(page: object) -> list[tuple[str, tuple[float, float, float, float]]]:
    """``(texto_normalizado, bbox)`` dos spans fantasma da página."""
    brutos = list(page.get_texttrace())  # type: ignore[attr-defined]
    invisiveis: list[tuple[str, tuple[float, float, float, float]]] = []
    for item in fantasma.spans_visiveis(page):
        if item.visivel:
            continue
        if not (0 <= item.indice < len(brutos)):
            continue
        span = brutos[item.indice]
        chave = _chave_texto(_texto_do_span(span))
        if chave:
            invisiveis.append((chave, _bbox(span)))
    return invisiveis


def filtrar_spans_fantasma(infos: list[SpanInfo], page: object) -> list[SpanInfo]:
    """Remove os spans do ``dict`` que casam com um span invisível do A1."""
    invisiveis = _spans_invisiveis(page)
    if not invisiveis:
        return infos
    mantidos: list[SpanInfo] = []
    for info in infos:
        chave = _chave_texto(info.span.texto)
        bbox = info.span.bbox
        fantasma_ = any(
            chave == chave_inv and _sobreposicao(bbox, bbox_inv) >= IOU_MINIMA
            for chave_inv, bbox_inv in invisiveis
        )
        if not fantasma_:
            mantidos.append(info)
    return mantidos


def _e_duplicata(a: SpanInfo, b: SpanInfo) -> bool:
    """True se ``a``/``b`` são a mesma superfície desenhada mais de uma vez.

    A diagramação reaproveita objetos e desenha o mesmo texto (ou letras
    capitulares desmembradas) em posições sobrepostas. O sintoma no
    ``get_text`` é o run repetido/intercalado: ``R RENATO ENATO``,
    ``Sylvia Soares Oliveira Portela Sylvia Soares Oliveira Portela``. Dois
    spans contam como duplicata quando:

    - as caixas se sobrepõem (fração da menor área ≥ :data:`IOU_DUPLICATA`); e
    - o texto normalizado é igual **ou** um é prefixo/sufixo do outro (capitular
      ``R`` ⊂ ``RENATO``; cópia truncada ``PRODATE`` ⊂ ``PRODATER``).

    Exige sobreposição geométrica forte: spans legítimos lado a lado (sem
    sobreposição) nunca são comparados.
    """
    ta = _chave_texto(a.span.texto)
    tb = _chave_texto(b.span.texto)
    if not ta or not tb:
        return False
    if _sobreposicao(a.span.bbox, b.span.bbox) < IOU_DUPLICATA:
        return False
    if ta == tb:
        return True
    return (
        ta.startswith(tb)
        or tb.startswith(ta)
        or ta.endswith(tb)
        or tb.endswith(ta)
    )


def deduplicar_spans(infos: list[SpanInfo]) -> list[SpanInfo]:
    """Remove spans sobrepostos com o mesmo texto (camadas/capitulares).

    Roda **antes** de montar as linhas lógicas (``montar_linhas_info``), para que
    o texto intercalado do expediente não chegue à segmentação. Conservador: só
    descarta um span quando há outro fortemente sobreposto com texto igual ou
    prefixo/sufixo do primeiro; na dúvida, mantém.
    """
    if len(infos) < 2:
        return infos
    grupos: dict[int, list[SpanInfo]] = {}
    for info in infos:
        chave = round(info.baseline / TOL_BASELINE_DUPLICATA_PT)
        grupos.setdefault(chave, []).append(info)

    descartar: set[int] = set()
    for grupo in grupos.values():
        if len(grupo) < 2:
            continue
        grupo.sort(key=lambda info: info.span.bbox[0])
        for a in range(len(grupo)):
            if id(grupo[a]) in descartar:
                continue
            for b in range(a + 1, len(grupo)):
                if id(grupo[b]) in descartar:
                    continue
                if grupo[b].span.bbox[0] - grupo[a].span.bbox[2] > VAO_DUPLICATA_PT:
                    break
                if not _e_duplicata(grupo[a], grupo[b]):
                    continue
                # Mantém a superfície MAIS LONGA: a capitular ``R`` e a cópia
                # truncada ``ENATO`` cedem para ``RENATO``. Em empate (cópia
                # exata) mantém a primeira.
                if len(grupo[a].span.texto) < len(grupo[b].span.texto):
                    descartar.add(id(grupo[a]))
                    break
                descartar.add(id(grupo[b]))
    if not descartar:
        return infos
    return [info for info in infos if id(info) not in descartar]


def extrair_linhas_visiveis(
    caminho: str | Path,
    pagina: int = 1,
    ate: int | None = None,
    *,
    aplicar_fantasma: bool = True,
) -> list[LinhaLogica]:
    """Linhas lógicas das páginas pedidas, já sem o texto fantasma (A1).

    ``aplicar_fantasma=False`` cai no ``page.get_text("dict")`` clássico (paridade
    com o ``inspect``). O A1 é conservador: se não puder decidir, mantém o span.
    """
    if pagina < 1:
        raise ValueError(f"página inicial inválida: {pagina} (use >= 1)")
    fim = pagina if ate is None else ate
    if fim < pagina:
        raise ValueError(f"--ate ({fim}) menor que --pagina ({pagina})")

    resultado: list[LinhaLogica] = []
    with pymupdf.open(str(caminho)) as doc:
        if fim > doc.page_count:
            raise ValueError(
                f"página final {fim} além do PDF ({doc.page_count} páginas)"
            )
        for numero in range(pagina, fim + 1):
            page = doc[numero - 1]
            infos = extrair_spans(page, numero)
            if aplicar_fantasma:
                infos = filtrar_spans_fantasma(infos, page)
            # Camadas sobrepostas/capitulares são removidas ANTES de montar as
            # linhas lógicas (T025): o texto intercalado não chega à segmentação.
            infos = deduplicar_spans(infos)
            resultado.extend(
                info.linha for info in montar_linhas_info(infos, numero)
            )
    return resultado


def alturas_do_pdf(caminho: str | Path) -> dict[int, float]:
    """Altura (pontos) de cada página do PDF (1-based)."""
    with pymupdf.open(str(caminho)) as doc:
        return {n: float(doc[n - 1].rect.height) for n in range(1, doc.page_count + 1)}

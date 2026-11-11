"""Diagnóstico ``inspect --trace``: clip, z-order e ordem de pintura (somente leitura).

Ferramenta de apoio ao time de dados/ingestão (item 2 da ordem de implementação
de ``reports/ideias/I004-ingestao-sintese.md``). Ela **não** decide nem remove
fantasmas: apenas despeja, por página, as APIs públicas do PyMuPDF usadas pela
proposta A1 e calcula um resumo para validar o pareamento clip × texto **antes**
de codar o filtro anti-fantasma.

Por página são emitidos registros JSONL (um por item):

- ``kind="texto"``  — um span de :meth:`pymupdf.Page.get_texttrace`, com
  ``seqno``, ``type``, ``opacity``, ``font``/``size``, ``bbox`` e, resumidamente,
  origem/bbox dos caracteres (``origem_primeira``/``origem_ultima`` e o texto
  reconstruído, já que ``get_texttrace`` entrega os glifos por ``unicode``).
- ``kind="desenho"`` — paths de :meth:`pymupdf.Page.get_drawings`
  (``extended=True``): ``clip`` e ``group`` sempre; paths de preenchimento
  (``f``/``fs``) incluídos conforme ``incluir_todos_fills`` (por padrão só os
  **brancos**, relevantes para o z-order).
- ``kind="pintura"`` — itens de :meth:`pymupdf.Page.get_bboxlog` (a ordem de
  pintura; o índice do item é o ``seqno``).
- ``kind="resumo"`` — um por página (ver :func:`analisar_pagina`).

Metodologia do resumo (documentada porque é uma **aproximação medida**):

- O PyMuPDF 1.28.2 **não** expõe ``seqno`` nos paths ``clip`` de
  ``get_drawings(extended=True)`` (o ``seqno`` existe só nos paths de pintura).
  Para localizar um clip na ordem de pintura usamos o intervalo
  ``(seqno_anterior, seqno_proximo)`` inferido dos paths de pintura vizinhos na
  lista de ``get_drawings`` (que sai em ordem de stream e com ``seqno``
  monotônico). Um clip só afeta um glifo cujo ``seqno`` cai estritamente nesse
  intervalo.
- ``glifos_fora_do_clip`` conta glifos cuja origem **não** está na interseção dos
  ``scissor`` dos clips ativos para o ``seqno`` do span.
- ``glifos_cobertos_por_branco_depois`` conta glifos cujo bbox intersecta um path
  de preenchimento **branco** (RGB ~ 1,1,1 e opacidade > 0) com ``seqno``
  **maior** que o do span.

Esta ferramenta é **somente leitura**: nunca grava no PDF nem em ``diarios/``.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

try:  # PyMuPDF >= 1.24 expõe ``pymupdf``; versões antigas só ``fitz``.
    import pymupdf
except ImportError:  # pragma: no cover - compatibilidade com versões antigas
    import fitz as pymupdf  # type: ignore[no-redef]

__all__ = [
    "AnalisePagina",
    "ClipAtivo",
    "KINDS",
    "analisar_pagina",
    "inspecionar_trace",
    "intervalos_de_clip",
    "registros_bboxlog",
    "registros_desenhos",
    "registros_texto",
    "resumo_vazio",
]

#: Tipos de registro emitidos (campo ``kind``).
KINDS: tuple[str, ...] = ("texto", "desenho", "pintura", "resumo")

#: Tolerância para considerar um preenchimento "branco".
BRANCO_MIN = 0.999

#: Casas decimais usadas ao arredondar coordenadas no JSONL.
CASAS = 2

#: Texto reconstruído do span é truncado neste tamanho (campo ``texto``).
MAX_TEXTO_PADRAO = 80

#: Tipos de `get_drawings` com preenchimento (candidatos a z-order).
TIPOS_FILL = ("f", "fs")


def _arredondar(valor: float) -> float:
    return round(float(valor), CASAS)


def _caixa(rect: Any) -> list[float] | None:
    """Converte um ``Rect``/tupla em ``[x0, y0, x1, y1]`` arredondado."""
    if rect is None:
        return None
    try:
        valores = [float(v) for v in rect]
    except TypeError:
        return None
    if len(valores) != 4:
        return None
    return [_arredondar(v) for v in valores]


def _cor(valor: Any) -> list[float] | None:
    """Converte cor (tupla de floats) em lista JSON; ``None`` fica ``None``."""
    if valor is None:
        return None
    try:
        return [_arredondar(float(v)) for v in valor]
    except TypeError:
        return None


def _ponto(valor: Any) -> list[float] | None:
    """Converte origem/ponto em ``[x, y]`` arredondado."""
    if valor is None:
        return None
    try:
        valores = [float(v) for v in valor]
    except TypeError:
        return None
    if len(valores) < 2:
        return None
    return [_arredondar(valores[0]), _arredondar(valores[1])]


def _texto_do_span(span: dict) -> str:
    """Reconstrói o texto do span a partir dos ``unicode`` dos caracteres."""
    partes: list[str] = []
    for char in span.get("chars") or ():
        try:
            codigo = int(char[0])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= codigo <= 0x10FFFF:
            partes.append(chr(codigo))
    return "".join(partes)


def _partes_char(char: Any) -> tuple[int, tuple[float, ...], tuple[float, ...]] | None:
    """Extrai ``(unicode, origem, bbox)`` de um char de ``get_texttrace``."""
    if char is None:
        return None
    try:
        codigo = int(char[0])
    except (TypeError, ValueError, IndexError):
        return None
    try:
        origem = tuple(float(v) for v in char[2])
        bbox = tuple(float(v) for v in char[3])
    except (TypeError, ValueError, IndexError):
        return None
    if len(origem) < 2 or len(bbox) < 4:
        return None
    return codigo, origem, bbox


def _uniao_caixas(caixas: Iterable[tuple[float, ...]]) -> list[float] | None:
    validas = [c for c in caixas if c and len(c) >= 4]
    if not validas:
        return None
    return [
        _arredondar(min(c[0] for c in validas)),
        _arredondar(min(c[1] for c in validas)),
        _arredondar(max(c[2] for c in validas)),
        _arredondar(max(c[3] for c in validas)),
    ]


def _registro_texto(
    span: dict, pagina: int, *, max_texto: int = MAX_TEXTO_PADRAO
) -> dict[str, Any]:
    """Monta o registro JSONL de um span de ``get_texttrace``."""
    chars = list(span.get("chars") or ())
    partes = [p for p in (_partes_char(c) for c in chars) if p is not None]
    origens = [p[1] for p in partes]
    caixas = [p[2] for p in partes]
    return {
        "kind": "texto",
        "pagina": pagina,
        "seqno": int(span["seqno"]) if span.get("seqno") is not None else None,
        "type": span.get("type"),
        "opacity": (
            round(float(span["opacity"]), 3)
            if span.get("opacity") is not None
            else None
        ),
        "font": span.get("font"),
        "size": (
            round(float(span["size"]), 3) if span.get("size") is not None else None
        ),
        "color": _cor(span.get("color")),
        "flags": span.get("flags"),
        "layer": span.get("layer"),
        "dir": _ponto(span.get("dir")),
        "bbox": _caixa(span.get("bbox")),
        "n_chars": len(chars),
        "origem_primeira": _ponto(origens[0]) if origens else None,
        "origem_ultima": _ponto(origens[-1]) if origens else None,
        "chars_bbox": _uniao_caixas(caixas),
        "texto": _texto_do_span(span)[:max_texto],
    }


def registros_texto(
    page: object, pagina: int, *, max_texto: int = MAX_TEXTO_PADRAO
) -> Iterator[dict[str, Any]]:
    """Gera um registro JSONL por span de :meth:`pymupdf.Page.get_texttrace`."""
    for span in page.get_texttrace():  # type: ignore[attr-defined]
        yield _registro_texto(span, pagina, max_texto=max_texto)


def _e_branco(fill: Any) -> bool:
    if fill is None:
        return False
    try:
        return all(float(v) >= BRANCO_MIN for v in fill)
    except TypeError:
        return False


def registros_desenhos(
    page: object,
    pagina: int,
    *,
    incluir_todos_fills: bool = False,
    drawings: list[dict[str, Any]] | None = None,
) -> Iterator[dict[str, Any]]:
    """Gera um registro por path relevante de ``get_drawings(extended=True)``.

    Inclui sempre ``clip`` e ``group``; de preenchimento (``f``/``fs``) inclui os
    brancos e, se ``incluir_todos_fills``, todos. ``drawings`` permite reaproveitar
    um ``get_drawings(extended=True)`` já lido.
    """
    if drawings is None:
        drawings = list(page.get_drawings(extended=True))  # type: ignore[attr-defined]
    for path in drawings:
        tipo = path.get("type")
        fill = path.get("fill")
        if tipo in ("clip", "group"):
            pass
        elif tipo in TIPOS_FILL and (incluir_todos_fills or _e_branco(fill)):
            pass
        else:
            continue
        registro: dict[str, Any] = {
            "kind": "desenho",
            "pagina": pagina,
            "tipo": tipo,
            "nivel": path.get("level"),
            "seqno": path.get("seqno"),
            "layer": path.get("layer"),
            "scissor": _caixa(path.get("scissor")),
            "rect": _caixa(path.get("rect")),
            "fill": _cor(fill),
            "fill_opacity": path.get("fill_opacity"),
            "even_odd": path.get("even_odd"),
        }
        if tipo == "group":
            registro["grupo"] = {
                "blendmode": path.get("blendmode"),
                "opacity": path.get("opacity"),
                "isolated": path.get("isolated"),
                "knockout": path.get("knockout"),
            }
        yield registro


def registros_bboxlog(page: object, pagina: int) -> Iterator[dict[str, Any]]:
    """Gera um registro por item de :meth:`pymupdf.Page.get_bboxlog`."""
    for ordem, item in enumerate(page.get_bboxlog()):  # type: ignore[attr-defined]
        if not item:
            continue
        registro: dict[str, Any] = {
            "kind": "pintura",
            "pagina": pagina,
            "ordem": ordem,
            "tipo": item[0],
            "bbox": _caixa(item[1]) if len(item) > 1 else None,
        }
        if len(item) > 2:
            registro["ocg"] = item[2]
        yield registro


@dataclass(frozen=True)
class ClipAtivo:
    """Clip localizado na ordem de pintura por seu intervalo de ``seqno``.

    Como o PyMuPDF não dá ``seqno`` aos paths ``clip``, o intervalo é inferido
    dos paths de pintura vizinhos: um clip atende aos itens com
    ``seqno_anterior < seqno < seqno_proximo``.
    """

    seqno_anterior: int | None
    seqno_proximo: int | None
    scissor: tuple[float, float, float, float] | None
    nivel: int | None

    def cobre(self, seqno: int | None) -> bool:
        """True se o ``seqno`` informado cai no intervalo deste clip."""
        if seqno is None:
            return False
        if self.seqno_anterior is not None and not (self.seqno_anterior < seqno):
            return False
        if self.seqno_proximo is not None and not (seqno < self.seqno_proximo):
            return False
        return True


def intervalos_de_clip(
    page: object, *, drawings: list[dict[str, Any]] | None = None
) -> list[ClipAtivo]:
    """Localiza os paths ``clip`` de uma página por intervalo de ``seqno``.

    ``drawings`` permite reaproveitar um ``get_drawings(extended=True)`` já lido.
    """
    if drawings is None:
        drawings = list(page.get_drawings(extended=True))  # type: ignore[attr-defined]
    indices: list[int] = []
    seqnos: list[int] = []
    for indice, path in enumerate(drawings):
        if path.get("seqno") is not None:
            indices.append(indice)
            seqnos.append(int(path["seqno"]))
    intervalos: list[ClipAtivo] = []
    for indice, path in enumerate(drawings):
        if path.get("type") != "clip":
            continue
        pos = bisect.bisect_left(indices, indice)
        anterior = seqnos[pos - 1] if pos > 0 else None
        proximo = seqnos[pos] if pos < len(seqnos) else None
        scissor = path.get("scissor")
        caixa = (
            tuple(float(v) for v in scissor) if scissor is not None else None
        )
        intervalos.append(
            ClipAtivo(
                seqno_anterior=anterior,
                seqno_proximo=proximo,
                scissor=caixa,
                nivel=path.get("level"),
            )
        )
    return intervalos


def _intersecao_scissors(
    intervalos: Iterable[ClipAtivo],
) -> tuple[float, float, float, float] | None:
    """Interseção dos ``scissor`` dos clips ativos; ``None`` se algum for vazio."""
    resultado: tuple[float, float, float, float] | None = None
    for clip in intervalos:
        if clip.scissor is None:
            continue
        if resultado is None:
            resultado = clip.scissor
            continue
        x0 = max(resultado[0], clip.scissor[0])
        y0 = max(resultado[1], clip.scissor[1])
        x1 = min(resultado[2], clip.scissor[2])
        y1 = min(resultado[3], clip.scissor[3])
        if x1 <= x0 or y1 <= y0:
            return (x0, y0, x1, y1)
        resultado = (x0, y0, x1, y1)
    return resultado


def _ponto_em_caixa(x: float, y: float, caixa: tuple[float, ...]) -> bool:
    return caixa[0] <= x <= caixa[2] and caixa[1] <= y <= caixa[3]


def _caixas_intersectam(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def resumo_vazio(pagina: int) -> dict[str, Any]:
    """Resumo de página sem conteúdo (para páginas em branco)."""
    return {
        "kind": "resumo",
        "pagina": pagina,
        "n_spans": 0,
        "n_glifos": 0,
        "glifos_fora_do_clip": 0,
        "spans_fora_do_clip": 0,
        "glifos_cobertos_por_branco_depois": 0,
        "spans_cobertos_por_branco_depois": 0,
        "clips_que_casam_com_texto": 0,
        "spans_em_intervalo_de_clip": 0,
        "n_clips": 0,
        "n_grupos": 0,
        "clips_sem_seqno": 0,
        "pareamento_clip_texto_confiavel": False,
        "n_fills": 0,
        "n_fills_brancas": 0,
        "n_bboxlog": 0,
        "n_spans_ocultos": 0,
        "n_spans_opacity_menor_1": 0,
        "metodo_clip": (
            "intervalo (seqno_anterior, seqno_proximo) inferido dos paths de "
            "get_drawings(extended=True); clip sem seqno proprio no PyMuPDF"
        ),
    }


@dataclass
class AnalisePagina:
    """Registros JSONL e resumo de uma página."""

    registros: list[dict[str, Any]] = field(default_factory=list)
    resumo: dict[str, Any] = field(default_factory=dict)


def analisar_pagina(
    page: object,
    pagina: int,
    *,
    incluir_todos_fills: bool = False,
    max_texto: int = MAX_TEXTO_PADRAO,
) -> AnalisePagina:
    """Monta os registros e o resumo de trace de uma página (somente leitura)."""
    draws = list(page.get_drawings(extended=True))  # type: ignore[attr-defined]
    intervalos = intervalos_de_clip(page, drawings=draws)

    # Paths de preenchimento branco com seqno, para o teste de z-order.
    brancos: list[tuple[int, tuple[float, ...]]] = []
    fills_total = 0
    for path in draws:
        if path.get("type") not in TIPOS_FILL:
            continue
        fills_total += 1
        seqno = path.get("seqno")
        rect = path.get("rect")
        if seqno is None or rect is None:
            continue
        if _e_branco(path.get("fill")):
            opacidade = path.get("fill_opacity")
            if opacidade is None or float(opacidade) > 0.0:
                brancos.append((int(seqno), tuple(float(v) for v in rect)))

    brancos_ordenados = sorted(brancos)
    seqnos_brancos = [s for s, _rect in brancos_ordenados]

    registros: list[dict[str, Any]] = []
    n_spans = 0
    n_glifos = 0
    glifos_fora = 0
    spans_fora = 0
    glifos_cobertos = 0
    spans_cobertos = 0
    n_ocultos = 0
    n_opacidade = 0
    seqnos_texto: list[int] = []

    for span in page.get_texttrace():  # type: ignore[attr-defined]
        n_spans += 1
        seqno = int(span["seqno"]) if span.get("seqno") is not None else None
        if seqno is not None:
            seqnos_texto.append(seqno)
        if span.get("type") == 3:
            n_ocultos += 1
        opacidade = span.get("opacity")
        if opacidade is not None and float(opacidade) < 1.0:
            n_opacidade += 1

        ativos = [c for c in intervalos if c.cobre(seqno)]
        clip = _intersecao_scissors(ativos)
        chars = list(span.get("chars") or ())
        partes = [p for p in (_partes_char(c) for c in chars) if p is not None]
        n_glifos += len(partes)

        if seqno is None:
            brancos_futuros: list[tuple[float, ...]] = []
        else:
            posicao = bisect.bisect_right(seqnos_brancos, seqno)
            brancos_futuros = [rect for _s, rect in brancos_ordenados[posicao:]]

        fora_span = False
        coberto_span = False
        for _codigo, origem, bbox in partes:
            if clip is not None and not _ponto_em_caixa(origem[0], origem[1], clip):
                glifos_fora += 1
                fora_span = True
            if brancos_futuros and any(
                _caixas_intersectam(bbox, rect) for rect in brancos_futuros
            ):
                glifos_cobertos += 1
                coberto_span = True
        spans_fora += int(fora_span)
        spans_cobertos += int(coberto_span)

        registros.append(_registro_texto(span, pagina, max_texto=max_texto))

    registros.extend(
        registros_desenhos(
            page, pagina, incluir_todos_fills=incluir_todos_fills, drawings=draws
        )
    )

    n_pintura = 0
    for registro in registros_bboxlog(page, pagina):
        n_pintura += 1
        registros.append(registro)

    n_clips = sum(1 for path in draws if path.get("type") == "clip")
    n_grupos = sum(1 for path in draws if path.get("type") == "group")
    # "casa com o texto": intervalos de clip que contêm ao menos um seqno de texto.
    clips_casam = sum(
        1 for c in intervalos if any(c.cobre(s) for s in seqnos_texto)
    )
    spans_em_clip = sum(1 for s in seqnos_texto if any(c.cobre(s) for c in intervalos))

    resumo = {
        "kind": "resumo",
        "pagina": pagina,
        "n_spans": n_spans,
        "n_glifos": n_glifos,
        "glifos_fora_do_clip": glifos_fora,
        "spans_fora_do_clip": spans_fora,
        "glifos_cobertos_por_branco_depois": glifos_cobertos,
        "spans_cobertos_por_branco_depois": spans_cobertos,
        "clips_que_casam_com_texto": clips_casam,
        "spans_em_intervalo_de_clip": spans_em_clip,
        "n_clips": n_clips,
        "n_grupos": n_grupos,
        "clips_sem_seqno": n_clips,
        "pareamento_clip_texto_confiavel": False,
        "n_fills": fills_total,
        "n_fills_brancas": len(brancos),
        "n_bboxlog": n_pintura,
        "n_spans_ocultos": n_ocultos,
        "n_spans_opacity_menor_1": n_opacidade,
        "metodo_clip": (
            "intervalo (seqno_anterior, seqno_proximo) inferido dos paths de "
            "get_drawings(extended=True); clip sem seqno proprio no PyMuPDF"
        ),
    }
    return AnalisePagina(registros=registros, resumo=resumo)


def _validar_intervalo(caminho: str | Path, pagina: int, ate: int | None) -> int:
    """Valida o intervalo de páginas e devolve a última página (1-based)."""
    if pagina < 1:
        raise ValueError(f"página inicial inválida: {pagina} (use >= 1)")
    fim = pagina if ate is None else ate
    if fim < pagina:
        raise ValueError(f"--ate ({fim}) menor que --pagina ({pagina})")
    with pymupdf.open(str(caminho)) as doc:
        if fim > doc.page_count:
            raise ValueError(
                f"página final {fim} além do PDF ({doc.page_count} páginas)"
            )
    return fim


def _gerar_trace(
    caminho: str | Path,
    pagina: int,
    fim: int,
    *,
    incluir_todos_fills: bool,
    max_texto: int,
) -> Iterator[str]:
    with pymupdf.open(str(caminho)) as doc:
        for numero in range(pagina, fim + 1):
            analise = analisar_pagina(
                doc[numero - 1],
                numero,
                incluir_todos_fills=incluir_todos_fills,
                max_texto=max_texto,
            )
            for registro in analise.registros:
                yield json.dumps(registro, ensure_ascii=False)
            yield json.dumps(analise.resumo, ensure_ascii=False)


def inspecionar_trace(
    caminho: str | Path,
    *,
    pagina: int,
    ate: int | None = None,
    incluir_todos_fills: bool = False,
    max_texto: int = MAX_TEXTO_PADRAO,
) -> Iterator[str]:
    """Ponto de entrada de ``harness inspect --trace`` (somente leitura).

    Levanta :class:`ValueError` para entrada inválida (a CLI converte em erro).
    Devolve um iterador de linhas JSONL; cada página termina com um registro
    ``kind="resumo"``.
    """
    fim = _validar_intervalo(caminho, pagina, ate)
    return _gerar_trace(
        caminho,
        pagina,
        fim,
        incluir_todos_fills=incluir_todos_fills,
        max_texto=max_texto,
    )

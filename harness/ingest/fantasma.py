"""A1 — filtro anti-fantasma por interpretação própria do content stream (D018).

O PyMuPDF **não respeita** o *clipping* ao extrair texto e os paths ``clip`` de
``get_drawings`` não carregam ``seqno`` (D018). Este módulo interpreta o
*content stream* da página (e dos Form XObjects) e decide, por **glifo** e por
**span**, se o texto de :meth:`pymupdf.Page.get_texttrace` é visível:

- **(a) fora do clip ativo** — o operador ``re W n``/``W* n`` (inclusive dentro
  de Form XObjects) intersecciona o clip corrente; um glifo cuja ``origin`` cai
  fora do clip ativo é fantasma. Regra do diário §2.1: casar pela ``origin``,
  não pelo bbox.
- **(b) coberto por fill opaco OU imagem pintado DEPOIS** — o z-order importa:
  um fundo branco pintado **antes** do texto **não** o oculta (§2.2). Qualquer
  preenchimento (``f``/``f*``/``B``/``B*``/``b``/``b*``), de qualquer cor,
  esconde o glifo anterior se for opaco e contiver a ``origin``; imagens
  (``Do`` de XObject ``/Image``) idem. Isso cobre o caso real de DOM4197 p.2
  (fantasma 2014/2017 coberto por fill cinza + imagem, não branco).

Coordenadas: o clip, os paths e as imagens são rastreados no espaço do usuário
do PDF e convertidos para o espaço de ``get_text`` com
``page.transformation_matrix`` (origem no topo, como no §2.2).

A API principal é :func:`spans_visiveis`, que devolve uma lista de
:class:`VisibilidadeSpan` alinhada 1:1 com ``page.get_texttrace()``, cada uma
com ``visivel`` e o motivo (``fora_do_clip``/``coberto``/``fora_e_coberto``).
:func:`mascara_visivel` devolve só os booleanos, :func:`spans_fantasma` só os
invisíveis e :func:`texto_visivel` o texto reconstruído dos spans visíveis.

O pareamento entre os spans do ``get_texttrace`` e as operações de texto do
content stream usa o ``seqno`` do span (= índice do item de ``get_bboxlog``) e a
geometria da ``origin``; quando um mesmo texto é desenhado duas vezes (fantasma
+ versão atual), o PyMuPDF pode deduplicar o span e apontar para o desenho
errado. Por isso o resultado é **conservador**: usa a ocorrência do item do
``bboxlog`` e só mantém o span se ao menos um glifo escapar do clip e da
cobertura.

Robustez: nenhuma página é derrubada por content stream malformado. Operandos
inválidos e coordenadas não finitas degradam o operador (com ``warnings`` e
contador em :func:`degradacoes`); falha na interpretação inteira devolve tudo
como visível com ``motivo="erro_interpretacao"``. A interpretação é cacheada por
página (LRU pequeno) e invalidável por :func:`limpar_cache`.

Limitações registradas (honestidade de medição):

- Apenas preenchimentos/``Do`` contam para cobertura; traços (``S``/``s``) não.
- Um teste de cobertura usa a **origin** do glifo (mesma filosofia do clip);
  glifos parcialmente cobertos contam como visíveis — conservador contra a
  remoção de linhas legítimas (ex.: 1ª linha de tabela).
- Opacidade vem do ExtGState (``/ca``) corrente; default 1. Máscaras de
  transparência de imagem (``/SMask``) são tratadas como opacas.
- A geometria de paths não retangulares usa o bbox dos pontos de controle
  (aproximação); no DOM os clips e os fills de cobertura são retângulos.
- Desenhos duplicados do mesmo texto (fantasma + atual): o pareamento por item
  do ``bboxlog`` pode atribuir o span a uma ocorrência oculta; medido em
  DOM4197 p.2 (nome de contrato de 2017 que vazou para o ouro de 2026), a
  ocorrência legítima continua visível e não regride nos 241 documentos.
"""

from __future__ import annotations

import math
import re
import warnings
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, NamedTuple

try:  # PyMuPDF >= 1.24 expõe ``pymupdf``; versões antigas só ``fitz``.
    import pymupdf
except ImportError:  # pragma: no cover - compatibilidade com versões antigas
    import fitz as pymupdf  # type: ignore[no-redef]

__all__ = [
    "ALPHA_OPACA",
    "AvisoFantasma",
    "Caixa",
    "EventoPintura",
    "EventoTexto",
    "VisibilidadeSpan",
    "Interpretacao",
    "degradacoes",
    "extrair_spans",
    "interpretar_pagina",
    "limpar_cache",
    "mascara_visivel",
    "resetar_degradacoes",
    "spans_fantasma",
    "spans_visiveis",
    "texto_visivel",
]


class AvisoFantasma(UserWarning):
    """Aviso emitido quando a interpretação degrada para o modo conservador."""


#: Contador de degradações (motivo → quantidade), por processo.
_DEGRADACOES: Counter[str] = Counter()


def degradacoes() -> dict[str, int]:
    """Cópia do contador de degradações (motivo → quantidade)."""
    return dict(_DEGRADACOES)


def resetar_degradacoes() -> None:
    """Zera o contador de degradações (uso em testes)."""
    _DEGRADACOES.clear()


def _degradar(motivo: str, exc: object | None = None) -> None:
    """Registra e avisa sobre uma degradação, seguindo o processamento."""
    _DEGRADACOES[motivo] += 1
    detalhe = f": {exc}" if exc is not None else ""
    warnings.warn(
        f"fantasma: degradando ({motivo}){detalhe}", AvisoFantasma, stacklevel=3
    )

#: Alpha mínimo para considerar um preenchimento/imagem "opaco" (z-order).
ALPHA_OPACA: float = 0.99

#: Folga (pt) ao casar a origem de um evento de texto com o bbox do item do
#: ``get_bboxlog`` (as duas geometrias vêm de camadas diferentes do PyMuPDF).
TOL_CASAMENTO_PT: float = 3.0

#: Tamanho (pt) da célula da grade espacial usada na consulta de cobertura.
CELULA_GRADE_PT: float = 16.0

#: Tamanho (pt) da célula da grade de eventos de texto (fallback de casamento).
CELULA_EVENTO_PT: float = 24.0

#: Acima deste número de células um retângulo de cobertura vai para a lista de
#: retângulos grandes (consultados linearmente) em vez da grade.
MAX_CELULAS_GRANDE: int = 200

#: Operadores de pintura de path que incluem preenchimento.
OPS_FILL = frozenset({"f", "f*", "F", "B", "B*", "b", "b*"})

#: Operadores de pintura de path sem preenchimento (traço/no-op).
OPS_PAINT = OPS_FILL | {"S", "s", "n"}

#: Whitespace e delimitadores do PDF.
_WS = frozenset(b"\x00\t\n\x0c\r ")
_DELIM = frozenset(b"()<>[]{}/%")


class Caixa(NamedTuple):
    """Retângulo no espaço de ``get_text`` (origem no topo)."""

    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def de_pontos(cls, pontos: Iterable[tuple[float, float]]) -> "Caixa | None":
        """Menor caixa que contém os pontos finitos; ``None`` se vazia."""
        xs: list[float] = []
        ys: list[float] = []
        for x, y in pontos:
            if math.isfinite(x) and math.isfinite(y):
                xs.append(x)
                ys.append(y)
        if not xs:
            return None
        return cls(min(xs), min(ys), max(xs), max(ys))

    def contem(self, x: float, y: float) -> bool:
        if not (math.isfinite(x) and math.isfinite(y)):
            return False
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def intersecao(self, outra: "Caixa") -> "Caixa | None":
        x0 = max(self.x0, outra.x0)
        y0 = max(self.y0, outra.y0)
        x1 = min(self.x1, outra.x1)
        y1 = min(self.y1, outra.y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return Caixa(x0, y0, x1, y1)


@dataclass(frozen=True)
class EventoTexto:
    """Um ``Tj``/``TJ``/``'``/``"`` interpretado do content stream."""

    ordem: int
    origem: tuple[float, float]
    clip: Caixa | None
    tamanho: float


@dataclass(frozen=True)
class EventoPintura:
    """Um preenchimento/``Do`` de imagem interpretado do content stream."""

    ordem: int
    tipo: str  # "fill" | "imagem"
    caixa: Caixa
    efeito: Caixa
    alpha: float


@dataclass
class Interpretacao:
    """Eventos de texto/pintura de uma página, em ordem de pintura."""

    textos: list[EventoTexto] = field(default_factory=list)
    pinturas: list[EventoPintura] = field(default_factory=list)
    n_ops: int = 0
    erro: str | None = None


@dataclass(frozen=True)
class VisibilidadeSpan:
    """Resultado da visibilidade de um span de ``get_texttrace``."""

    indice: int
    seqno: int | None
    bbox: tuple[float, float, float, float]
    texto: str
    visivel: bool
    motivo: str
    n_glifos: int
    n_glifos_visiveis: int
    clip: Caixa | None

    @property
    def id(self) -> int:
        return self.indice


# --------------------------------------------------------------------------- #
# Tokenizador do content stream                                               #
# --------------------------------------------------------------------------- #


def _tokenizar(dados: bytes) -> Iterator[tuple[str, Any]]:
    """Gera ``(tipo, valor)`` do content stream: ``num``, ``name`` ou ``op``.

    Strings (``(...)``), strings hex (``<...>``) e dicionários (``<<...>>``) são
    consumidos mas **não** emitidos: o filtro não precisa decodificar glifos, só
    saber que houve um ``Tj``/``TJ`` e onde a origem caiu.
    """
    i = 0
    n = len(dados)
    while i < n:
        c = dados[i]
        if c in _WS:
            i += 1
            continue
        if c == 0x25:  # '%' comentário até fim de linha
            j = dados.find(b"\n", i)
            i = n if j < 0 else j + 1
            continue
        if c == 0x28:  # '(' string literal balanceada com escapes
            i += 1
            profundidade = 1
            while i < n and profundidade:
                ch = dados[i]
                if ch == 0x5C:  # '\\'
                    i += 2
                    continue
                if ch == 0x28:
                    profundidade += 1
                elif ch == 0x29:
                    profundidade -= 1
                i += 1
            continue
        if c == 0x3C:  # '<'
            if i + 1 < n and dados[i + 1] == 0x3C:
                i += 2
                profundidade = 1
                while i < n and profundidade:
                    if dados[i] == 0x3C and i + 1 < n and dados[i + 1] == 0x3C:
                        profundidade += 1
                        i += 2
                    elif dados[i] == 0x3E and i + 1 < n and dados[i + 1] == 0x3E:
                        profundidade -= 1
                        i += 2
                    else:
                        i += 1
                continue
            j = dados.find(b">", i + 1)
            i = n if j < 0 else j + 1
            continue
        if c == 0x2F:  # '/' nome
            i += 1
            inicio = i
            while i < n and dados[i] not in _WS and dados[i] not in _DELIM:
                i += 1
            yield "name", dados[inicio:i].decode("latin-1")
            continue
        if c in b"[]{}":
            i += 1
            continue
        if c in b")>":  # delimitadores não tratados acima: consome sem emitir
            i += 1
            continue
        inicio = i
        while i < n and dados[i] not in _WS and dados[i] not in _DELIM:
            i += 1
        tok = dados[inicio:i]
        try:
            yield "num", float(tok)
        except ValueError:
            yield "op", tok.decode("latin-1")


# --------------------------------------------------------------------------- #
# Recursos: ExtGState (alpha) e XObjects                                      #
# --------------------------------------------------------------------------- #

_RE_EXGSTATE = re.compile(r"/([^\s/<>\[\]]+)\s+(\d+)\s+\d+\s+R")
_RE_CA = re.compile(r"/ca\s+([0-9.]+)")
_RE_REF = re.compile(r"\s*(\d+)\s+\d+\s+R")
_RE_MATRIX = re.compile(r"/Matrix\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\]")
_RE_BBOX = re.compile(r"/BBox\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\]")


def _objeto_ou_ref(doc: Any, valor: str | None) -> str:
    """Resolve ``valor`` (dict inline ou ``N 0 R``) para o texto do objeto."""
    if not valor:
        return ""
    achado = _RE_REF.match(valor.strip())
    if achado:
        try:
            return doc.xref_object(int(achado.group(1)))
        except Exception:  # noqa: BLE001
            return ""
    return valor


def _corpo_subdicionario(doc: Any, bruto: str, chave: str) -> str:
    """Conteúdo de ``/chave`` em ``bruto``, resolvendo dict inline ou indireto."""
    m = re.search(rf"/{re.escape(chave)}\s*<<(.*?)>>", bruto, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(rf"/{re.escape(chave)}\s+(\d+)\s+\d+\s+R", bruto)
    if m:
        try:
            return doc.xref_object(int(m.group(1)))
        except Exception:  # noqa: BLE001
            return ""
    return ""


def _resolver_recursos(doc: Any, xref: int) -> dict[str, Any]:
    """Resolve ``/ExtGState`` e ``/XObject`` de recursos diretos ou indiretos."""
    try:
        _, valor = doc.xref_get_key(xref, "Resources")
    except Exception:  # noqa: BLE001 - recurso ausente/ilegível
        return {"extgstate": {}, "xobject": {}}
    bruto = _objeto_ou_ref(doc, valor)
    if not bruto:
        return {"extgstate": {}, "xobject": {}}

    extg: dict[str, float] = {}
    for nome, xr in _RE_EXGSTATE.findall(_corpo_subdicionario(doc, bruto, "ExtGState")):
        try:
            obj = doc.xref_object(int(xr))
        except Exception:  # noqa: BLE001
            continue
        achado = _RE_CA.search(obj)
        try:
            extg[nome] = float(achado.group(1)) if achado else 1.0
        except (TypeError, ValueError):
            extg[nome] = 1.0

    xobj: dict[str, tuple[str, int]] = {}
    for nome, xr in _RE_EXGSTATE.findall(_corpo_subdicionario(doc, bruto, "XObject")):
        try:
            obj = doc.xref_object(int(xr))
        except Exception:  # noqa: BLE001
            continue
        if "/Subtype/Image" in obj.replace(" ", "") or "/Subtype /Image" in obj:
            xobj[nome] = ("image", int(xr))
        else:
            xobj[nome] = ("form", int(xr))
    return {"extgstate": extg, "xobject": xobj}


def _matriz_do_xobject(doc: Any, xref: int) -> tuple[float, float, float, float, float, float]:
    """Matriz ``/Matrix`` do Form XObject (identidade se ausente)."""
    try:
        obj = doc.xref_object(xref)
    except Exception:  # noqa: BLE001
        return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    m = _RE_MATRIX.search(obj)
    if not m:
        return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    return tuple(float(v) for v in m.groups())  # type: ignore[return-value]


def _bbox_do_xobject(doc: Any, xref: int) -> tuple[float, float, float, float] | None:
    """``/BBox`` do Form XObject, se houver."""
    try:
        obj = doc.xref_object(xref)
    except Exception:  # noqa: BLE001
        return None
    m = _RE_BBOX.search(obj)
    if not m:
        return None
    return tuple(float(v) for v in m.groups())  # type: ignore[return-value]


def _stream_do_xobject(doc: Any, xref: int) -> bytes:
    try:
        return doc.xref_stream(xref) or b""
    except Exception:  # noqa: BLE001
        return b""


# --------------------------------------------------------------------------- #
# Interpretador                                                               #
# --------------------------------------------------------------------------- #


Matriz = tuple[float, float, float, float, float, float]

_IDENTIDADE: Matriz = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _mat6(matriz: Any) -> Matriz:
    """Converte ``pymupdf.Matrix`` (ou sequência de 6) em tupla ``(a,b,c,d,e,f)``."""
    if isinstance(matriz, tuple) and len(matriz) == 6:
        return matriz
    a = matriz.a
    return (float(a), float(matriz.b), float(matriz.c), float(matriz.d), float(matriz.e), float(matriz.f))


def _mul(m1: Matriz, m2: Matriz) -> Matriz:
    """Composição ``m1`` **depois** ``m2`` (convenção de vetor-linha do PyMuPDF)."""
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
        e1 * a2 + f1 * c2 + e2,
        e1 * b2 + f1 * d2 + f2,
    )


def _ponto(matriz: Matriz, x: float, y: float) -> tuple[float, float]:
    """Aplica ``matriz`` a ``(x, y)`` (multiplicação por vetor-linha)."""
    a, b, c, d, e, f = matriz
    return (a * x + c * y + e, b * x + d * y + f)


def _como_float(valor: Any) -> float | None:
    """Converte um operando em float finito; ``None`` para nome/inválido/NaN/∞."""
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        return None
    numero = float(valor)
    return numero if math.isfinite(numero) else None


def _operandos(pilha: list[Any], quantidade: int) -> tuple[float, ...] | None:
    """Últimos ``quantidade`` operandos como floats finitos; ``None`` se inválidos."""
    if len(pilha) < quantidade:
        return None
    valores = pilha[-quantidade:]
    numeros: list[float] = []
    for valor in valores:
        numero = _como_float(valor)
        if numero is None:
            return None
        numeros.append(numero)
    return tuple(numeros)


class _Interpretador:
    """Máquina de estados do content stream (q/Q/cm/re/W n/BT/Tm/Td/Tj/TJ/Do...)."""

    def __init__(self, doc: Any, page: Any) -> None:
        self.doc = doc
        self.page = page
        self.page_matrix: Matriz = _mat6(page.transformation_matrix)
        self.eventos = Interpretacao()
        self._ordem = 0

    # -- helpers -------------------------------------------------------- #

    def _registrar_texto(self, origem: tuple[float, float], clip: Caixa | None, tamanho: float) -> None:
        self._ordem += 1
        self.eventos.textos.append(
            EventoTexto(ordem=self._ordem, origem=origem, clip=clip, tamanho=tamanho)
        )

    def _registrar_pintura(
        self, tipo: str, caixa: Caixa, clip: Caixa | None, alpha: float
    ) -> None:
        efeito = caixa if clip is None else (caixa.intersecao(clip) or Caixa(0, 0, 0, 0))
        if efeito.x1 <= efeito.x0 or efeito.y1 <= efeito.y0:
            return
        self._ordem += 1
        self.eventos.pinturas.append(
            EventoPintura(ordem=self._ordem, tipo=tipo, caixa=caixa, efeito=efeito, alpha=alpha)
        )

    # -- execução ------------------------------------------------------- #

    def executar(self) -> Interpretacao:
        try:
            data = self.page.read_contents()
        except Exception as exc:  # noqa: BLE001 - página sem content stream legível
            self.eventos.erro = f"read_contents: {exc}"
            _degradar("read_contents", exc)
            return self.eventos
        recursos = _resolver_recursos(self.doc, self.page.xref)
        self._executar(data, _IDENTIDADE, None, recursos, 0, 1.0)
        self.eventos.n_ops = self._ordem
        return self.eventos

    def _executar(
        self,
        data: bytes,
        ctm: Matriz,
        clip: Caixa | None,
        recursos: dict[str, Any],
        profundidade: int,
        alpha_in: float = 1.0,
    ) -> None:
        if not data or profundidade > 12:
            return
        extgstate: dict[str, float] = recursos.get("extgstate", {})
        xobjects: dict[str, tuple[str, int]] = recursos.get("xobject", {})

        pilha: list[Any] = []
        q_pilha: list[tuple[Matriz, Caixa | None, float]] = []
        alpha = alpha_in
        tamanho = 0.0

        tm: Matriz = _IDENTIDADE
        tlm: Matriz = tm
        leading = 0.0
        pts: list[tuple[float, float]] = []
        rects: list[tuple[float, float, float, float]] = []

        try:
            for tipo_tok, valor in _tokenizar(data):
                if tipo_tok != "op":
                    pilha.append(valor)
                    continue
                op = valor
                if op == "q":
                    q_pilha.append((ctm, clip, alpha))
                elif op == "Q":
                    if q_pilha:
                        ctm, clip, alpha = q_pilha.pop()
                elif op == "cm":
                    nums = _operandos(pilha, 6)
                    if nums is None:
                        _degradar("operando_invalido", op)
                    else:
                        ctm = _mul(nums, ctm)  # type: ignore[arg-type]
                elif op == "re":
                    nums = _operandos(pilha, 4)
                    if nums is None:
                        _degradar("operando_invalido", op)
                    else:
                        rects.append(nums)  # type: ignore[arg-type]
                elif op in ("m", "l"):
                    nums = _operandos(pilha, 2)
                    if nums is None:
                        _degradar("operando_invalido", op)
                    else:
                        pts.append(nums)  # type: ignore[arg-type]
                elif op == "c":
                    nums = _operandos(pilha, 6)
                    if nums is None:
                        _degradar("operando_invalido", op)
                    else:
                        pts.extend([nums[0:2], nums[2:4], nums[4:6]])  # type: ignore[arg-type]
                elif op in ("v", "y"):
                    nums = _operandos(pilha, 4)
                    if nums is None:
                        _degradar("operando_invalido", op)
                    else:
                        pts.extend([nums[0:2], nums[2:4]])  # type: ignore[arg-type]
                elif op == "h":
                    if rects:
                        x, y, _w, _h = rects[-1]
                        pts.append((x, y))
                elif op in ("W", "W*"):
                    caixa_clip = self._caixa_path(ctm, pts, rects)
                    if caixa_clip is None:
                        _degradar("clip_invalido", op)
                    else:
                        clip = (
                            caixa_clip
                            if clip is None
                            else (clip.intersecao(caixa_clip) or Caixa(0, 0, 0, 0))
                        )
                elif op in OPS_PAINT:
                    if op in OPS_FILL:
                        caixa = self._caixa_path(ctm, pts, rects)
                        if caixa is not None:
                            self._registrar_pintura("fill", caixa, clip, alpha)
                    pts = []
                    rects = []
                elif op == "BT":
                    tm = _IDENTIDADE
                    tlm = tm
                elif op == "ET":
                    pass
                elif op == "Tm":
                    nums = _operandos(pilha, 6)
                    if nums is None:
                        _degradar("operando_invalido", op)
                    else:
                        tm = nums  # type: ignore[assignment]
                        tlm = tm
                elif op in ("Td", "TD"):
                    nums = _operandos(pilha, 2)
                    if nums is None:
                        _degradar("operando_invalido", op)
                    else:
                        tx, ty = nums
                        tlm = _mul((1.0, 0.0, 0.0, 1.0, tx, ty), tlm)
                        tm = tlm
                        if op == "TD":
                            leading = -ty
                elif op == "TL":
                    nums = _operandos(pilha, 1)
                    if nums is None:
                        _degradar("operando_invalido", op)
                    else:
                        leading = nums[0]
                elif op == "T*":
                    tlm = _mul((1.0, 0.0, 0.0, 1.0, 0.0, -leading), tlm)
                    tm = tlm
                elif op == "Tf":
                    tamanho_ = _como_float(pilha[-1]) if pilha else None
                    if tamanho_ is None:
                        _degradar("operando_invalido", op)
                    else:
                        tamanho = tamanho_
                elif op in ("Tj", "TJ", "'", '"'):
                    if op in ("'", '"'):
                        tlm = _mul((1.0, 0.0, 0.0, 1.0, 0.0, -leading), tlm)
                        tm = tlm
                    origem = _ponto(_mul(tm, _mul(ctm, self.page_matrix)), 0.0, 0.0)
                    self._registrar_texto(origem, clip, tamanho)
                elif op == "Do":
                    nome = pilha[-1] if pilha else None
                    if isinstance(nome, str) and nome in xobjects:
                        kind, xref = xobjects[nome]
                        if kind == "image":
                            caixa = self._caixa_imagem(ctm)
                            if caixa is not None:
                                self._registrar_pintura("imagem", caixa, clip, alpha)
                        else:
                            self._executar_form(xref, ctm, clip, profundidade, alpha)
                elif op == "gs":
                    nome = pilha[-1] if pilha else None
                    if isinstance(nome, str) and nome in extgstate:
                        alpha = extgstate[nome]
                pilha = []
        except Exception as exc:  # noqa: BLE001 - degrada em vez de derrubar a página
            self.eventos.erro = str(exc)
            _degradar("execucao_interrompida", exc)

    def _executar_form(
        self,
        xref: int,
        ctm: Matriz,
        clip: Caixa | None,
        profundidade: int,
        alpha: float,
    ) -> None:
        matriz = _matriz_do_xobject(self.doc, xref)
        ctm_form = _mul(matriz, ctm)
        bbox = _bbox_do_xobject(self.doc, xref)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            transformacao = _mul(ctm_form, self.page_matrix)
            caixa_form = Caixa.de_pontos(
                _ponto(transformacao, x, y)
                for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))
            )
            if caixa_form is not None:
                clip = caixa_form if clip is None else (clip.intersecao(caixa_form) or Caixa(0, 0, 0, 0))
        recursos = _resolver_recursos(self.doc, xref)
        self._executar(
            _stream_do_xobject(self.doc, xref),
            ctm_form,
            clip,
            recursos,
            profundidade + 1,
            alpha,
        )

    def _caixa_path(
        self,
        ctm: Matriz,
        pts: list[tuple[float, float]],
        rects: list[tuple[float, float, float, float]],
    ) -> Caixa | None:
        transformacao = _mul(ctm, self.page_matrix)
        pontos: list[tuple[float, float]] = []
        for x, y in pts:
            pontos.append(_ponto(transformacao, x, y))
        for x, y, w, h in rects:
            pontos.extend(
                _ponto(transformacao, px, py)
                for px, py in ((x, y), (x + w, y), (x, y + h), (x + w, y + h))
            )
        return Caixa.de_pontos(pontos)

    def _caixa_imagem(self, ctm: Matriz) -> Caixa | None:
        return Caixa.de_pontos(
            _ponto(_mul(ctm, self.page_matrix), x, y)
            for x, y in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
        )


def interpretar_pagina(page: Any) -> Interpretacao:
    """Interpreta o content stream da página (somente leitura).

    Nunca levanta exceção: em falha devolve uma :class:`Interpretacao` vazia com
    ``erro`` preenchido (o chamador degrada para "tudo visível") e emite aviso.
    """
    try:
        return _Interpretador(page.parent, page).executar()
    except Exception as exc:  # noqa: BLE001 - degrada, nunca derruba a página
        _degradar("interpretacao", exc)
        return Interpretacao(erro=str(exc))


# --------------------------------------------------------------------------- #
# Casamento span × evento e máscara de visibilidade                           #
# --------------------------------------------------------------------------- #


def _texto_do_span(span: dict) -> str:
    partes: list[str] = []
    for char in span.get("chars") or ():
        try:
            codigo = int(char[0])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= codigo <= 0x10FFFF:
            partes.append(chr(codigo))
    return "".join(partes)


def _partes_char(char: Any) -> tuple[tuple[float, float], tuple[float, float, float, float]] | None:
    try:
        origem = (float(char[2][0]), float(char[2][1]))
        bbox = tuple(float(v) for v in char[3])
    except (TypeError, ValueError, IndexError):
        return None
    if len(bbox) != 4:
        return None
    return origem, bbox  # type: ignore[return-value]


def _expandir(caixa: tuple[float, float, float, float], folga: float) -> Caixa:
    return Caixa(caixa[0] - folga, caixa[1] - folga, caixa[2] + folga, caixa[3] + folga)


def _celula(x: float, y: float, tamanho: float) -> tuple[int, int] | None:
    """Índice de célula da grade para ``(x, y)``; ``None`` se não finito."""
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    try:
        return (int(x // tamanho), int(y // tamanho))
    except (OverflowError, ValueError):
        return None


def _evento_mais_proximo(
    origem: tuple[float, float], candidatos: list[EventoTexto]
) -> EventoTexto | None:
    if not candidatos:
        return None
    ox, oy = origem
    melhor: EventoTexto | None = None
    melhor_dist = float("inf")
    for evento in candidatos:
        dx = evento.origem[0] - ox
        dy = evento.origem[1] - oy
        dist = dx * dx + dy * dy
        if dist < melhor_dist:
            melhor_dist = dist
            melhor = evento
    return melhor


class _GradeEventos:
    """Grade espacial dos eventos de texto, para o fallback por vizinhança."""

    def __init__(self, eventos: list[EventoTexto]) -> None:
        self.celulas: dict[tuple[int, int], list[EventoTexto]] = {}
        for evento in eventos:
            chave = _celula(evento.origem[0], evento.origem[1], CELULA_EVENTO_PT)
            if chave is None:
                continue
            self.celulas.setdefault(chave, []).append(evento)

    def locais(self, x: float, y: float) -> list[EventoTexto]:
        base = _celula(x, y, CELULA_EVENTO_PT)
        if base is None:
            return []
        ix, iy = base
        achados: list[EventoTexto] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                achados.extend(self.celulas.get((ix + dx, iy + dy), ()))
        return achados


class _GradeCobertura:
    """Grade espacial das pinturas opacas, consultada com varredura por ordem.

    As consultas (glifos) e as pinturas são varridas em ordem **decrescente**:
    quando se consulta um glifo de ordem ``o``, a grade contém exatamente as
    pinturas com ``ordem > o`` — o teste de z-order correto.
    """

    def __init__(self) -> None:
        self.celulas: dict[tuple[int, int], list[Caixa]] = {}
        self.grandes: list[Caixa] = []

    def inserir(self, caixa: Caixa) -> None:
        cantos = (
            _celula(caixa.x0, caixa.y0, CELULA_GRADE_PT),
            _celula(caixa.x1, caixa.y1, CELULA_GRADE_PT),
        )
        if any(canto is None for canto in cantos):
            return
        (ix0, iy0), (ix1, iy1) = cantos  # type: ignore[misc]
        n_celulas = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
        if n_celulas > MAX_CELULAS_GRANDE:
            self.grandes.append(caixa)
            return
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                self.celulas.setdefault((ix, iy), []).append(caixa)

    def contem(self, x: float, y: float) -> bool:
        for caixa in self.grandes:
            if caixa.contem(x, y):
                return True
        base = _celula(x, y, CELULA_GRADE_PT)
        if base is None:
            return False
        return any(c.contem(x, y) for c in self.celulas.get(base, ()))


def _cobertura_por_glifo(
    consultas: list[tuple[tuple[float, float], int]],
    pinturas: list[EventoPintura],
) -> list[bool]:
    """Para cada ``(origem, ordem)``, True se **não** coberta por pintura opaca posterior."""
    opacas = sorted(
        (p for p in pinturas if p.alpha >= ALPHA_OPACA),
        key=lambda p: p.ordem,
        reverse=True,
    )
    ordem_consultas = sorted(
        range(len(consultas)), key=lambda i: consultas[i][1], reverse=True
    )
    grade = _GradeCobertura()
    resultado = [True] * len(consultas)
    ponteiro = 0
    for indice in ordem_consultas:
        origem, limite = consultas[indice]
        while ponteiro < len(opacas) and opacas[ponteiro].ordem > limite:
            grade.inserir(opacas[ponteiro].efeito)
            ponteiro += 1
        resultado[indice] = not grade.contem(*origem)
    return resultado


def _mapear_eventos_para_itens(
    eventos: list[EventoTexto], bboxlog: list[Any]
) -> dict[int, list[EventoTexto]]:
    """Agrupa os eventos de texto pelo item correspondente do ``get_bboxlog``.

    Os dois lados estão em ordem de pintura; o item do bboxlog agrega um ou mais
    ``Tj``/``TJ`` consecutivos. Uma grade espacial acha os itens cujo bbox contém
    a origem; entre eles escolhe-se o de menor índice **desde o último item**,
    o que mantém o pareamento monotônico sem varrer milhares de itens.
    """
    indices = [i for i, item in enumerate(bboxlog) if item and item[0] == "fill-text"]
    caixas = {
        i: _expandir(tuple(float(v) for v in bboxlog[i][1]), TOL_CASAMENTO_PT)
        for i in indices
    }
    celulas: dict[tuple[int, int], list[tuple[int, Caixa]]] = {}
    grandes: list[tuple[int, Caixa]] = []
    for idx in indices:
        caixa = caixas[idx]
        cantos = (
            _celula(caixa.x0, caixa.y0, CELULA_GRADE_PT),
            _celula(caixa.x1, caixa.y1, CELULA_GRADE_PT),
        )
        if any(canto is None for canto in cantos):
            continue
        (ix0, iy0), (ix1, iy1) = cantos  # type: ignore[misc]
        if (ix1 - ix0 + 1) * (iy1 - iy0 + 1) > MAX_CELULAS_GRANDE:
            grandes.append((idx, caixa))
            continue
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                celulas.setdefault((ix, iy), []).append((idx, caixa))

    def candidatos(x: float, y: float) -> list[int]:
        achados = [idx for idx, caixa in grandes if caixa.contem(x, y)]
        base = _celula(x, y, CELULA_GRADE_PT)
        if base is None:
            return achados
        for idx, caixa in celulas.get(base, ()):
            if caixa.contem(x, y):
                achados.append(idx)
        return achados

    por_item: dict[int, list[EventoTexto]] = {}
    j = -1
    for evento in eventos:
        x, y = evento.origem
        achados = [idx for idx in candidatos(x, y) if idx >= j]
        if not achados:
            achados = candidatos(x, y)
        if not achados:
            continue
        idx = min(achados)
        por_item.setdefault(idx, []).append(evento)
        if idx > j:
            j = idx
    return por_item


#: Cache LRU (por página) da interpretação do content stream.
_CACHE_INTERP: "OrderedDict[tuple[int, int], tuple[Any, Interpretacao, list[Any]]]" = (
    OrderedDict()
)
_CACHE_TAMANHO = 8


def limpar_cache() -> None:
    """Limpa o cache de interpretação por página (uso em testes)."""
    _CACHE_INTERP.clear()


def _interpretar_com_cache(page: Any) -> tuple[Interpretacao, list[Any]]:
    """Interpreta a página reusando o resultado anterior (LRU pequeno)."""
    try:
        chave: tuple[int, int] | None = (id(page.parent), int(page.number))
    except Exception:  # noqa: BLE001 - página sem pai/número válidos
        chave = None
    if chave is not None:
        entrada = _CACHE_INTERP.get(chave)
        if entrada is not None and entrada[0] is page.parent:
            _CACHE_INTERP.move_to_end(chave)
            return entrada[1], entrada[2]
    interp = interpretar_pagina(page)
    try:
        bboxlog = list(page.get_bboxlog())
    except Exception as exc:  # noqa: BLE001
        _degradar("bboxlog", exc)
        bboxlog = []
    if chave is not None:
        _CACHE_INTERP[chave] = (page.parent, interp, bboxlog)
        while len(_CACHE_INTERP) > _CACHE_TAMANHO:
            _CACHE_INTERP.popitem(last=False)
    return interp, bboxlog


def _bbox_do_span(span: dict) -> tuple[float, float, float, float]:
    try:
        return tuple(float(v) for v in span.get("bbox", (0, 0, 0, 0)))  # type: ignore[return-value]
    except (TypeError, ValueError):
        return (0.0, 0.0, 0.0, 0.0)


def _spans_todos_visiveis(
    spans: list[dict], motivo: str = "erro_interpretacao"
) -> list[VisibilidadeSpan]:
    """Fallback conservador: mantém todos os spans como visíveis."""
    return [
        VisibilidadeSpan(
            indice=indice,
            seqno=span.get("seqno"),
            bbox=_bbox_do_span(span),
            texto=_texto_do_span(span),
            visivel=True,
            motivo=motivo,
            n_glifos=len(span.get("chars") or ()),
            n_glifos_visiveis=len(span.get("chars") or ()),
            clip=None,
        )
        for indice, span in enumerate(spans)
    ]


def _visibilidade_normal(
    spans: list[dict], interp: Interpretacao, bboxlog: list[Any]
) -> list[VisibilidadeSpan]:
    """Calcula a máscara de visibilidade a partir dos eventos interpretados."""
    eventos = interp.textos
    por_item = _mapear_eventos_para_itens(eventos, bboxlog)
    grade_eventos = _GradeEventos(eventos)

    # 1) Monta as consultas por glifo (origem, evento de texto casado).
    consultas: list[tuple[tuple[float, float], EventoTexto | None]] = []
    limites_por_span: list[tuple[int, int]] = []
    for span in spans:
        seqno = span.get("seqno")
        candidatos = por_item.get(int(seqno), []) if seqno is not None else []
        inicio = len(consultas)
        for char in span.get("chars") or ():
            partes = _partes_char(char)
            if partes is None:
                continue
            origem, _bbox_char = partes
            if not (math.isfinite(origem[0]) and math.isfinite(origem[1])):
                evento = None
            elif candidatos:
                evento = _evento_mais_proximo(origem, candidatos)
            else:
                evento = _evento_mais_proximo(origem, grade_eventos.locais(*origem))
            consultas.append((origem, evento))
        limites_por_span.append((inicio, len(consultas)))

    # 2) Cobertura (z-order) para todos os glifos de uma vez.
    cobertos = _cobertura_por_glifo(
        [(origem, ev.ordem if ev is not None else -1) for origem, ev in consultas],
        interp.pinturas,
    )

    resultado: list[VisibilidadeSpan] = []
    for indice, span in enumerate(spans):
        bbox = _bbox_do_span(span)
        texto = _texto_do_span(span)
        inicio, fim = limites_por_span[indice]

        n_glifos = 0
        n_visiveis = 0
        fora_clip = False
        coberto = False
        clip_do_span: Caixa | None = None

        for pos in range(inicio, fim):
            origem, evento = consultas[pos]
            n_glifos += 1
            if evento is None:
                n_visiveis += 1
                continue
            if clip_do_span is None and evento.clip is not None:
                clip_do_span = evento.clip
            dentro_clip = evento.clip is None or evento.clip.contem(*origem)
            if not dentro_clip:
                fora_clip = True
            elif not cobertos[pos]:
                coberto = True
            else:
                n_visiveis += 1

        if n_visiveis > 0:
            motivo = "visivel"
            visivel = True
        elif fora_clip and coberto:
            motivo = "fora_e_coberto"
            visivel = False
        elif fora_clip:
            motivo = "fora_do_clip"
            visivel = False
        elif coberto:
            motivo = "coberto"
            visivel = False
        else:
            motivo = "sem_evento"
            visivel = True

        resultado.append(
            VisibilidadeSpan(
                indice=indice,
                seqno=span.get("seqno"),
                bbox=bbox,  # type: ignore[arg-type]
                texto=texto,
                visivel=visivel,
                motivo=motivo,
                n_glifos=n_glifos,
                n_glifos_visiveis=n_visiveis,
                clip=clip_do_span,
            )
        )
    return resultado


def spans_visiveis(page: Any) -> list[VisibilidadeSpan]:
    """Máscara de visibilidade por span de ``page.get_texttrace()``.

    Devolve uma lista alinhada 1:1 com ``get_texttrace()``. Cada span traz
    ``visivel``, ``motivo`` e a contagem de glifos visíveis. Nunca levanta:
    qualquer erro de interpretação degrada para ``visivel=True`` com
    ``motivo="erro_interpretacao"`` e aviso (contador em :func:`degradacoes`).
    """
    spans = list(page.get_texttrace())
    if not spans:
        return []

    interp, bboxlog = _interpretar_com_cache(page)
    if interp.erro:
        return _spans_todos_visiveis(spans)
    try:
        return _visibilidade_normal(spans, interp, bboxlog)
    except Exception as exc:  # noqa: BLE001 - degrada em vez de derrubar a página
        _degradar("pos_processamento", exc)
        return _spans_todos_visiveis(spans)


def mascara_visivel(page: Any) -> list[bool]:
    """Só os booleanos de visibilidade, alinhados a ``get_texttrace()``."""
    return [s.visivel for s in spans_visiveis(page)]


def spans_fantasma(page: Any) -> list[VisibilidadeSpan]:
    """Apenas os spans considerados fantasma (invisíveis), na ordem da extração."""
    return [s for s in spans_visiveis(page) if not s.visivel]


def texto_visivel(page: Any, *, separador: str = "\n") -> str:
    """Texto reconstruído apenas com os spans visíveis (limpeza de NUL incluída)."""
    partes = [s.texto for s in spans_visiveis(page) if s.visivel and s.texto.strip()]
    limpo = separador.join(partes)
    return re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", limpo)


def extrair_spans(page: Any) -> list[dict]:
    """Spans visíveis (dicts crus de ``get_texttrace``) — conveniência de API."""
    brutos = list(page.get_texttrace())
    return [brutos[v.indice] for v in spans_visiveis(page) if v.visivel]

"""Inspeção somente-leitura de páginas de PDF (ferramenta do time de dados).

Reconstrói **linhas lógicas** a partir dos spans do PyMuPDF para que o time de
dados enxergue negrito, bbox, fonte e colunas sem depender do ``pdftotext``
(ver ``knowledge/regras_diario.md``, seções 1 e 2).

Regras implementadas:

- **Negrito** = ``flags & 16`` **ou** nome da fonte contendo
  ``Bold``/``Black``/``Semibold`` (etc.). O critério que disparou é preservado
  para diagnóstico.
- **Linha lógica** = spans na mesma altura (baseline do glifo) fundidos quando o
  vão horizontal é ≤ 8pt; vão maior separa colunas em linhas distintas.
- **Limpeza** de caracteres de controle/NUL, preservando ``\\n``/``\\t``.

Esta ferramenta é **somente leitura**: nunca grava no PDF nem em ``diarios/``.
O filtro anti-fantasma (clipping do content stream) **não** é implementado aqui —
apenas uma marcação heurística opcional ``[?fantasma]`` em linhas que contêm ano
fora do ano da edição no masthead.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

try:  # PyMuPDF >= 1.24 expõe ``pymupdf``; versões antigas só ``fitz``.
    import pymupdf
except ImportError:  # pragma: no cover - compatibilidade com versões antigas
    import fitz as pymupdf  # type: ignore[no-redef]

from harness.contracts import LinhaLogica, Span

__all__ = [
    "FORMATOS",
    "LinhaInfo",
    "SpanInfo",
    "ano_do_masthead",
    "detectar_negrito",
    "extrair_linhas",
    "extrair_linhas_logicas",
    "filtrar_linhas",
    "formatar_jsonl",
    "formatar_texto",
    "inspecionar",
    "limpar_texto",
    "linha_tem_ano_fora",
    "montar_linhas_info",
    "nome_indica_negrito",
    "normalizar_busca",
]

#: Formatos de saída aceitos por :func:`inspecionar`.
FORMATOS: tuple[str, ...] = ("texto", "jsonl")

#: Tolerância (pt) de diferença de baseline para considerar spans na mesma linha.
TOLERANCIA_Y = 2.0

#: Vão horizontal máximo (pt) para fundir dois spans numa mesma linha lógica.
VAO_MAX_PT = 8.0

#: Distância mínima (pt) para inserir um espaço entre textos fundidos.
ESPACO_MIN_PT = 0.5

#: Sobreposição (pt) a partir da qual tratamos como span duplicado (camada fantasma).
OVERLAP_MIN_PT = -1.0

#: Caracteres de controle C0/C1 e DEL — só ``\t`` (0x09) e ``\n`` (0x0A) são preservados.
_CONTROLE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")

#: Runs de espaço/tab são colapsados a um único espaço na linha lógica.
_ESPACOS = re.compile(r"[ \t]{2,}")

#: Palavras que indicam fonte em negrito no nome do PostScript name.
_PALAVRAS_NEGRITO = ("bold", "black", "semibold", "semi-bold", "extrabold", "heavy")

#: ``Ano 2026`` no masthead.
_ANO_MASTHEAD = re.compile(r"\bAno\s+(\d{4})\b", re.IGNORECASE)

#: Fallback: ``... de 12 de fevereiro de 2026``.
_DATA_MASTHEAD = re.compile(r"\bde\s+\d{1,2}\s+de\s+\S+\s+de\s+(\d{4})\b", re.IGNORECASE)

#: Anos plausíveis (usados só na heurística de fantasma).
_ANO_PLAUSIVEL = re.compile(r"\b(19\d{2}|20\d{2})\b")


def limpar_texto(texto: str) -> str:
    """Remove caracteres de controle/NUL preservando ``\\n`` e ``\\t``."""
    if not texto:
        return ""
    return _CONTROLE.sub("", texto)


def nome_indica_negrito(fonte: str) -> bool:
    """True se o nome da fonte sugere negrito (Bold/Black/Semibold...)."""
    base = (fonte or "").casefold().replace(" ", "").replace("-", "")
    return any(palavra in base for palavra in _PALAVRAS_NEGRITO)


def detectar_negrito(flags: int, fonte: str) -> tuple[bool, str]:
    """Detecta negrito por flag do PyMuPDF e/ou nome da fonte.

    Retorna ``(negrito, criterio)`` onde ``criterio`` é ``"flags"``, ``"fonte"``,
    ``"flags+fonte"`` ou ``""`` (não negrito) — é o "qual critério disparou".
    """
    por_flags = bool(int(flags) & 16)
    por_fonte = nome_indica_negrito(fonte)
    if por_flags and por_fonte:
        return True, "flags+fonte"
    if por_flags:
        return True, "flags"
    if por_fonte:
        return True, "fonte"
    return False, ""


def normalizar_busca(texto: str) -> str:
    """Normaliza para busca insensível a caixa e acentos."""
    decomposto = unicodedata.normalize("NFKD", texto or "")
    return "".join(c for c in decomposto if not unicodedata.combining(c)).casefold()


def _baseline_do_span(bruto: dict) -> float:
    """Baseline (origem y) do span; cai para o rodapé do bbox se ausente."""
    origem = bruto.get("origin")
    bbox = bruto["bbox"]
    if origem and len(origem) >= 2:
        return float(origem[1])
    return float(bbox[3])


@dataclass(frozen=True)
class SpanInfo:
    """Span do contrato + metadados internos de inspeção (não vão para o JSONL)."""

    span: Span
    criterio_negrito: str
    baseline: float


@dataclass(frozen=True)
class LinhaInfo:
    """Linha lógica + critério de negrito por span (paralelo a ``linha.spans``)."""

    linha: LinhaLogica
    criterios: tuple[str, ...]

    @property
    def criterio_inicio(self) -> str:
        """Critério de negrito do primeiro span (início) da linha, se houver."""
        return self.criterios[0] if self.criterios else ""


def extrair_spans(page: object, pagina: int) -> list[SpanInfo]:
    """Extrai os spans de uma página como :class:`SpanInfo` (ordem de leitura crua)."""
    dados = page.get_text("dict")  # type: ignore[attr-defined]
    infos: list[SpanInfo] = []
    for bloco in dados.get("blocks", []):
        if bloco.get("type") != 0:
            continue
        for linha in bloco.get("lines", []):
            for bruto in linha.get("spans", []):
                texto = limpar_texto(bruto.get("text", ""))
                if not texto.strip():
                    # Descarta spans só de espaço/tab: não carregam conteúdo útil.
                    continue
                bbox = tuple(float(v) for v in bruto["bbox"])
                fonte = str(bruto.get("font", ""))
                flags = int(bruto.get("flags", 0))
                negrito, criterio = detectar_negrito(flags, fonte)
                infos.append(
                    SpanInfo(
                        span=Span(
                            texto=texto,
                            bbox=bbox,
                            fonte=fonte,
                            tamanho=float(bruto.get("size", 0.0)),
                            negrito=negrito,
                        ),
                        criterio_negrito=criterio,
                        baseline=_baseline_do_span(bruto),
                    )
                )
    return infos


def _clusterizar_por_baseline(infos: list[SpanInfo]) -> list[tuple[list[SpanInfo], float]]:
    """Agrupa spans em faixas de mesma baseline (tolerância :data:`TOLERANCIA_Y`)."""
    ordenados = sorted(infos, key=lambda info: (info.baseline, info.span.bbox[0]))
    grupos: list[list[SpanInfo]] = []
    bases: list[float] = []
    for info in ordenados:
        for indice, base in enumerate(bases):
            if abs(base - info.baseline) <= TOLERANCIA_Y:
                grupos[indice].append(info)
                n = len(grupos[indice])
                bases[indice] = (base * (n - 1) + info.baseline) / n
                break
        else:
            grupos.append([info])
            bases.append(info.baseline)
    return list(zip(grupos, bases))


def _juntar_textos(run: list[SpanInfo]) -> str:
    """Concatena os spans de um run sem gerar espaços consecutivos.

    Os spans do PyMuPDF costumam já terminar em espaço; por isso só inserimos um
    separador quando necessário e, no fim, colapsamos runs de espaço/tab. Sem isso
    o `--busca` com espaçamento natural falha (ex.: ``"JANAINA E SILVA AUXILIAR"``).
    """
    partes: list[str] = []
    prev_x1: float | None = None
    for info in run:
        x0, x1 = info.span.bbox[0], info.span.bbox[2]
        if prev_x1 is not None:
            vao = x0 - prev_x1
            anterior = partes[-1] if partes else ""
            proximo = info.span.texto
            if (vao > ESPACO_MIN_PT or vao < OVERLAP_MIN_PT) and not (
                anterior.endswith((" ", "\t")) or proximo.startswith((" ", "\t"))
            ):
                partes.append(" ")
        partes.append(info.span.texto)
        prev_x1 = x1
    return _ESPACOS.sub(" ", "".join(partes)).strip()


def _linha_de_run(pagina: int, baseline: float, run: list[SpanInfo]) -> LinhaInfo:
    """Constrói a :class:`LinhaInfo` de um run horizontal contíguo."""
    x0 = min(info.span.bbox[0] for info in run)
    x1 = max(info.span.bbox[2] for info in run)
    linha = LinhaLogica(
        pagina=pagina,
        y=round(baseline, 1),
        x0=round(x0, 1),
        x1=round(x1, 1),
        texto=_juntar_textos(run),
        negrito_inicio=run[0].span.negrito,
        spans=[info.span for info in run],
    )
    return LinhaInfo(linha=linha, criterios=tuple(info.criterio_negrito for info in run))


def montar_linhas_info(infos: list[SpanInfo], pagina: int) -> list[LinhaInfo]:
    """Monta linhas lógicas de uma página a partir de :class:`SpanInfo`.

    Spans na mesma baseline são fundidos enquanto o vão horizontal for ≤ 8pt;
    vãos maiores (colunas) geram linhas separadas.
    """
    resultado: list[LinhaInfo] = []
    for grupo, baseline in _clusterizar_por_baseline(infos):
        grupo.sort(key=lambda info: info.span.bbox[0])
        run: list[SpanInfo] = [grupo[0]]
        for info in grupo[1:]:
            x1_atual = max(item.span.bbox[2] for item in run)
            if info.span.bbox[0] - x1_atual <= VAO_MAX_PT:
                run.append(info)
            else:
                resultado.append(_linha_de_run(pagina, baseline, run))
                run = [info]
        resultado.append(_linha_de_run(pagina, baseline, run))
    resultado.sort(key=lambda info: (info.linha.y, info.linha.x0))
    return resultado


def extrair_linhas(
    caminho: str | Path, pagina: int, ate: int | None = None
) -> list[LinhaInfo]:
    """Extrai linhas lógicas (com diagnóstico de negrito) das páginas pedidas.

    ``pagina`` e ``ate`` são 1-based e inclusivos; ``ate`` padrão é ``pagina``.
    Levanta :class:`ValueError` para intervalo inválido ou fora do PDF.
    """
    if pagina < 1:
        raise ValueError(f"página inicial inválida: {pagina} (use >= 1)")
    fim = pagina if ate is None else ate
    if fim < pagina:
        raise ValueError(f"--ate ({fim}) menor que --pagina ({pagina})")

    resultado: list[LinhaInfo] = []
    with pymupdf.open(str(caminho)) as doc:
        if fim > doc.page_count:
            raise ValueError(
                f"página final {fim} além do PDF ({doc.page_count} páginas)"
            )
        for numero in range(pagina, fim + 1):
            infos = extrair_spans(doc[numero - 1], numero)
            resultado.extend(montar_linhas_info(infos, numero))
    return resultado


def extrair_linhas_logicas(
    caminho: str | Path, pagina: int, ate: int | None = None
) -> list[LinhaLogica]:
    """Como :func:`extrair_linhas`, mas devolve apenas os contratos ``LinhaLogica``."""
    return [info.linha for info in extrair_linhas(caminho, pagina, ate)]


def filtrar_linhas(
    linhas: list[LinhaInfo],
    *,
    so_negrito: bool = False,
    busca: str | None = None,
) -> list[LinhaInfo]:
    """Aplica os filtros de ``--so-negrito`` e ``--busca`` (acento/caixa-insensível)."""
    filtradas = linhas
    if so_negrito:
        filtradas = [info for info in filtradas if info.linha.negrito_inicio]
    if busca:
        alvo = normalizar_busca(busca)
        filtradas = [
            info for info in filtradas if alvo in normalizar_busca(info.linha.texto)
        ]
    return filtradas


def ano_do_masthead(linhas: list[LinhaInfo]) -> int | None:
    """Heurística: ano da edição a partir da primeira linha ``Ano AAAA`` do topo.

    Faz duas passadas (padrão ``Ano AAAA`` e, se falhar, data por extenso). Não é
    análise do content stream — serve só para marcar candidatos a texto fantasma.
    """
    ordenadas = sorted(linhas, key=lambda info: (info.linha.pagina, info.linha.y, info.linha.x0))
    for info in ordenadas:
        achado = _ANO_MASTHEAD.search(info.linha.texto)
        if achado:
            return int(achado.group(1))
    for info in ordenadas:
        achado = _DATA_MASTHEAD.search(info.linha.texto)
        if achado:
            return int(achado.group(1))
    return None


def linha_tem_ano_fora(texto: str, ano_edicao: int | None) -> bool:
    """Heurística de fantasma: contém algum ano plausível diferente de ``ano_edicao``.

    ATENÇÃO: é heurística grosseira. Documentos citam leis antigas legitimamente;
    a palavra final é do filtro anti-fantasma da ingestão (clipping).
    """
    if ano_edicao is None:
        return False
    return any(int(ano) != ano_edicao for ano in _ANO_PLAUSIVEL.findall(texto))


def _prefixo_texto(
    info: LinhaInfo,
    *,
    edicao_ano: int | None,
    detalhar_negrito: bool,
    marcar_fantasma: bool,
) -> str:
    """Cabeçalho ``pN y=.. x=..-.. [B] [?fantasma]`` de uma linha."""
    linha = info.linha
    partes = [f"p{linha.pagina} y={linha.y:.1f} x={linha.x0:.1f}-{linha.x1:.1f}"]
    if linha.negrito_inicio:
        if detalhar_negrito and info.criterio_inicio:
            partes.append(f"[B:{info.criterio_inicio}]")
        else:
            partes.append("[B]")
    if marcar_fantasma and linha_tem_ano_fora(linha.texto, edicao_ano):
        partes.append("[?fantasma]")
    return " ".join(partes)


def formatar_texto(
    linhas: list[LinhaInfo],
    *,
    edicao_ano: int | None = None,
    detalhar_negrito: bool = False,
    marcar_fantasma: bool = False,
) -> Iterator[str]:
    """Renderiza uma linha de texto por linha lógica."""
    for info in linhas:
        prefixo = _prefixo_texto(
            info,
            edicao_ano=edicao_ano,
            detalhar_negrito=detalhar_negrito,
            marcar_fantasma=marcar_fantasma,
        )
        yield f"{prefixo} {info.linha.texto}"


def formatar_jsonl(linhas: list[LinhaInfo]) -> Iterator[str]:
    """Renderiza um ``LinhaLogica`` por linha (JSON Lines)."""
    for info in linhas:
        yield info.linha.model_dump_json()


def _ano_da_edicao(caminho: str | Path, linhas: list[LinhaInfo]) -> int | None:
    """Ano da edição pelo masthead; se o intervalo inspecionado não tem masthead, tenta a página 1."""
    ano = ano_do_masthead(linhas)
    if ano is not None:
        return ano
    if any(info.linha.pagina == 1 for info in linhas):
        return None
    return ano_do_masthead(extrair_linhas(caminho, 1))


def inspecionar(
    caminho: str | Path,
    *,
    pagina: int,
    ate: int | None = None,
    formato: str = "texto",
    so_negrito: bool = False,
    busca: str | None = None,
    detalhar_negrito: bool = False,
    fantasma: bool = False,
) -> Iterator[str]:
    """Ponto de entrada do comando ``harness inspect`` (somente leitura).

    Levanta :class:`ValueError` para entrada inválida; a CLI converte em erro.
    """
    if formato not in FORMATOS:
        raise ValueError(f"formato inválido: {formato!r} (use: {', '.join(FORMATOS)})")

    todas = extrair_linhas(caminho, pagina, ate)
    filtradas = filtrar_linhas(todas, so_negrito=so_negrito, busca=busca)

    if formato == "jsonl":
        # O contrato LinhaLogica proíbe campos extras: a marca de fantasma só
        # existe na saída texto.
        return formatar_jsonl(filtradas)

    edicao_ano = _ano_da_edicao(caminho, todas) if fantasma else None
    return formatar_texto(
        filtradas,
        edicao_ano=edicao_ano,
        detalhar_negrito=detalhar_negrito,
        marcar_fantasma=fantasma,
    )

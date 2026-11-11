"""Interface de linha de comando do harness.

``inspect`` está implementado (T005), com o diagnóstico ``--trace`` (T011),
``ingest`` liga a segmentação C1 em fluxo (T018) e ``gate`` expõe o medidor de
extração (T023). Os demais subcomandos ainda são esqueleto: existem, documentam
o que farão e respondem "não implementado". A implementação pertence aos módulos
donos (``ingest``, ``index``, ``search``, ``agent``, ``eval``).

Uso:
    uv run harness --help
    uv run harness inspect diarios/DOM4196-11022026-ASSINADO.pdf --pagina 1
    uv run harness inspect diarios/DOM4197-12022026-ASSINADO.pdf -p 2 --trace
    uv run harness ingest diarios --saida dados/snapshots/dev.jsonl
    uv run harness ingest "diarios/ANEXO AO DOM3159-ASSINADO.pdf" -s lote.jsonl
    uv run harness gate dados/snapshots/dev.jsonl --loo
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import typer

from harness.contracts import normalizar_data
from harness.tests import gate_extracao as gate_mod
from harness.tools import inspect as inspect_mod
from harness.tools import trace as trace_mod

app = typer.Typer(
    name="harness",
    help="Harness de busca conversacional sobre o Diário Oficial (DOM Teresina).",
    no_args_is_help=True,
    add_completion=False,
)


def _stub(nome: str) -> None:
    """Sinaliza que o subcomando ainda não foi implementado."""
    typer.echo(f"{nome}: não implementado")
    raise typer.Exit(code=1)


@app.command()
def inspect(
    pdf: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="PDF do diário (aberto somente para leitura).",
    ),
    pagina: int = typer.Option(..., "--pagina", "-p", min=1, help="Página inicial (1-based)."),
    ate: int | None = typer.Option(
        None, "--ate", "-a", min=1, help="Página final, inclusive (padrão: a inicial)."
    ),
    formato: str = typer.Option(
        "texto", "--formato", "-f", help="Formato de saída: texto | jsonl."
    ),
    so_negrito: bool = typer.Option(
        False, "--so-negrito", help="Mostra só linhas que começam em negrito."
    ),
    busca: str | None = typer.Option(
        None, "--busca", "-b", help="Filtra linhas contendo o texto (sem acento/caixa)."
    ),
    detalhar_negrito: bool = typer.Option(
        False,
        "--detalhar-negrito",
        help="Mostra o critério do negrito: [B:flags], [B:fonte] ou [B:flags+fonte].",
    ),
    fantasma: bool = typer.Option(
        False,
        "--fantasma/--no-fantasma",
        help=(
            "Heurística (desligada por padrão): marca [?fantasma] em anos fora do "
            "ano da edição (só saída texto)."
        ),
    ),
    trace: bool = typer.Option(
        False,
        "--trace",
        help=(
            "Despeja o trace de clip/z-order em JSONL (get_texttrace, "
            "get_drawings(extended=True), get_bboxlog e um resumo por página). "
            "Somente leitura; ignora --formato/--busca/--so-negrito."
        ),
    ),
    trace_fills: bool = typer.Option(
        False,
        "--trace-fills",
        help="Com --trace, inclui TODOS os paths de preenchimento (padrão: só brancos).",
    ),
) -> None:
    """Extrai texto do PDF com negrito, linhas lógicas, bbox e fonte (T005).

    Com ``--trace``, despeja o diagnóstico de clip/z-order usado pela proposta
    anti-fantasma A1 (T011). Somente leitura. Exemplos:

        uv run harness inspect diarios/DOM4197-12022026-ASSINADO.pdf -p 1 --ate 2
        uv run harness inspect diarios/DOM4197-12022026-ASSINADO.pdf -p 2 --trace
    """
    if trace:
        try:
            saida_trace = trace_mod.inspecionar_trace(
                pdf,
                pagina=pagina,
                ate=ate,
                incluir_todos_fills=trace_fills,
            )
        except ValueError as exc:
            typer.secho(f"inspect: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
        for linha in saida_trace:
            typer.echo(linha)
        return

    try:
        saida = inspect_mod.inspecionar(
            pdf,
            pagina=pagina,
            ate=ate,
            formato=formato,
            so_negrito=so_negrito,
            busca=busca,
            detalhar_negrito=detalhar_negrito,
            fantasma=fantasma,
        )
    except ValueError as exc:
        typer.secho(f"inspect: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    for linha in saida:
        typer.echo(linha)


def _normalizar_edicao(valor: str) -> str:
    """Número de edição canônico: só dígitos, sem zeros à esquerda (``4.196`` → ``4196``)."""
    digitos = re.sub(r"\D", "", valor)
    return (digitos.lstrip("0") or "0") if digitos else valor.strip()


def _edicao_do_conteudo(pdf: Path) -> tuple[str | None, date | None]:
    """Lê a página 1 (com anti-fantasma) e extrai ``(numero, data)`` do masthead."""
    from harness.ingest import segmentar as segmentar_mod

    return segmentar_mod.edicao_do_conteudo(pdf)


def _resolver_edicao(
    pdf: Path,
    *,
    edicao: str | None,
    data: date | None,
) -> tuple[str, date | str]:
    """Resolve ``(numero, data)`` de um PDF: opção da CLI > nome > conteúdo da pág. 1.

    Levanta :class:`ValueError` quando a data não pode ser obtida de nenhuma
    fonte; o chamador registra o erro do arquivo e segue o lote.
    """
    from harness.ingest import segmentar as segmentar_mod

    numero_nome, data_nome = segmentar_mod.edicao_do_nome(pdf)
    numero = edicao if edicao is not None else numero_nome
    valor_data: date | str | None = data if data is not None else data_nome
    if not numero or valor_data is None:
        numero_conteudo, data_conteudo = _edicao_do_conteudo(pdf)
        if not numero:
            numero = numero_conteudo or ""
        if valor_data is None:
            valor_data = data_conteudo
    if valor_data is None:
        raise ValueError(
            f"não foi possível obter a data da edição de {pdf.name!r}; "
            "informe --data (ou renomeie o arquivo como DOM<num>-<ddmmaaaa>)"
        )
    return numero, valor_data


def _ingerir_em_fluxo(
    entrada: Path,
    saida: Path,
    ordem: str,
    proveniencia: Path | None,
    *,
    edicao: str | None = None,
    data: date | None = None,
) -> tuple[int, dict[str, int], list[tuple[str, str]]]:
    """Segmenta a entrada **um PDF por vez** e grava snapshot (e sidecar) em fluxo.

    Evita materializar todos os ``DocumentoSegmentado`` de uma vez: cada PDF é
    classificado, ordenado e segmentado isoladamente; seus documentos são
    escritos e liberados antes de abrir o próximo. Um PDF com erro **não** aborta
    o lote: o erro é registrado em ``erros`` e o próximo PDF é processado; a
    saída já gravada não é truncada. Retorna ``(total, por_arquivo, erros)``.
    """
    from harness.ingest import segmentar as segmentar_mod

    if ordem not in segmentar_mod.METODOS_ORDEM:
        raise ValueError(
            f"método de ordem inválido: {ordem!r} (use: {', '.join(segmentar_mod.METODOS_ORDEM)})"
        )

    saida.parent.mkdir(parents=True, exist_ok=True)
    if proveniencia is not None:
        proveniencia.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    por_arquivo: dict[str, int] = {}
    erros: list[tuple[str, str]] = []
    pdfs: Iterator[Path] = iter(segmentar_mod.listar_pdfs(entrada))
    with saida.open("w", encoding="utf-8") as arquivo_saida:
        arquivo_prov = (
            proveniencia.open("w", encoding="utf-8") if proveniencia is not None else None
        )
        try:
            for pdf in pdfs:
                try:
                    numero, valor_data = _resolver_edicao(pdf, edicao=edicao, data=data)
                    itens = segmentar_mod.segmentar_arquivo(
                        pdf,
                        metodo_ordem=ordem,
                        edicao_numero=numero,
                        edicao_data=valor_data,
                    )
                except Exception as exc:  # noqa: BLE001 - um PDF ruim não derruba o lote
                    mensagem = f"{type(exc).__name__}: {exc}"
                    erros.append((str(pdf), mensagem))
                    typer.secho(
                        f"ingest: erro em {pdf.name}: {mensagem} (seguindo)",
                        fg=typer.colors.RED,
                        err=True,
                    )
                    continue
                for item in itens:
                    arquivo_saida.write(item.documento.model_dump_json() + "\n")
                    if arquivo_prov is not None:
                        arquivo_prov.write(
                            json.dumps(item.para_dict(), ensure_ascii=False) + "\n"
                        )
                    total += 1
                    nome = item.documento.arquivo
                    por_arquivo[nome] = por_arquivo.get(nome, 0) + 1
                # Saída incremental: o que já foi segmentado fica em disco mesmo
                # que um PDF posterior derrube o processo.
                arquivo_saida.flush()
                if arquivo_prov is not None:
                    arquivo_prov.flush()
        finally:
            if arquivo_prov is not None:
                arquivo_prov.close()

    return total, por_arquivo, erros


@app.command()
def ingest(
    entrada: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help="PDF do diário ou diretório com os PDFs.",
    ),
    saida: Path = typer.Option(
        ..., "--saida", "-s", help="Snapshot JSONL de Documento (ex.: dados/snapshots/lote.jsonl)."
    ),
    ordem: str = typer.Option(
        "propria", "--ordem", help="Ordem de leitura: propria | column_boxes | nao."
    ),
    proveniencia: Path | None = typer.Option(
        None, "--proveniencia", help="Sidecar JSONL opcional com flags/proveniência."
    ),
    edicao: str | None = typer.Option(
        None,
        "--edicao",
        help="Número da edição (sobrepõe nome/conteúdo; ex.: 4196 ou 4.196).",
    ),
    data: str | None = typer.Option(
        None,
        "--data",
        help="Data da edição (sobrepõe nome/conteúdo): ISO, dd/mm/aaaa ou ddmmaaaa.",
    ),
) -> None:
    """Segmenta (C1) os diários em documentos e grava o snapshot JSONL (T018/T027).

    Processa um PDF por vez (memória constante) e delega ao pipeline já existente
    ``harness.ingest.segmentar`` (mesma rotina de ``python -m harness.ingest``).

    Quando o nome do arquivo não traz ``DOM<num>-<ddmmaaaa>`` (ex.: ``ANEXO AO
    DOM3159-ASSINADO.pdf``), a edição/data são lidas do masthead da página 1;
    ``--edicao``/``--data`` sobrepõem. Um PDF com erro é registrado e o lote
    continua; a CLI sai com código 1 no fim, com o resumo dos erros.
    """
    data_normalizada: date | None = None
    if data is not None:
        try:
            valor_data = normalizar_data(data)
        except ValueError as exc:
            typer.secho(f"ingest: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
        if not isinstance(valor_data, date):
            typer.secho(f"ingest: data inválida: {data!r}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        data_normalizada = valor_data
    edicao_normalizada = _normalizar_edicao(edicao) if edicao is not None else None

    try:
        total, por_arquivo, erros = _ingerir_em_fluxo(
            entrada,
            saida,
            ordem,
            proveniencia,
            edicao=edicao_normalizada,
            data=data_normalizada,
        )
    except (ValueError, OSError) as exc:
        typer.secho(f"ingest: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"ingest: {total} documento(s) -> {saida}")
    for arquivo, quantidade in sorted(por_arquivo.items()):
        typer.echo(f"  {Path(arquivo).name}: {quantidade}")

    if erros:
        typer.secho(
            f"\ningest: {len(erros)} arquivo(s) com erro (lote continuou):",
            fg=typer.colors.RED,
            err=True,
        )
        for arquivo, mensagem in erros:
            typer.secho(f"  {Path(arquivo).name}: {mensagem}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command()
def gate(
    snapshot: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="JSONL com os Documento produzidos pela ingestão.",
    ),
    anotacao: Path = typer.Option(
        gate_mod.DIR_ANOTACAO,
        "--anotacao",
        help="Diretório do ouro (padrão: dados/anotacao).",
    ),
    exclusoes: Path | None = typer.Option(
        None,
        "--exclusoes",
        help="Arquivo de exclusões por campo (padrão: dados/anotacao/exclusoes.json).",
    ),
    limite: float = typer.Option(
        gate_mod.LIMITE_GATE,
        "--limite",
        help="Limite de P/R do gate (padrão: 0.95; D003).",
    ),
    loo: bool = typer.Option(
        False, "--loo", help="Imprime também o relatório leave-one-edition-out."
    ),
    sem_gate: bool = typer.Option(
        False, "--sem-gate", help="Imprime a tabela mas sempre retorna 0 (não reprova)."
    ),
    max_divergencias: int = typer.Option(
        40,
        "--max-divergencias",
        help="Quantas divergências listar por arquivo (padrão: 40; use -1 para todas).",
    ),
) -> None:
    """Mede o gate de extração (T010) sobre um snapshot (T023).

    Delega a ``harness.tests.gate_extracao`` mantendo a mesma tabela, o mesmo
    relatório ``--loo`` e o mesmo veredito (sai com código 1 quando reprovado).
    Exemplo:

        uv run harness gate dados/snapshots/dev.jsonl --loo
    """
    try:
        resultados = gate_mod.avaliar_snapshot(
            snapshot,
            dir_anotacao=anotacao,
            # `gate_extracao.main` não expõe `--exclusoes` (caminho fixo) e não
            # pode ser alterado nesta tarefa; delegamos à API do módulo para
            # apontar um arquivo alternativo (ex.: validação) sem duplicar a
            # medição nem a formatação.
            exclusoes=(
                gate_mod.carregar_exclusoes(exclusoes) if exclusoes is not None else None
            ),
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(f"gate: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    max_div = None if max_divergencias < 0 else max_divergencias
    typer.echo(gate_mod.formatar(resultados, max_divergencias=max_div))
    if loo:
        typer.echo()
        typer.echo(gate_mod.formatar_loo(gate_mod.leave_one_edition_out(resultados)))

    reprovacoes: list[str] = []
    for resultado in resultados:
        # `_reprovacoes` é a mesma rotina usada por `gate_extracao.main`.
        for motivo in gate_mod._reprovacoes(resultado, limite):  # noqa: SLF001
            reprovacoes.append(f"{resultado.arquivo}: {motivo}")
    if reprovacoes and not sem_gate:
        typer.echo(f"\nGATE REPROVADO (< {limite:.0%}): " + "; ".join(reprovacoes))
        raise typer.Exit(code=1)
    typer.echo(f"\nGATE APROVADO (P/R ≥ {limite:.0%}).")


@app.command()
def index() -> None:
    """Constrói os índices (SQLite FTS5 + entidades + vetorial LanceDB)."""
    _stub("index")


@app.command()
def search() -> None:
    """Executa a busca híbrida e mostra resultados com trace."""
    _stub("search")


@app.command()
def chat() -> None:
    """Abre o agente de chat (CrewAI Flow) com as tools de busca."""
    _stub("chat")


@app.command()
def eval() -> None:
    """Roda a avaliação (hit@5 etc.) sobre o golden set dev/holdout."""
    _stub("eval")

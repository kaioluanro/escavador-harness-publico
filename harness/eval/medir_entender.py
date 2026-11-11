"""Baseline do :class:`DecisorRegras` no golden dev (E001a) — CLI de avaliação.

A medição saiu de ``harness/search/entender.py`` (E001d): aquele módulo é só o
caminho de produção, sem CLI e sem constante vinda do golden. Este módulo guarda
o comparativo original — ``tipo``/``modo_busca``/``suficiente`` contra os rótulos
de ``harness/eval/dev/golden_dev.jsonl``. O T031 usa o MESMO comparativo para
medir o laya.

Uso::

    uv run python -m harness.eval.medir_entender
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from harness.contracts import ModoBusca, TipoIntencao
from harness.search.entender import entender

__all__ = ["medir_golden"]


# --------------------------------------------------------------------------- #
# Baseline no golden dev (E001a)                                               #
# --------------------------------------------------------------------------- #

#: ``categoria`` do golden → ``modo_busca`` esperado (vocabulário do R032).
_MODOS_POR_CATEGORIA: dict[str, ModoBusca] = {
    "conversa_sem_busca": "contexto_tema",
    "nome_unico": "um_nome",
    "nomes_multiplos": "varios_nomes",
    "id_dom": "id_dom",
    "numero": "numero_ato",
    "cpf_cnpj": "cpf_cnpj",
    "contexto": "contexto_tema",
    "nome_com_erro": "um_nome",
    "ambiguo": "numero_ato",
}
#: follow_up (10 casos): modo da PRIMEIRA fala, rotulado manualmente — é ela que
#: define o canal que recupera o alvo retomado pelo follow-up.
_MODOS_FOLLOW_UP: dict[str, ModoBusca] = {
    "tem algo com Antonio Francisco da Rocha?": "um_nome",
    "ID: 000440419600172026": "id_dom",
    "vai ter licitação para escolher a empresa?": "contexto_tema",
    "Portaria 43/2026": "numero_ato",
    "quem foi escalado para cuidar das parcerias firmadas?": "contexto_tema",
    "tem algo com AIRTON NASCIMENTO DOS SANTOS?": "um_nome",
    "000440419600102026": "id_dom",
    "deram título de cidadania a alguém?": "contexto_tema",
    "tem portaria 050/2026/GAB/SEMEC?": "numero_ato",
    "contrataram trabalhadores temporários?": "contexto_tema",
}


def _esperado(caso: dict[str, Any]) -> tuple[TipoIntencao, ModoBusca | None, bool]:
    """Rótulos do golden: ``esperado.buscar/perguntar`` → ``tipo``; categoria → modo."""
    esperado = caso["esperado"]
    categoria = caso["categoria"]
    if esperado["perguntar"]:
        tipo: TipoIntencao = "esclarecimento"
    elif not esperado["buscar"]:
        tipo = "conversa"
    elif categoria == "follow_up":
        tipo = "follow_up"
    else:
        tipo = "busca"
    if categoria == "follow_up":
        modo = _MODOS_FOLLOW_UP.get(caso["turnos"][0])
    else:
        modo = _MODOS_POR_CATEGORIA.get(categoria)
    suficiente = bool(esperado["buscar"]) and not bool(esperado["perguntar"])
    return tipo, modo, suficiente


def medir_golden(
    caminho: str | None = None,
) -> dict[str, Any]:
    """Baseline do :class:`DecisorRegras` no golden dev (não é gate — só mede).

    Compara ``tipo``/``modo_busca``/``suficiente`` de :func:`entender` com os
    rótulos do golden dev e devolve acurácia por campo + as 10 maiores
    discordâncias. O T031 usa o MESMO comparativo para medir o laya.
    """
    import json
    from pathlib import Path

    from harness.contracts import MensagemHistorico as _Mensagem

    raiz = Path(__file__).resolve().parents[2]
    alvo = Path(caminho) if caminho else raiz / "harness" / "eval" / "dev" / "golden_dev.jsonl"
    casos = [json.loads(linha) for linha in alvo.read_text(encoding="utf-8").splitlines() if linha]

    acertos = {"tipo": 0, "modo_busca": 0, "suficiente": 0}
    contados = {"tipo": 0, "modo_busca": 0, "suficiente": 0}
    divergencias: list[dict[str, Any]] = []
    for caso in casos:
        turnos = caso["turnos"]
        historico = [_Mensagem(papel="usuario", texto=t) for t in turnos[:-1]]
        intencao = entender(turnos[-1], historico)
        tipo_esp, modo_esp, suf_esp = _esperado(caso)
        campos = {
            "tipo": (tipo_esp, intencao.tipo),
            "modo_busca": (modo_esp, intencao.modo_busca),
            "suficiente": (suf_esp, intencao.suficiente),
        }
        for campo, (esperado, obtido) in campos.items():
            if esperado is None:
                continue
            contados[campo] += 1
            if esperado == obtido:
                acertos[campo] += 1
            else:
                divergencias.append(
                    {
                        "id": caso["id"],
                        "campo": campo,
                        "frase": turnos[-1],
                        "esperado": esperado,
                        "obtido": obtido,
                    }
                )
    divergencias.sort(key=lambda d: (d["campo"], d["id"]))
    acuracia = {
        campo: (acertos[campo] / contados[campo] if contados[campo] else 0.0)
        for campo in acertos
    }
    return {
        "casos": len(casos),
        "acertos": acertos,
        "contados": contados,
        "acuracia": acuracia,
        "divergencias": divergencias,
    }


def _main(argv: Sequence[str] | None = None) -> int:
    """Imprime o baseline (``uv run python -m harness.eval.medir_entender``)."""
    resultado = medir_golden(argv[0] if argv else None)
    print(f"casos: {resultado['casos']}")
    for campo, valor in resultado["acuracia"].items():
        acertos = resultado["acertos"][campo]
        total = resultado["contados"][campo]
        print(f"{campo:>12}: {valor:.3f}  ({acertos}/{total})")
    print("top 10 discordâncias:")
    for d in resultado["divergencias"][:10]:
        print(
            f"  [{d['campo']}] {d['id']}: {d['frase']!r} "
            f"esperado={d['esperado']!r} obtido={d['obtido']!r}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI de baseline
    raise SystemExit(_main())

"""Pipeline único de ingestão: A1 → B1 → C1 → D-meta-1 → D-ent-1 (+ NER).

Encadeia os passos já implementados e devolve :class:`Documento` **completo**
(metadados e entidades preenchidos), preservando as flags/proveniência do C1 no
invólucro :class:`~harness.ingest.segmentar.DocumentoSegmentado`.

Ordem (T020/D013):

1. **segmentar** — extração com filtro anti-fantasma A1 (``extrair``), B1
   (``classificar``) e C1 (``segmentar``);
2. **metadados** — ``metadados.preencher_metadados`` (tipo/numero/ano/orgao);
3. **entidades** — regras determinísticas de ``entidades`` + **NER híbrido**
   (D032) quando o extra opcional ``ner`` está instalado. O documento **gigante**
   (> :data:`harness.ingest.ner_nomes.LIMITE_GIGANTE`) não recebe NER — só a
   extração determinística.

O carregamento do NER é **preguiçoso e cacheado**: uma única carga por processo;
sem ``transformers``/``torch``/modelo, o pipeline degrada para as regras e
segue. O processamento é **um PDF por vez** (memória constante).
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Iterator

from harness.contracts import Documento, Entidade
from harness.ingest import entidades as _ent
from harness.ingest import metadados as _meta
from harness.ingest import ner_nomes as _ner
from harness.ingest import segmentar as _seg

__all__ = [
    "LIMITE_GIGANTE",
    "NERDisponivel",
    "enriquecer",
    "enriquecer_segmentado",
    "ingerir",
    "ingerir_pdf",
]

#: Acima deste tamanho o documento é o "gigante" (D012): NER não roda nele.
LIMITE_GIGANTE: int = _ner.LIMITE_GIGANTE

#: Cache do NER (``None`` = indisponível nesta execução).
_NER: object | None = None
_NER_CARREGADO = False


def NERDisponivel() -> bool:  # noqa: N802 - nome de fábrica documentada
    """True se o extra opcional ``ner`` puder carregar o modelo (D032)."""
    return _carregar_ner() is not None


#: Variável de ambiente que desliga o NER (testes/model-free/execução offline).
VAR_SEM_NER = "HARNESS_INGEST_SEM_NER"


def _carregar_ner() -> object | None:
    """Carrega o NER uma única vez; ``None`` se indisponível (degrada p/ regras)."""
    global _NER, _NER_CARREGADO
    if _NER_CARREGADO:
        return _NER
    _NER_CARREGADO = True
    if os.environ.get(VAR_SEM_NER):
        _NER = None
        return _NER
    try:
        _NER = _ner.carregar_ner()
    except Exception:  # noqa: BLE001 - NER é opcional
        _NER = None
    return _NER


def _entidades_com_ner(
    documento: Documento, *, ner: object | None, limite_gigante: int = LIMITE_GIGANTE
) -> list[Entidade]:
    """Entidades do documento: regras de ``entidades`` + nomes híbridos (D032).

    Se o NER estiver disponível e o documento não for gigante, os **nomes** vêm do
    híbrido (NER filtrado ∪ regras); as demais entidades (CPF/CNPJ/números) vêm
    sempre das regras. Sem NER (ou no gigante), tudo vem das regras.
    """
    texto = "\n".join(parte for parte in (documento.texto, *documento.tabelas) if parte)
    gigante = len(texto) > limite_gigante
    # Gigante (D012): só a coluna NOME — sem CPF/CNPJ nem números citados.
    base = _ent.extrair_entidades(
        texto,
        pagina=documento.pagina_inicio,
        incluir_cpf=not gigante,
        incluir_numeros=not gigante,
    )
    if ner is None or gigante:
        return base

    nomes = _ner.extrair_nomes_hibrido(texto, ner=ner)  # type: ignore[arg-type]
    outras = [entidade for entidade in base if entidade.tipo != "nome"]
    vistos: set[str] = set()
    entidades: list[Entidade] = []
    for nome in nomes:
        chave = _ent.chave_nome(nome)
        if not chave or chave in vistos:
            continue
        vistos.add(chave)
        entidades.append(
            Entidade(
                tipo="nome",
                valor=nome,
                valor_normalizado=chave,
                pagina=documento.pagina_inicio,
            )
        )
    entidades.extend(outras)
    return entidades


def enriquecer(
    documento: Documento, *, ner: object | None = None, usar_ner: bool = True
) -> Documento:
    """``metadados`` + ``entidades`` (+NER) de um documento, sem mutar a entrada.

    Com ``usar_ner=True`` (padrão) e ``ner=None``, carrega o NER uma vez
    (cacheado). Sem NER disponível ou desligado (``HARNESS_INGEST_SEM_NER``),
    cai nas regras determinísticas.
    """
    if usar_ner and ner is None:
        ner = _carregar_ner()
    documento = _meta.preencher_metadados(documento)
    entidades = _entidades_com_ner(documento, ner=ner)
    # D024: o número PRÓPRIO do ato vai em ``numero``; não é "número citado".
    # (ex.: CONTRATO Nº 01/2026 → ``01/2026`` não vira entidade ``contrato``).
    # A comparação tolera sufixo de órgão (``02/2026/SEMPLAN`` × ``02/2026``).
    numero = documento.numero
    if numero:
        chave_numero = _ent.normalizar_numero_citado(numero)
        if chave_numero:
            def _e_o_proprio(entidade: Entidade) -> bool:
                if entidade.tipo not in ("processo", "contrato", "numero_ato"):
                    return False
                chave = entidade.valor_normalizado
                return chave == chave_numero or chave.startswith(
                    chave_numero
                ) or chave_numero.startswith(chave)

            entidades = [e for e in entidades if not _e_o_proprio(e)]
    return documento.model_copy(update={"entidades": entidades})


def enriquecer_segmentado(
    item: _seg.DocumentoSegmentado, *, ner: object | None = None, usar_ner: bool = True
) -> _seg.DocumentoSegmentado:
    """Enriquece o ``Documento`` de um :class:`DocumentoSegmentado` (flags intactas)."""
    return dataclasses.replace(
        item, documento=enriquecer(item.documento, ner=ner, usar_ner=usar_ner)
    )


def ingerir_pdf(
    caminho: str | Path,
    *,
    metodo_ordem: str = "propria",
    ner: object | None = None,
    usar_ner: bool = True,
    aplicar_fantasma: bool = True,
) -> Iterator[_seg.DocumentoSegmentado]:
    """Pipeline de um PDF: segmenta e enriquece, um documento por vez."""
    carregado = ner is not None or not usar_ner
    for item in _seg.segmentar_arquivo(
        caminho,
        metodo_ordem=metodo_ordem,
        aplicar_fantasma=aplicar_fantasma,
        enriquecer=False,
    ):
        if not carregado:
            ner = _carregar_ner()
            carregado = True
        yield enriquecer_segmentado(item, ner=ner, usar_ner=usar_ner)


def ingerir(
    entrada: str | Path,
    *,
    metodo_ordem: str = "propria",
    usar_ner: bool = True,
    aplicar_fantasma: bool = True,
) -> Iterator[_seg.DocumentoSegmentado]:
    """Pipeline de um PDF ou de um diretório (um PDF por vez), como iterador."""
    carregado = not usar_ner
    ner: object | None = None
    for pdf in _seg.listar_pdfs(entrada):
        for item in _seg.segmentar_arquivo(
            pdf,
            metodo_ordem=metodo_ordem,
            aplicar_fantasma=aplicar_fantasma,
            enriquecer=False,
        ):
            if not carregado:
                ner = _carregar_ner()
                carregado = True
            yield enriquecer_segmentado(item, ner=ner, usar_ner=usar_ner)

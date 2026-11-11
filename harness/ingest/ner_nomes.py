"""NER opcional para nomes de pessoas — experimento do passo 10 do I004 (D014).

Este módulo é o **experimento gateado** de NER de nomes de pessoas previsto em
``reports/ideias/I004-ingestao-sintese.md`` (item 10) e decidido em D014: usa o
modelo ``Luciano/bertimbau-base-lener_br`` (licença MIT), que classifica
``PESSOA`` / ``ORGANIZACAO`` / ``LOCAL`` / ``TEMPO`` / ``LEGISLACAO`` /
``JURISPRUDENCIA``.

Ele é **opcional e não entra no caminho crítico da ingestão**:

* nada é importado de ``transformers``/``torch`` no topo do módulo — as
  dependências são carregadas de forma preguiçosa em :func:`carregar_ner`, de
  modo que o pacote base continue funcionando sem elas;
* ``harness/ingest/entidades.py`` (regras determinísticas) **não é alterado**;
  as regras são reutilizadas via :func:`harness.ingest.entidades.extrair_nomes`;
* o documento gigante (regra de segurança §1.8 / D012) não recebe NER.

Três variantes são comparáveis no gate de extração (nomes):

``a``  só regras — baseline atual (``entidades.extrair_nomes``);
``b``  só NER — todos os spans ``PESSOA`` do modelo, sem filtro;
``c``  híbrido — NER propõe, a lista negra/validação de pessoa filtra falsos
       positivos (lugares/instituições/empresas que o modelo promoveu a PESSOA)
       e as regras contextuais (``NOME``/``CPF``/tabela, gatilhos de assinatura)
       completam o recall.

As dependências ficam no extra opcional ``ner`` (torch CPU via
``[[tool.uv.index]]`` do PyTorch)::

    uv run --extra ner python -m harness.ingest.ner_nomes --variantes a,b,c

Sem o extra, a variante ``a`` (só regras) funciona normalmente. O resultado
medido e os padrões de erro são reportados no relatório do agente
``prog-ingestao`` (T019).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from harness.contracts import Documento, Entidade
from harness.ingest import entidades as _ent

__all__ = [
    "LIMITE_BLOCO",
    "LIMITE_GIGANTE",
    "MODELO_PADRAO",
    "NERIndisponivel",
    "ROTULOS_PESSOA",
    "VARIANTES",
    "ResultadoVariante",
    "aplicar_variante",
    "carregar_ner",
    "extrair_nomes_hibrido",
    "extrair_nomes_ner",
    "main",
    "medir_variantes",
    "nome_parece_pessoa",
    "nomes_da_variante",
    "revisao_regras",
]

#: Modelo decidido em D014 (MIT). NÃO trocar pelo card sem licença.
MODELO_PADRAO = "Luciano/bertimbau-base-lener_br"

#: Rótulos que contam como PESSOA na saída do modelo/pipeline.
ROTULOS_PESSOA: frozenset[str] = frozenset({"PESSOA", "B-PESSOA", "I-PESSOA"})

#: Variantes medidas no gate.
VARIANTES: tuple[str, ...] = ("a", "b", "c")

#: Tamanho-alvo de cada bloco enviado ao BERT (o limite duro é 512 tokens, que
#: em português/BERTimbau fica bem acima de 900 caracteres — folga proposital).
LIMITE_BLOCO = 900

#: Acima disto o documento é o "gigante" (§1.8/D012): NER não roda nele.
LIMITE_GIGANTE = 50_000

#: Assinatura de uma função NER: bloco de texto → superfícies ``PESSOA`` cruas.
Ner = Callable[[str], Sequence[str]]


class NERIndisponivel(RuntimeError):
    """``transformers``/``torch`` (ou o modelo) não estão disponíveis."""


def _minusculo(texto: str) -> str:
    return _ent.sem_acento(texto or "").casefold()


# --------------------------------------------------------------------------- #
# Carregamento preguiçoso do modelo                                           #
# --------------------------------------------------------------------------- #


def carregar_ner(
    modelo: str = MODELO_PADRAO,
    *,
    dispositivo: int = -1,
    lote: int = 16,
) -> Ner:
    """Carrega o pipeline de token-classification e devolve uma função NER.

    ``dispositivo=-1`` força **CPU** (padrão; use CPU quando a GPU estiver
    ocupada pelo LM Studio). A função devolvida recebe um bloco de texto e
    devolve os spans classificados como ``PESSOA`` (superfície, sem filtro).

    Levanta :class:`NERIndisponivel` se ``transformers``/``torch`` não estiverem
    instalados; nenhuma dessas dependências é importada no topo do módulo.
    """
    try:  # pragma: no cover - depende de ambiente com transformers/torch
        from transformers import pipeline
    except Exception as exc:  # noqa: BLE001
        raise NERIndisponivel(
            "transformers/torch ausentes; instale o grupo opcional 'ner' "
            "(uv add --optional ner transformers torch) ou rode com "
            "uv run --with transformers --with torch"
        ) from exc

    try:  # pragma: no cover - carrega pesos do Hugging Face
        pipe = pipeline(
            "token-classification",
            model=modelo,
            tokenizer=modelo,
            aggregation_strategy="simple",
            device=dispositivo,
            batch_size=lote,
        )
    except Exception as exc:  # noqa: BLE001
        raise NERIndisponivel(f"não foi possível carregar {modelo!r}: {exc}") from exc

    def _ner(texto: str) -> list[str]:
        if not texto.strip():
            return []
        # transformers v5: o pipeline trunca sozinho em ``tokenizer.model_max_length``
        # (512); por isso os blocos de :func:`_blocos` ficam bem abaixo disso.
        saida = pipe(texto)
        spans: list[tuple[int, int]] = []
        for item in saida:
            rotulo = str(item.get("entity_group") or item.get("entity") or "")
            rotulo = rotulo.upper()
            if rotulo.startswith(("B-", "I-")):
                rotulo = rotulo[2:]
            if rotulo != "PESSOA":
                continue
            spans.append((int(item["start"]), int(item["end"])))
        return _fundir_adjacentes(spans, texto)

    return _ner


def _fundir_adjacentes(spans: Sequence[tuple[int, int]], texto: str) -> list[str]:
    """Funde spans ``PESSOA`` contíguos (ou separados só por espaço na mesma linha).

    O checkpoint fragmenta palavras acentuadas em maiúsculas (``SÍLVIO`` vira
    ``S``+``##Í``+…) e marca cada pedaço como ``B-PESSOA``; a agregação ``simple``
    então devolve vários spans. Fundir por proximidade no texto-fonte reconstitui
    o nome. Não funde através de quebra de linha ou pontuação (``,``, ``;``, ``e``
    entre pessoas).
    """
    ordenados = sorted(spans)
    fundidos: list[tuple[int, int]] = []
    for inicio, fim in ordenados:
        if fundidos:
            ant_inicio, ant_fim = fundidos[-1]
            vao = texto[ant_fim:inicio]
            if inicio <= ant_fim or (vao.strip() == "" and "\n" not in vao):
                fundidos[-1] = (ant_inicio, max(ant_fim, fim))
                continue
        fundidos.append((inicio, fim))
    return [re.sub(r"\s+", " ", texto[inicio:fim]).strip() for inicio, fim in fundidos]


# --------------------------------------------------------------------------- #
# Blocos e validação de pessoa                                                #
# --------------------------------------------------------------------------- #

#: Tokens simples (mesma noção do tokenizador de ``entidades``).
_TOK = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’]*\.?")

#: Tokens que podem aparecer em nomes reais mas estão na lista negra geral por
#: outro motivo (siglas de empresa/órgão); liberar aqui evita perder sobrenomes.
_PERMITIR_EM_NOME: frozenset[str] = frozenset(
    {"sa", "dom", "norte", "sul", "leste", "oeste", "central", "jardim", "parque", "vila"}
)

#: Acréscimos à lista negra para a saída de NER (cargos/funções que o modelo
#: pode promover a PESSOA).
_NAO_PESSOA_EXTRA: frozenset[str] = frozenset(
    """
    secretario secretaria subsecretario subsecretaria superintendente presidente
    prefeito prefeita vereador vereadora relator relatora diretor diretora
    coordenador coordenadora assessor assessora procurador procuradora auditor
    auditora ouvidor ouvidora corregedor corregedora promotor promotora
    comandante delegado delegada ministro ministra governador governadora
    deputado deputada senador senadora conselheiro conselheira substituto
    substituta temporario temporaria signatario signatarios legal pregoeiro
    pregoeira chefe fiscal gestor gestora titular suplente efetivo efetiva
    """.split()
)


def _dehyphen(texto: str) -> str:
    """Junta palavras partidas na quebra de linha (``SIL-\\nVIO`` → ``SILVIO``)."""
    return re.sub(r"(?<=[^\W\d_])-\n\s*(?=[^\W\d_])", "", texto or "")


def _blocos(texto: str, max_chars: int = LIMITE_BLOCO) -> list[str]:
    """Divide o texto em blocos de linhas inteiras, para não cortar nomes.

    O BERT tem limite de 512 tokens; blocos de ~``max_chars`` caracteres ficam
    com folga. A quebra só acontece em ``\\n`` (linha inteira), de modo que um
    nome de uma linha nunca é partido.
    """
    if max_chars <= 0:
        max_chars = LIMITE_BLOCO
    blocos: list[str] = []
    buffer: list[str] = []
    tamanho = 0
    for linha in texto.split("\n"):
        extra = len(linha) + 1
        if buffer and tamanho + extra > max_chars:
            blocos.append("\n".join(buffer))
            buffer, tamanho = [], 0
        buffer.append(linha)
        tamanho += extra
    if buffer:
        blocos.append("\n".join(buffer))
    return blocos


def nome_parece_pessoa(nome: str) -> bool:
    """True se a superfície passa no filtro de pessoa (lista negra + forma).

    Usado no híbrido para derrubar falsos positivos do NER: lugares,
    instituições, empresas e cargos promovidos a ``PESSOA``. Reusa
    :func:`harness.ingest.entidades.limpar_superficie_nome` (limpa prefixos de
    comando, sufixos empresariais e conectores finais; rejeita categorias
    proibidas) e ainda exige 2+ tokens-núcleo, com exceções de sobrenome.
    """
    if not nome or not nome.strip():
        return False
    if re.search(r"\d", nome):
        return False
    limpo = _ent.limpar_superficie_nome(nome)
    if limpo is None:
        return False
    tokens = _TOK.findall(limpo)
    bases = [_minusculo(token).strip(".") for token in tokens]
    if not (2 <= len(bases) <= 9):
        return False
    nucleos = [base for base in bases if len(base) > 1]
    if len(nucleos) < 2:
        return False
    if not any(len(base) >= 3 for base in bases):
        return False
    for base in bases:
        if base in _PERMITIR_EM_NOME or base in _ent.CONECTORES:
            continue
        if base in _ent.PALAVRAS_NAO_PESSOA or base in _NAO_PESSOA_EXTRA:
            return False
    return True


def _absorver_prefixos(nomes: Iterable[str]) -> list[str]:
    """Remove nomes que são prefixo-de-tokens de outro (``JOSÉ GOMES`` ⊂ ``JOSÉ GOMES DA SILVA``).

    Evita o par FP+FN clássico das regras (nome truncado + nome completo). A
    comparação é por tokens normalizados (sem acento/caixa).
    """
    itens = [nome for nome in nomes if nome.strip()]
    sequencias = [tuple(_minusculo(tok).strip(".") for tok in _TOK.findall(nome)) for nome in itens]
    resultado: list[str] = []
    for i, nome in enumerate(itens):
        seq = sequencias[i]
        if not seq:
            continue
        coberto = any(
            j != i
            and len(sequencias[j]) > len(seq)
            and sequencias[j][: len(seq)] == seq
            for j in range(len(itens))
        )
        if not coberto:
            resultado.append(nome)
    return resultado


def _unicos(nomes: Iterable[str]) -> list[str]:
    vistos: set[str] = set()
    unicos: list[str] = []
    for nome in nomes:
        chave = _ent.chave_nome(nome)
        if chave and chave not in vistos:
            vistos.add(chave)
            unicos.append(nome.strip())
    return unicos


# --------------------------------------------------------------------------- #
# Variantes                                                                   #
# --------------------------------------------------------------------------- #


def extrair_nomes_ner(
    texto: str,
    *,
    ner: Ner,
    max_chars: int = LIMITE_BLOCO,
    limite_gigante: int = LIMITE_GIGANTE,
) -> list[str]:
    """Nomes ``PESSOA`` do modelo, sem filtro. O gigante é ignorado."""
    if len(texto) > limite_gigante:
        return []
    achados: list[str] = []
    for bloco in _blocos(_dehyphen(texto), max_chars=max_chars):
        for nome in ner(bloco):
            limpo = re.sub(r"\s+", " ", str(nome)).strip(" .,;:-")
            if limpo:
                achados.append(limpo)
    return _unicos(achados)


def extrair_nomes_hibrido(
    texto: str,
    *,
    ner: Ner,
    max_chars: int = LIMITE_BLOCO,
    limite_gigante: int = LIMITE_GIGANTE,
) -> list[str]:
    """Híbrido: NER filtrado pela lista negra ∪ regras contextuais.

    O NER propõe; :func:`nome_parece_pessoa` derruba os falsos positivos de
    lugar/instituição/empresa. As regras de ``entidades.extrair_nomes`` entram
    como complemento de recall (contexto ``NOME``/``CPF``/tabela e gatilhos de
    assinatura), que é onde o NER puro costuma falhar (listas, campos separados).
    """
    if len(texto) > limite_gigante:
        # Gigante: NER não roda (D012); vale a extração determinística da coluna.
        return _ent.extrair_nomes(texto)

    do_ner: list[str] = []
    for bruto in extrair_nomes_ner(texto, ner=ner, max_chars=max_chars):
        for nome in _ent.separar_nomes_coordenados(bruto, texto=texto):
            limpo = _ent.limpar_superficie_nome(nome)
            if limpo and nome_parece_pessoa(limpo) and not _ent.em_contexto_lugar(texto, limpo):
                do_ner.append(limpo)
    das_regras: list[str] = []
    for bruto in _ent.extrair_nomes(texto):
        partes = _ent.separar_nomes_coordenados(bruto, texto=texto)
        if len(partes) == 1:
            das_regras.append(bruto)
            continue
        das_regras.extend(
            limpo for limpo in map(_ent.limpar_superficie_nome, partes) if limpo
        )
    # T030b (F002 padrão 5): candidato cortado antes do conector ``e`` é
    # completado com o run que segue (``… MOURA`` → ``… MOURA E VASCONCELOS``);
    # a separação de nomes coordenados só roda de novo quando algo mudou, já
    # com o texto à mão (``e`` no fim de linha não separa).
    completos: list[str] = []
    for nome in _unicos([*do_ner, *das_regras]):
        expandido = _ent.expandir_conector_e(texto, nome)
        if _ent.chave_nome(expandido) == _ent.chave_nome(nome):
            completos.append(nome)
            continue
        completos.extend(
            limpo
            for limpo in map(
                _ent.limpar_superficie_nome,
                _ent.separar_nomes_coordenados(expandido, texto=texto),
            )
            if limpo
        )
    # T030: descarta candidatos sem ocorrência "de pessoa" no texto (escola,
    # firma, sigla, fragmento de quebra de linha) ANTES de absorver prefixos.
    candidatos = _ent.filtrar_nomes_por_contexto(texto, _unicos(completos))
    return _absorver_prefixos(candidatos)


def nomes_da_variante(
    texto: str,
    variante: str,
    *,
    ner: Ner | None = None,
    max_chars: int = LIMITE_BLOCO,
    limite_gigante: int = LIMITE_GIGANTE,
) -> list[str]:
    """Nomes de uma variante (``a`` regras, ``b`` NER, ``c`` híbrido)."""
    variante = variante.lower()
    if variante == "a":
        return _ent.extrair_nomes(texto)
    if variante == "b":
        if ner is None:
            raise ValueError("variante 'b' exige uma função NER (ner=...)")
        return extrair_nomes_ner(
            texto, ner=ner, max_chars=max_chars, limite_gigante=limite_gigante
        )
    if variante == "c":
        if ner is None:
            raise ValueError("variante 'c' exige uma função NER (ner=...)")
        return extrair_nomes_hibrido(
            texto, ner=ner, max_chars=max_chars, limite_gigante=limite_gigante
        )
    raise ValueError(f"variante desconhecida: {variante!r} (use a, b ou c)")


def aplicar_variante(
    documento: Documento,
    variante: str,
    *,
    ner: Ner | None = None,
    max_chars: int = LIMITE_BLOCO,
    limite_gigante: int = LIMITE_GIGANTE,
) -> Documento:
    """Cópia do documento com ``entidades`` de nome da variante escolhida.

    As demais entidades (CPF/CNPJ/números) não são tocadas — o experimento mede a
    classe **nomes**; a lista devolvida fica só com entidades ``nome`` para isolar
    a variável.
    """
    texto = "\n".join(parte for parte in (documento.texto, *documento.tabelas) if parte)
    nomes = nomes_da_variante(
        texto, variante, ner=ner, max_chars=max_chars, limite_gigante=limite_gigante
    )
    entidades = [
        Entidade(
            tipo="nome",
            valor=nome,
            valor_normalizado=_ent.chave_nome(nome),
            pagina=documento.pagina_inicio,
        )
        for nome in nomes
        if nome and nome.strip()
    ]
    return documento.model_copy(update={"entidades": entidades})


# --------------------------------------------------------------------------- #
# Medição no gate                                                             #
# --------------------------------------------------------------------------- #

#: Exclusões de diagnóstico usadas no T016/I004 (documentos com fantasma no corpo
#: e anotações não exaustivas). NÃO substituem ``dados/anotacao/exclusoes.json``,
#: que pertence ao time de Dados — servem para comparar maçãs com maçãs.
EXCLUSOES_DIAGNOSTICO: tuple[dict[str, object], ...] = (
    {
        "arquivo": "DOM4197-12022026-ASSINADO.pdf",
        "campo": "nomes",
        "doc_seq": 5,
        "motivo": "fantasma/expediente no corpo (masthead_fantasma)",
    },
    {
        "arquivo": "DOM4198-13022026-ASSINADO.pdf",
        "campo": "nomes",
        "doc_seq": 4,
        "motivo": "fantasma/expediente no corpo (masthead_fantasma)",
    },
    {
        "arquivo": "DOM4196-11022026-ASSINADO.pdf",
        "campo": "nomes",
        "doc_seq": 2,
        "motivo": "lista de nomes não exaustiva (I004)",
    },
    {
        "arquivo": "DOM4197-12022026-ASSINADO.pdf",
        "campo": "nomes",
        "doc_seq": 23,
        "motivo": "anexo gigante: lista de nomes não enumerada (I004)",
    },
)


@dataclass
class ResultadoVariante:
    """Métricas e custo de uma variante de nomes no gate."""

    variante: str
    precisao: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    segundos: float
    paginas: int
    paginas_processadas: int
    caracteres: int
    caracteres_processados: int
    docs: int
    pico_rss_mb: float
    exemplos_fp: list[str] = field(default_factory=list)
    exemplos_fn: list[str] = field(default_factory=list)

    def como_dict(self) -> dict[str, object]:
        return {
            "variante": self.variante,
            "precisao": round(self.precisao, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "segundos": round(self.segundos, 2),
            "paginas": self.paginas,
            "paginas_processadas": self.paginas_processadas,
            "caracteres": self.caracteres,
            "caracteres_processados": self.caracteres_processados,
            "docs": self.docs,
            "pico_rss_mb": round(self.pico_rss_mb, 1),
            "exemplos_fp": self.exemplos_fp,
            "exemplos_fn": self.exemplos_fn,
        }


#: Hash de ``entidades.py`` capturado no IMPORT — congela a revisão usada pela
#: variante ``a`` mesmo que o arquivo mude em paralelo (outra tarefa o edita).
_REVISAO_REGRAS: str = ""
try:  # pragma: no cover - trivial
    import hashlib as _hashlib

    _REVISAO_REGRAS = _hashlib.sha256(Path(_ent.__file__).read_bytes()).hexdigest()[:16]
except Exception:  # noqa: BLE001  # pragma: no cover
    _REVISAO_REGRAS = "indisponivel"


def revisao_regras() -> str:
    """Hash curto da revisão de ``entidades.py`` importada neste processo."""
    return _REVISAO_REGRAS


def _pico_rss_mb() -> float:
    try:
        import resource

        valor = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return valor / 1024.0 if valor else 0.0
    except Exception:  # pragma: no cover - ambiente sem resource
        return 0.0


def medir_variantes(
    snapshot: Path | str,
    *,
    variantes: Sequence[str] = VARIANTES,
    ner: Ner | None = None,
    exclusoes: object | None = None,
    diagnostico: bool = False,
    max_chars: int = LIMITE_BLOCO,
    limite_gigante: int = LIMITE_GIGANTE,
    max_exemplos: int = 20,
) -> list[ResultadoVariante]:
    """Roda as variantes sobre ``snapshot`` e mede a classe **nomes** no gate.

    Importa ``harness.tests.gate_extracao`` de forma preguiçosa (o medidor é a
    referência do time). ``exclusoes`` pode ser um caminho para ``exclusoes.json``
    ou um objeto :class:`~harness.tests.gate_extracao.Exclusoes`; ``diagnostico``
    acrescenta :data:`EXCLUSOES_DIAGNOSTICO` para comparar com T016/I004.
    """
    # Import preguiçoso: mantém o módulo de produto independente do medidor.
    from harness.tests.gate_extracao import (  # noqa: PLC0415
        CLASSE_NOMES,
        Exclusoes,
        agregar,
        avaliar_arquivos,
        carregar_exclusoes,
        carregar_snapshot,
    )

    documentos = carregar_snapshot(snapshot)

    if exclusoes is None or isinstance(exclusoes, (str, Path)):
        base = carregar_exclusoes(exclusoes) if exclusoes is not None else carregar_exclusoes()
        exclusoes_obj = base
    else:
        exclusoes_obj = exclusoes

    if diagnostico:
        lista = list(EXCLUSOES_DIAGNOSTICO)
        if isinstance(exclusoes_obj, Exclusoes):
            # Preserva as exclusões do arquivo, se houver.
            for chave, itens in exclusoes_obj._por_arquivo.items():  # noqa: SLF001
                for item in itens:
                    lista.append(
                        {
                            "arquivo": item.arquivo,
                            "campo": item.campo,
                            "motivo": item.motivo,
                            "doc_seq": item.doc_seq,
                            "id_dom": item.id_dom,
                            "pagina_inicio": item.pagina_inicio,
                            "enunciado": item.enunciado,
                        }
                    )
        exclusoes_obj = Exclusoes.de_lista(lista)

    paginas = sum(d.pagina_fim - d.pagina_inicio + 1 for d in documentos)
    caracteres = sum(len(d.texto) + sum(len(t) for t in d.tabelas) for d in documentos)
    # O gigante não recebe NER; o custo por página/char usa só o que foi processado.
    processaveis = [
        d
        for d in documentos
        if len(d.texto) + sum(len(t) for t in d.tabelas) <= limite_gigante
    ]
    paginas_proc = sum(d.pagina_fim - d.pagina_inicio + 1 for d in processaveis)
    caracteres_proc = sum(
        len(d.texto) + sum(len(t) for t in d.tabelas) for d in processaveis
    )

    resultados: list[ResultadoVariante] = []
    for variante in variantes:
        inicio = time.perf_counter()
        processados = [
            aplicar_variante(
                documento,
                variante,
                ner=ner,
                max_chars=max_chars,
                limite_gigante=limite_gigante,
            )
            for documento in documentos
        ]
        segundos = time.perf_counter() - inicio
        avaliados = avaliar_arquivos(processados, exclusoes=exclusoes_obj)
        total = agregar(avaliados)
        contagem = total.medida(CLASSE_NOMES)
        divergencias = [
            d
            for resultado in avaliados
            for d in resultado.divergencias
            if d.classe == CLASSE_NOMES
        ]
        fps = sorted({d.extraido for d in divergencias if d.ouro is None and d.extraido})
        fns = sorted({d.ouro for d in divergencias if d.extraido is None and d.ouro})
        resultados.append(
            ResultadoVariante(
                variante=variante,
                precisao=contagem.precisao,
                recall=contagem.recall,
                f1=contagem.f1,
                tp=contagem.tp,
                fp=contagem.fp,
                fn=contagem.fn,
                segundos=segundos,
                paginas=paginas,
                paginas_processadas=paginas_proc,
                caracteres=caracteres,
                caracteres_processados=caracteres_proc,
                docs=len(documentos),
                pico_rss_mb=_pico_rss_mb(),
                exemplos_fp=fps[:max_exemplos],
                exemplos_fn=fns[:max_exemplos],
            )
        )
    return resultados


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

RAIZ = Path(__file__).resolve().parents[2]
SNAPSHOT_PADRAO = RAIZ / "dados" / "snapshots" / "dev.jsonl"
EXCLUSOES_PADRAO = RAIZ / "dados" / "anotacao" / "exclusoes.json"


def _formatar(resultados: Sequence[ResultadoVariante]) -> str:
    linhas = [
        f"{'var':<4} {'P':>7} {'R':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5} "
        f"{'s':>7} {'ms/pág':>8} {'ms/kchar':>9} {'RSS MB':>8}",
    ]
    for r in resultados:
        ms_pag = (
            (r.segundos * 1000 / r.paginas_processadas) if r.paginas_processadas else 0.0
        )
        ms_char = (
            (r.segundos * 1000 / (r.caracteres_processados / 1000))
            if r.caracteres_processados
            else 0.0
        )
        linhas.append(
            f"{r.variante:<4} {r.precisao:>7.3f} {r.recall:>7.3f} {r.f1:>7.3f} "
            f"{r.tp:>5} {r.fp:>5} {r.fn:>5} {r.segundos:>7.1f} {ms_pag:>8.1f} "
            f"{ms_char:>9.2f} {r.pico_rss_mb:>8.1f}"
        )
    return "\n".join(linhas)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m harness.ingest.ner_nomes [--variantes a,b,c]``."""
    parser = argparse.ArgumentParser(
        description="Experimento de NER de nomes (passo 10 do I004; T019)."
    )
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PADRAO)
    parser.add_argument("--exclusoes", type=Path, default=EXCLUSOES_PADRAO)
    parser.add_argument(
        "--diagnostico",
        action="store_true",
        help="Acrescenta EXCLUSOES_DIAGNOSTICO (4 docs de T016/I004).",
    )
    parser.add_argument(
        "--variantes", default="a,b,c", help="Variantes a medir (padrão: a,b,c)."
    )
    parser.add_argument("--modelo", default=MODELO_PADRAO)
    parser.add_argument(
        "--dispositivo", type=int, default=-1, help="-1 = CPU (padrão)."
    )
    parser.add_argument("--max-chars", type=int, default=LIMITE_BLOCO)
    parser.add_argument("--limite-gigante", type=int, default=LIMITE_GIGANTE)
    parser.add_argument(
        "--exemplos", type=int, default=15, help="Exemplos de FP/FN por variante."
    )
    parser.add_argument(
        "--sem-gate",
        action="store_true",
        help="Compatibilidade: o experimento nunca reprova o gate.",
    )
    parser.add_argument("--json", type=Path, default=None, help="Salva o resultado.")
    args = parser.parse_args(argv)

    print(f"entidades.py (variante a) sha256={revisao_regras()}")
    variantes = [v.strip() for v in args.variantes.split(",") if v.strip()]
    ner: Ner | None = None
    if any(v in ("b", "c") for v in variantes):
        ner = carregar_ner(args.modelo, dispositivo=args.dispositivo)
        # Aquece o modelo para não medir o carregamento no primeiro doc medido.
        ner("Teresina, 1 de janeiro de 2026. João da Silva.")

    resultados = medir_variantes(
        args.snapshot,
        variantes=variantes,
        ner=ner,
        exclusoes=args.exclusoes,
        diagnostico=args.diagnostico,
        max_chars=args.max_chars,
        limite_gigante=args.limite_gigante,
        max_exemplos=args.exemplos,
    )
    print(_formatar(resultados))
    for r in resultados:
        print(f"\n# variante {r.variante} — exemplos de erro")
        print("  FP:", ", ".join(r.exemplos_fp) or "(nenhum)")
        print("  FN:", ", ".join(r.exemplos_fn) or "(nenhum)")
        print(
            f"  docs={r.docs} páginas={r.paginas} (processadas={r.paginas_processadas}) "
            f"chars={r.caracteres} (processados={r.caracteres_processados}) "
            f"segundos={r.segundos:.1f} RSS={r.pico_rss_mb:.1f}MB"
        )
    if args.json is not None:
        args.json.write_text(
            json.dumps([r.como_dict() for r in resultados], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

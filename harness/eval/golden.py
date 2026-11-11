"""Geração e verificação do GOLDEN SET de avaliação do chat (T009).

O golden set é construído **na geração a partir da anotação de referência**
(``dados/anotacao/DOM*.jsonl``) e cada caso é **verificado por código contra o
texto do PDF**: o ``trecho_literal`` da evidência precisa existir na página
indicada do PDF (comparação sem acento/caixa, com hífen de quebra de linha e
espaços normalizados). Casos que falham na verificação são **descartados** —
nunca ajustamos o gabarito à saída de um pipeline em execução
(``knowledge/regras_diario.md``, seção 4).

Convenção de ``doc_id`` (ver também ``harness/eval/README.md``)::

    doc_id = "<arquivo_sem_extensao>#<doc_seq>"

onde ``arquivo_sem_extensao`` é o nome-base do PDF em ``diarios/`` e
``doc_seq`` é a ordem do documento no PDF (1-based), exatamente como anotada.

Categorias e semântica de ``esperado`` (detalhada no README):

- ``buscar``: True se a mensagem deve disparar busca; False em conversa.
- ``perguntar``: True se a resposta correta é desambiguar (perguntar ao usuário).
- ``doc_ids_alvo``: docs que contam como acerto (hit@5) na resposta final.
- ``opcoes_doc_ids``: opções reais esperadas quando ``perguntar`` é True.

Uso::

    uv run python -m harness.eval.golden            # grava dev/ e holdout
    uv run python -m harness.eval.golden --sem-holdout  # só dev (testes)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

try:  # PyMuPDF >= 1.24 expõe ``pymupdf``; versões antigas só ``fitz``.
    import pymupdf
except ImportError:  # pragma: no cover - compatibilidade com versões antigas
    import fitz as pymupdf  # type: ignore[no-redef]

__all__ = [
    "ALVO_POR_SPLIT",
    "CATEGORIAS",
    "CasoGolden",
    "Descarte",
    "DocumentoAnotado",
    "Esperado",
    "Evidencia",
    "LeitorPDF",
    "RelatorioGeracao",
    "canon_numero",
    "carregar_anotacao",
    "doc_com_numero_unico",
    "doc_id_de",
    "dividir_por_categoria",
    "formato_cpf_cnpj",
    "gerar_casos",
    "gerar_golden",
    "localizar_evidencia",
    "nome_arquivo_de",
    "nomes_distintos",
    "normalizar_verificacao",
    "numero_compacto",
    "numero_especifico",
    "parafrase_valida",
    "parafrases_do_tema",
]

# --------------------------------------------------------------------------- #
# Constantes e caminhos                                                       #
# --------------------------------------------------------------------------- #

#: Categorias obrigatórias do golden set, na ordem do briefing.
CATEGORIAS: tuple[str, ...] = (
    "conversa_sem_busca",
    "nome_unico",
    "nomes_multiplos",
    "id_dom",
    "numero",
    "cpf_cnpj",
    "contexto",
    "nome_com_erro",
    "ambiguo",
    "follow_up",
)

#: Casos desejados por categoria em cada split.
ALVO_POR_SPLIT = 10

#: Quantos candidatos por categoria construímos antes da verificação/recorte.
#: Precisa sobrar folga para os descartes de verificação.
MARGEM_CANDIDATOS = 4

#: Semente determinística da divisão dev/holdout.
SEED = 20260925

RAIZ = Path(__file__).resolve().parents[2]
DIR_ANOTACAO = RAIZ / "dados" / "anotacao"
DIR_DIARIOS = RAIZ / "diarios"
DEV_PATH = Path(__file__).resolve().parent / "dev" / "golden_dev.jsonl"
HOLDOUT_PATH = Path.home() / "escavador_holdout" / "golden_holdout.jsonl"

#: Mensagens de :data:`CATEGORIAS` que NÃO devem disparar busca. Ficam fora da
#: anotação de propósito: são conversa pura, sem documento-alvo nem evidência.
MENSAGENS_CONVERSA: tuple[str, ...] = (
    "Olá! Tudo bem?",
    "Bom dia, você pode me ajudar?",
    "Oi, quem é você?",
    "O que é o Diário Oficial do Município?",
    "Qual a diferença entre um decreto e uma portaria?",
    "Você consegue ler arquivos PDF?",
    "O que significa a sigla DOM?",
    "Me explique o que é um termo aditivo.",
    "Para que serve um edital de licitação?",
    "Você tem acesso à internet?",
    "Obrigado pela ajuda!",
    "Até logo, tchau!",
    "Quais são os tipos de atos oficiais?",
    "Como funciona a numeração das portarias?",
    "O que é um decreto legislativo?",
    "Você guarda o histórico das nossas conversas?",
    "Pode me explicar o que é um contrato administrativo?",
    "O que é um extrato de contrato?",
    "Quais órgãos publicam no diário oficial?",
    "Como faço para citar um documento do diário?",
    "Qual o seu nome?",
    "Você é um robô?",
    "Boa tarde! Está funcionando?",
    "Você entende português?",
)


# --------------------------------------------------------------------------- #
# Modelos do caso                                                             #
# --------------------------------------------------------------------------- #


class Evidencia(BaseModel):
    """Trecho literal que comprova o gabarito, verificado contra o PDF."""

    model_config = ConfigDict(extra="forbid")

    arquivo: str = Field(description="Nome-base do PDF de origem em diarios/.")
    pagina: int = Field(ge=1, description="Página (1-based) onde o trecho aparece.")
    trecho_literal: str = Field(
        min_length=1, description="Fragmento literal que deve existir na página."
    )


class Esperado(BaseModel):
    """Gabarito de um caso de avaliação."""

    model_config = ConfigDict(extra="forbid")

    doc_ids_alvo: list[str] = Field(
        default_factory=list,
        description="doc_ids que contam como acerto (hit@5) na resposta final.",
    )
    buscar: bool = Field(description="True se a mensagem deve disparar busca.")
    perguntar: bool = Field(
        default=False, description="True se a resposta correta é desambiguar."
    )
    opcoes_doc_ids: list[str] = Field(
        default_factory=list,
        description="doc_ids das opções reais quando `perguntar` é True.",
    )
    posicao_resultado: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Follow-up por posição ('e o segundo?'): índice 1-based na ordem "
            "canônica de exibição (ranking devolvido no turno anterior). Quando "
            "presente, o alvo é o resultado nessa posição e `doc_ids_candidatos` "
            "lista os docs que o turno 1 deve ter exibido."
        ),
    )
    doc_ids_candidatos: list[str] = Field(
        default_factory=list,
        description=(
            "Follow-up por posição: docs esperados no top-k do turno anterior; "
            "serve para auditar a recuperação do turno 1 e para não repetir "
            "documento-alvo entre dev e holdout."
        ),
    )


class CasoGolden(BaseModel):
    """Um caso do golden set (uma linha de JSONL)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Identificador único do caso.")
    categoria: str = Field(description="Uma de CATEGORIAS.")
    turnos: list[str] = Field(min_length=1, description="Mensagens do usuário, em ordem.")
    esperado: Esperado
    evidencia: Evidencia | None = Field(
        default=None,
        description="Evidência verificada; None apenas em conversa_sem_busca.",
    )


@dataclass(frozen=True)
class Descarte:
    """Registro de um candidato rejeitado na verificação."""

    categoria: str
    motivo: str
    detalhe: str = ""


@dataclass(frozen=True)
class DocumentoAnotado:
    """Um documento da anotação de referência (``dados/anotacao/*.jsonl``)."""

    arquivo: str
    edicao_numero: str
    edicao_data: str
    doc_seq: int
    pagina_inicio: int
    pagina_fim: int
    enunciado: str
    tipo: str
    numero: str | None
    ano: int | None
    id_dom: str | None
    orgao: str | None
    inicio_literal: str
    nomes: tuple[str, ...] = ()
    cpfs: tuple[str, ...] = ()
    cnpjs: tuple[str, ...] = ()
    outros_numeros: tuple[str, ...] = ()
    tem_tabela: bool = False

    @property
    def doc_id(self) -> str:
        return doc_id_de(self.arquivo, self.doc_seq)

    @property
    def basename(self) -> str:
        return nome_arquivo_de(self.arquivo)

    @property
    def paginas(self) -> range:
        return range(self.pagina_inicio, self.pagina_fim + 1)


# --------------------------------------------------------------------------- #
# Carregamento da anotação                                                    #
# --------------------------------------------------------------------------- #


def nome_arquivo_de(arquivo: str) -> str:
    """Nome-base do PDF, sem diretório (a anotação às vezes guarda ``diarios/``)."""
    return Path(arquivo.replace("\\", "/")).name


def doc_id_de(arquivo: str, doc_seq: int) -> str:
    """Convenção única de ``doc_id``: ``<arquivo_sem_ext>#<doc_seq>``."""
    return f"{Path(nome_arquivo_de(arquivo)).stem}#{doc_seq}"


def carregar_anotacao(diretorio: Path | str = DIR_ANOTACAO) -> list[DocumentoAnotado]:
    """Lê todos os ``DOM*.jsonl`` do diretório de anotação.

    Ignora os ``bold_*.jsonl`` (anotação de linhas, não de documentos).
    """
    diretorio = Path(diretorio)
    documentos: list[DocumentoAnotado] = []
    for caminho in sorted(diretorio.glob("DOM*.jsonl")):
        with caminho.open(encoding="utf-8") as arquivo:
            for linha in arquivo:
                linha = linha.strip()
                if not linha:
                    continue
                bruto = json.loads(linha)
                documentos.append(
                    DocumentoAnotado(
                        arquivo=bruto["arquivo"],
                        edicao_numero=str(bruto.get("edicao_numero", "")),
                        edicao_data=str(bruto.get("edicao_data", "")),
                        doc_seq=int(bruto["doc_seq"]),
                        pagina_inicio=int(bruto["pagina_inicio"]),
                        pagina_fim=int(bruto["pagina_fim"]),
                        enunciado=str(bruto.get("enunciado", "")),
                        tipo=str(bruto.get("tipo", "")),
                        numero=bruto.get("numero"),
                        ano=bruto.get("ano"),
                        id_dom=bruto.get("id_dom"),
                        orgao=bruto.get("orgao"),
                        inicio_literal=str(bruto.get("inicio_literal", "")),
                        nomes=tuple(bruto.get("nomes", []) or ()),
                        cpfs=tuple(bruto.get("cpfs", []) or ()),
                        cnpjs=tuple(bruto.get("cnpjs", []) or ()),
                        outros_numeros=tuple(bruto.get("outros_numeros", []) or ()),
                        tem_tabela=bool(bruto.get("tem_tabela", False)),
                    )
                )
    return documentos


# --------------------------------------------------------------------------- #
# Normalização e verificação contra o PDF                                     #
# --------------------------------------------------------------------------- #


def _sem_acento(texto: str) -> str:
    decomposto = unicodedata.normalize("NFKD", texto or "")
    return "".join(c for c in decomposto if not unicodedata.combining(c))


def normalizar_verificacao(texto: str) -> str:
    """Normaliza texto para casamento literal contra o PDF.

    - sem acento e caixa-insensível;
    - **hífen de quebra de linha** removido (``Es-\\ntrutura`` → ``Estrutura``);
    - espaços (inclusive ``\\n``/``\\t``) colapsados.
    """
    t = _sem_acento(texto).casefold()
    t = re.sub(r"-\s*\n\s*", "", t)  # hífen ao fim da linha + quebra
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def chave_nome(nome: str) -> str:
    """Chave canônica de nome (sem acento/caixa, espaços colapsados)."""
    return normalizar_verificacao(nome)


def _tokens_nome(nome: str) -> list[str]:
    return chave_nome(nome).split()


def _distancia_ate_1(a: str, b: str) -> bool:
    """True se ``a`` e ``b`` diferem por no máximo 1 edição (ins/del/troca)."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = j = 0
    edicoes = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        edicoes += 1
        if edicoes > 1:
            return False
        if len(a) == len(b):
            i += 1
            j += 1
        else:
            j += 1
    edicoes += (len(a) - i) + (len(b) - j)
    return edicoes <= 1


def nomes_distintos(nome_a: str, nome_b: str) -> bool:
    """True se os nomes parecem **pessoas distintas** (R009/B3).

    Rejeita variantes do MESMO nome: mesmo conjunto de tokens, subconjunto
    (nome mais/menos completo) ou 1 token diferindo por 1 letra (ex.:
    ``SOUZA`` × ``SOUSA``). É isso que impede ``nomes_multiplos`` de usar duas
    grafias da mesma pessoa.
    """
    tokens_a = _tokens_nome(nome_a)
    tokens_b = _tokens_nome(nome_b)
    if not tokens_a or not tokens_b:
        return False
    if tokens_a == tokens_b:
        return False
    conj_a, conj_b = set(tokens_a), set(tokens_b)
    comuns = conj_a & conj_b
    if comuns and (conj_a <= conj_b or conj_b <= conj_a) and min(
        len(conj_a), len(conj_b)
    ) >= 2:
        return False
    if len(tokens_a) == len(tokens_b) and len(comuns) >= len(tokens_a) - 1:
        diferentes = [
            (a, b) for a, b in zip(tokens_a, tokens_b) if a != b
        ]
        if len(diferentes) <= 1:
            if not diferentes or _distancia_ate_1(*diferentes[0]):
                return False
    return True


def _sem_hifen(texto: str) -> str:
    return texto.replace("-", "")


# --------------------------------------------------------------------------- #
# Paráfrases de contexto (sem copiar o texto do documento)                    #
# --------------------------------------------------------------------------- #

#: Stopwords usadas só na checagem de "palavras de conteúdo" das paráfrases.
_STOPWORDS_PARAFRASE = {
    "a", "o", "os", "as", "de", "da", "do", "dos", "das", "e", "em", "no", "na",
    "nos", "nas", "um", "uma", "uns", "umas", "para", "por", "com", "sem", "que",
    "se", "ao", "aos", "às", "tem", "ter", "foi", "ser", "sao", "é", "qual",
    "quais", "algum", "alguma", "alguns", "algumas", "mais", "menos", "muito",
    "como", "onde", "quando", "quem", "ja", "já", "nao", "não", "sim", "ou",
    "sobre", "entre", "seu", "sua", "seus", "suas", "este", "esta", "esse",
    "essa", "isso", "aquele", "aquela", "vai", "vao", "deu", "dar", "fazer",
    "fez", "faz", "está", "estao", "precisa", "preciso", "quero", "mostra",
    "mostre", "procura", "procure", "busca", "busque", "acha", "ache",
    "documento", "documentos", "ato", "atos", "publicacao", "publicacoes",
    "saiu", "sairam", "teve", "houve", "foi", "são", "sao",
}


def _tokens(texto: str) -> list[str]:
    """Tokens normalizados (sem acento/caixa) para n-gramas e conteúdo."""
    return re.findall(r"[a-z0-9/.\-]+", normalizar_verificacao(texto))


def _ngramas(tokens: Sequence[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _palavras_conteudo(tokens: Sequence[str]) -> set[str]:
    return {t for t in tokens if len(t) >= 4 and t not in _STOPWORDS_PARAFRASE}


def parafrase_valida(consulta: str, trechos: Sequence[str]) -> bool:
    """True se a paráfrase não copia o texto (regra 2 do T009b).

    Exige **zero 3-gramas em comum** e no máximo **2 palavras de conteúdo** em
    comum entre a consulta e qualquer trecho do documento.
    """
    tokens_consulta = _tokens(consulta)
    gramas_consulta = _ngramas(tokens_consulta, 3)
    gramas_doc: set[tuple[str, ...]] = set()
    conteudo_doc: set[str] = set()
    for trecho in trechos:
        tokens = _tokens(trecho)
        gramas_doc |= _ngramas(tokens, 3)
        conteudo_doc |= _palavras_conteudo(tokens)
    if gramas_consulta & gramas_doc:
        return False
    comuns = _palavras_conteudo(tokens_consulta) & conteudo_doc
    return len(comuns) <= 2


class LeitorPDF:
    """Lê e cacheia o texto normalizado das páginas dos PDFs de ``diarios/``."""

    def __init__(self, diretorio: Path | str = DIR_DIARIOS) -> None:
        self.diretorio = Path(diretorio)
        self._cache: dict[str, list[str]] = {}

    def _textos(self, arquivo: str) -> list[str]:
        nome = nome_arquivo_de(arquivo)
        if nome not in self._cache:
            caminho = self.diretorio / nome
            if not caminho.exists():
                raise FileNotFoundError(f"PDF não encontrado: {caminho}")
            with pymupdf.open(str(caminho)) as doc:
                self._cache[nome] = [
                    normalizar_verificacao(pagina.get_text("text")) for pagina in doc
                ]
        return self._cache[nome]

    def texto_pagina(self, arquivo: str, pagina: int) -> str:
        """Texto normalizado de uma página (1-based)."""
        textos = self._textos(arquivo)
        if not 1 <= pagina <= len(textos):
            raise ValueError(f"página {pagina} fora do PDF {arquivo}")
        return textos[pagina - 1]

    def contem(self, arquivo: str, pagina: int, trecho: str) -> bool:
        """True se ``trecho`` (normalizado) aparece na página indicada."""
        alvo = normalizar_verificacao(trecho)
        if not alvo:
            return False
        texto = self.texto_pagina(arquivo, pagina)
        if alvo in texto:
            return True
        # Fallback: hífens não mapeados (quebra de coluna, formatação quebrada).
        alvo_sem = _sem_hifen(alvo)
        return len(alvo_sem) >= 6 and alvo_sem in _sem_hifen(texto)

    def localizar(
        self, arquivo: str, paginas: Iterable[int], trecho: str
    ) -> int | None:
        """Primeira página (1-based) onde ``trecho`` aparece; ``None`` se ausente."""
        for pagina in paginas:
            if self.contem(arquivo, pagina, trecho):
                return pagina
        return None


def localizar_evidencia(
    leitor: LeitorPDF, documento: DocumentoAnotado, trecho: str
) -> Evidencia | None:
    """Monta a evidência se o trecho existir no intervalo de páginas do documento."""
    pagina = leitor.localizar(documento.arquivo, documento.paginas, trecho)
    if pagina is None:
        return None
    return Evidencia(arquivo=documento.basename, pagina=pagina, trecho_literal=trecho)


def verificar_caso(leitor: LeitorPDF, caso: CasoGolden) -> bool:
    """Confere a evidência do caso contra o PDF (None é aceito só sem alvo)."""
    if caso.evidencia is None:
        return not caso.esperado.doc_ids_alvo and not caso.esperado.opcoes_doc_ids
    return leitor.contem(
        caso.evidencia.arquivo, caso.evidencia.pagina, caso.evidencia.trecho_literal
    )


# --------------------------------------------------------------------------- #
# Numeração canônica (para unicidade e ambiguidade)                           #
# --------------------------------------------------------------------------- #

_TIPOS_DECRETO = re.compile(r"\b(decreto|lei)\b", re.IGNORECASE)


def canon_numero(tipo: str, numero: str | None) -> str | None:
    """Chave canônica de número de ato, para comparar escritas diferentes.

    - Decretos/leis usam os dígitos do número (``28.597`` → ``num-28597``), pois
      o ponto é separador de milhar.
    - Demais atos usam ``base/ano`` (``055/2026/GAB/SEMEC`` → ``ato-55/2026``),
      ignorando zeros à esquerda do número e sufixos de órgão.
    """
    if not numero:
        return None
    if _TIPOS_DECRETO.search(tipo or ""):
        digitos = re.sub(r"\D", "", numero)
        return f"num-{digitos}" if digitos else None
    grupos = re.findall(r"\d+", numero)
    if not grupos:
        return None
    base = grupos[0].lstrip("0") or "0"
    ano = ""
    for grupo in grupos[1:]:
        if len(grupo) == 4 and grupo.startswith(("19", "20")):
            ano = grupo
    return f"ato-{base}/{ano}" if ano else f"ato-{base}"


def token_ambiguo(chave: str) -> str:
    """Token legível a partir da chave canônica (ex.: ``ato-55/2026`` → ``55/2026``)."""
    if chave.startswith("ato-"):
        return chave[4:]
    if chave.startswith("num-"):
        return chave[4:]
    return chave


def numero_especifico(chave: str | None) -> bool:
    """True se a chave identifica o número com ano (ou decreto/lei completo).

    Número solto sem ano (``ato-2``) é genérico demais: "extrato número 2" pode
    casar vários documentos, então não serve para a categoria ``numero``.
    """
    if not chave:
        return False
    if chave.startswith("num-"):
        return True
    return bool(re.fullmatch(r"ato-\d+/\d{4}", chave))


def numero_compacto(numero: str) -> str:
    """Forma ``base/ano`` sem zeros à esquerda nem sufixo de órgão."""
    achado = re.search(r"(\d+)\s*/\s*(\d{4})", numero)
    if not achado:
        return numero.strip()
    base = achado.group(1).lstrip("0") or "0"
    return f"{base}/{achado.group(2)}"


# --------------------------------------------------------------------------- #
# Índices auxiliares                                                          #
# --------------------------------------------------------------------------- #


@dataclass
class _Indices:
    """Índices derivados da anotação, reusados pelos construtores."""

    documentos: list[DocumentoAnotado]
    por_doc_id: dict[str, DocumentoAnotado] = field(default_factory=dict)
    docs_por_nome: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    docs_por_numero: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    docs_por_id_dom: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    docs_por_cpf_cnpj: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    #: Cache dos âncoras de contexto (evita recomputar/duplicar descartes).
    cache_contexto: tuple[list[dict], list["Descarte"]] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for doc in self.documentos:
            self.por_doc_id[doc.doc_id] = doc
            if doc.id_dom:
                self.docs_por_id_dom[doc.id_dom].add(doc.doc_id)
            for nome in doc.nomes:
                chave = chave_nome(nome)
                if len(chave) >= 6:
                    self.docs_por_nome[chave].add(doc.doc_id)
            if doc.numero:
                chave = canon_numero(doc.tipo, doc.numero)
                if chave:
                    self.docs_por_numero[chave].add(doc.doc_id)
            for identificador in (*doc.cpfs, *doc.cnpjs):
                chave = chave_cpf_cnpj(identificador)
                if chave:
                    self.docs_por_cpf_cnpj[chave].add(doc.doc_id)


def chave_cpf_cnpj(valor: str) -> str | None:
    """Chave de CPF/CNPJ: só dígitos quando completo; senão o literal normalizado."""
    digitos = re.sub(r"\D", "", valor)
    if "*" in valor or len(digitos) < 8:
        return normalizar_verificacao(valor) if valor.strip() else None
    return digitos


_RE_CPF_FORMATADO = re.compile(r"^\d{3}\.\d{3}\.\d{3}-\d{2}$")
_RE_CNPJ_FORMATADO = re.compile(r"^\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}$")


def formato_cpf_cnpj(valor: str) -> str:
    """Classifica a superfície: ``cpf``, ``cnpj``, ``mascarado`` ou ``quebrado``.

    ``quebrado`` cobre os formatos reais do diário que não seguem a máscara
    (ex.: ``898.19187-20``, ``062.979.903.28``, ``45.517.180./0001-11``,
    ``19590049/0001-70``).
    """
    if "*" in valor:
        return "mascarado"
    if _RE_CPF_FORMATADO.match(valor):
        return "cpf"
    if _RE_CNPJ_FORMATADO.match(valor):
        return "cnpj"
    return "quebrado"


def montar_indices(documentos: Sequence[DocumentoAnotado]) -> _Indices:
    return _Indices(documentos=list(documentos))


# --------------------------------------------------------------------------- #
# Construtores por categoria                                                  #
# --------------------------------------------------------------------------- #

#: Marcas usadas para montar consultas sem repetição excessiva.
_TIPOS_TITULO = {
    "PORTARIA": "portaria",
    "DECRETO": "decreto",
    "DECRETO LEGISLATIVO": "decreto legislativo",
    "LEI": "lei",
    "LEI COMPLEMENTAR": "lei complementar",
    "CONTRATO": "contrato",
    "EXTRATO": "extrato",
    "EDITAL": "edital",
    "ERRATA": "errata",
    "RESOLUÇÃO": "resolução",
    "ACÓRDÃO": "acórdão",
    "TERMO ADITIVO": "termo aditivo",
    "INSTRUÇÃO NORMATIVA": "instrução normativa",
    "DISPENSA": "dispensa",
    "ATA": "ata",
    "DECISÃO": "decisão",
}


def _tipo_legivel(tipo: str) -> str:
    return _TIPOS_TITULO.get(tipo, (tipo or "documento").lower())


def _conversa() -> tuple[list[CasoGolden], list[Descarte]]:
    casos = [
        CasoGolden(
            id="",
            categoria="conversa_sem_busca",
            turnos=[mensagem],
            esperado=Esperado(buscar=False, perguntar=False),
            evidencia=None,
        )
        for mensagem in MENSAGENS_CONVERSA
    ]
    return casos, []


def _nome_unico(ix: _Indices, leitor: LeitorPDF) -> tuple[list[CasoGolden], list[Descarte]]:
    casos: list[CasoGolden] = []
    descartes: list[Descarte] = []
    vistos: set[str] = set()
    limite = ALVO_POR_SPLIT * 2 * MARGEM_CANDIDATOS
    for doc in ix.documentos:
        if len(casos) >= limite:
            break
        for nome in doc.nomes:
            chave = chave_nome(nome)
            if len(chave) < 8 or len(nome.split()) < 2 or chave in vistos:
                continue
            alvo = sorted(ix.docs_por_nome.get(chave, set()))
            if not alvo or len(alvo) > 3:
                continue
            evidencia = localizar_evidencia(leitor, doc, nome)
            if evidencia is None:
                descartes.append(
                    Descarte("nome_unico", "trecho não encontrado no PDF", nome)
                )
                continue
            vistos.add(chave)
            templates = (
                f"Quero o documento que cita {nome}.",
                f"Em qual ato aparece {nome}?",
                f"Procure publicações que mencionem {nome}.",
                f"Você tem o documento com o nome {nome}?",
                f"{nome}",
                f"onde acham {nome}?",
                f"tem algo com {nome}?",
                f"me diz em que ato está {nome}",
            )
            turno = templates[len(casos) % len(templates)]
            casos.append(
                CasoGolden(
                    id="",
                    categoria="nome_unico",
                    turnos=[turno],
                    esperado=Esperado(doc_ids_alvo=alvo, buscar=True),
                    evidencia=evidencia,
                )
            )
            break
    return casos, descartes


def _nomes_multiplos(
    ix: _Indices, leitor: LeitorPDF
) -> tuple[list[CasoGolden], list[Descarte]]:
    """2–3 pessoas **distintas** que aparecem no mesmo documento (R009/B3).

    Rejeita combinações com grafias do mesmo nome (`nomes_distintos`). O alvo é
    a interseção dos documentos que citam TODAS as pessoas; o acerto do caso é
    ≥1 alvo no top-5 (pode haver mais de um documento com o mesmo conjunto).
    """
    casos: list[CasoGolden] = []
    descartes: list[Descarte] = []
    chaves_usadas: set[tuple[str, ...]] = set()
    limite = ALVO_POR_SPLIT * 2 * MARGEM_CANDIDATOS
    for doc in sorted(ix.documentos, key=lambda d: -len(d.nomes)):
        if len(casos) >= limite:
            break
        candidatos = [
            n for n in doc.nomes if len(chave_nome(n)) >= 8 and len(n.split()) >= 2
        ]
        if len(candidatos) < 2:
            continue
        # Nomes mais raros primeiro para o alvo ficar discriminativo.
        candidatos.sort(key=lambda n: len(ix.docs_por_nome.get(chave_nome(n), set())))
        base = candidatos[:10]  # limita o custo das combinações

        escolhido: tuple | None = None
        for quantidade in (2, 3):
            if len(casos) >= limite or escolhido is not None:
                break
            for combo in combinations(base, quantidade):
                if not all(
                    nomes_distintos(a, b) for a, b in combinations(combo, 2)
                ):
                    continue
                chaves = tuple(sorted(chave_nome(n) for n in combo))
                if chaves in chaves_usadas:
                    continue
                conjuntos = [ix.docs_por_nome.get(k, set()) for k in chaves]
                alvo = set.intersection(*conjuntos) if conjuntos else set()
                if not alvo or len(alvo) > 2:
                    continue
                mostraveis = [n for n in combo if n in doc.nomes]
                evidencias = [
                    localizar_evidencia(leitor, doc, n) for n in mostraveis
                ]
                if any(e is None for e in evidencias):
                    descartes.append(
                        Descarte(
                            "nomes_multiplos",
                            "nome não encontrado no PDF",
                            " | ".join(mostraveis),
                        )
                    )
                    continue
                escolhido = (combo, chaves, alvo, mostraveis, evidencias)
                break
        if escolhido is None:
            continue
        _combo, chaves, alvo, mostraveis, evidencias = escolhido
        chaves_usadas.add(chaves)
        if len(mostravels := mostraveis) == 1:
            citacao = mostraveis[0]
        else:
            citacao = " e ".join([", ".join(mostraveis[:-1]), mostraveis[-1]])
        templates = (
            f"Me mostre o ato que cita {citacao}.",
            f"Qual documento menciona {citacao}?",
            f"Busque o documento em que aparecem {citacao}.",
            f"{citacao}",
            f"tem um ato com {citacao}?",
            f"onde aparecem juntos {citacao}?",
        )
        turno = templates[len(casos) % len(templates)]
        casos.append(
            CasoGolden(
                id="",
                categoria="nomes_multiplos",
                turnos=[turno],
                esperado=Esperado(doc_ids_alvo=sorted(alvo), buscar=True),
                evidencia=evidencias[0],
            )
        )
    return casos, descartes


def _id_dom(ix: _Indices, leitor: LeitorPDF) -> tuple[list[CasoGolden], list[Descarte]]:
    casos: list[CasoGolden] = []
    descartes: list[Descarte] = []
    for doc in ix.documentos:
        if len(casos) >= ALVO_POR_SPLIT * 2 * MARGEM_CANDIDATOS:
            break
        if not doc.id_dom:
            continue
        evidencia = localizar_evidencia(leitor, doc, doc.id_dom)
        if evidencia is None:
            descartes.append(
                Descarte("id_dom", "ID não encontrado no PDF", doc.id_dom)
            )
            continue
        espacado = " ".join(
            doc.id_dom[i : i + 3] for i in range(0, len(doc.id_dom), 3)
        )
        meia = f"{doc.id_dom[:9]} {doc.id_dom[9:]}"
        templates = (
            f"Localize o documento de ID {doc.id_dom}.",
            f"ID: {doc.id_dom}",
            f"Quero o documento cujo id é {doc.id_dom}.",
            f"Busque pelo ID {espacado}.",
            f"{doc.id_dom}",
            f"id {espacado}",
            f"{meia}",
        )
        turno = templates[len(casos) % len(templates)]
        casos.append(
            CasoGolden(
                id="",
                categoria="id_dom",
                turnos=[turno],
                esperado=Esperado(doc_ids_alvo=[doc.doc_id], buscar=True),
                evidencia=evidencia,
            )
        )
    return casos, descartes


def doc_com_numero_unico(ix: _Indices, doc: DocumentoAnotado) -> bool:
    """True se o número do documento é específico E único no corpus (T009b).

    A unicidade é conferida por código contra a anotação: a chave canônica
    (ignora zeros à esquerda e sufixos ``/GAB/SEMEC``) precisa apontar para um
    único documento. Número solto sem ano é rejeitado por ser genérico demais.
    """
    if not doc.numero:
        return False
    chave = canon_numero(doc.tipo, doc.numero)
    if not numero_especifico(chave):
        return False
    return len(ix.docs_por_numero.get(chave, set())) == 1


def _variante_numero(doc: DocumentoAnotado, indice: int) -> str:
    """Variações de escrita do número (T009b): nº, zeros à esquerda, sufixo."""
    numero = (doc.numero or "").strip()
    tipo = _tipo_legivel(doc.tipo)
    sem_zero = re.sub(r"^0+(\d)", r"\1", numero)
    compacto = numero_compacto(numero)
    opcoes = [
        f"{tipo.capitalize()} {numero}",
        f"{tipo} nº {numero}",
        f"{tipo} número {sem_zero}",
        f"tem {tipo} {numero}?",
        f"me acha {tipo} {numero}",
        f"{tipo.capitalize()} {compacto}",
    ]
    # Decreto/lei: o número completo (com milhar) já é específico.
    if tipo.startswith(("decreto", "lei")):
        opcoes.append(numero)
    return opcoes[indice % len(opcoes)]


#: Processo/contrato citado no corpo: `00045.070264/2025-16`, `00070.002054/2023-61`.
_RE_PROCESSO = re.compile(r"\d{2,6}[./]\d{2,6}/\d{4}(?:-\d{2})?")


def _processos(doc: DocumentoAnotado) -> list[str]:
    """Números de processo plausíveis em ``outros_numeros`` (só o número)."""
    achados = []
    for bruto in doc.outros_numeros:
        achado = _RE_PROCESSO.search(bruto)
        if achado:
            achados.append(achado.group(0))
    return achados


def _evidencia_do_numero(
    leitor: LeitorPDF, doc: DocumentoAnotado
) -> Evidencia | None:
    """Evidência que CONTÉM o número consultado (R009/M1).

    Prefere o enunciado quando ele traz o número; senão usa o próprio número (ou
    uma variante compacta/compactada) como ``trecho_literal`` — sempre
    verificado na página do PDF.
    """
    numero = (doc.numero or "").strip()
    if not numero:
        return None
    if normalizar_verificacao(numero) in normalizar_verificacao(doc.enunciado):
        evidencia = localizar_evidencia(leitor, doc, doc.enunciado)
        if evidencia is not None:
            return evidencia
    candidatos = [
        numero,
        re.sub(r"^0+(\d)", r"\1", numero),
        numero_compacto(numero),
    ]
    achado = re.search(r"\d+\s*/\s*\d{4}", numero)
    if achado:
        candidatos.append(achado.group(0))
    vistos: set[str] = set()
    for candidato in candidatos:
        candidato = candidato.strip()
        if not candidato or candidato in vistos:
            continue
        vistos.add(candidato)
        evidencia = localizar_evidencia(leitor, doc, candidato)
        if evidencia is not None:
            return evidencia
    return None


def _numero(ix: _Indices, leitor: LeitorPDF) -> tuple[list[CasoGolden], list[Descarte]]:
    casos: list[CasoGolden] = []
    descartes: list[Descarte] = []
    # Processos só entram se forem únicos no corpus (verificação por código).
    contagem_processos: Counter[str] = Counter()
    for doc in ix.documentos:
        contagem_processos.update(set(_processos(doc)))
    for doc in ix.documentos:
        if len(casos) >= ALVO_POR_SPLIT * 2 * MARGEM_CANDIDATOS:
            break
        if doc.numero:
            if not doc_com_numero_unico(ix, doc):
                continue  # não é único / genérico: fica fora de `numero`
            evidencia = _evidencia_do_numero(leitor, doc)
            if evidencia is None:
                descartes.append(
                    Descarte(
                        "numero", "número não encontrado no PDF", doc.numero or ""
                    )
                )
                continue
            turno = _variante_numero(doc, len(casos))
            casos.append(
                CasoGolden(
                    id="",
                    categoria="numero",
                    turnos=[turno],
                    esperado=Esperado(doc_ids_alvo=[doc.doc_id], buscar=True),
                    evidencia=evidencia,
                )
            )
            continue
        # Sem número de ato: tenta um processo único citado no corpo.
        for processo in _processos(doc):
            if contagem_processos[processo] != 1:
                continue
            evidencia = localizar_evidencia(leitor, doc, processo)
            if evidencia is None:
                continue
            casos.append(
                CasoGolden(
                    id="",
                    categoria="numero",
                    turnos=[f"Qual o ato do processo {processo}?"],
                    esperado=Esperado(doc_ids_alvo=[doc.doc_id], buscar=True),
                    evidencia=evidencia,
                )
            )
            break
    return casos, descartes


def _variantes_cpf_cnpj(identificador: str) -> list[str]:
    """Formas de escrita aceitas para um CPF/CNPJ (com e sem pontuação)."""
    variantes = [identificador]
    if "*" not in identificador:
        digitos = re.sub(r"\D", "", identificador)
        if digitos and digitos != identificador:
            variantes.append(digitos)
    return variantes


def _turno_cpf_cnpj(escolhida: str, formato: str, indice: int) -> str:
    """Consulta com registro variado (T009b.5), sem inventar fatos."""
    templates = [
        f"Procure o processo do CPF/CNPJ {escolhida}.",
        f"Qual ato menciona o CPF/CNPJ {escolhida}?",
        f"Busque pelo número {escolhida}.",
        f"quem é o dono do {escolhida}?",
        f"e o {escolhida}?",
        escolhida,
    ]
    if formato == "mascarado":
        templates[0] = f"Ache o ato com o CPF {escolhida}."
    elif formato == "quebrado":
        templates[1] = f"Onde aparece esse número {escolhida}?"
    return templates[indice % len(templates)]


def _cpf_cnpj(ix: _Indices, leitor: LeitorPDF) -> tuple[list[CasoGolden], list[Descarte]]:
    """Mistura CPF mascarado, CNPJ/CPF completos e formatos quebrados reais."""
    casos: list[CasoGolden] = []
    descartes: list[Descarte] = []
    # Superfície representativa por chave (a 1ª encontrada), com formato e doc.
    info: dict[str, tuple[str, str, DocumentoAnotado]] = {}
    for doc in ix.documentos:
        for identificador in (*doc.cpfs, *doc.cnpjs):
            chave = chave_cpf_cnpj(identificador)
            if not chave or chave in info:
                continue
            info[chave] = (identificador, formato_cpf_cnpj(identificador), doc)
    buckets: dict[str, list[tuple[str, str, DocumentoAnotado]]] = defaultdict(list)
    for chave, dados in info.items():
        buckets[dados[1]].append((chave, dados[0], dados[2]))
    for lista in buckets.values():
        # Identificadores mais discriminativos primeiro (menos documentos-alvo).
        lista.sort(
            key=lambda item: (len(ix.docs_por_cpf_cnpj.get(item[0], set())), item[2].doc_id)
        )

    ordem = ("cnpj", "cpf", "mascarado", "quebrado")
    ponteiros = {formato: 0 for formato in ordem}
    limite = ALVO_POR_SPLIT * 2  # pool exato: garante os 4 formatos nos 2 splits
    indice = 0
    avancou = True
    usados_docs: set[str] = set()  # mantém alvos disjuntos entre candidatos
    while len(casos) < limite and avancou:
        avancou = False
        for formato in ordem:
            if len(casos) >= limite:
                break
            lista = buckets.get(formato, [])
            while ponteiros[formato] < len(lista):
                chave, identificador, doc = lista[ponteiros[formato]]
                ponteiros[formato] += 1
                alvo = set(ix.docs_por_cpf_cnpj.get(chave, set()))
                if alvo & usados_docs:
                    continue
                evidencia = localizar_evidencia(leitor, doc, identificador)
                if evidencia is None:
                    descartes.append(
                        Descarte(
                            "cpf_cnpj",
                            "CPF/CNPJ não encontrado no PDF",
                            identificador,
                        )
                    )
                    continue
                usados_docs |= alvo
                # Formato quebrado/mascarado é usado literal (como publicado).
                if formato in ("mascarado", "quebrado"):
                    variantes = [identificador]
                else:
                    variantes = _variantes_cpf_cnpj(identificador)
                escolhida = variantes[indice % len(variantes)]
                caso = CasoGolden(
                    id="",
                    categoria="cpf_cnpj",
                    turnos=[_turno_cpf_cnpj(escolhida, formato, indice)],
                    esperado=Esperado(
                        doc_ids_alvo=sorted(ix.docs_por_cpf_cnpj.get(chave, set())),
                        buscar=True,
                    ),
                    evidencia=evidencia,
                )
                casos.append(caso)
                indice += 1
                avancou = True
                break
    return casos, descartes


# --- contexto --------------------------------------------------------------- #

_TEMA_DISPOE = re.compile(r"^disp[õo]e sobre\s+(.+)", re.IGNORECASE | re.DOTALL)
_TEMA_ABRE = re.compile(r"^abre cr[ée]dito", re.IGNORECASE)
_TEMA_RESERVA = re.compile(r"^reserva\s+as\s+sextas", re.IGNORECASE)
_TEMA_DESIGNA_SERVIDORES = re.compile(r"^designa\s+servidores", re.IGNORECASE)
_TEMA_DESIGNA_FISCAL = re.compile(
    r"^(designa\s+(gestor|fiscal)|fiscal\s+titular)", re.IGNORECASE
)
_TEMA_CONTRATO_DE = re.compile(r"^contrato de\s+(.+)", re.IGNORECASE | re.DOTALL)
_TEMA_PRESENTE = re.compile(
    r"^o presente documento trata-se d[ao]\s+(.+)", re.IGNORECASE | re.DOTALL
)
_TEMA_RELATIVO = re.compile(r"^\(?relativ[oa]\s+[àa]\s+(.+)", re.IGNORECASE | re.DOTALL)
_TEMA_ATA = re.compile(r"^ata de registro de pre[çc]os", re.IGNORECASE)
_TEMA_CONCORRENCIA = re.compile(r"^concorr[êe]ncia p[úu]blica", re.IGNORECASE)
_TEMA_PREGAO = re.compile(r"^preg[ãa]o eletr[ôo]nico", re.IGNORECASE)
_RE_CORTE_TEMA = re.compile(
    r"\s+(ao senhor|à senhora|ao sr\.?|à sra\.?|que entre|entre si|celebram|"
    r"relativamente)\b",
    re.IGNORECASE,
)
#: Sufixos que denunciam nome de empresa — não servem como "assunto".
_SUFIXOS_EMPRESA = re.compile(r"\b(ltda|eireli|s/?a|me|epp|cnpj)\b", re.IGNORECASE)
#: Conectores que não podem encerrar um tema.
_CONECTORES_FINAIS = {
    "no", "na", "de", "da", "do", "dos", "das", "e", "nº", "n°", "n.", "ano", "que",
}


def _validar_tema(tema: str | None) -> str | None:
    """Rejeita temas com número, nome de empresa ou menos de 3 palavras."""
    if not tema:
        return None
    if re.search(r"\d", tema) or _SUFIXOS_EMPRESA.search(tema):
        return None
    if len(tema.split()) < 3:
        return None
    return tema


def _limpar_tema(texto: str) -> str | None:
    """Limpa um tema: corta em nome/número, remove conectores e limita palavras."""
    texto = _RE_CORTE_TEMA.split(" ".join(texto.split()))[0]
    texto = re.split(r"\d", texto)[0]
    palavras = [p for p in texto.replace(",", " ").split() if p]
    while palavras and palavras[-1].lower().strip(".,;:-") in _CONECTORES_FINAIS:
        palavras.pop()
    if palavras and palavras[0].lower() in {"a", "o", "os", "as"}:
        palavras = palavras[1:]
    if len(palavras) < 3:
        return None
    return " ".join(palavras[:12]).strip(" .,;:-")


def _tema_de(doc: DocumentoAnotado) -> str | None:
    """Tema/assunto em linguagem natural derivado da anotação (sem nome/número).

    Só aceita início de documento com padrão temático explícito; inícios que são
    boilerplate ou nome de empresa/contratante são rejeitados (evita "assunto"
    que na verdade é nome ou cabeçalho de tabela).
    """
    base = " ".join(doc.inicio_literal.split())
    if not base:
        return None
    achado = _TEMA_DISPOE.match(base)
    if achado:
        return _validar_tema(_limpar_tema(achado.group(1)))
    if _TEMA_ABRE.match(base):
        return "abertura de crédito suplementar no orçamento"
    if _TEMA_RESERVA.match(base):
        return "expediente interno nas sextas-feiras"
    if _TEMA_DESIGNA_SERVIDORES.match(base):
        return "designação de servidores para gestão de parceria"
    if _TEMA_DESIGNA_FISCAL.match(base):
        return "designação de fiscal ou gestor de contratos"
    if _TEMA_ATA.match(base):
        return "ata de registro de preços"
    if _TEMA_CONCORRENCIA.match(base):
        return "concorrência pública"
    if _TEMA_PREGAO.match(base):
        return "pregão eletrônico"
    for padrao in (_TEMA_CONTRATO_DE, _TEMA_PRESENTE, _TEMA_RELATIVO):
        achado = padrao.match(base)
        if achado:
            return _validar_tema(_limpar_tema(achado.group(1)))
    return None


#: Marcador (substring do tema canônico, sem acento) → paráfrases de cidadão.
#: As paráfrases são verificadas por `parafrase_valida` contra o texto do
#: documento (zero 3-gramas em comum; no máximo 2 palavras de conteúdo iguais).
_PARAFRASES_CONTEXTO: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "calendario anual de pagamento",
        (
            "como ficam as datas de pagamento dos servidores da câmara?",
            "divulgaram o cronograma de salários do legislativo municipal?",
        ),
    ),
    (
        "feriados e declara",
        (
            "a câmara deu folga em algum feriado?",
            "o legislativo definiu descanso em datas comemorativas?",
        ),
    ),
    (
        "ponto facultativo",
        (
            "o serviço público vai emendar feriado?",
            "decretaram descanso prolongado para os servidores?",
            "vai ter folga na repartição municipal?",
        ),
    ),
    (
        "expediente interno nas sextas",
        (
            "o atendimento ao público muda às sextas-feiras?",
            "as repartições vão fechar na sexta?",
        ),
    ),
    (
        "fiscal substituto",
        (
            "quem são os substitutos que vão acompanhar os contratos?",
            "indicaram fiscais temporários para os acordos?",
        ),
    ),
    (
        "gestao de parceria",
        (
            "quem foi escalado para cuidar das parcerias firmadas?",
            "nomearam equipe para administrar os convênios?",
        ),
    ),
    (
        "acompanhamento e fiscalizacao",
        (
            "quem vai supervisionar a execução desse serviço?",
            "designaram alguém para vigiar o andamento do contrato?",
        ),
    ),
    (
        "designacao de fiscal e gestor",
        (
            "quem ficou responsável por acompanhar os contratos da prefeitura?",
            "quem toma conta da execução dos acordos firmados?",
        ),
    ),
    (
        "designacao de fiscal ou gestor",
        (
            "quem acompanha os contratos da fundação de saúde?",
            "quem foi designado para gerir os acordos da fundação?",
        ),
    ),
    (
        "substituicao de conselheiros",
        (
            "trocaram quem representa a população no conselho de saúde?",
            "mudaram os representantes do conselho de saúde?",
        ),
    ),
    (
        "concessao de titulo honorifico",
        (
            "quem recebeu homenagem de cidadão teresinense?",
            "deram título de cidadania a alguém?",
        ),
    ),
    (
        "autorizacao para implantacao",
        (
            "liberaram a abertura de um novo serviço na rede municipal?",
            "algum atendimento novo foi autorizado na cidade?",
        ),
    ),
    (
        "isencao das taxas",
        (
            "os blocos de rua vão ter desconto de imposto?",
            "liberaram perdão de tributo para o carnaval?",
        ),
    ),
    (
        "lancamento do imposto",
        (
            "saiu o carnê do imposto do imóvel deste ano?",
            "já cobraram o tributo sobre a casa?",
        ),
    ),
    (
        "aquisicao de equipamentos",
        (
            "compraram computadores novos?",
            "houve compra de aparelhos de tecnologia?",
        ),
    ),
    (
        "locacao de mao de obra",
        (
            "contrataram trabalhadores temporários?",
            "houve contratação de mão de obra terceirizada?",
        ),
    ),
    (
        "dispensa de chamamento",
        (
            "abriram dispensa de licitação para firmar um termo?",
            "contrataram sem licitação por dispensa?",
        ),
    ),
    (
        "ata de registro de precos",
        (
            "alguma compra foi registrada com preço fixo?",
            "publicaram registro para compras por preço fixo?",
        ),
    ),
    (
        "concorrencia publica",
        (
            "abriram disputa para contratar uma obra?",
            "vai ter licitação para escolher a empresa?",
        ),
    ),
    (
        "pregao eletronico",
        (
            "tem licitação online aberta?",
            "publicaram aviso de compra eletrônica?",
        ),
    ),
    (
        "abertura de credito suplementar",
        (
            "liberaram dinheiro extra no orçamento municipal?",
            "teve verba adicional aprovada pela prefeitura?",
        ),
    ),
)


def parafrases_do_tema(tema_normalizado: str) -> tuple[str, ...]:
    """Paráfrases cadastradas para um tema canônico (vazio se não houver)."""
    for marcador, parafrases in _PARAFRASES_CONTEXTO:
        if marcador in tema_normalizado:
            return parafrases
    return ()


def _temas_contexto(
    ix: _Indices, leitor: LeitorPDF
) -> tuple[list[dict], list[Descarte]]:
    """Âncoras de contexto: ``{docs, turno1, evidencia}`` por tema válido.

    Reusado por ``_contexto`` (consulta de assunto) e por ``_follow_up``
    (turno 1 de contexto para referência por posição).
    """
    if ix.cache_contexto is not None:
        return ix.cache_contexto
    ancoras: list[dict] = []
    descartes: list[Descarte] = []
    grupos: dict[str, list[DocumentoAnotado]] = defaultdict(list)
    for doc in ix.documentos:
        tema = _tema_de(doc)
        if tema:
            grupos[normalizar_verificacao(tema)].append(doc)

    ordenados = sorted(grupos.items(), key=lambda kv: (len(kv[1]), kv[0]))
    for chave_tema, membros in ordenados:
        parafrases = parafrases_do_tema(chave_tema)
        if not parafrases:
            descartes.append(
                Descarte(
                    "contexto", "sem paráfrase cadastrada para o tema", chave_tema[:70]
                )
            )
            continue
        trechos = [d.inicio_literal for d in membros] + [d.enunciado for d in membros]
        escolhida = None
        for deslocamento in range(len(parafrases)):
            candidata = parafrases[(len(ancoras) + deslocamento) % len(parafrases)]
            if parafrase_valida(candidata, trechos):
                escolhida = candidata
                break
        if escolhida is None:
            descartes.append(
                Descarte(
                    "contexto",
                    "paráfrase copia o documento (3-grama/palavras em comum)",
                    chave_tema[:70],
                )
            )
            continue
        evidencia = localizar_evidencia(leitor, membros[0], membros[0].inicio_literal)
        if evidencia is None:
            descartes.append(
                Descarte(
                    "contexto", "início literal não encontrado no PDF", chave_tema[:70]
                )
            )
            continue
        ancoras.append(
            {
                "docs": sorted(d.doc_id for d in membros),
                "turno1": escolhida,
                "evidencia": evidencia,
            }
        )
    ix.cache_contexto = (ancoras, descartes)
    return ancoras, descartes


def _contexto(ix: _Indices, leitor: LeitorPDF) -> tuple[list[CasoGolden], list[Descarte]]:
    """Consultas de contexto como PARÁFRASES (T009b.2), verificadas por 3-gramas."""
    ancoras, descartes = _temas_contexto(ix, leitor)
    casos = [
        CasoGolden(
            id="",
            categoria="contexto",
            turnos=[ancora["turno1"]],
            esperado=Esperado(doc_ids_alvo=list(ancora["docs"]), buscar=True),
            evidencia=ancora["evidencia"],
        )
        for ancora in ancoras[: ALVO_POR_SPLIT * 2 * MARGEM_CANDIDATOS]
    ]
    return casos, descartes


# --- nome_com_erro ---------------------------------------------------------- #


def _aplicar_erro(nome: str, indice: int) -> str:
    """Aplica 1–2 erros determinísticos a um nome real."""
    palavras = nome.split()
    if indice % 5 == 0:  # troca duas letras internas da primeira palavra
        p = palavras[0]
        if len(p) >= 4:
            p = p[0] + p[2] + p[1] + p[3:]
        return " ".join([p, *palavras[1:]])
    if indice % 5 == 1:  # omissão de letra
        p = palavras[0]
        if len(p) >= 4:
            p = p[:2] + p[3:]
        return " ".join([p, *palavras[1:]])
    if indice % 5 == 2:  # letra duplicada (erro real de digitação)
        p = palavras[0]
        if len(p) >= 3:
            p = p[:2] + p[1] + p[2:]
        return " ".join([p, *palavras[1:]])
    if indice % 5 == 3:  # abrevia o nome do meio
        if len(palavras) >= 3:
            return f"{palavras[0]} {palavras[1][0]}. {' '.join(palavras[2:])}"
        return nome
    # último token trocado
    if len(palavras) >= 2 and len(palavras[-1]) >= 4:
        p = palavras[-1]
        return " ".join([*palavras[:-1], p[:-1] + p[-2]])
    return nome


def _nome_com_erro(
    ix: _Indices, leitor: LeitorPDF
) -> tuple[list[CasoGolden], list[Descarte]]:
    casos: list[CasoGolden] = []
    descartes: list[Descarte] = []
    todos_nomes = set(ix.docs_por_nome)
    vistos: set[str] = set()
    tentativas = 0
    for doc in ix.documentos:
        if len(casos) >= ALVO_POR_SPLIT * 2 * MARGEM_CANDIDATOS:
            break
        for nome in doc.nomes:
            chave = chave_nome(nome)
            if chave in vistos or len(chave) < 10 or len(nome.split()) < 2:
                continue
            alvo = sorted(ix.docs_por_nome.get(chave, set()))
            if len(alvo) != 1:
                continue
            errado = _aplicar_erro(nome, tentativas)
            tentativas += 1
            if chave_nome(errado) in todos_nomes or chave_nome(errado) == chave:
                continue
            evidencia = localizar_evidencia(leitor, doc, nome)
            if evidencia is None:
                descartes.append(
                    Descarte("nome_com_erro", "nome real não encontrado no PDF", nome)
                )
                continue
            vistos.add(chave)
            templates = (
                f"Você acha o documento de {errado}?",
                f"Procure atos que citam {errado}.",
                f"Busque por {errado} (acho que escrevi o nome errado).",
                f"é {errado}? acho que errei o nome",
                f"{errado}",
                f"tem alguém com o nome {errado}?",
            )
            turno = templates[len(casos) % len(templates)]
            casos.append(
                CasoGolden(
                    id="",
                    categoria="nome_com_erro",
                    turnos=[turno],
                    esperado=Esperado(doc_ids_alvo=alvo, buscar=True),
                    evidencia=evidencia,
                )
            )
            break
    return casos, descartes


# --- ambiguo ---------------------------------------------------------------- #


def _ambiguo(ix: _Indices, leitor: LeitorPDF) -> tuple[list[CasoGolden], list[Descarte]]:
    """Consulta com mais de uma leitura real: número repetido em >1 documento.

    Inclui grupos com ano (``55/2026``) e sem ano (``termo aditivo nº 05``), em
    que o token é a forma escrita comum às opções. Unicidade de ``numero`` já
    foi verificada — quem não é único vem para cá.
    """
    casos: list[CasoGolden] = []
    descartes: list[Descarte] = []
    grupos: list[tuple[str, list[str], str, bool]] = []
    for chave, docs in ix.docs_por_numero.items():
        if len(docs) < 2:
            continue
        com_ano = bool(re.fullmatch(r"ato-\d+/\d{4}", chave))
        if not com_ano and not re.fullmatch(r"ato-\d+", chave):
            continue
        ordenados = sorted(docs)
        if com_ano:
            token = token_ambiguo(chave)
        else:
            superficies = [
                (ix.por_doc_id[d].numero or "").strip() for d in ordenados
            ]
            token = Counter(s for s in superficies if s).most_common(1)[0][0]
        if not token:
            continue
        grupos.append((chave, ordenados, token, com_ano))

    def ordenar(item: tuple[str, list[str], str, bool]) -> tuple:
        chave, docs, token, com_ano = item
        tipos = {ix.por_doc_id[d].tipo for d in docs}
        edicoes = {ix.por_doc_id[d].edicao_numero for d in docs}
        # Preferência: (1) sem ano com token por extenso; (2) com ano e poucas
        # opções; (3) demais. Garante representação das ambiguidades "soltas".
        if not com_ano and len(token) >= 2:
            faixa = 0
        elif com_ano and 2 <= len(docs) <= 4:
            faixa = 1
        else:
            faixa = 2
        return (faixa, len(docs), -len(tipos), -len(edicoes), chave)

    # Pool exato do split: garante que os casos raros (sem ano) sejam usados.
    limite = ALVO_POR_SPLIT * 2
    for chave, docs, token, _com_ano in sorted(grupos, key=ordenar):
        if len(casos) >= limite:
            break
        # O token comum precisa existir em TODOS os documentos das opções.
        localizados: list[tuple[str, int]] = []
        for doc_id in docs:
            doc = ix.por_doc_id[doc_id]
            pagina = leitor.localizar(doc.arquivo, doc.paginas, token)
            if pagina is None:
                break
            localizados.append((doc_id, pagina))
        if len(localizados) != len(docs):
            descartes.append(
                Descarte("ambiguo", "token não encontrado em todas as opções", token)
            )
            continue
        tipos = {ix.por_doc_id[d].tipo for d in docs}
        template = len(casos) % 3
        if template == 0:
            turno = f"Quero o {token}."
        elif template == 1 and len(tipos) == 1:
            turno = f"tem {_tipo_legivel(next(iter(tipos)))} {token}?"
        else:
            turno = f"Me mostra o documento número {token}."
        primeira_id, primeira_pagina = localizados[0]
        casos.append(
            CasoGolden(
                id="",
                categoria="ambiguo",
                turnos=[turno],
                esperado=Esperado(
                    doc_ids_alvo=[],
                    buscar=True,
                    perguntar=True,
                    opcoes_doc_ids=docs,
                ),
                evidencia=Evidencia(
                    arquivo=ix.por_doc_id[primeira_id].basename,
                    pagina=primeira_pagina,
                    trecho_literal=token,
                ),
            )
        )
    return casos, descartes


# --- follow_up -------------------------------------------------------------- #


_ORDINAIS = {1: "primeiro", 2: "segundo", 3: "terceiro"}


def _montar_follow_up(ancora: dict, indice: int) -> CasoGolden:
    """Monta o caso de follow-up (foco/conteúdo ou referência por posição)."""
    docs = list(ancora["docs"])
    turno1 = ancora["turno1"]
    evidencia = ancora["evidencia"]
    posicao = ancora["posicao"]
    if posicao is None:
        variacao = indice % 3
        if variacao == 0:
            turnos = [turno1, "Abra esse documento.", "Quem assina esse ato?"]
        elif variacao == 1:
            turnos = [turno1, "E qual a data de publicação?"]
        else:
            turnos = [turno1, "Sobre o que dispõe esse documento?"]
        return CasoGolden(
            id="",
            categoria="follow_up",
            turnos=turnos,
            esperado=Esperado(doc_ids_alvo=docs, buscar=True),
            evidencia=evidencia,
        )
    ordinal = _ORDINAIS[posicao]
    variacao = indice % 3
    if variacao == 0:
        turnos = [turno1, f"e o {ordinal}?"]
    elif variacao == 1:
        turnos = [turno1, f"abre o {ordinal}"]
    else:
        turnos = [turno1, f"me mostra o {ordinal}", "quem assina?"]
    return CasoGolden(
        id="",
        categoria="follow_up",
        turnos=turnos,
        esperado=Esperado(
            doc_ids_alvo=[],
            buscar=True,
            posicao_resultado=posicao,
            doc_ids_candidatos=docs,
        ),
        evidencia=evidencia,
    )


def _follow_up(ix: _Indices, leitor: LeitorPDF) -> tuple[list[CasoGolden], list[Descarte]]:
    """Conversas de 2–3 turnos: âncora por número, ID, nome ou contexto.

    Inclui referência por POSIÇÃO ("e o segundo?"), resolvida pela ordem
    canônica de exibição (ranking do turno anterior) — ver README.
    """
    casos: list[CasoGolden] = []
    descartes: list[Descarte] = []
    filas: dict[str, list[dict]] = {
        "numero": [],
        "id": [],
        "nome": [],
        "contexto": [],
    }
    for doc in ix.documentos:
        if not doc_com_numero_unico(ix, doc):
            continue
        evidencia = localizar_evidencia(leitor, doc, doc.enunciado)
        if evidencia is None:
            descartes.append(
                Descarte("follow_up", "enunciado não encontrado no PDF", doc.enunciado)
            )
            continue
        filas["numero"].append(
            {
                "docs": [doc.doc_id],
                "turno1": _variante_numero(doc, len(filas["numero"])),
                "evidencia": evidencia,
                "posicao": None,
            }
        )
    templates_id = (
        "Localize o documento de ID {0}.",
        "ID: {0}",
        "quero o id {0}",
        "id {0}",
        "{0}",
    )
    for doc in ix.documentos:
        if not doc.id_dom:
            continue
        evidencia = localizar_evidencia(leitor, doc, doc.id_dom)
        if evidencia is None:
            descartes.append(Descarte("follow_up", "ID não encontrado no PDF", doc.id_dom))
            continue
        turno1 = templates_id[len(filas["id"]) % len(templates_id)].format(doc.id_dom)
        filas["id"].append(
            {
                "docs": [doc.doc_id],
                "turno1": turno1,
                "evidencia": evidencia,
                "posicao": None,
            }
        )
    templates_nome = (
        "Em qual documento aparece {0}?",
        "quem é {0}?",
        "onde citam {0}?",
        "tem algo com {0}?",
    )
    for doc in ix.documentos:
        nome = next(
            (
                n
                for n in doc.nomes
                if len(ix.docs_por_nome.get(chave_nome(n), set())) == 1
                and len(n.split()) >= 2
            ),
            None,
        )
        if not nome:
            continue
        evidencia = localizar_evidencia(leitor, doc, nome)
        if evidencia is None:
            continue
        turno1 = templates_nome[len(filas["nome"]) % len(templates_nome)].format(nome)
        filas["nome"].append(
            {
                "docs": [doc.doc_id],
                "turno1": turno1,
                "evidencia": evidencia,
                "posicao": None,
            }
        )
    ancoras_contexto, descartes_contexto = _temas_contexto(ix, leitor)
    descartes.extend(descartes_contexto)
    for indice, ancora in enumerate(ancoras_contexto):
        if len(ancora["docs"]) < 2:
            continue
        posicao = (indice % 3) + 1
        if posicao > len(ancora["docs"]):
            posicao = len(ancora["docs"])
        filas["contexto"].append(
            {
                "docs": ancora["docs"],
                "turno1": ancora["turno1"],
                "evidencia": ancora["evidencia"],
                "posicao": posicao,
            }
        )

    ordem = ("numero", "id", "nome", "contexto")
    ponteiros = {tipo: 0 for tipo in ordem}
    usados: set[str] = set()
    limite = ALVO_POR_SPLIT * 2 + 4  # 24: sobra folga sem perder diversidade
    avancou = True
    while len(casos) < limite and avancou:
        avancou = False
        for tipo in ordem:
            if len(casos) >= limite:
                break
            fila = filas[tipo]
            while ponteiros[tipo] < len(fila):
                ancora = fila[ponteiros[tipo]]
                ponteiros[tipo] += 1
                docs = set(ancora["docs"])
                if docs & usados:
                    continue
                usados |= docs
                casos.append(_montar_follow_up(ancora, len(casos)))
                avancou = True
                break
    return casos, descartes


_BUILDERS = {
    "conversa_sem_busca": lambda ix, leitor: _conversa(),
    "nome_unico": _nome_unico,
    "nomes_multiplos": _nomes_multiplos,
    "id_dom": _id_dom,
    "numero": _numero,
    "cpf_cnpj": _cpf_cnpj,
    "contexto": _contexto,
    "nome_com_erro": _nome_com_erro,
    "ambiguo": _ambiguo,
    "follow_up": _follow_up,
}


def gerar_casos(
    documentos: Sequence[DocumentoAnotado], leitor: LeitorPDF | None = None
) -> tuple[dict[str, list[CasoGolden]], list[Descarte]]:
    """Constrói candidatos de todas as categorias e os verifica contra o PDF.

    Retorna ``(casos_por_categoria, descartes)``. Todo caso retornado (exceto
    ``conversa_sem_busca``) passou na verificação de evidência.
    """
    leitor = leitor or LeitorPDF()
    indice = montar_indices(documentos)
    casos_por_categoria: dict[str, list[CasoGolden]] = {}
    descartes: list[Descarte] = []
    for categoria in CATEGORIAS:
        construtor = _BUILDERS[categoria]
        candidatos, rejeitados = construtor(indice, leitor)
        descartes.extend(rejeitados)
        aprovados: list[CasoGolden] = []
        for caso in candidatos:
            if verificar_caso(leitor, caso):
                aprovados.append(caso)
            else:
                alvo = caso.evidencia.trecho_literal if caso.evidencia else caso.id
                descartes.append(
                    Descarte(categoria, "falhou na verificação contra o PDF", alvo[:80])
                )
        casos_por_categoria[categoria] = aprovados
    # Remove descartes repetidos (o cache de contexto é reusado por 2 builders).
    vistos: set[tuple[str, str, str]] = set()
    unicos: list[Descarte] = []
    for descarte in descartes:
        chave = (descarte.categoria, descarte.motivo, descarte.detalhe)
        if chave in vistos:
            continue
        vistos.add(chave)
        unicos.append(descarte)
    return casos_por_categoria, unicos


# --------------------------------------------------------------------------- #
# Divisão dev/holdout                                                         #
# --------------------------------------------------------------------------- #


def _docs_alvo(caso: CasoGolden) -> set[str]:
    return (
        set(caso.esperado.doc_ids_alvo)
        | set(caso.esperado.opcoes_doc_ids)
        | set(caso.esperado.doc_ids_candidatos)
    )


def dividir_por_categoria(
    casos_por_categoria: dict[str, list[CasoGolden]],
    *,
    alvo_por_split: int = ALVO_POR_SPLIT,
    seed: int = SEED,
) -> tuple[dict[str, list[CasoGolden]], dict[str, list[CasoGolden]], dict[str, int]]:
    """Divide cada categoria em dev/holdout sem repetir doc-alvo entre splits.

    Invariante (T009e): um documento-alvo não pode aparecer nos DOIS splits;
    dois casos do MESMO split podem compartilhar documento. A colocação só é
    aceita se os docs do caso não colidem com o split **oposto** — isso mantém
    ``usados["dev"] ∩ usados["hold"] = ∅`` durante todo o greedy, então
    ``conflitos`` é 0 por construção enquanto houver candidato colocável.

    Retorna ``(dev, holdout, conflitos)`` onde ``conflitos`` conta, por
    categoria, quantos documentos-alvo ficaram nos dois splits (só > 0 se o
    pool inteiro não permitiu preencher os dois splits sem conflito).
    """
    rng = random.Random(seed)
    dev: dict[str, list[CasoGolden]] = {}
    holdout: dict[str, list[CasoGolden]] = {}
    conflitos: dict[str, int] = {}
    for categoria in CATEGORIAS:
        casos = list(casos_por_categoria.get(categoria, []))
        rng.shuffle(casos)
        usados = {"dev": set(), "hold": set()}
        lista = {"dev": [], "hold": []}
        oposto = {"dev": "hold", "hold": "dev"}
        colocados: set[int] = set()

        def cabe(nome: str, docs: set[str]) -> bool:
            """Cabe no split se há vaga e não colide com o split oposto."""
            return len(lista[nome]) < alvo_por_split and not (docs & usados[oposto[nome]])

        def colocar(nome: str, caso: CasoGolden, docs: set[str]) -> None:
            lista[nome].append(caso)
            usados[nome] |= docs
            colocados.add(id(caso))

        def cheio() -> bool:
            return (
                len(lista["dev"]) >= alvo_por_split
                and len(lista["hold"]) >= alvo_por_split
            )

        # 1ª passada: só coloca sem criar conflito entre os splits.
        for caso in casos:
            if cheio():
                break
            docs = _docs_alvo(caso)
            opcoes = [nome for nome in ("dev", "hold") if cabe(nome, docs)]
            if not opcoes:
                continue
            escolhido = min(opcoes, key=lambda n: len(lista[n]))
            colocar(escolhido, caso, docs)

        # 2ª passada: completa com os demais candidatos do pool (sem conflito).
        if not cheio():
            for caso in casos:
                if cheio():
                    break
                if id(caso) in colocados:
                    continue
                docs = _docs_alvo(caso)
                opcoes = [nome for nome in ("dev", "hold") if cabe(nome, docs)]
                if not opcoes:
                    continue
                escolhido = min(opcoes, key=lambda n: len(lista[n]))
                colocar(escolhido, caso, docs)

        # 3ª passada (best-effort): só se o pool inteiro não bastou, aceita
        # conflito para não entregar menos que o alvo por split.
        if not cheio():
            for caso in casos:
                if cheio():
                    break
                if id(caso) in colocados:
                    continue
                docs = _docs_alvo(caso)
                nome = "dev" if len(lista["dev"]) <= len(lista["hold"]) else "hold"
                colocar(nome, caso, docs)

        dev[categoria] = lista["dev"][:alvo_por_split]
        holdout[categoria] = lista["hold"][:alvo_por_split]
        conflitos[categoria] = len(usados["dev"] & usados["hold"])
    return dev, holdout, conflitos


def _renumerar(casos: list[CasoGolden], split: str) -> None:
    for indice, caso in enumerate(casos, start=1):
        caso.id = f"{caso.categoria}-{split}-{indice:03d}"


# --------------------------------------------------------------------------- #
# Escrita e relatório                                                         #
# --------------------------------------------------------------------------- #


@dataclass
class RelatorioGeracao:
    """Resumo da geração: contagens, descartes e conflitos de split."""

    contagem: dict[str, dict[str, int]]
    descartes_por_categoria: dict[str, int]
    descartes_por_motivo: dict[str, int]
    exemplos_descarte: list[tuple[str, str, str]]
    conflitos: dict[str, int]
    total_documentos: int

    def resumo(self) -> dict:
        return {
            "total_documentos_anotacao": self.total_documentos,
            "contagem": self.contagem,
            "totais": {
                "dev": sum(v["dev"] for v in self.contagem.values()),
                "holdout": sum(v["holdout"] for v in self.contagem.values()),
            },
            "descartes_por_categoria": self.descartes_por_categoria,
            "descartes_por_motivo": self.descartes_por_motivo,
            "conflitos_doc_entre_splits": self.conflitos,
        }


def _escrever_jsonl(caminho: Path, casos: Iterable[CasoGolden]) -> int:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with caminho.open("w", encoding="utf-8") as arquivo:
        for caso in casos:
            arquivo.write(caso.model_dump_json() + "\n")
            total += 1
    return total


def gerar_golden(
    *,
    anotacao: Path | str = DIR_ANOTACAO,
    diarios: Path | str = DIR_DIARIOS,
    dev_path: Path | str = DEV_PATH,
    holdout_path: Path | str | None = HOLDOUT_PATH,
    seed: int = SEED,
    alvo_por_split: int = ALVO_POR_SPLIT,
) -> RelatorioGeracao:
    """Gera o golden set, grava dev (e holdout) e devolve o relatório."""
    documentos = carregar_anotacao(anotacao)
    leitor = LeitorPDF(diarios)
    casos_por_categoria, descartes = gerar_casos(documentos, leitor)
    dev, holdout, conflitos = dividir_por_categoria(
        casos_por_categoria, alvo_por_split=alvo_por_split, seed=seed
    )
    for categoria in CATEGORIAS:
        _renumerar(dev.get(categoria, []), "dev")
        _renumerar(holdout.get(categoria, []), "holdout")
    _escrever_jsonl(
        Path(dev_path),
        [caso for categoria in CATEGORIAS for caso in dev.get(categoria, [])],
    )
    if holdout_path is not None:
        _escrever_jsonl(
            Path(holdout_path),
            [caso for categoria in CATEGORIAS for caso in holdout.get(categoria, [])],
        )

    contagem = {
        categoria: {
            "dev": len(dev.get(categoria, [])),
            "holdout": len(holdout.get(categoria, [])),
        }
        for categoria in CATEGORIAS
    }
    por_categoria = Counter(d.categoria for d in descartes)
    por_motivo = Counter(d.motivo for d in descartes)
    exemplos = [(d.categoria, d.motivo, d.detalhe) for d in descartes[:40]]
    return RelatorioGeracao(
        contagem=contagem,
        descartes_por_categoria=dict(por_categoria),
        descartes_por_motivo=dict(por_motivo),
        exemplos_descarte=exemplos,
        conflitos=conflitos,
        total_documentos=len(documentos),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gera o golden set de avaliação (T009).")
    parser.add_argument("--anotacao", default=str(DIR_ANOTACAO))
    parser.add_argument("--diarios", default=str(DIR_DIARIOS))
    parser.add_argument("--dev", default=str(DEV_PATH))
    parser.add_argument("--holdout", default=str(HOLDOUT_PATH))
    parser.add_argument(
        "--sem-holdout",
        action="store_true",
        help="Não grava o holdout (útil para testes/sandbox).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    relatorio = gerar_golden(
        anotacao=args.anotacao,
        diarios=args.diarios,
        dev_path=args.dev,
        holdout_path=None if args.sem_holdout else args.holdout,
        seed=args.seed,
    )
    print(json.dumps(relatorio.resumo(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

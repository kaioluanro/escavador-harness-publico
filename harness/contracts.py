"""Contratos de dados do harness.

Estes modelos são a fronteira entre os módulos: a ingestão produz
:class:`Documento`/``LinhaLogica``, a busca devolve :class:`ResultadoBusca` e o
agente consome/produz :class:`TraceBusca`. Mantenha-os pequenos e estáveis —
mudanças aqui afetam todos os times.

Regras refletidas nos contratos (ver ``knowledge/regras_diario.md``):
- O ID do diário (``id_dom``) é **metadado opcional**, não fronteira: pode
  faltar e nunca delimita documento. ``doc_id`` é gerado na ingestão.
- **Tabela pertence ao corpo do documento**: ela vai em ``Documento.tabelas`` e
  também aparece em ``Documento.texto``; nunca inicia um documento.

Os contratos de **CHAT** e **TOOLS** (T006) também ficam aqui, e não em
``harness/agent/contracts.py``: as tools (``harness/tools``), o agente
(``harness/agent``) e a avaliação (``harness/eval``) precisam exatamente dos
mesmos schemas (``RespostaChat.doc_ids_recuperados`` é o que permite medir hit@5).
Um módulo raiz único evita inversão de camadas (``tools`` não importa de
``agent``) e imports circulares; se o arquivo crescer demais, ele pode virar um
pacote mantendo estes nomes reexportados, sem quebrar ninguém.

**Invariante de schema plano (D008/D009; D045/D046):** ``IntencaoMensagem`` e as
ENTRADAS de tool são consumidas como JSON plano. A intenção é preenchida por
**laya** (decisões: ``tipo``, ``modo_busca``, ``suficiente``, ``ambigua`` — D045)
somada aos **extratores determinísticos** de ``harness/ingest/`` (trechos: nomes,
ID, números, CPF/CNPJ), dentro do pipeline Python ``entender → buscar →
responder`` (D046: CrewAI sai; a intenção não é mais gerada por um LLM de 8B).
Logo seus JSON schemas **não podem conter ``$defs``/``$ref``**. Para isso usam só
campos simples (``str``, ``int``, ``bool``, ``list[str]``, ``date``) com
descrição em cada campo e sem modelos aninhados. O teste
``test_schemas_de_entrada_sem_refs`` garante a invariante. As SAÍDAS
(``SaidaBusca``, ``SaidaPerguntarAoUsuario``, ``RespostaChat``) podem ter listas
de modelos, pois são montadas pelo harness, não pelo modelo. Em particular,
``EntradaPerguntarAoUsuario`` só carrega ``pergunta`` e ``criterio``; a própria
tool monta as opções reais com doc_ids (D009).
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

#: Tipos de entidade reconhecidos na extração estruturada.
TipoEntidade = Literal[
    "nome",
    "cpf",
    "cnpj",
    "numero_ato",
    "processo",
    "contrato",
    "id_dom",
]

#: Classificação de uma mensagem recebida pelo chat.
TipoIntencao = Literal["conversa", "busca", "esclarecimento", "follow_up"]

#: Modo de busca decidido pelo **laya** (D045). O roteador de busca
#: (``harness/search``) escolhe o CANAL de recuperação por este valor:
#: ``um_nome``/``varios_nomes`` → entidades fuzzy (com agregação por documento);
#: ``id_dom``/``numero_ato``/``cpf_cnpj`` → casamento exato; ``contexto_tema`` →
#: BM25 + vetorial com RRF (canal de fallback, quando o modo não foi decidido).
ModoBusca = Literal[
    "um_nome",
    "varios_nomes",
    "id_dom",
    "numero_ato",
    "cpf_cnpj",
    "contexto_tema",
]

#: Papéis possíveis no histórico da conversa.
PapelMensagem = Literal["usuario", "assistente", "sistema"]


class _Contrato(BaseModel):
    """Base dos contratos: campos extras proibidos para manter o schema estável."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Normalização de campos (R001)                                               #
# --------------------------------------------------------------------------- #

#: Formatos aceitos para datas que entram no harness, na ordem de tentativa.
_FORMATOS_DATA = ("%Y-%m-%d", "%d/%m/%Y", "%d%m%Y")


def normalizar_data(valor: object) -> object:
    """Normaliza data para :class:`datetime.date` (R001).

    Aceita, além de ``date``/``datetime``, as formas ``ISO`` (``aaaa-mm-dd``),
    ``dd/mm/aaaa`` e ``ddmmaaaa`` (usadas nos nomes dos PDFs, ex.:
    ``DOM4196-11022026``). Qualquer outra forma levanta ``ValueError``.
    """
    if isinstance(valor, datetime):
        return valor.date()
    if isinstance(valor, date):
        return valor
    if isinstance(valor, str):
        texto = valor.strip()
        for formato in _FORMATOS_DATA:
            try:
                return datetime.strptime(texto, formato).date()
            except ValueError:
                continue
        raise ValueError(
            f"data inválida: {valor!r}; use ISO (aaaa-mm-dd), dd/mm/aaaa ou ddmmaaaa"
        )
    return valor


#: Data que aceita ISO, dd/mm/aaaa ou ddmmaaaa (usar em ``edicao_data`` e períodos).
DataNormalizada = Annotated[date, BeforeValidator(normalizar_data)]

_ID_DOM = re.compile(r"^\d{18}$")


def normalizar_id_dom(valor: object) -> object:
    """Valida o ID do diário como exatamente 18 dígitos (ou ``None``) — R001.

    O ID continua **opcional** (metadado, não fronteira); quando presente, porém,
    precisa ter o formato de 18 dígitos: ``id_secretaria(5) + id_diario(5) +
    id_documento(4) + id_ano(4)``.
    """
    if valor is None:
        return None
    if isinstance(valor, str) and _ID_DOM.match(valor.strip()):
        return valor.strip()
    raise ValueError("id_dom, quando presente, deve conter exatamente 18 dígitos")


#: ID do diário opcional, mas de 18 dígitos quando presente.
IdDom = Annotated[str, BeforeValidator(normalizar_id_dom)]


class Span(_Contrato):
    """Fragmento de texto contíguo com o mesmo estilo — unidade crua do PyMuPDF."""

    texto: str = Field(description="Texto do span, já limpo de caracteres de controle (\\x00).")
    bbox: tuple[float, float, float, float] = Field(
        description="Caixa (x0, y0, x1, y1) em pontos PDF, origem no canto superior esquerdo."
    )
    fonte: str = Field(description="Nome da fonte (ex.: 'Times-Bold').")
    tamanho: float = Field(description="Tamanho da fonte em pontos.")
    negrito: bool = Field(
        description="True se o span está em negrito (PyMuPDF: span['flags'] & 16)."
    )


class LinhaLogica(_Contrato):
    """Linha lógica: spans fundidos na mesma altura ``y`` com vão horizontal ≤ 8pt."""

    pagina: int = Field(ge=1, description="Página (1-based) onde a linha aparece.")
    y: float = Field(description="Altura da linha em pontos PDF; chave da fusão de spans.")
    x0: float = Field(description="Coordenada x inicial da linha (pontos PDF).")
    x1: float = Field(description="Coordenada x final da linha (pontos PDF).")
    texto: str = Field(description="Texto da linha após fusão dos spans e limpeza.")
    negrito_inicio: bool = Field(
        description="True se o início da linha é negrito; pista para achar enunciados que abrem documento."
    )
    spans: list[Span] = Field(
        default_factory=list,
        description="Spans originais que compõem a linha, em ordem de leitura.",
    )


class Entidade(_Contrato):
    """Entidade extraída do documento (nome, CPF/CNPJ, números, ID)."""

    tipo: TipoEntidade = Field(
        description="Categoria: nome, cpf, cnpj, numero_ato, processo, contrato ou id_dom."
    )
    valor: str = Field(description="Forma de superfície como aparece no texto.")
    valor_normalizado: str = Field(
        description="Forma canônica para casamento (CPF/CNPJ só dígitos; nomes sem acento/caixa etc.)."
    )
    pagina: int = Field(ge=1, description="Página onde a entidade ocorre.")


class Documento(_Contrato):
    """Documento segmentado de uma edição do diário."""

    doc_id: str = Field(
        description="Identificador estável e único do documento, gerado na ingestão (não derivado do id_dom)."
    )
    arquivo: str = Field(description="Caminho/nome do PDF de origem em diarios/.")
    edicao_numero: str = Field(description="Número/label da edição do diário (ex.: '4196').")
    edicao_data: DataNormalizada = Field(
        description="Data de publicação da edição; aceita ISO, dd/mm/aaaa ou ddmmaaaa."
    )
    pagina_inicio: int = Field(ge=1, description="Primeira página do documento (1-based).")
    pagina_fim: int = Field(ge=1, description="Última página do documento (1-based).")
    enunciado: str = Field(
        description="Enunciado em negrito que ABRE o documento (ex.: 'PORTARIA Nº 159/2021')."
    )
    tipo: str = Field(
        description="Tipo do ato, um dos enunciados canônicos (PORTARIA, EXTRATO, EDITAL, ...)."
    )
    numero: str | None = Field(
        default=None, description="Número do ato como publicado (ex.: '159/2021'); None se não houver."
    )
    ano: int | None = Field(default=None, description="Ano do ato; None se não identificado.")
    id_dom: IdDom | None = Field(
        default=None,
        description="ID de 18 dígitos do diário. METADADO OPCIONAL, não fronteira: pode faltar e não delimita documento.",
    )
    orgao: str | None = Field(
        default=None,
        description="Órgão/secretaria herdado do cabeçalho de seção em negrito; None se não identificado.",
    )
    texto: str = Field(
        description="Corpo integral do documento (linhas lógicas concatenadas); inclui as tabelas, que pertencem ao corpo."
    )
    tabelas: list[str] = Field(
        default_factory=list,
        description="Tabelas extraídas via page.find_tables(); pertencem ao corpo e NUNCA iniciam um documento.",
    )
    entidades: list[Entidade] = Field(
        default_factory=list, description="Entidades extraídas deste documento."
    )


class DocumentoResumo(_Contrato):
    """Projeção leve de :class:`Documento` para resultados (sem o texto integral)."""

    doc_id: str = Field(description="Identificador estável do documento.")
    arquivo: str = Field(description="PDF de origem.")
    edicao_numero: str = Field(description="Número/label da edição.")
    edicao_data: DataNormalizada = Field(
        description="Data de publicação da edição; aceita ISO, dd/mm/aaaa ou ddmmaaaa."
    )
    pagina_inicio: int = Field(ge=1, description="Primeira página do documento.")
    pagina_fim: int = Field(ge=1, description="Última página do documento.")
    enunciado: str = Field(description="Enunciado que abre o documento.")
    tipo: str = Field(description="Tipo canônico do ato.")
    numero: str | None = Field(default=None, description="Número do ato, se houver.")
    ano: int | None = Field(default=None, description="Ano do ato, se identificado.")
    id_dom: IdDom | None = Field(
        default=None, description="ID do diário (metadado opcional, 18 dígitos quando presente)."
    )
    orgao: str | None = Field(default=None, description="Órgão/secretaria, se identificado.")

    @classmethod
    def de_documento(cls, doc: Documento) -> "DocumentoResumo":
        """Extrai os metadados citáveis de um :class:`Documento`."""
        return cls(**{nome: getattr(doc, nome) for nome in cls.model_fields})


class ResultadoBusca(_Contrato):
    """Um resultado de busca com score, canais e trecho citável."""

    doc_id: str = Field(description="ID do documento recuperado.")
    score: float = Field(description="Score final de fusão/ranking (maior = mais relevante).")
    canais: dict[str, float] = Field(
        default_factory=dict,
        description="Score por canal de recuperação (ex.: {'bm25': ..., 'denso': ..., 'exato': ...}).",
    )
    trecho: str = Field(description="Trecho/snippet do documento que justifica o resultado.")
    documento: DocumentoResumo = Field(
        description="Metadados resumidos do documento (sem o texto integral)."
    )


class TraceBusca(_Contrato):
    """Registro do caminho de uma busca, para diagnóstico de falhas."""

    query_original: str = Field(description="Query como o usuário escreveu.")
    query_entendida: str = Field(
        description="Query normalizada/reescrita pelo entendimento (determinístico + laya, D045)."
    )
    filtros_aplicados: dict[str, Any] = Field(
        default_factory=dict,
        description="Filtros efetivamente usados (edição, ano, órgão, data, ...).",
    )
    filtros_relaxados: list[str] = Field(
        default_factory=list,
        description="Nomes dos filtros relaxados/descartados por não ancorarem resultado.",
    )
    resultados_por_canal: dict[str, list[str]] = Field(
        default_factory=dict,
        description="doc_ids retornados em cada canal antes da fusão.",
    )
    final: list[str] = Field(
        default_factory=list, description="doc_ids do ranking final, em ordem decrescente de score."
    )


# --------------------------------------------------------------------------- #
# Estado da conversa (chat multi-turno)                                       #
# --------------------------------------------------------------------------- #


class MensagemHistorico(_Contrato):
    """Um turno do histórico da conversa (entrada para o agente)."""

    papel: PapelMensagem = Field(
        description="Quem produziu o turno: 'usuario', 'assistente' ou 'sistema'."
    )
    texto: str = Field(description="Conteúdo textual do turno, como apareceu no chat.")
    doc_ids: list[str] = Field(
        default_factory=list,
        description="doc_ids citados/retornados neste turno; permitem retomar o contexto.",
    )


class EstadoConversa(_Contrato):
    """Estado multi-turno do chat: histórico, resultados exibidos e documento em foco.

    As listas preservam a ORDEM EXIBIDA, em base 1-based: ``ultimos_resultados[0]``
    é 'o primeiro', ``[1]`` é 'o segundo'; ``resultado_por_indice(2)`` devolve o
    segundo. É esse estado que resolve follow-ups como "o segundo" e "abre esse".
    """

    historico: list[MensagemHistorico] = Field(
        default_factory=list, description="Turnos anteriores, do mais antigo ao mais recente."
    )
    ultimos_resultados: list[ResultadoBusca] = Field(
        default_factory=list,
        description="Resultados mostrados ao usuário na última resposta, na ordem exibida (1-based).",
    )
    documento_em_foco: DocumentoResumo | None = Field(
        default=None,
        description="Documento 'aberto' no momento; alvo de 'abre esse'/'esse documento'.",
    )
    turno: int = Field(default=0, ge=0, description="Contador de turnos já processados.")

    def resultado_por_indice(self, indice: int) -> ResultadoBusca | None:
        """Resultado na posição 1-based da lista exibida ('o segundo' -> 2)."""
        if 1 <= indice <= len(self.ultimos_resultados):
            return self.ultimos_resultados[indice - 1]
        return None

    def doc_ids_exibidos(self) -> list[str]:
        """doc_ids dos últimos resultados, na ordem em que foram mostrados."""
        return [resultado.doc_id for resultado in self.ultimos_resultados]


class ConsultaEstruturada(_Contrato):
    """Query estruturada extraída da mensagem, em campos planos.

    Período é representado por ``periodo_inicio``/``periodo_fim`` (planos) para
    caber sem aninhamento no JSON plano (D008), consumido pelo laya na intenção e
    pelo modelo nas entradas de tool.
    """

    nomes: list[str] = Field(
        default_factory=list,
        description="Nomes de pessoas/empresas citados, como vieram (aceita 1 ou vários do mesmo documento).",
    )
    id_dom: IdDom | None = Field(
        default=None,
        description="ID de 18 dígitos do diário, quando o usuário citá-lo; opcional.",
    )
    cpf_cnpj: list[str] = Field(
        default_factory=list,
        description="CPF/CNPJ citados, preferencialmente só dígitos (ex.: '12345678900'); aceita múltiplos.",
    )
    tipo_ato: str | None = Field(
        default=None,
        description="Tipo canônico do ato, quando explícito (PORTARIA, EXTRATO, EDITAL, ...).",
    )
    numero: str | None = Field(
        default=None,
        description="Número do ato/processo/contrato, como citado (ex.: '159' ou '159/2021').",
    )
    ano: int | None = Field(default=None, ge=1900, le=2100, description="Ano do ato (ex.: 2021).")
    edicao: str | None = Field(
        default=None, description="Número/label da edição do diário (ex.: '4196')."
    )
    orgao: str | None = Field(default=None, description="Órgão/secretaria citado.")
    periodo_inicio: DataNormalizada | None = Field(
        default=None,
        description="Início do período pedido; aceita ISO, dd/mm/aaaa ou ddmmaaaa.",
    )
    periodo_fim: DataNormalizada | None = Field(
        default=None,
        description="Fim do período pedido; aceita ISO, dd/mm/aaaa ou ddmmaaaa.",
    )
    texto_livre: str | None = Field(
        default=None,
        description="Texto restante da mensagem para busca textual (sem os campos já estruturados).",
    )


class IntencaoMensagem(ConsultaEstruturada):
    """Classificação da mensagem do usuário + query estruturada extraída.

    Preenchida pelo pipeline Python ``entender → buscar → responder`` (D046,
    CrewAI sai): **laya** decide ``tipo``, ``modo_busca``, ``suficiente`` e
    ``ambigua`` (D045) e os **extratores determinísticos** de ``harness/ingest/``
    preenchem os trechos (``nomes``, ``id_dom``, ``cpf_cnpj``, ``numero``...);
    decisão e spans NÃO vêm mais de um LLM generativo de 8B (D008 revogado por
    D046). Herda os campos planos de
    :class:`ConsultaEstruturada` (``nomes``, ``id_dom``, ``cpf_cnpj``,
    ``tipo_ato``, ``numero``, ``ano``, ``edicao``, ``orgao``, ``periodo_inicio``,
    ``periodo_fim``, ``texto_livre``) e acrescenta a classificação, o modo de
    busca e as pistas de follow-up. É deliberadamente PLANA (D008): o JSON gerado
    não pode ter ``$defs``/``$ref``.

    Defaults dos campos de decisão (mantêm válidas as construções que não os
    informam — ex.: ``IntencaoMensagem(tipo="busca")``):

    - ``modo_busca="contexto_tema"`` — canal de fallback do roteador de busca,
      usado quando o modo não foi decidido (ex.: ``tipo="conversa"``);
    - ``suficiente=True`` — segue para a busca até que o decisor declare o
      contrário (consultado só quando ``tipo == "busca"``);
    - ``ambigua=False`` — a mensagem não pede esclarecimento.
    """

    tipo: TipoIntencao = Field(
        description="'conversa', 'busca', 'esclarecimento' ou 'follow_up'.",
    )
    modo_busca: ModoBusca = Field(
        default="contexto_tema",
        description=(
            "Modo de busca decidido pelo laya (D045): o roteador de busca escolhe o "
            "CANAL por ele (um_nome/varios_nomes → fuzzy; id_dom/numero_ato/cpf_cnpj → "
            "exato; contexto_tema → BM25+vetorial). Default 'contexto_tema' (fallback) "
            "quando o modo não foi decidido."
        ),
    )
    suficiente: bool = Field(
        default=True,
        description=(
            "D045: True quando há informação suficiente para buscar; consultado só "
            "quando tipo == 'busca' (False leva a esclarecer). Default True."
        ),
    )
    ambigua: bool = Field(
        default=False,
        description=(
            "D045: True quando a mensagem é ambígua e pede esclarecimento. Default False."
        ),
    )
    indice_resultado: int | None = Field(
        default=None,
        ge=1,
        description="Follow-up: posição 1-based na lista de últimos resultados ('o segundo' -> 2).",
    )
    doc_id_alvo: str | None = Field(
        default=None,
        description="Follow-up: doc_id apontado quando o usuário não usa posição (ex.: veio de uma citação).",
    )
    usar_foco: bool = Field(
        default=False,
        description="True quando o follow-up se refere ao documento em foco do estado da conversa.",
    )
    confianca: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confiança na classificação (0 a 1)."
    )


# --------------------------------------------------------------------------- #
# Tools do chat — entradas e saídas                                            #
# --------------------------------------------------------------------------- #


class EntradaBuscar(ConsultaEstruturada):
    """Entrada da tool ``buscar``: busca híbrida com a query estruturada.

    Herda os campos planos de :class:`ConsultaEstruturada` e acrescenta ``limite``.
    """

    limite: int = Field(
        default=10, ge=1, le=50, description="Nº máximo de resultados (a avaliação hit@5 usa 5)."
    )


class EntradaBuscarPorNumero(_Contrato):
    """Entrada da tool ``buscar_por_numero``: localiza atos pelo número."""

    numero: str = Field(
        description="Número do ato/processo/contrato, como citado (ex.: '159' ou '159/2021')."
    )
    tipo_ato: str | None = Field(
        default=None, description="Restringe ao tipo canônico, quando conhecido (ex.: 'PORTARIA')."
    )
    ano: int | None = Field(default=None, ge=1900, le=2100, description="Ano do ato, se conhecido.")
    edicao: str | None = Field(
        default=None, description="Edição do diário a restringir, se o usuário citou (ex.: '4196')."
    )
    limite: int = Field(default=10, ge=1, le=50, description="Nº máximo de resultados.")


class EntradaBuscarDocumento(_Contrato):
    """Entrada da tool ``buscar_documento``: localiza por CPF/CNPJ, ID do diário ou doc_id."""

    cpf_cnpj: str | None = Field(
        default=None,
        description="CPF ou CNPJ que o usuário quer localizar, com ou sem máscara (ex.: '000.000.000-00').",
    )
    id_dom: IdDom | None = Field(
        default=None, description="ID de 18 dígitos do diário, quando citado pelo usuário."
    )
    doc_id: str | None = Field(
        default=None, description="doc_id interno já conhecido (ex.: vindo de um resultado anterior)."
    )
    limite: int = Field(default=10, ge=1, le=50, description="Nº máximo de resultados.")


class EntradaAbrirDocumento(_Contrato):
    """Entrada da tool ``abrir_documento``: abre um documento para leitura e citação."""

    doc_id: str = Field(
        description="doc_id do documento a abrir (vem de um ResultadoBusca ou do estado da conversa)."
    )
    pagina: int | None = Field(
        default=None, ge=1, description="Página específica a abrir; None abre desde o início."
    )
    max_chars: int = Field(
        default=6000, ge=200, le=50000, description="Limite de caracteres do texto devolvido."
    )


class OpcaoClarificacao(_Contrato):
    """Uma opção REAL, vinda do índice, para o usuário escolher."""

    texto: str = Field(
        description="Rótulo legível da opção (ex.: 'PORTARIA 159/2021 — edição 4196, 11/02/2026, p. 12')."
    )
    doc_id: str | None = Field(
        default=None,
        description="doc_id da opção quando ela aponta para um documento do índice; None para opções não documentais.",
    )


class EntradaPerguntarAoUsuario(_Contrato):
    """Entrada PLANA da tool ``perguntar_ao_usuario`` (D009: preenchida pelo modelo).

    NÃO carrega ``doc_ids``/opções: o modelo (RLM do turno de responder, D045)
    só descreve a pergunta e o critério da desambiguação. A própria tool monta
    as :class:`OpcaoClarificacao` (com doc_ids REAIS) a partir de
    ``EstadoConversa.ultimos_resultados`` e dos candidatos do índice — assim o
    modelo nunca precisa copiar doc_ids (D009).
    """

    pergunta: str = Field(
        description="Pergunta objetiva ao usuário (ex.: 'Você quer a PORTARIA ou o EXTRATO?')."
    )
    criterio: str = Field(
        default="qual_documento",
        description="Critério da desambiguação: 'ato_ou_edicao', 'qual_documento' ou 'qual_pessoa'.",
    )


class PerguntaClarificacao(_Contrato):
    """Pergunta de desambiguação com opções REAIS — SAÍDA, nunca preenchida pelo LLM."""

    pergunta: str = Field(description="Pergunta apresentada ao usuário.")
    criterio: str = Field(description="Critério usado para montar as opções (ex.: 'ato_ou_edicao').")
    opcoes: list[OpcaoClarificacao] = Field(
        default_factory=list,
        description="Opções REAIS montadas pela tool a partir do índice (nunca pelo modelo).",
    )
    permitir_texto_livre: bool = Field(
        default=True, description="True se o usuário pode responder algo fora das opções."
    )


class SaidaPerguntarAoUsuario(PerguntaClarificacao):
    """Saída da tool ``perguntar_ao_usuario``: pergunta + opções REAIS montadas pela tool.

    Pode conter modelos aninhados porque é produzida pela tool, não pelo LLM.
    """


class RespostaUsuario(_Contrato):
    """Saída da tool ``perguntar_ao_usuario``: resposta dada pelo usuário."""

    resposta: str = Field(description="Opção escolhida ou texto livre digitado pelo usuário.")
    indice: int | None = Field(
        default=None, ge=1, description="Índice 1-based da opção escolhida; None se texto livre."
    )
    doc_id: str | None = Field(
        default=None, description="doc_id da opção escolhida, quando a opção apontava para um documento."
    )
    cancelado: bool = Field(default=False, description="True se o usuário não quis responder.")


class SaidaBusca(_Contrato):
    """Saída das tools de busca (``buscar``, ``buscar_por_numero``, ``buscar_documento``)."""

    resultados: list[ResultadoBusca] = Field(
        default_factory=list,
        description="Resultados ordenados por relevância (posição 1 = mais relevante).",
    )
    trace: TraceBusca | None = Field(
        default=None, description="Trace da busca, para diagnóstico de falhas."
    )
    mensagem: str | None = Field(
        default=None,
        description="Aviso para o agente (ex.: 'nenhum resultado', 'filtros relaxados').",
    )

    @property
    def vazia(self) -> bool:
        """True quando a busca não retornou nenhum resultado."""
        return not self.resultados

    def doc_ids(self) -> list[str]:
        """doc_ids dos resultados, na ordem do ranking (base para hit@5)."""
        return [resultado.doc_id for resultado in self.resultados]


class DocumentoAberto(_Contrato):
    """Saída da tool ``abrir_documento``: conteúdo do documento para leitura/citação."""

    documento: DocumentoResumo = Field(description="Metadados citáveis do documento.")
    texto: str = Field(description="Corpo do documento (possivelmente truncado em ``max_chars``).")
    trechos: list[str] = Field(
        default_factory=list, description="Trechos relevantes destacados dentro do texto."
    )
    truncado: bool = Field(default=False, description="True se o texto foi cortado em ``max_chars``.")


# --------------------------------------------------------------------------- #
# Resposta final do chat                                                       #
# --------------------------------------------------------------------------- #


class Citacao(_Contrato):
    """Citação de um documento na resposta do chat (edição, data, página, tipo+número, doc_id)."""

    doc_id: str = Field(description="Identificador do documento citado.")
    edicao_numero: str = Field(description="Edição do diário em que o documento foi publicado.")
    edicao_data: DataNormalizada = Field(
        description="Data de publicação da edição; aceita ISO, dd/mm/aaaa ou ddmmaaaa."
    )
    pagina: int = Field(ge=1, description="Página do documento em que o trecho está (1-based).")
    tipo: str = Field(description="Tipo canônico do ato (PORTARIA, EXTRATO, ...).")
    numero: str | None = Field(default=None, description="Número do ato, se houver.")
    enunciado: str | None = Field(default=None, description="Enunciado que abre o documento.")
    arquivo: str | None = Field(default=None, description="PDF de origem em diarios/.")
    trecho: str | None = Field(default=None, description="Trecho textual que justifica a citação.")

    @classmethod
    def de_resultado(
        cls,
        resultado: ResultadoBusca,
        *,
        pagina: int | None = None,
        trecho: str | None = None,
    ) -> "Citacao":
        """Monta a citação a partir de um :class:`ResultadoBusca`."""
        doc = resultado.documento
        return cls(
            doc_id=doc.doc_id,
            edicao_numero=doc.edicao_numero,
            edicao_data=doc.edicao_data,
            pagina=pagina if pagina is not None else doc.pagina_inicio,
            tipo=doc.tipo,
            numero=doc.numero,
            enunciado=doc.enunciado,
            arquivo=doc.arquivo,
            trecho=trecho if trecho is not None else resultado.trecho,
        )


class RespostaChat(_Contrato):
    """Resposta final do chat ao usuário, com citações e rastreabilidade para hit@5."""

    texto: str = Field(description="Resposta em linguagem natural ao usuário.")
    citacoes: list[Citacao] = Field(
        default_factory=list,
        description="Citações dos documentos usados (edição, data, página, tipo+número, doc_id).",
    )
    doc_ids_recuperados: list[str] = Field(
        default_factory=list,
        description=(
            "doc_ids entregues ao agente como contexto, na ordem do ranking recebido "
            "(top-k). Lista usada para medir hit@5; NÃO contém texto de trecho."
        ),
    )
    nao_encontrado: bool = Field(
        default=False,
        description="True quando a busca não achou nada; nesse caso o chat avisa, sem pedir trechos ao usuário.",
    )
    precisa_esclarecimento: bool = Field(
        default=False, description="True quando a resposta é uma pergunta de desambiguação."
    )
    pergunta: PerguntaClarificacao | None = Field(
        default=None, description="Pergunta com opções reais, quando ``precisa_esclarecimento``."
    )
    intencao: TipoIntencao | None = Field(
        default=None, description="Classificação usada para produzir esta resposta (auditoria)."
    )

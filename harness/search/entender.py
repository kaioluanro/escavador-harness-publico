"""Entendimento da mensagem do chat — 1ª fase do pipeline ``entender → buscar → responder``.

Responsabilidade (D045/D046): transformar a fala do usuário em
:class:`harness.contracts.IntencaoMensagem`. A intenção é montada em duas camadas,
nesta ordem de prioridade:

1. **Trechos DETERMINÍSTICOS** — reaproveitados (por import, nunca copiados) de
   ``harness/ingest``: CPF/CNPJ (:func:`harness.ingest.entidades.extrair_cpf_cnpj`),
   números de ato/processo/contrato (:func:`~harness.ingest.entidades.extrair_numeros_citados`),
   nomes (:func:`~harness.ingest.entidades.extrair_nomes` +
   :func:`~harness.ingest.entidades.separar_nomes_coordenados`), tipo canônico
   (:func:`~harness.ingest.metadados.tipo_canonico`) e ID do diário (validado por
   :func:`harness.contracts.normalizar_id_dom`). Os extratores do ``ingest`` são
   feitos para o CORPO dos PDFs (gatilhos de nomeação/assinatura); a fala do chat
   é curta e sem gatilho, por isso os nomes passam também por um candidato de
   ``runs`` capitalizados da própria mensagem — que termina sempre nos MESMOS
   filtros do ingest (:func:`~harness.ingest.ner_nomes.nome_parece_pessoa` e
   :func:`~harness.ingest.entidades.em_contexto_lugar`).
2. **Decisões** (``tipo``, ``modo_busca``, ``suficiente``, ``ambigua``,
   ``confianca``) — hoje pelo :class:`DecisorRegras` (determinístico) e, a partir
   do T031, pelo :class:`LayaDecisor` (``laya``, ``Router(default="multilingual")``,
   P002 A3). ``import laya`` é preguiçoso: sem o pacote o ``entender`` continua
   funcionando, sem traceback.

**Precedência do D045 vale em qualquer caso** (aplicada em
:func:`_aplicar_precedencia`, ou seja, também por cima de um eventual ``laya``):

- trecho **forte** (``id_dom``, ``cpf_cnpj``, tipo+número de ato, nome) impõe
  ``tipo="busca"``, ``suficiente=True`` e o ``modo_busca`` do próprio trecho;
- número "solto" (sem tipo, ou de série anual sem ano — ``Portaria 02``) impõe
  ``tipo="esclarecimento"`` e ``ambigua=True``;
- ``confianca < LIMIAR_CONFIANCA`` impõe ``tipo="esclarecimento"`` e ``ambigua=True``.

Regras gerais (sem caso especial por string): 1 nome → ``um_nome``; ≥2 nomes →
``varios_nomes``; sem trecho e ≥ ``LIMIAR_PALAVRAS`` palavras → ``contexto_tema``;
mensagem curta sem trecho → ``suficiente=False`` (interrogação de conceito/
saudação/meta → ``conversa``). Follow-up usa o ``historico``: ordinal ("o segundo"
→ 2 em ``indice_resultado``), anáfora ("abre esse documento" → ``usar_foco``),
continuação ("E qual a data…") ou elipse ("quem assina? — fala curta que só se
entende pelo turno anterior"); o ``modo_busca`` herdado é o da mensagem anterior,
é ele que diz qual canal recupera o alvo do follow-up.

NER de nomes é **extra opcional** (``usar_ner=True`` em :func:`entender`): a
chamada é preguiçosa e cai para as regras se ``torch``/``transformers`` ou os
pesos não estiverem disponíveis — este módulo importa num ambiente **sem** torch.

Uso::

    from harness.search.entender import entender
    intencao = entender("Portaria 159/2021")

Este módulo é SÓ o caminho de produção (sem CLI, sem constante vinda do golden).
A medição no golden dev — baseline do E001a que o T031 repete com o laya — está
em ``harness/eval/medir_entender.py``::

    uv run python -m harness.eval.medir_entender
"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from harness.contracts import (
    IntencaoMensagem,
    MensagemHistorico,
    ModoBusca,
    TipoIntencao,
    normalizar_id_dom,
)
from harness.ingest.entidades import (
    CONECTORES,
    chave_cpf_cnpj,
    chave_nome,
    em_contexto_lugar,
    extrair_cpf_cnpj,
    extrair_nomes,
    extrair_numeros_citados,
    separar_nomes_coordenados,
)
from harness.ingest.ner_nomes import nome_parece_pessoa

__all__ = [
    "LIMIAR_CONFIANCA",
    "LIMIAR_PALAVRAS",
    "Decisao",
    "Decisor",
    "DecisorRegras",
    "EntradaDecisao",
    "LayaDecisor",
    "Trechos",
    "decisor_ativo",
    "definir_decisor",
    "entender",
]

#: Confiança mínima para uma decisão valer como decisão (D045): abaixo dela a
#: intenção vira ``esclarecimento``/``ambigua`` em vez de uma busca frágil.
LIMIAR_CONFIANCA: float = 0.5

#: Palavras mínimas do texto livre para uma mensagem sem trecho virar busca
#: ``suficiente=True`` (briefing E001a: ``< 4 palavras ⇒ suficiente=False``).
#: Com histórico, a MESMA faixa (``0 < palavras < 4``) significa o contrário: a
#: fala é elíptica ("quem assina?") e continua o turno anterior (``follow_up``).
LIMIAR_PALAVRAS: int = 4

#: Número de atos cuja série é ÚNICA no tempo (lei/decreto/resolução não
#: reiniciam no ano): sem ano eles ainda identificam o alvo. Todo o resto
#: (portaria/extrato/aviso/…) reinicia a cada ano → ``… sem ano`` é ambíguo.
SERIE_UNICA: frozenset[str] = frozenset(
    {"LEI", "DECRETO", "RESOLUCAO", "REGULAMENTO"}
)

_MODOS: frozenset[str] = frozenset(
    {"um_nome", "varios_nomes", "id_dom", "numero_ato", "cpf_cnpj", "contexto_tema"}
)
_TIPOS: frozenset[str] = frozenset(
    {"conversa", "busca", "esclarecimento", "follow_up"}
)


# --------------------------------------------------------------------------- #
# Sinais determinísticos entregues ao decisor                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Trechos:
    """Campos planos extraídos da mensagem (sem decisão de busca)."""

    nomes: tuple[str, ...] = ()
    id_dom: str | None = None
    cpf_cnpj: tuple[str, ...] = ()
    tipo_ato: str | None = None
    numero: str | None = None
    ano: int | None = None
    orgao: str | None = None
    edicao: str | None = None
    texto_livre: str | None = None


@dataclass(frozen=True)
class EntradaDecisao:
    """Sinais determinísticos da mensagem — entrada única de qualquer :class:`Decisor`."""

    mensagem: str
    trechos: Trechos
    #: palavras do texto livre (sem trechos já consumidos)
    palavras: int
    #: termina em ``?``
    pergunta: bool
    #: canal imposto por trecho forte (D045); ``None`` se não há trecho forte
    modo_forte: ModoBusca | None
    #: número sem tipo ou de série anual sem ano → precisa esclarecer
    ambigua: bool
    #: histórico + ordinal/anáfora/continuação/elipse → mensagem de continuação
    follow_up: bool
    #: histórico + texto livre curto (``0 < palavras < LIMIAR_PALAVRAS``): fala
    #: elíptica sem trecho próprio, que só se entende pelo turno anterior
    elipse: bool
    #: "o segundo" → 2 (1-based)
    ordinal: int | None
    #: "esse documento"/"abre esse" → alvo é o documento em foco
    anafora: bool
    #: saudação/meta sobre o assistente/pergunta de conhecimento → conversa
    sinal_conversa: bool
    tem_historico: bool
    #: últimas falas, do mais antigo ao mais recente (contexto do laya)
    turnos: tuple[str, ...] = ()


@dataclass(frozen=True)
class Decisao:
    """Decisão do :class:`Decisor` sobre uma :class:`EntradaDecisao`."""

    tipo: TipoIntencao
    modo_busca: ModoBusca
    suficiente: bool
    ambigua: bool
    confianca: float


# --------------------------------------------------------------------------- #
# Extração determinística                                                     #
# --------------------------------------------------------------------------- #

#: ID do diário: 18 dígitos, tolerando espaços de coluna/quebra ("000 440 419…").
_RE_ID_DOM = re.compile(r"(?<!\d)\d(?:[ \t]?\d){17}(?!\d)")
#: CPF/CNPJ "pelado" (só dígitos): os padrões do ingest exigem pontuação.
_RE_CPF_PELADO = re.compile(r"(?<![\d.,/*-])(\d{11}|\d{14})(?!\.?\d)(?![\d/])")
#: Número solto do chat ("o 01", "extrato 02"): não é ID nem CPF nem fração.
_RE_NUMERO_SOLTO = re.compile(r"(?<![\d.,/*-])(\d{1,4})(?!\.?\d)(?![\d/])")
_RE_MENCIONA_IDENT = re.compile(r"\b(?:cpf|cnpj)\b", re.IGNORECASE)
_RE_ANO = re.compile(r"/(\d{4})(?!\d)")
_RE_SIGLA = re.compile(r"[A-ZÀ-Þ]{3,}")

#: Token de nome do chat (aceita inicial "d." e apóstrofos).
_RE_TOKEN_NOME = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’]*\.?")
#: Separador permitido ENTRE tokens de um run de nome (sem "/" nem dígitos:
#: "GAB/SEMEC" e "Maria 123" não são nomes contínuos).
_RE_SEP_RUN = re.compile(r"^[\s'’,;:()\"]*$")
#: Partículas que podem aparecer dentro de um nome (reuso do ingest).
_PARTICULAS: frozenset[str] = frozenset(p.casefold() for p in CONECTORES)

_ORDINAIS: dict[str, int] = {
    "primeiro": 1,
    "primeira": 1,
    "segundo": 2,
    "segunda": 2,
    "terceiro": 3,
    "terceira": 3,
    "quarto": 4,
    "quarta": 4,
    "quinto": 5,
    "quinta": 5,
}
_RE_ORDINAL = re.compile(
    r"\b(primeir[oa]|segund[oa]|terceir[oa]|quart[oa]|quint[oa])\b", re.IGNORECASE
)
_RE_ORDINAL_NUM = re.compile(r"\b(\d{1,2})\s*[º°]")

#: Anáfora de documento: o alvo é o que já está em foco / foi listado.
_RE_ANAFORA = re.compile(
    r"\b(?:esse|essa|este|esta)\s+(?:documento|atos?|textos?|resultados?|pdf|"
    r"publica[çc][ãa]os?|pe[çc]as?|itens?|trechos?|processo|norma)s?\b"
    r"|\b(?:abra|abre|abrir|mostre|mostra|mostrar|exiba|exibe|exibir|diga|diz|"
    r"resuma|resume|apresente|apresenta|mostrame)\s+(?:esse|essa|este|esta)\b"
    r"|\b(?:nele|nela|nisso|disso|nisto|disto)\b",
    re.IGNORECASE,
)
#: Continuação de turno: a fala começa por "E …".
_RE_CONTINUACAO = re.compile(r"^\s*e[,\s]", re.IGNORECASE)

#: Saudação/despedida: conversa, nunca busca.
_RE_SAUDACAO = re.compile(
    r"^\s*(?:oi|ol[áa]|opa|hey|hello|bom dia|boa tarde|boa noite|boa semana|"
    r"at[ée]|obrigad[oa]|valeu|tchau|favor)\b",
    re.IGNORECASE,
)
#: Mensagem voltada ao PRÓPRIO assistente (meta), não ao diário.
_RE_META_ASSISTENTE = re.compile(
    r"\b(?:quem\s+(?:é|e)\s+voc[êe]|seu\s+nome|teu\s+nome|"
    r"voc[êe]\s+(?:é|e)\s+(?:um\s+)?(?:rob[ôo]|bot|assistente|ia|intelig[êe]ncia)|"
    r"voc[êe]\s+(?:consegue|sabe|entende|guarda|processa|responde)|"
    r"voc[êe]\s+pode\s+(?:me\s+)?ajuda|"
    r"consegue\s+(?:ler|escrever|processar|interpretar)|funciona(?:ndo)?)\b",
    re.IGNORECASE,
)
#: Pergunta de CONHECIMENTO (definição ou listagem): a resposta não é um documento.
_RE_CONHECIMENTO = re.compile(
    r"^\s*(?:o\s+qu[eé]\s+(?:é|e|s[aã]o|significa|quer\s+dizer)|"
    r"como\s+(?:fa[çc]o|faco|posso|fazer)|quais|quantas|quantos)\b",
    re.IGNORECASE,
)


def _digito_verificador(base: Sequence[int], pesos: Sequence[int]) -> int:
    """Dígito verificador módulo-11 de CPF/CNPJ (regra oficial)."""
    total = sum(int(d) * peso for d, peso in zip(base, pesos))
    resto = total % 11
    return 0 if resto < 2 else 11 - resto


def _cpf_valido(digitos: str) -> bool:
    """True se os 11 dígitos passam na verificação de CPF."""
    if len(digitos) != 11 or len(set(digitos)) == 1:
        return False
    d = [int(c) for c in digitos]
    if _digito_verificador(d[:9], range(10, 1, -1)) != d[9]:
        return False
    return _digito_verificador(d[:10], range(11, 1, -1)) == d[10]


def _cnpj_valido(digitos: str) -> bool:
    """True se os 14 dígitos passam na verificação de CNPJ."""
    if len(digitos) != 14 or len(set(digitos)) == 1:
        return False
    d = [int(c) for c in digitos]
    pesos1 = (5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)
    pesos2 = (6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)
    if _digito_verificador(d[:12], pesos1) != d[12]:
        return False
    return _digito_verificador(d[:13], pesos2) == d[13]


def _extrair_id_dom(texto: str) -> tuple[str | None, tuple[int, int] | None]:
    """ID de 18 dígitos do diário (aceita espaços de coluna/quebra)."""
    for casamento in _RE_ID_DOM.finditer(texto):
        compacto = re.sub(r"\s+", "", casamento.group(0))
        try:
            normalizar_id_dom(compacto)
        except ValueError:
            continue
        return compacto, casamento.span()
    return None, None


def _extrair_cpf_cnpj(texto: str) -> tuple[list[str], list[str]]:
    """CPF/CNPJ com pontuação (ingest) + só-dígitos com dígito verificador."""
    superficies: list[str] = []
    chaves: list[str] = []
    for _tipo, valor in extrair_cpf_cnpj(texto):
        chave = chave_cpf_cnpj(valor)
        if chave:
            superficies.append(valor)
            chaves.append(chave)
    menciona = bool(_RE_MENCIONA_IDENT.search(texto))
    for casamento in _RE_CPF_PELADO.finditer(texto):
        digitos = casamento.group(1)
        valido = _cpf_valido(digitos) if len(digitos) == 11 else _cnpj_valido(digitos)
        if not valido and not menciona:
            continue
        if digitos in chaves:
            continue
        superficies.append(digitos)
        chaves.append(digitos)
    return superficies, chaves


def _extrair_numero(texto: str) -> tuple[str | None, tuple[int, int] | None]:
    """Número de ato/processo/contrato citado; na falta, um número solto do chat.

    O número solto só entra quando não há ID nem CPF/CNPJ na mensagem (senão o
    "000" de um ID quebrado por espaço viraria número de ato) e nunca captura ano
    isolado (``2026``) nem fragmento de data/percentual.
    """
    achados = extrair_numeros_citados(texto)
    if achados:
        _tipo, valor = achados[0]
        intervalo = _intervalo_de(texto, valor)
        return valor, intervalo
    id_dom, _ = _extrair_id_dom(texto)
    if id_dom:
        return None, None
    if _extrair_cpf_cnpj(texto)[1]:
        return None, None
    for casamento in _RE_NUMERO_SOLTO.finditer(texto):
        valor = casamento.group(1)
        if len(valor) == 4 and 1900 <= int(valor) <= 2100:
            continue
        if int(valor) == 0:
            continue
        return valor, casamento.span()
    return None, None


def _extrair_tipo_ato(texto: str) -> str | None:
    """Tipo canônico do ato quando o chat cita um tipo conhecido (D007/D020).

    ``tipo_canonico`` devolve a 1ª palavra em maiúsculas para enunciado
    irreconhecível; só aceitamos se a base está no léxico canônico do ingest.
    """
    try:
        from harness.ingest.classificar import TOKENS_CANONICOS
        from harness.ingest.metadados import tipo_canonico
    except Exception:  # noqa: BLE001 - ambiente sem PyMuPDF: degrada, não quebra
        return None
    bruto = tipo_canonico(texto)
    if not bruto:
        return None
    base = bruto.split()[0]
    return bruto if base in TOKENS_CANONICOS else None


def _extrair_orgao(texto: str) -> str | None:
    """Sigla de órgão do catálogo do ingest citada na mensagem (``…/SEMEC``)."""
    try:
        from harness.ingest.metadados import ORGAOS_POR_ID_SECRETARIA
    except Exception:  # noqa: BLE001 - degrada sem PyMuPDF
        return None
    siglas = set(ORGAOS_POR_ID_SECRETARIA.values())
    for token in _RE_SIGLA.findall(texto):
        if token in siglas:
            return token
    return None


def _intervalo_de(texto: str, superficie: str) -> tuple[int, int] | None:
    """Posição da superfície no texto (tolerando espaços normalizados)."""
    pos = texto.find(superficie)
    if pos >= 0:
        return pos, pos + len(superficie)
    padrao = r"\s*".join(re.escape(parte) for parte in superficie.split())
    casamento = re.search(padrao, texto)
    return casamento.span() if casamento else None


def _e_maiuscula(token: str) -> bool:
    base = token.rstrip(".")
    return bool(base) and base[0].isupper()


def _particula(token: str) -> bool:
    return token.rstrip(".").casefold() in _PARTICULAS


def _runs_capitalizados(texto: str) -> list[str]:
    """Runs de tokens capitalizados (com partículas ``de/da/do/e`` no meio).

    É o que permite achar "ARDIANA FERREIRA MENDES" ou "Maria de Oliveira" na
    fala curta do chat, onde os gatilhos do ingest não disparam. O separador
    entre tokens precisa ser só pontuação leve: ``GAB/SEMEC`` e ``Maria 123``
    quebram o run.
    """
    runs: list[str] = []
    atual: list[re.Match[str]] = []
    fim: int | None = None

    def _fechar() -> None:
        nonlocal atual
        if atual:
            runs.append(texto[atual[0].start() : atual[-1].end()])
            atual = []

    for token in _RE_TOKEN_NOME.finditer(texto):
        texto_token = token.group(0)
        separador = texto[fim : token.start()] if fim is not None else ""
        separavel = fim is None or bool(_RE_SEP_RUN.match(separador))
        if atual and separavel and (_e_maiuscula(texto_token) or _particula(texto_token)):
            atual.append(token)
        else:
            _fechar()
            if _e_maiuscula(texto_token):
                atual = [token]
        fim = token.end()
    _fechar()
    return runs


def _candidatos_de_nome(texto: str) -> list[str]:
    """Candidatos a nome na fala do chat, validados pelos filtros do ingest."""
    achados: list[str] = []
    for bruto in _runs_capitalizados(texto):
        if len(_RE_TOKEN_NOME.findall(bruto)) < 2:
            continue
        partes = separar_nomes_coordenados(bruto, texto=texto) or [bruto]
        for parte in partes:
            limpo = re.sub(r"\s+", " ", parte).strip(" .,;:-")
            if not limpo:
                continue
            if not nome_parece_pessoa(limpo):
                continue
            if em_contexto_lugar(texto, limpo):
                continue
            achados.append(limpo)
    return achados


_NER: Any = None


def _ner() -> Any:
    """Modelo NER carregado uma única vez (extra ``ner``; ``None`` se falhar)."""
    global _NER
    if _NER is None:
        from harness.ingest.ner_nomes import carregar_ner

        _NER = carregar_ner()
    return _NER


def _nomes(texto: str, *, com_ner: bool) -> list[str]:
    """Nomes da mensagem: regras do ingest ∪ candidatos do chat (NER é opt-in).

    Com ``com_ner=True`` usa :func:`harness.ingest.ner_nomes.extrair_nomes_hibrido`
    (que já embute as regras); qualquer falha de import/carga cai para as regras.
    """
    if com_ner:
        try:
            from harness.ingest.ner_nomes import extrair_nomes_hibrido

            hibrido = extrair_nomes_hibrido(texto, ner=_ner())
        except Exception:  # noqa: BLE001 - sem torch/pesos: regras puras
            hibrido = None
        if hibrido is not None:
            return _unicos(hibrido)
    return _unicos([*extrair_nomes(texto), *_candidatos_de_nome(texto)])


def _unicos(nomes: Iterable[str]) -> list[str]:
    vistos: set[str] = set()
    saida: list[str] = []
    for nome in nomes:
        chave = chave_nome(nome)
        if chave and chave not in vistos:
            vistos.add(chave)
            saida.append(nome)
    return saida


def _ano_de(numero: str | None) -> int | None:
    if not numero:
        return None
    casamento = _RE_ANO.search(numero)
    if not casamento:
        return None
    ano = int(casamento.group(1))
    return ano if 1900 <= ano <= 2100 else None


def _base_tipo(tipo: str) -> str:
    return tipo.split()[0] if tipo else ""


def _e_ambigua(trechos: Trechos) -> bool:
    """Número que não fecha sozinho: sem tipo ou de série anual sem ano."""
    if not trechos.numero:
        return False
    if not trechos.tipo_ato:
        return True
    if trechos.ano is not None:
        return False
    return _base_tipo(trechos.tipo_ato) not in SERIE_UNICA


def _modo_por_trechos(trechos: Trechos) -> ModoBusca | None:
    """Canal que aqueles trechos pedem (D045). ``None`` se não há trecho."""
    if trechos.id_dom:
        return "id_dom"
    if trechos.cpf_cnpj:
        return "cpf_cnpj"
    if trechos.numero:
        return "numero_ato"
    if len(trechos.nomes) >= 2:
        return "varios_nomes"
    if trechos.nomes:
        return "um_nome"
    return None


def _modo_forte(trechos: Trechos) -> ModoBusca | None:
    """Como :func:`_modo_por_trechos`, mas número ambíguo NÃO ancora a busca."""
    modo = _modo_por_trechos(trechos)
    if modo == "numero_ato" and _e_ambigua(trechos):
        return None
    return modo


def _apagar(texto: str, intervalos: Iterable[tuple[int, int]]) -> str:
    saida = texto
    for inicio, fim in sorted(intervalos, reverse=True):
        saida = saida[:inicio] + " " * (fim - inicio) + saida[fim:]
    return saida


def _texto_livre(texto: str, trechos: Trechos, intervalos: list[tuple[int, int]]) -> str | None:
    """Mensagem menos os campos já estruturados (alimenta BM25/vetorial)."""
    for superficie in [*trechos.cpf_cnpj, trechos.numero or "", *trechos.nomes]:
        if not superficie:
            continue
        intervalo = _intervalo_de(texto, superficie)
        if intervalo:
            intervalos.append(intervalo)
    restante = " ".join(_apagar(texto, intervalos).split())
    return restante or None


def _extrair_trechos(texto: str, *, com_ner: bool = False) -> Trechos:
    """Extrai todos os campos planos da mensagem (determinístico, sem decisão)."""
    id_dom, intervalo_id = _extrair_id_dom(texto)
    intervalos: list[tuple[int, int]] = [intervalo_id] if intervalo_id else []
    superficies_cpf, chaves_cpf = _extrair_cpf_cnpj(texto)
    numero, intervalo_numero = _extrair_numero(texto)
    nomes = _nomes(texto, com_ner=com_ner)
    trechos = Trechos(
        nomes=tuple(nomes),
        id_dom=id_dom,
        cpf_cnpj=tuple(chaves_cpf),
        tipo_ato=_extrair_tipo_ato(texto),
        numero=numero,
        ano=_ano_de(numero),
        orgao=_extrair_orgao(texto),
        edicao=str(int(id_dom[5:10])) if id_dom else None,
    )
    if intervalo_numero:
        intervalos.append(intervalo_numero)
    return Trechos(**{**trechos.__dict__, "texto_livre": _texto_livre(texto, trechos, intervalos)})


def _ordinal(texto: str) -> int | None:
    casamento = _RE_ORDINAL.search(texto)
    if casamento:
        return _ORDINAIS[casamento.group(1).lower()]
    casamento = _RE_ORDINAL_NUM.search(texto)
    if casamento:
        return int(casamento.group(1)) or None
    return None


def _e_conversa(texto: str) -> bool:
    return bool(
        _RE_SAUDACAO.search(texto)
        or _RE_META_ASSISTENTE.search(texto)
        or _RE_CONHECIMENTO.search(texto)
    )


def _avaliar(texto: str, trechos: Trechos, turnos: Sequence[MensagemHistorico]) -> EntradaDecisao:
    """Sinais prontos para o decisor (trechos + flags determinísticas)."""
    conteudo = trechos.texto_livre or texto
    historico = bool(turnos)
    palavras = len(re.findall(r"\w+", conteudo))
    ordinal = _ordinal(texto) if historico else None
    anafora = bool(_RE_ANAFORA.search(texto)) if historico else False
    continuacao = bool(_RE_CONTINUACAO.match(texto)) if historico else False
    # Elipse: fala curta que não fecha consulta sozinha e só vem depois de outro
    # turno — "quem assina?" é continuação; sem histórico, a mesma faixa é uma
    # consulta incompleta (``suficiente=False``).
    elipse = historico and 0 < palavras < LIMIAR_PALAVRAS
    return EntradaDecisao(
        mensagem=texto,
        trechos=trechos,
        palavras=palavras,
        pergunta="?" in texto,
        modo_forte=_modo_forte(trechos),
        ambigua=_e_ambigua(trechos),
        follow_up=historico and (ordinal is not None or anafora or continuacao or elipse),
        elipse=elipse,
        ordinal=ordinal,
        anafora=anafora,
        sinal_conversa=_e_conversa(texto),
        tem_historico=historico,
        turnos=tuple(t.texto for t in turnos),
    )


# --------------------------------------------------------------------------- #
# Decisores                                                                   #
# --------------------------------------------------------------------------- #


@runtime_checkable
class Decisor(Protocol):
    """Interface mínima do decisor de ``tipo``/``modo_busca``/``suficiente``/``ambigua``."""

    def decidir(self, entrada: EntradaDecisao) -> Decisao:
        """Decide a intenção de uma mensagem já extraída."""
        ...


class DecisorRegras:
    """Decisor determinístico — fallback do D045 enquanto o T031 não instala o laya.

    Regras gerais (nenhuma ligada a string do dev), nesta ordem:

    1. trecho forte → ``busca`` no canal do trecho;
    2. número que não fecha sozinho → ``esclarecimento``/``ambigua``;
    3. saudação/meta/pergunta de conhecimento → ``conversa`` (vale até com
       histórico: "oi" é conversa, não continuação);
    4. continuação de turno (ordinal/anáfora/"E …"/elipse) → ``follow_up``;
    5. texto livre ≥ ``LIMIAR_PALAVRAS`` → ``busca`` por ``contexto_tema``;
    6. mensagem curta sem trecho → ``busca`` ainda ``suficiente=False``
       (o pipeline esclarece antes de procurar).
    """

    def decidir(self, entrada: EntradaDecisao) -> Decisao:
        if entrada.modo_forte:
            return Decisao("busca", entrada.modo_forte, True, False, 0.95)
        modo: ModoBusca = "numero_ato" if entrada.trechos.numero else "contexto_tema"
        if entrada.ambigua:
            return Decisao("esclarecimento", modo, False, True, 0.5)
        if entrada.sinal_conversa:
            return Decisao("conversa", "contexto_tema", False, False, 0.85)
        if entrada.follow_up:
            return Decisao("follow_up", modo, True, False, 0.9)
        if entrada.palavras >= LIMIAR_PALAVRAS:
            return Decisao("busca", "contexto_tema", True, False, 0.7)
        return Decisao("busca", "contexto_tema", False, False, 0.55)


#: Perguntas do laya (P002 A3: `Router.predict(state, questions)`); rótulos
#: booleanos evitados (README "Honest limits") — ``sim``/``nao`` no lugar de
#: ``true``/``false``. O T031 confere o formato real contra a doc instalada.
_PERGUNTAS_LAYA: tuple[dict[str, Any], ...] = (
    {
        "pergunta": "Qual a intenção da mensagem?",
        "opcoes": ["conversa", "busca", "esclarecimento", "follow_up"],
    },
    {
        "pergunta": "Qual o modo de busca?",
        "opcoes": ["um_nome", "varios_nomes", "id_dom", "numero_ato", "cpf_cnpj", "contexto_tema"],
    },
    {"pergunta": "A informação é suficiente para buscar?", "opcoes": ["sim", "nao"]},
    {"pergunta": "A mensagem é ambígua?", "opcoes": ["sim", "nao"]},
)


def _estado_laya(entrada: EntradaDecisao) -> dict[str, Any]:
    """Estado enviado ao laya: a fala + o que os extratores já acharam."""
    trechos = entrada.trechos
    return {
        "mensagem": entrada.mensagem,
        "turnos": list(entrada.turnos),
        "nomes": list(trechos.nomes),
        "id_dom": trechos.id_dom,
        "cpf_cnpj": list(trechos.cpf_cnpj),
        "tipo_ato": trechos.tipo_ato,
        "numero": trechos.numero,
        "ano": trechos.ano,
        "texto_livre": trechos.texto_livre,
        "palavras": entrada.palavras,
        "seguiu_de_historico": entrada.tem_historico,
    }


def _bool(valor: Any, padrao: bool) -> bool:
    if isinstance(valor, bool):
        return valor
    if isinstance(valor, str):
        baixo = valor.strip().casefold()
        if baixo in {"sim", "s", "yes", "a", "true"}:
            return True
        if baixo in {"nao", "não", "n", "no", "b", "false"}:
            return False
    return padrao


def _decisao_de_laya(resposta: Any) -> Decisao | None:
    """Lê a resposta do laya; ``None`` quando não dá para confiar nela."""
    if isinstance(resposta, dict):
        pegar = resposta.get
    else:
        pegar = lambda chave: getattr(resposta, chave, None)  # noqa: E731
    tipo = pegar("tipo")
    modo = pegar("modo_busca")
    if tipo not in _TIPOS or modo not in _MODOS:
        return None
    confianca = pegar("confianca")
    if not isinstance(confianca, (int, float)):
        confianca = 1.0
    return Decisao(
        tipo=tipo,  # type: ignore[arg-type]
        modo_busca=modo,  # type: ignore[arg-type]
        suficiente=_bool(pegar("suficiente"), True),
        ambigua=_bool(pegar("ambigua"), False),
        confianca=min(max(float(confianca), 0.0), 1.0),
    )


class LayaDecisor:
    """Decisor via **laya** (D045) — ``import`` preguiçoso e fallback garantido.

    O pacote ``laya`` só existe depois do T031; enquanto isso (e sempre que a
    inferência falhar) a decisão vem do :class:`DecisorRegras`, sem traceback.
    """

    def __init__(self, fallback: Decisor | None = None) -> None:
        self._fallback: Decisor = fallback if fallback is not None else DecisorRegras()
        self._roteador: Any = None
        self._carregado = False

    def _router(self) -> Any:
        if not self._carregado:
            self._carregado = True
            try:
                from laya import Router  # noqa: PLC0415 - import preguiçoso (T031)

                self._roteador = Router(default="multilingual")
            except Exception:  # noqa: BLE001 - pacote ausente/quebrado → regras
                self._roteador = None
        return self._roteador

    def decidir(self, entrada: EntradaDecisao) -> Decisao:
        roteador = self._router()
        if roteador is None:
            return self._fallback.decidir(entrada)
        try:
            resposta = roteador.predict(_estado_laya(entrada), _PERGUNTAS_LAYA)
        except Exception:  # noqa: BLE001 - API diferente/erro de inferência
            return self._fallback.decidir(entrada)
        return _decisao_de_laya(resposta) or self._fallback.decidir(entrada)


_DECISOR: Decisor | None = None


def _laya_disponivel() -> bool:
    try:
        return importlib.util.find_spec("laya") is not None
    except Exception:  # noqa: BLE001
        return False


def decisor_ativo() -> Decisor:
    """Decisor usado por :func:`entender`: laya se instalado, senão regras."""
    global _DECISOR
    if _DECISOR is None:
        _DECISOR = LayaDecisor() if _laya_disponivel() else DecisorRegras()
    return _DECISOR


def definir_decisor(decisor: Decisor | None) -> None:
    """Troca o decisor (T031 recalibra o laya; ``None`` volta a escolher sozinho)."""
    global _DECISOR
    _DECISOR = decisor


# --------------------------------------------------------------------------- #
# Precedência D045 e função pública                                            #
# --------------------------------------------------------------------------- #


def _aplicar_precedencia(entrada: EntradaDecisao, decisao: Decisao) -> Decisao:
    """D045 por cima de qualquer decisor (inclusive o laya)."""
    if entrada.modo_forte:
        # Trecho forte: busca, com o canal do próprio trecho.
        return Decisao("busca", entrada.modo_forte, True, False, 0.95)
    if entrada.ambigua:
        modo: ModoBusca = "numero_ato" if entrada.trechos.numero else decisao.modo_busca
        return Decisao("esclarecimento", modo, False, True, 0.5)
    if decisao.confianca < LIMIAR_CONFIANCA:
        return Decisao("esclarecimento", decisao.modo_busca, False, True, decisao.confianca)
    return decisao


def _modo_do_historico(turnos: Sequence[MensagemHistorico]) -> ModoBusca | None:
    """Canal da pergunta anterior — é o que recupera o alvo do follow-up."""
    for turno in reversed(turnos):
        if turno.papel != "usuario":
            continue
        modo = _modo_por_trechos(_extrair_trechos((turno.texto or "").strip()))
        if modo:
            return modo
    return None


def entender(
    mensagem: str,
    historico: list[MensagemHistorico] | None = None,
    *,
    usar_ner: bool = False,
) -> IntencaoMensagem:
    """Classifica a mensagem do usuário e extrai a query estruturada.

    :param mensagem: fala do usuário (pode ser vazia → ``suficiente=False``).
    :param historico: turnos anteriores; habilita o follow-up (``indice_resultado``,
        ``usar_foco``) e a herança do ``modo_busca`` da pergunta anterior.
    :param usar_ner: extra opcional ``ner`` (NER de nomes); preguiçoso e cai para
        as regras se ``torch``/os pesos não estiverem disponíveis.
    """
    texto = (mensagem or "").strip()
    turnos = list(historico or [])
    trechos = _extrair_trechos(texto, com_ner=usar_ner)
    entrada = _avaliar(texto, trechos, turnos)
    decisao = _aplicar_precedencia(entrada, decisor_ativo().decidir(entrada))

    tipo = decisao.tipo
    modo = decisao.modo_busca
    if tipo == "follow_up":
        modo = _modo_do_historico(turnos) or modo

    campos: dict[str, Any] = {
        "nomes": list(trechos.nomes),
        "id_dom": trechos.id_dom,
        "cpf_cnpj": list(trechos.cpf_cnpj),
        "tipo_ato": trechos.tipo_ato,
        "numero": trechos.numero,
        "ano": trechos.ano,
        "edicao": trechos.edicao,
        "orgao": trechos.orgao,
        "texto_livre": trechos.texto_livre,
        "tipo": tipo,
        "modo_busca": modo,
        "suficiente": decisao.suficiente,
        "ambigua": decisao.ambigua,
        "confianca": decisao.confianca,
    }
    if tipo == "follow_up":
        campos["indice_resultado"] = entrada.ordinal
        campos["usar_foco"] = entrada.ordinal is None
    return IntencaoMensagem(**campos)

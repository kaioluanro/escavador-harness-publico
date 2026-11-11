"""D-ent-1 — extração de entidades (nomes, CPF/CNPJ e números citados).

Implementa o item (d) de ``reports/ideias/I004-ingestao-sintese.md`` e as
críticas de ``reports/ideias/I002-ingestao-contraponto.md`` para o passo D-ent-1
da ingestão (D013), respeitando D015 (o que é do enunciado × citado) e D019
(comparação por identificador normalizado).

A função central é :func:`extrair_entidades`, **pura**: recebe o texto de um
documento (corpo + tabelas, como em :attr:`harness.contracts.Documento.texto`) e
devolve a lista de :class:`harness.contracts.Entidade`. Nada de NER/LLM: são
regras + contexto + lista negra, como decidido em D013. O NER (camada v2) só
entra depois de a base determinística estar medida (I004, tarefa 10).

Nomes de pessoas
----------------
Não basta "run de tokens capitalizados": o corpus tem ``Praça Marechal Deodoro``,
``Palácio da Cidade``, ``FUNDAÇÃO MUNICIPAL DE SAÚDE`` e ``MVP COMERCIO E
SERVIÇOS`` — runs capitalizados que **não** são pessoas (I002 E-3). Por isso a
extração é **ancorada em contexto de pessoa** (I004, plano v1):

1. **Gatilhos de nomeação/assinatura** (:data:`GATILHOS_FRENTE`): ``Sr.``/``Exmo.``,
   ``Relator:``/``Secretário:``, ``NOME:``, ``ASSINAM``, data de assinatura
   (``… de 2026.``), ``(assinado eletronicamente)`` e verbos
   *nomear/exonerar/designar/nomeado*.
2. **Gatilhos reversos** (:data:`GATILHOS_TRAS`): nome seguido de cargo
   (``Fulano, Secretário``) ou de dado pessoal (``Fulano, brasileiro, portador…``).
3. **Linha de tabela**: linha que é só nome, seguida (ou precedida) da linha de um
   CPF/CNPJ — a coluna ``NOME`` das portarias de exoneração.
4. **Lista negra de tipos** (:data:`PALAVRAS_NAO_PESSOA`): um token proibido
   **quebra** o run; nomes nunca atravessam lugar/instituição/empresa/cargo/ato.
   Todo nome exige **2+ tokens-núcleo** (no ouro há de 2 a 7).

CPF/CNPJ
--------
Recorta os três formatos exigidos por D019:
* **completo** (``000.000.000-00``, ``05.522.917/0001-70`` ou 11/14 dígitos);
* **mascarado** (``***.382.303 -**`` e variantes ``***.567.**-87``);
* **quebrado** real (``898.19187-20``, ``517.047.063.00``, ``062.979.903.28``,
  ``19590049/0001-70``, ``45.517.180./0001-11``, ``18.893.557.0001-64``).
``valor_normalizado`` guarda só dígitos (ou dígitos + ``*`` no mascarado), como
o medidor (``harness.tests.gate_extracao.chave_cpf_cnpj``).

Números citados
---------------
``processo`` (SEI/licitação), ``contrato``/instrumento (ata, aditivo, convênio,
empenho) e ``ato_citado`` (lei/decreto/portaria citados) são extraídos por
identificador normalizado (D019): processos SEI (``00045.064078/2024-07``),
números com barra (``90009/2025``), com separador de milhar (``24.759``) e após
marcador (``nº 24.759``). O tipo vem do líder textual mais próximo
(:func:`_classificar_numero`); leis/decretos apenas citados ficam como
``numero_ato`` (D027). Datas, valores monetários e CPF/CNPJ não viram número
citado.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from harness.contracts import Documento, Entidade, TipoEntidade

__all__ = [
    "CONECTORES",
    "GAP_LEADER_NUMERO",
    "GATILHOS_FRENTE",
    "LIMITE_GIGANTE",
    "GATILHOS_TRAS",
    "JANELA_ASSINAM",
    "JANELAS_FRENTE",
    "PALAVRAS_NAO_PESSOA",
    "TITULOS",
    "chave_cpf_cnpj",
    "chave_nome",
    "com_entidades",
    "em_contexto_lugar",
    "entidades_do_documento",
    "expandir_conector_e",
    "extrair_cpf_cnpj",
    "extrair_entidades",
    "extrair_nomes",
    "extrair_numeros_citados",
    "filtrar_nomes_por_contexto",
    "limpar_superficie_nome",
    "normalizar_numero_citado",
    "separar_nomes_coordenados",
    "sem_acento",
]

# --------------------------------------------------------------------------- #
# Normalização                                                                #
# --------------------------------------------------------------------------- #


def sem_acento(texto: str) -> str:
    """Remove acentos por decomposição NFKD (mantém a caixa)."""
    decomposto = unicodedata.normalize("NFKD", texto or "")
    return "".join(c for c in decomposto if not unicodedata.combining(c))


def _minusculo(texto: str) -> str:
    return sem_acento(texto).casefold()


def chave_nome(nome: str) -> str:
    """Chave de nome: sem acento/caixa e com hífen e espaço equivalentes.

    Mesma regra de ``gate_extracao.chave_nome`` (T010b item 8): NÃO canonicaliza
    grafias (SOUZA/SOUSA), decisão do ouro (D015).
    """
    texto = sem_acento(nome).replace("-", " ")
    return re.sub(r"\s+", " ", texto).strip().casefold()


def chave_cpf_cnpj(valor: str) -> str | None:
    """Chave de CPF/CNPJ: completo → dígitos; mascarado → dígitos + ``*``."""
    texto = re.sub(r"\s+", "", valor or "")
    if not texto:
        return None
    if "*" in texto:
        limpo = "".join(c for c in texto if c.isdigit() or c == "*")
        return limpo or None
    digitos = re.sub(r"\D", "", texto)
    if len(digitos) >= 8:
        return digitos
    return _minusculo(texto) or None


def normalizar_numero_citado(valor: str) -> str:
    """Chave de número citado: normaliza e remove espaços ao redor de pontuação."""
    texto = _minusculo(valor)
    return re.sub(r"\s*([.,/\-])\s*", r"\1", texto).strip(".,;:-/ ")


# --------------------------------------------------------------------------- #
# Léxicos de nomes                                                            #
# --------------------------------------------------------------------------- #

#: Conectores que podem aparecer DENTRO de um nome de pessoa.
CONECTORES: frozenset[str] = frozenset({"de", "da", "do", "das", "dos", "e", "d"})

#: Tratamentos que precedem o nome e NÃO o compõem.
TITULOS: frozenset[str] = frozenset(
    {
        "sr",
        "sra",
        "srs",
        "sras",
        "exmo",
        "exma",
        "exmos",
        "exmas",
        "ilmo",
        "ilma",
        "ilmos",
        "ilmas",
        "dr",
        "dra",
        "drs",
        "dras",
        "dom",
        "dona",
        "senhor",
        "senhora",
        "senhores",
        "senhoras",
        "eng",
        "arq",
        "adv",
        "bel",
    }
)

#: Cargos/funções: não fazem parte do nome e funcionam como âncora.
CARGOS: frozenset[str] = frozenset(
    """
    prefeito prefeita presidente vicepresidente vice-presidente vice
    secretario secretaria secretaria subsecretario subsecretaria
    vereador vereadora relator relatora diretor diretora gerente
    coordenador coordenadora assessor assessora superintendente
    procurador procuradora auditor auditora ouvidor ouvidora
    chefe comandante delegado delegada fiscal fiscais gestor gestora
    ministro ministra governador governadora deputado deputada
    senador senadora conselheiro conselheira membro membros titular
    suplente corregedor corregedora promotor promotora
    """.split()
)

#: Lista negra de tipos: tokens que NUNCA fazem parte de um nome de pessoa
#: (lugares, instituições, empresas, unidades, atos, adjetivos de contexto,
#: siglas de órgão). Um token desta lista **quebra** o run de nome.
PALAVRAS_NAO_PESSOA: frozenset[str] = frozenset(
    """
    rua avenida praca travessa alameda rodovia estrada viela vila bairro parque
    jardim conjunto loteamento distrito zona regiao residencial centro cidade
    estado pais municipio teresina piaui brasil norte sul leste oeste central
    nacional federal estadual municipal publico publica publicos publicas
    marechal monsenhor professor professora prof profa
    fundacao hospital hospitais universidade escola faculdade instituto colegio
    creche secretaria superintendencia prefeitura camara tribunal ministerio
    conselho coordenadoria gerencia departamento diretoria divisao assessoria
    gabinete palacio agencia banco caixa companhia empresa promotoria junta
    vara cartorio serventia procuradoria defensoria controladoria ouvidoria
    nucleo setor unidade consorcio autarquia fundo
    ltda me epp eireli sa cia comercio servico servicos construcao construcoes
    distribuidora farmacia mercado drogaria material materiais equipamento
    equipamentos medicamento medicamentos transporte transportes locacao
    locacoes publicidade engenharia tecnologia solucoes negocios papelaria
    alimentos confeccoes grafica editora papel restaurante supermercado posto
    oficina laboratorio clinica plano operadora
    lei decretos decreto portarias portaria plano termo termos contratos
    contrato extrato editais edital errata aviso atas ata ato atos apostilamento
    apostila aditivo notificacao suspensao resolucao retificacao regulamento
    ratificacao republicacao inexigibilidade dispensa processo processos decisao
    justificativa relatorio comunicado acordao instrucao normativa pregao
    concorrencia tomada leilao carta oficio despacho memorando certidao anexo
    anexos apendice
    art artigo artigos inciso incisos alinea paragrafo paragrafos item itens
    caput
    brasileiro brasileira brasileiros brasileiras casado casada solteiro
    solteira viuvo viuva viuvA divorciado divorciada portador portadora inscrito
    inscrita residente domiciliado domiciliada nacionalidade nascido nascida
    data assinatura assinaram assinam assinado registro precos preco
    eletronico eletronica diario oficial tempo integral regimento interno
    demonstrativo saldos creditos suplementares autorizado limite total anual
    orcamentaria orcamento dotacao vigencia objeto clausula clausulas exposicao
    justificativas administrativo administrativa administrativos orgao orgaos
    planejamento coordenacao desenvolvimento urbano saude educacao assistencia
    social politicas integradas sistema unico fonte recursos vinculados
    impostos elemento despesa geral consumo contrapartida transferencia salario
    valor valores unitario quantidade marca memorial descritivo planta
    requerimento requer deferimento indeferimento considerando resolve
    primeira segunda terceira quarta quinta sexta setima oitava nona decima
    vigesima ciclo primeiro segundo terceiro executivo executiva especial
    pela pelo contratante contratada contratantes contratadas indenizante
    indenizada firmado firma firmada entre si celebram conforme documentos
    fundamento legal objeto referente assunto intermediario gestao prestacao
    prestador escrito declarante declarado outorgante outorgado
    por intermedio meio tendo como mao obra para responder ao aos
    sec mun secao exercicio titulo honorifico cidadania mobilidade
    habitacao mulheres juventude cultura transporte defesa civil
    ambiental ambientalmente resíduos residuos solidos disposicao final
    desenvolvimento economico turismo social ensino fundamental
    servidores servidor servidora previdencia assistencia
    esporte esportes lazer simbolo entidade entidades autarquica teresinense
    melhoria qualidade atendimento educacional emenda parlamentar associacao
    matricula funcional governo assistente juridico supervisor humanos
    rural sudeste
    avaliacao execucao parceria decorrente comissao permanente representante
    folha pagamento cargo funcao condicao condicoes contratuais manutencao
    computadores onde que tem relacionados sindicancia frutos
    substituto substituta temporario temporaria signatario signatarios legal
    regulacao auditoria centros referencia referencias administracao pessoal
    gestao pessoas
    nome nomes completo
    semec semplan semcaspi semel semusp strans semgov semjuv fms semf ipmt
    sdu cpl sei srp cnpj cpf rg dom sus inss catmat governe pmt
    # T022 (P3/e): profissões, patentes, estabelecimentos, siglas e palavras de
    # processo/valores por extenso — categorias gerais (não nomes do dev).
    jornalista poeta madre monsenhor padre frei pastor bispo engenheiro arquiteto
    advogado medico medica dentista enfermeiro enfermeira tecnico tecnica auxiliar
    motorista coronel tenente sargento soldado cabo major capitao almirante
    brigadeiro marechal
    odontologia clinica farmacia laboratorio distribuidora empreendimento
    empreendimentos telecomunicacoes copiadoras papelaria
    inovacao projetos projeto inicio acao analise documentacao fase amostras
    recebimento divulgacao resultado parcial conhecimento provimento recurso
    voluntario participaram confissao debito setoriais equipe materiais
    patrimoniais obrigacoes despesas manutencao especializada fornecimento
    contratado contratante contratada gestor gestora signatario substituicao
    determinada seja sua
    turma supremo redator min
    av al rod
    progressao promocao
    telecomunicacao telecomunicacoes sistemas analista contabilidade previdenciario
    ate ou mais auto infracao destes ajustes ajuste
    mail gmail durante toda parcela vencimento
    janeiro fevereiro abril maio junho julho agosto setembro outubro
    novembro dezembro
    ii iii iv vi vii viii ix
    um dois tres quatro cinco seis sete oito nove dez onze doze treze quatorze
    catorze quinze dezesseis dezessete dezoito dezenove vinte trinta quarenta
    cinquenta sessenta setenta oitenta noventa cem cento duzentos trezentos
    quatrocentos quinhentos seiscentos setecentos oitocentos novecentos mil
    milhao milhoes reais real centavos centavo
    # T030: palavras gramaticais (preposição/pronome) nunca compõem nome.
    em se no na nos nas com sem sob
    # T030b (F002 padrão 1): pronomes possessivos e substantivos comuns de
    # produto/domésticos do texto do diário — nenhum compõe nome de pessoa.
    nosso vossa cartolina aba elastico gas cozinha
    # T030c (F002 padrão 1): substantivos comuns de cargo/nível/serviço que o
    # gatilho de matrícula devolve de célula de tabela (``TÉCNICO DE NÍVEL
    # SUPERIOR``, ``AGENTE DE TRÂNSITO``, ``ATENÇÃO BÁSICA``) — nenhum compõe
    # nome de pessoa.
    nivel superior agente transito atencao basica acompanhamento apoiadora
    institucional
    """.split()
)

#: Tokens que são lugares/siglas mas podem ser sobrenomes/parte de nome real.
_PERMITIR_EM_NOME: frozenset[str] = frozenset(
    {"sa", "dom", "norte", "sul", "leste", "oeste", "central", "jardim", "parque", "vila"}
)

#: Prefixos de comando/rótulo que não fazem parte de um nome (T022).
_PREFIXOS_COMANDO: frozenset[str] = frozenset(
    {
        "exonerar",
        "nomear",
        "designar",
        "determinar",
        "ratifico",
        "resolve",
        "pessoa",
        "fisica",
        "física",
        "min",
        "relator",
        "redator",
        "presidenta",
        "presidente",
        "ver",
        "vereador",
        "vereadora",
        "contratado",
        "contratada",
        "setoriais",
        "equipe",
        "tecnica",
        "técnica",
        "ii",
        "iii",
        "iv",
        "v",
        "vi",
        "vii",
        "viii",
        "ix",
        "x",
    }
)

#: Sufixos empresariais/estabelecimento que não fazem parte do nome (T022/P5).
_SUFIXOS_EMPRESA: frozenset[str] = frozenset(
    {
        "ltda",
        "me",
        "epp",
        "eireli",
        "s/a",
        "odontologia",
        "clinica",
        "farmacia",
        "laboratorio",
        "distribuidora",
        "comercio",
        "servicos",
        "construcoes",
        "construtora",
        "empreendimento",
        "telecomunicacoes",
        "copiadoras",
        "engenharia",
        "tecnologia",
        "solucoes",
        "negocios",
    }
)

#: Siglas de partido que não fazem parte do nome (D038: ``(PSB)``).
#: Só são removidas quando grafadas em CAIXA ALTA (evita casar a palavra comum
#: ``solidariedade``).
_PARTIDOS: frozenset[str] = frozenset(
    {
        "psb",
        "psdb",
        "pt",
        "ptb",
        "pdt",
        "pp",
        "pp",
        "psd",
        "mdb",
        "pmdb",
        "dem",
        "psl",
        "psol",
        "pcdob",
        "pv",
        "rede",
        "pros",
        "pps",
        "psc",
        "ptc",
        "prb",
        "pr",
        "pl",
        "novo",
        "avante",
        "solidariedade",
        "podemos",
        "patriota",
        "republicanos",
        "cidadania",
        "ptn",
        "phs",
        "pmn",
        "prp",
        "prtb",
        "pstu",
        "pcb",
        "pco",
        "pst",
        "pfl",
        "psdc",
        "prona",
        "up",
        "dc",
        "mobiliza",
        "agir",
        "uniao",
    }
)

#: Parênteses que cercam rótulo/partido (``(PSB)``, ``(em exercício)``) — saem do
#: nome antes da tokenização (T025/D038).
_RE_PARENTESES = re.compile(r"\([^)]*\)")

#: Palavras da lista negra que podem ser SOBRENOME/parte de nome de pessoa
#: (``ADMILSON BRASIL LUSTOSA FILHO``). São as únicas "fracas" que o run
#: capitalizado pode absorver, e só no interior (nunca abrindo o nome).
_FRACOS_PERMITIDOS_EM_NOME: frozenset[str] = frozenset(
    {
        "brasil",
        "teresina",
        "piaui",
        "norte",
        "sul",
        "leste",
        "oeste",
        "central",
        "sao",
        "santa",
        "santo",
        "dom",
        "dona",
        "nossa",
        "senhora",
        "jardim",
        "parque",
        "vila",
        "rio",
        "serra",
        "lagoa",
        "praia",
        "campo",
        "ponte",
    }
)



def _e_partido(token: str) -> bool:
    """True para sigla de partido grafada em CAIXA ALTA (``PSB``, ``PSDB``)."""
    base = _minusculo(token).strip(".")
    return bool(base) and token.isupper() and base in _PARTIDOS


#: Fim de frase dentro de uma superfície de nome: ". " / "; " depois de uma
#: palavra de 3+ letras (não depois de inicial abreviada como "T.").
_RE_QUEBRA_FRASE = re.compile(r"(?<=[A-Za-zÀ-ÿ]{3})[.;]\s+(?=[A-ZÀ-Þ])")


def _conector_no_fim_da_linha(texto: str, esquerda: str) -> bool:
    """True se o conector ``e`` que fecha ``esquerda`` está no FIM da linha.

    Em ``RAIMUNDO NETO E\\nSILVA NOGUEIRA LIMA`` o ``e`` pendura a linha: ele
    não liga dois nomes, é um nome só quebrado pela coluna (F002 padrão 5). Se
    ao menos UMA ocorrência do nome traz o ``e`` nessa posição, não se separa.
    """
    if not texto or not esquerda:
        return False
    partes = [re.escape(tok) for tok in _TOK.findall(esquerda)]
    if len(partes) < 2:
        return False
    padrao = re.compile(r"\s+".join(partes) + r"\s+[eE](?![^\W\d_])")
    return any(re.match(r"\s*\n", texto[casamento.end() :]) for casamento in padrao.finditer(texto))


def separar_nomes_coordenados(nome: str, texto: str | None = None) -> list[str]:
    """Separa "FULANO DE TAL e BELTRANO DA SILVA" em dois nomes (regra geral).

    Só separa no conector ``e`` quando o lado esquerdo tem ao menos 2
    tokens-núcleo e o direito tem ao menos 3 tokens e 2 núcleos, sem contar
    agnomes (Filho/Neto/Júnior/Sobrinho) (D043): ``José da Silva e Souza`` e
    ``FRANCISCO DAS CHAGAS E SILVA FILHO`` continuam um nome só. A caixa do
    conector original é preservada.

    Com ``texto`` (T030b), o ``e`` no fim de uma linha também impede a separação:
    o conector pendurado é de um único nome quebrado pela coluna
    (``RAIMUNDO NETO E\\nSILVA NOGUEIRA LIMA``), e não de dois signatários
    (``VANICE … SOUSA E\\nWILSON …`` só separa se o ``e`` não fecha a linha).
    """
    pedacos = re.split(r"(\s+[eE]\s+)", nome or "")
    if len(pedacos) < 3:
        return [nome]
    grupos: list[str] = []
    atual = pedacos[0]
    for i in range(1, len(pedacos), 2):
        conector, parte = pedacos[i], pedacos[i + 1]
        nuc_atual = [t for t in _TOK.findall(atual) if _classe_token(t) == "core"]
        toks_parte = _TOK.findall(parte)
        nuc_parte = [
            t
            for t in toks_parte
            if _classe_token(t) == "core" and _minusculo(t).strip(".") not in _AGNOMES
        ]
        if (
            len(nuc_atual) >= 2
            and len(toks_parte) >= 3
            and len(nuc_parte) >= 2
            and not (texto and _conector_no_fim_da_linha(texto, atual))
        ):
            grupos.append(atual)
            atual = parte
        else:
            atual = f"{atual}{conector}{parte}"
    grupos.append(atual)
    return grupos


#: Agnomes: nunca são fragmento nem contam como núcleo ao separar por "e" (D043).
_AGNOMES = frozenset({"jr", "sr", "junior", "filho", "neto", "sobrinho"})


def limpar_superficie_nome(nome: str) -> str | None:
    """Limpa uma superfície de nome (regras gerais, T022 P3/P5 e T025/D038).

    Remove prefixos de comando/rótulo/título (``EXONERAR``, ``NOMEAR``, ``Ver.``,
    ``Vereador``, ``PESSOA FÍSICA``, ``MIN.``, ``RELATOR``…), parênteses
    (``(PSB)``, ``(em exercício)``), siglas de partido e sufixos
    empresariais/quebra (``LTDA``/``ME``/``ODONTOLOGIA``…, conectores finais).
    Devolve ``None`` se sobrar menos de 2 tokens-núcleo ou se algum token for de
    categoria proibida.
    """
    sem_parenteses = _RE_PARENTESES.sub(" ", nome or "")
    # (a) Pontuação de fim de frase dentro da superfície ("FINANÇAS. HENRY …"):
    # separa os trechos e fica com o melhor nome. Não quebra após inicial ("T.").
    trechos = _RE_QUEBRA_FRASE.split(sem_parenteses)
    if len(trechos) > 1:
        candidatos = [limpar_superficie_nome(t) for t in trechos]
        validos = [c for c in candidatos if c]
        return max(validos, key=lambda c: len(c.split())) if validos else None
    tokens = [tok for tok in _TOK.findall(sem_parenteses) if not _e_partido(tok)]
    if not tokens:
        return None
    # (b) Palavra comum em minúscula colada a um nome em maiúsculas
    # ("eletronicamente KATIA …", "básica SELENE …"): fora das pontas.
    if any(t[:1].isupper() and t.isupper() and len(t) > 2 for t in tokens):
        while tokens and tokens[0][:1].islower():
            tokens = tokens[1:]
        while tokens and tokens[-1][:1].islower():
            tokens = tokens[:-1]
    # (d) Advérbio em "-mente" nunca é nome ("Eletronicamente CARLOS …"). Só
    # token com ≥ 12 caracteres (D043): protege Clemente/Vicente.
    while tokens and _minusculo(tokens[0]).endswith("mente") and len(tokens[0]) >= 12:
        tokens = tokens[1:]
    # (c) Fragmentos nas pontas (resíduo de sobreposição/quebra) — D043: só
    # 1 letra sem ponto, ou 2 letras em MINÚSCULA na superfície. Iniciais
    # ("T."), partículas/sobrenomes curtos em caixa de nome ("LI", "Di") e
    # agnomes (JR/SR/FILHO…) nunca são fragmento.
    def _fragmento(tok: str) -> bool:
        base = _minusculo(tok).strip(".")
        if tok.endswith(".") or base in _AGNOMES:
            return False
        if base in CONECTORES or base in _PERMITIR_EM_NOME:
            return False
        return len(base) == 1 or (len(base) == 2 and tok.islower())
    while tokens and _fragmento(tokens[0]):
        tokens = tokens[1:]
    while tokens and _fragmento(tokens[-1]):
        tokens = tokens[:-1]
    if not tokens:
        return None
    inicio, fim = 0, len(tokens)
    while inicio < fim and _minusculo(tokens[inicio]).strip(".") in _PREFIXOS_COMANDO:
        inicio += 1
    while fim > inicio and (
        _minusculo(tokens[fim - 1]).strip(".") in _SUFIXOS_EMPRESA
        or _minusculo(tokens[fim - 1]).strip(".") in CONECTORES
    ):
        fim -= 1
    limpos = tokens[inicio:fim]
    # Rejeita se ainda houver token de categoria proibida.
    for token in limpos:
        base = _minusculo(token).strip(".")
        if base in _PERMITIR_EM_NOME:
            continue
        if _classe_token(token) == "proibido":
            return None
    nucleos = [tok for tok in limpos if _classe_token(tok) == "core"]
    if len(nucleos) < 2 or len(limpos) > 9:
        return None
    return re.sub(r"\s+", " ", " ".join(limpos)).strip().rstrip(".,;:") or None


#: Marcadores de lugar/estabelecimento que antecedem nomes de escola/rua/avenida
#: (T022/P3): o nome que segue é topônimo, não pessoa. Palavras inteiras
#: (``\\b``): sem o limite, ``UNIDADE`` casa dentro de ``imunidade`` e um nome
#: de pessoa legítimo é descartado.
_RE_CONTEXTO_LUGAR = re.compile(
    r"(E\.\s*M\.?|AV\.|\b(?:ESCOLA|AVENIDA|RUA|PRA[ÇC]A|TRAVESSA|ALAMEDA|"
    r"UNIDADE|CENTRO|HOSPITAL|FUNDA[ÇC][ÃA]O|PAL[ÁA]CIO|MERCADO|PARQUE|"
    r"POSTO|CL[ÍI]NICA|LABORAT[ÓO]RIO|HOTEL)\b)",
    re.IGNORECASE,
)


def em_contexto_lugar(texto: str, nome: str, *, janela: int = 40) -> bool:
    """True se TODAS as ocorrências de ``nome`` vêm logo após marcador de lugar.

    Usado para não extrair como pessoa o topônimo de ``E.M. JÚLIO LOPES``/
    ``AV. CONQUISTA ...``. Se o mesmo nome aparecer em outro contexto (pessoa),
    ele é preservado. A busca tolera quebra de linha/espaços dentro do nome
    (``JOÃO EMILIO\\nFALCÃO``), como o texto extraído costuma trazer.
    """
    if not nome or not texto:
        return False
    partes = [re.escape(parte) for parte in nome.split()]
    padrao = r"\s+".join(partes)
    ocorrencias = 0
    for achado in re.finditer(padrao, texto, re.IGNORECASE):
        ocorrencias += 1
        # Um caractere a mais que a janela: sem ele a janela pode COMEÇAR no
        # meio da palavra (``… imunidade`` cortado em ``unidade``) e o ``\\b``
        # do marcador casa sem motivo.
        antes = texto[max(0, achado.start() - janela - 1) : achado.start()]
        if not _RE_CONTEXTO_LUGAR.search(antes):
            return False
    return ocorrencias > 0


# --------------------------------------------------------------------------- #
# Filtro de nomes pelo contexto das ocorrências (T030)                        #
# --------------------------------------------------------------------------- #

#: Marcador de lugar/estabelecimento IMEDIATAMENTE antes do nome (no máximo 2
#: palavras entre o marcador e o nome, sem pontuação): ``E.M. PROFESSOR JOSÉ …``,
#: ``US MARIANO …``, ``AV. CONQUISTA FRANCISCO …``, ``ASSOCIAÇÃO MARIA …``.
#: Aplicado ao texto dobrado (minúsculo e sem acento).
_RE_LUGAR_IMEDIATO = re.compile(
    r"(?:\be\.\s*m\b\.?|\bescola|\bu\.?\s?b?\.?\s?s\b\.?|\bav\.|\bavenida|\brua|\bpraca|"
    r"\btravessa|\balameda|\bassociacao|\bcreche|\binstituto|\bhospital)"
    r"\s+(?:[^\s,;:()]+\s+){0,2}$"
)

#: ``MUNICIPIO DE <NOME>`` só vale COLADO no nome (gap 0): em
#: ``MUNICIPIO DE SELVIRIA PROC.`` a palavra seguinte é o topônimo, mas em
#: ``MUNICIPIO DE TERESINA DR. TIAGO…`` há cidade + cargo entre o marcador e a
#: pessoa (T030b/F002: ``tiago lira pontes`` não pode cair).
_RE_MUNICIPIO_COLADO = re.compile(r"\bmunicipio\s+de\s+$")

#: Sufixo societário logo depois do nome: é razão social, não pessoa.
_RE_SUFIXO_SOCIETARIO = re.compile(
    r"^\s*[-–—,]?\s*(?:ME|MEE|(?i:epp|eireli|ltda)|S\.\s?A\.?|S/A)(?![^\W\d_])"
)

#: Registro cadastral de PESSOA JURÍDICA logo depois do nome
#: (``INSCRITA NO CNPJ``, ``inscrita sob o CNPJ``). O ``, CNPJ`` solto NÃO
#: basta: MEI é pessoa física com CNPJ (``TARCILA MACHADO MACIEL, CNPJ …``).
_RE_CNPJ_DEPOIS = re.compile(
    r"^\s*[,;:.]?\s*(?:inscrit[oa]|sob)\b[^.;\n]{0,30}?\bCNPJ\b", re.I
)

#: Abreviatura processual com marca de plural (``EMBTE. (S)``, ``PROC. (A/S)``).
#: O PONTO é obrigatório: `` `` (PP)`` sem ponto é só um nome entre parênteses
#: (``PEDRO ALCÂNTARA (PP)``), e não uma abreviatura de cargo.
_RE_ABREV_PROCESSUAL = re.compile(r"^\s*\.\s*\([A-Za-z]{1,2}(?:/[A-Za-z]{1,2})?\)")

#: Iniciais soltas de razão social antes do nome (``M H BARROS…``, ``J A DIAS…``,
#: ``MARIA C DA COSTA…``): letra maiúscula consoante sem ponto (não é o
#: conector ``E`` nem artigo ``A``/``O``), opcionalmente seguida de outras letras
#: soltas e de ``DA/DE/DO(S)``.
_RE_INICIAIS_ANTES = re.compile(
    r"(?<![^\W\d_])[B-DF-HJ-NP-TV-Z](?:\s+[A-Z])*\s+(?:D[AEO]S?\s+)?$"
)


def _dobrar(texto: str) -> str:
    """Minúsculo e sem acento, preservando o comprimento (1 caractere → 1)."""
    return "".join(
        (unicodedata.normalize("NFD", ch)[:1] or ch).casefold()[:1] or ch for ch in texto
    )


def _padrao_nome(nome: str) -> re.Pattern[str] | None:
    """Regex (no texto dobrado) que acha o nome com espaço/quebra/hífen entre tokens.

    Cada separador é um grupo, para o chamador inspecionar hífen/parêntese. Não
    exige fronteira à esquerda: quem chama verifica se a ocorrência começa no
    meio de uma palavra.
    """
    tokens = [_dobrar(tok) for tok in _TOK.findall(nome or "")]
    if len(tokens) < 2:
        return None
    partes = [re.escape(tokens[0])]
    for anterior, token in zip(tokens, tokens[1:]):
        sep = r"([\s\-–—()]*)" if anterior.endswith(".") else r"([\s\-–—()]+)"
        partes.append(sep + re.escape(token))
    return re.compile("".join(partes) + r"(?![^\W\d_])")


def _e_sigla(token: str) -> bool:
    letras = token.strip(".")
    return 2 <= len(letras) <= 5 and letras.isalpha() and letras.isupper()


#: Vogais: palavra portuguesa nunca fica sem vogal nenhuma (F002 padrão 3).
_VOGAIS: frozenset[str] = frozenset("aeiouáàâãéêíóôõúü")


def _sigla_nas_pontas(
    original: str, dobrado: str, casamento: re.Match[str]
) -> bool:
    """True se a 1ª/última palavra desta ocorrência é SIGLA, não nome (F002/3).

    Sigla é CAIXA ALTA de 2–5 letras que (a) não tem nenhuma vogal — impossível
    em palavra portuguesa: ``STC ARAUJO`` — ou (b) aparece em OUTRO trecho do
    documento como sigla, isto é, entre parênteses (``UBS (HUT-FMS)``) ou logo
    após barra (``GEAFA/UBS``). Só o texto FORA das ocorrências do próprio nome
    conta como evidência: ``NEVES/`` em outra linha da mesma tabela é o
    separador da célula (``CRISLAINE DE AQUINO NEVES/ CARGO: …``), não uso de
    sigla — sem essa exclusão o nome inteiro é descartado.
    """
    tokens = list(_TOK.finditer(original, casamento.start(), casamento.end()))
    if not tokens:
        return False
    removidos = sorted(m.span() for m in casamento.re.finditer(dobrado))
    fora_partes: list[str] = []
    ultimo = 0
    for inicio, fim in removidos:
        if inicio > ultimo:
            fora_partes.append(original[ultimo:inicio])
        ultimo = max(ultimo, fim)
    fora_partes.append(original[ultimo:])
    fora = "".join(fora_partes)
    for encontrado in (tokens[0], tokens[-1]):
        bruto = encontrado.group()
        letras = bruto.strip(".")
        if not (2 <= len(letras) <= 5) or not letras.isalpha() or not bruto.isupper():
            continue
        if letras.casefold() in _AGNOMES:
            # Agnome (``JOSÉ DA SILVA JR``): sem vogal, mas é parte do nome.
            continue
        if not any(c.casefold() in _VOGAIS for c in letras):
            return True
        padrao = rf"(?<![^\W\d_])(?:\([^)\n]*?\b{re.escape(letras)}\b|{re.escape(letras)}/)"
        if re.search(padrao, fora):
            return True
    return False


def _ocorrencia_nao_pessoa(original: str, dobrado: str, casamento: re.Match[str]) -> bool:
    """True se esta ocorrência do nome, pelo contexto, não é pessoa (T030).

    Sinais gerais: começa no meio de uma palavra (``NOMEAR`` → ``AR …``) ou
    continua uma sigla com ponto (``E.M.`` → ``M. …``); lugar/estabelecimento
    imediatamente antes; sufixo societário ou registro cadastral de PJ logo
    depois; iniciais soltas de razão social antes; abreviatura processual
    (``EMBTE. (S)``); sigla nas pontas (``STC ARAUJO``); e separador interno
    com parêntese ou hífen junto de sigla (``GEAFA-DAB``,
    ``Almoxarifado- HUT``, ``BÁSICA)\\nSELENE``).
    """
    inicio, fim = casamento.span()
    anterior = original[inicio - 1] if inicio > 0 else ""
    if anterior.isalpha():
        return True
    primeiro = _TOK.match(original, inicio)
    if (
        primeiro
        and len(primeiro.group().strip(".")) == 1
        and re.search(r"(?<![^\W\d_])[^\W\d_]\.$", original[max(0, inicio - 2) : inicio])
    ):
        return True
    antes = dobrado[max(0, inicio - 60) : inicio].replace("\n", " ")
    if _RE_LUGAR_IMEDIATO.search(antes) or _RE_MUNICIPIO_COLADO.search(antes):
        return True
    if _RE_INICIAIS_ANTES.search(original, max(0, inicio - 12), inicio):
        return True
    if _sigla_nas_pontas(original, dobrado, casamento):
        return True
    depois = original[fim : fim + 60]
    if (
        _RE_SUFIXO_SOCIETARIO.match(depois)
        or _RE_CNPJ_DEPOIS.match(depois)
        or _RE_ABREV_PROCESSUAL.match(depois)
    ):
        return True
    posicao = inicio
    for indice in range(1, casamento.re.groups + 1):
        separador = casamento.group(indice)
        s_ini, s_fim = casamento.span(indice)
        if "(" in separador or ")" in separador:
            return True
        if re.search(r"[-–—]", separador):
            esquerda = original[posicao:s_ini]
            direita = _TOK.match(original, s_fim)
            if _e_sigla(esquerda.split()[-1] if esquerda.split() else "") or (
                direita and _e_sigla(direita.group())
            ):
                return True
        posicao = s_fim
    return False


def filtrar_nomes_por_contexto(texto: str, nomes: Sequence[str]) -> list[str]:
    """Descarta nomes sem nenhuma ocorrência "de pessoa" no texto (T030).

    Para cada candidato, localiza TODAS as ocorrências no texto (tolerando
    quebra de linha e hifenização). O nome é descartado quando **toda**
    ocorrência é (a) contexto de não-pessoa (:func:`_ocorrencia_nao_pessoa`) ou
    (b) parte de uma ocorrência de outro nome mantido, mais longo, do mesmo
    documento (fragmento de quebra de linha: ``Machado Filho`` só aparece
    dentro de ``Edgar Carneiro\\nMachado Filho``). Basta UMA ocorrência
    independente para o nome ficar. Nome não localizado no texto é mantido.

    Se todas as ocorrências começam no meio de uma palavra, o 1º token é
    fragmento e o nome é reancorado sem ele (``AR DEAN RUSK…`` → ``DEAN RUSK…``).
    """
    original = _dehyphen(texto)
    dobrado = _dobrar(original)

    def _ocorrencias(nome: str) -> list[re.Match[str]]:
        padrao = _padrao_nome(nome)
        return list(padrao.finditer(dobrado)) if padrao else []

    # 1) Reancora nomes que só aparecem começando no meio de uma palavra.
    candidatos: list[str] = []
    for nome in nomes:
        achados = _ocorrencias(nome)
        if achados and all(
            m.start() > 0 and original[m.start() - 1].isalpha() for m in achados
        ):
            resto = _TOK.findall(nome)[1:]
            nome = limpar_superficie_nome(" ".join(resto)) if resto else None
            if not nome:
                continue
        candidatos.append(nome)

    # 2) Classifica cada ocorrência.
    info: list[tuple[str, int, list[tuple[int, int, bool]]]] = []
    for nome in candidatos:
        spans = [
            (m.start(), m.end(), _ocorrencia_nao_pessoa(original, dobrado, m))
            for m in _ocorrencias(nome)
        ]
        info.append((nome, len(_TOK.findall(nome)), spans))

    # 3) Descarta quem só tem ocorrências de não-pessoa.
    vivos = [item for item in info if not item[2] or not all(s[2] for s in item[2])]

    # 4) Descarta fragmento cujas ocorrências independentes estão todas dentro
    #    de uma ocorrência de outro nome vivo mais longo.
    def _coberto(inicio: int, fim: int, tamanho: int, proprio: str) -> bool:
        return any(
            outro != proprio
            and n_outro > tamanho
            and any(a <= inicio and fim <= b for a, b, _ in spans_outro)
            for outro, n_outro, spans_outro in vivos
        )

    resultado: list[str] = []
    for nome, tamanho, spans in vivos:
        if spans and all(
            nao_pessoa or _coberto(inicio, fim, tamanho, nome)
            for inicio, fim, nao_pessoa in spans
        ):
            continue
        resultado.append(nome)
    return resultado


def expandir_conector_e(texto: str, nome: str) -> str:
    """Completa candidato que termina logo antes de um conector ``e`` (F002/5).

    O extrator de lista corta na vírgula e o NER para no conector, então
    ``JOSÉ MARIA DE MOURA`` sai sozinho em ``… MOURA E VASCONCELOS …``: vira FP
    e o nome do ouro vira FN. Só expande quando **TODAS** as ocorrências do
    candidato no documento são seguidas de `` e <Maiúscula>`` — assim um nome
    que também aparece sozinho (ou seguido de vírgula/fim de frase) não muda.
    A expansão é o run de nome que segue, validado como nome (2+ núcleos, no
    máximo 9 tokens); na dúvida devolve o candidato original.
    """
    if not texto or not nome:
        return nome
    partes = [re.escape(tok) for tok in _TOK.findall(nome)]
    if len(partes) < 2:
        return nome
    padrao = re.compile(r"\s+".join(partes), re.IGNORECASE)
    achados = list(padrao.finditer(texto))
    if not achados:
        return nome
    for casamento in achados:
        if not re.match(r"\s+[eE]\s+[A-ZÀ-Þ]", texto[casamento.end() :]):
            return nome
    tokens = _tokens(texto)
    for casamento in achados:
        # O run precisa COMEÇAR no 1º token do casamento: começar no último
        # perde o prefixo (``VANICE … SOUSA`` → ``SOUSA E WILSON …``) e o nome
        # vira FP ao mesmo tempo que o ouro vira FN.
        indice = None
        for k, token in enumerate(tokens):
            if token[0] >= casamento.end():
                break
            if token[1] <= casamento.end() and token[0] >= casamento.start():
                indice = k
                break
        if indice is None:
            continue
        expandido, j = _consumir_frente(
            tokens, texto, indice, min(len(tokens), indice + 12)
        )
        if not expandido or len(_TOK.findall(expandido)) <= len(partes):
            continue
        # Título depois do conector (``… PONTES E DR. IVAN …``): é outro
        # signatário, não continuação do nome — não funde (F002/5).
        if any(
            token[3] == "titulo" and token[0] >= casamento.end()
            for token in tokens[indice:j]
        ):
            return nome
        return expandido
    return nome


# --------------------------------------------------------------------------- #
# Tokenização                                                                 #
# --------------------------------------------------------------------------- #

_TOK = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’]*\.?")


def _dehyphen(texto: str) -> str:
    """Junta palavras partidas na quebra de linha: ``SIL-\\nVIO`` → ``SILVIO``.

    Só junta **letra-hífen-quebra-letra** (hifenização). Números partidos no
    hífen (``027.254.173-\\n76``) são preservados: o hífen é separador real e os
    padrões de CPF/CNPJ já aceitam quebra de linha. O hífen seguido de **espaço**
    na mesma linha é limpo no enunciado (``segmentar._limpar_enunciado``, T024
    item 5), não aqui — juntar em nomes colava ``MELO- DIRETOR``.
    """
    return re.sub(r"(?<=[^\W\d_])-\n\s*(?=[^\W\d_])", "", texto or "")


def _classe_token(token: str) -> str:
    """Classifica um token: ``core``, ``conector``, ``titulo`` ou ``proibido``."""
    base = _minusculo(token).strip(".")
    if not base:
        return "proibido"
    if base in CONECTORES:
        return "conector"
    if base in TITULOS:
        return "titulo"
    if base in PALAVRAS_NAO_PESSOA or base in CARGOS:
        return "proibido"
    if not token[0].isupper():
        return "proibido"
    # Letra solta (``O``, ``A``, ``F`` de fragmento de coluna) não é nome;
    # inicial pontuada (``M.``) é.
    if len(base) == 1 and not token.endswith("."):
        return "proibido"
    return "core"


def _normalizar_nome(partes: Sequence[str]) -> str | None:
    """Limpa um run: tira conectores/títulos das bordas; exige 2+ núcleos."""
    inicio, fim = 0, len(partes)
    while inicio < fim and _classe_token(partes[inicio]) in ("conector", "titulo"):
        inicio += 1
    while fim > inicio and _classe_token(partes[fim - 1]) in ("conector", "titulo"):
        fim -= 1
    limpo = list(partes[inicio:fim])
    nucleos = [p for p in limpo if _classe_token(p) == "core"]
    reais = [p for p in nucleos if len(_minusculo(p).strip(".")) > 1]
    if len(nucleos) < 2 or not reais or len(limpo) > 9:
        return None
    return limpar_superficie_nome(" ".join(limpo))


#: Separadores permitidos DENTRO de um nome (não inclui vírgula: nomes não têm
#: vírgula interna; isso impede fundir ``Finanças, Edgar`` num run só).
_SEP_INTERNO = re.compile(r"[\s'’\-]*")


def _dentro_parenteses(texto: str, token: tuple[int, int, str, str]) -> bool:
    """True se o token está cercado por parênteses (``(PSDB)``, ``(em exercício)``)."""
    antes = texto[: token[0]].rstrip()
    depois = texto[token[1] :].lstrip()
    return antes.endswith("(") and depois.startswith(")")


def _separador_interno_ok(separador: str) -> bool:
    """True se o separador entre dois tokens é interno a um nome.

    Parênteses são ignorados (``RENATO PIRES (PSDB) BERGER`` continua um run);
    vírgula/ponto/quebra de oração interrompem.
    """
    return bool(_SEP_INTERNO.fullmatch(separador.replace("(", "").replace(")", "")))


def _consumir_frente(tokens: Sequence[tuple[int, int, str, str]], texto: str, i: int,
                     limite: int) -> tuple[str | None, int]:
    """Consome o run de nome começando em ``tokens[i]`` (que deve ser ``core``)."""
    partes = [tokens[i][2]]
    j = i + 1
    while j < limite and tokens[j][3] in ("core", "conector", "titulo"):
        if _dentro_parenteses(texto, tokens[j]):
            j += 1
            continue
        separador = texto[tokens[j - 1][1] : tokens[j][0]]
        if not _separador_interno_ok(separador):
            break
        if "(" in separador and not _dentro_parenteses(texto, tokens[j]):
            # Abre parêntese que não fecha neste token (``… MORAES (Tema 796 …``):
            # o que segue é texto entre parênteses, não continuação do nome.
            break
        partes.append(tokens[j][2])
        j += 1
    return _normalizar_nome(partes), j


def _nome_apos(
    tokens: Sequence[tuple[int, int, str, str]],
    texto: str,
    posicao: int,
    *,
    janela: int = 10,
) -> str | None:
    """Primeiro run de nome (2+ núcleos) após ``posicao``, saltando tokens vazios."""
    inicio = 0
    for indice, token in enumerate(tokens):
        if token[0] >= posicao:
            inicio = indice
            break
    else:
        return None
    limite = min(len(tokens), inicio + janela)
    i = inicio
    while i < limite:
        if tokens[i][3] == "core":
            nome, j = _consumir_frente(
                tokens, texto, i, min(len(tokens), i + 12)
            )
            if nome is not None:
                return nome
            i = max(j, i + 1)
        else:
            i += 1
    return None


def _consumir_tras(tokens: Sequence[tuple[int, int, str, str]], texto: str,
                   i: int, limite: int) -> tuple[str | None, int]:
    """Consome o run de nome terminando em ``tokens[i]`` (que deve ser ``core``)."""
    partes = [tokens[i][2]]
    j = i - 1
    while j >= limite and tokens[j][3] in ("core", "conector", "titulo"):
        if _dentro_parenteses(texto, tokens[j]):
            j -= 1
            continue
        separador = texto[tokens[j][1] : tokens[j + 1][0]]
        if not _separador_interno_ok(separador):
            break
        if "(" in separador and not _dentro_parenteses(texto, tokens[j]):
            break
        partes.insert(0, tokens[j][2])
        j -= 1
    return _normalizar_nome(partes), j


def _nome_antes(
    tokens: Sequence[tuple[int, int, str, str]],
    texto: str,
    posicao: int,
    *,
    janela: int = 16,
) -> str | None:
    """Primeiro run de nome (2+ núcleos) terminando antes de ``posicao``."""
    limite_inferior = 0
    fim = 0
    for indice in range(len(tokens) - 1, -1, -1):
        if tokens[indice][1] <= posicao:
            fim = indice
            break
    else:
        return None
    inicio = max(0, fim - janela + 1)
    i = fim
    while i >= inicio:
        if tokens[i][3] == "core":
            nome, j = _consumir_tras(tokens, texto, i, max(0, i - 12))
            if nome is not None:
                return nome
            i = min(j, i - 1)
        else:
            i -= 1
    return None


#: Palavras "ponte" entre um cargo e o nome colado (artigos, pronomes, títulos).
_PONTE_CARGO: frozenset[str] = frozenset(
    set(TITULOS)
    | set(CONECTORES)
    | {
        "o",
        "a",
        "os",
        "as",
        "seu",
        "sua",
        "seus",
        "suas",
        "este",
        "esta",
        "esse",
        "essa",
        "em",
    }
)


def _nome_colado_ao_cargo(
    tokens: Sequence[tuple[int, int, str, str]], texto: str, posicao: int
) -> str | None:
    """Nome **imediatamente** após um cargo (``O SUPERINTENDENTE JOÃO EULÁLIO…``).

    Só pula títulos/artigos/conectores; um token da lista negra (``Municipal``,
    ``Teresina``, ``Educação``) interrompe, evitando pegar um nome de outra oração.
    """
    i = 0
    for indice, token in enumerate(tokens):
        if token[0] >= posicao:
            i = indice
            break
    else:
        return None
    while i < len(tokens):
        base = _minusculo(tokens[i][2]).strip(".")
        if base in _PONTE_CARGO:
            i += 1
            continue
        break
    if i < len(tokens) and tokens[i][3] == "core":
        nome, _ = _consumir_frente(tokens, texto, i, len(tokens))
        return nome
    return None


def _runs_de_nomes(
    tokens: Sequence[tuple[int, int, str, str]], texto: str, inicio: int, fim: int
) -> list[str]:
    """Todos os runs de nome (2+ núcleos) no intervalo de tokens ``[inicio, fim)``."""
    nomes: list[str] = []
    i = inicio
    while i < fim:
        if tokens[i][3] == "core":
            nome, j = _consumir_frente(tokens, texto, i, fim)
            if nome is not None:
                nomes.append(nome)
            i = max(j, i + 1)
        else:
            i += 1
    return nomes


# --------------------------------------------------------------------------- #
# Gatilhos de contexto                                                        #
# --------------------------------------------------------------------------- #

_CARGO_PADRAO = (
    r"prefeito|prefeita|secret[aá]ri[oa]|subsecret[aá]ri[oa]|presidente|"
    r"vice-?presidente|vereador[a]?|relator[a]?|diretor[a]?|gerente|"
    r"coordenador[a]?|assessor[a]?|superintendente|ouvidor[a]?|procurador[a]?|"
    r"auditor[a]?|corregedor[a]?|promotor[a]?|comandante|delegado[a]?|"
    r"ministro[a]?|governador[a]?|deputado[a]?|senador[a]?|chefe|fiscal|"
    r"gestor[a]?|conselheir[oa]|titular|suplente"
)

#: Gatilhos "antes do nome": do fim do gatilho procura-se o próximo run de nome.
GATILHOS_FRENTE: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:Exm[oa]|Ilm[oa]|Sr|Sra|Senhor|Senhora|Dom|Dona)\.?\b", re.I),
    re.compile(rf"\b(?:{_CARGO_PADRAO})\s*:\s*", re.I),
    re.compile(r"\bnome(?:\s+completo)?\s*:\s*", re.I),
    re.compile(r"\(?\s*assinado\s+eletronicamente\s*\)?\s*", re.I),
    re.compile(
        r"(?:\bem\s+\d{1,2}\s+de\s+[^\W\d_]+\s+de\s+\d{4}"
        r"|Teresina\b[^\n]{0,60}?\d{1,2}\s+de\s+[^\W\d_]+\s+de\s+\d{4})"
        r"\s*[.,]\s*",
        re.I,
    ),
    re.compile(r"\d{2}/\d{2}/\d{4}\.\s*"),
    re.compile(
        r"\b(?:nomear|nomeia|nomeou|nomead[oa]|exonerar|exonera|exonerou|"
        r"exonerad[oa]|designar|designa|designou|designad[oa]|empossar|empossou)"
        r"\b\s*(?:o|a|os|as|servidor(?:a)?|senhor(?:a)?)?\s*",
        re.I,
    ),
    # Título abreviado colado ao nome (``voto do Min. ALEXANDRE DE MORAES``).
    # O ponto é obrigatório: sem ele ``min`` de ``minuto``/``min`` dispararia.
    re.compile(r"\b(?:Min|Ministra|Ministro)\.\s*", re.I),
    # Cargo colado ao nome (``O SUPERINTENDENTE JOÃO EULÁLIO…``, ``o Sr. …``).
    re.compile(rf"\b(?:{_CARGO_PADRAO})\b\s*", re.I),
)

#: Janela de tokens de cada gatilho de :data:`GATILHOS_FRENTE` (nome pode estar
#: perto do cargo/verbo, mas não a um parágrafo de distância).
JANELAS_FRENTE: tuple[int, ...] = (10, 10, 10, 10, 10, 6, 5, 6, 6)

#: Gatilhos "depois do nome" (nome imediatamente antes do gatilho).
GATILHOS_TRAS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"\b(?:{_CARGO_PADRAO})\b", re.I),
    re.compile(
        r"\b(?:brasileir[oa]|portador[a]?|inscrit[oa]|residente|domiciliad[oa]|"
        r"casado|casada|solteir[oa]|vi[uú]v[oa]|nascid[oa]|nomead[oa]|"
        r"cpf|rg)\b",
        re.I,
    ),
    # Servidor identificado por matrícula (``… ARAUJO BRASIL, matrícula nº 50753``).
    re.compile(r"\bmatr[ií]cula\b", re.I),
)

_RE_ASSINAM = re.compile(r"\bASSINAM\b", re.I)

#: Janela de tokens do bloco de assinatura após ``ASSINAM`` (signatários podem
#: estar em linhas diferentes).
JANELA_ASSINAM = 30


def _tokens(texto: str) -> list[tuple[int, int, str, str]]:
    return [(m.start(), m.end(), m.group(0), _classe_token(m.group(0))) for m in _TOK.finditer(texto)]


def _gatilho_adjacente(
    texto: str, tokens: Sequence[tuple[int, int, str, str]], posicao: int
) -> bool:
    """True se o gatilho reverso está colado a um nome (``Fulano, Cargo``).

    Evita disparar em menções soltas de cargo (``o Secretário``,
    ``Secretaria Municipal``): é preciso o token anterior ser ``core`` ou haver
    pontuação de separação (vírgula, hífen, ponto e vírgula ou quebra de linha).
    """
    anterior: tuple[int, int, str, str] | None = None
    for token in tokens:
        if token[1] <= posicao:
            anterior = token
        else:
            break
    if anterior is None:
        return False
    if anterior[3] == "core":
        return True
    return bool(re.search(r"[,;:\-–\n]", texto[anterior[1] : posicao]))


def _nomes_de_linha_nome(linhas: Sequence[str]) -> list[str]:
    """Linhas que são SÓ um nome (2+ núcleos), em contexto de tabela/assinatura.

    Exige que a linha limpa cubra a linha inteira; a lista negra impede que
    frases de corpo em maiúsculas virem nome. Aceita vírgula final.
    """
    nomes: list[str] = []
    for indice, bruta in enumerate(linhas):
        linha = bruta.strip().rstrip(".,;:")
        if not linha or len(linha) > 70:
            continue
        tokens = _tokens(linha)
        if not tokens:
            continue
        if tokens[0][0] != 0 or tokens[-1][1] != len(linha):
            continue
        if any(token[3] == "proibido" for token in tokens):
            continue
        cores = [token for token in tokens if token[3] == "core"]
        if len(cores) < 2 or len(cores) > 8:
            continue
        # Contexto: CPF/CNPJ na linha seguinte/anterior, ou data/assinatura antes.
        anterior = linhas[indice - 1] if indice > 0 else ""
        seguinte = linhas[indice + 1] if indice + 1 < len(linhas) else ""
        contexto = bool(
            _RE_CPF_QUALQUER.search(seguinte)
            or _RE_CPF_QUALQUER.search(anterior)
            or re.search(r"assinado\s+eletronicamente", anterior, re.I)
            or re.search(r"\b\d{4}\s*[.,]?\s*$", anterior)
        )
        if contexto:
            limpo = limpar_superficie_nome(linha)
            if limpo:
                nomes.append(limpo)
    return nomes


#: Palavras de cargo/órgão que interrompem a expansão do run capitalizado
#: (``Secretário Municipal de Finanças ADMILSON…``): não são nome de pessoa.
_LIMITES_EXPANSAO: frozenset[str] = frozenset(
    {
        "financas",
        "planejamento",
        "gestao",
        "obras",
        "comunicacao",
        "seguranca",
        "mobilidade",
        "habitacao",
        "trabalho",
        "fazenda",
        "procuradoria",
        "controladoria",
        "adjunto",
        "adjunta",
        "interino",
        "interina",
        "substituto",
        "substituta",
        "exercicio",
        "geral",
        "municipal",
        "estadual",
        "federal",
        "publico",
        "publica",
        "executivo",
        "executiva",
        "especial",
        "regional",
        "urbano",
        "rural",
        "social",
        "integrada",
        "integradas",
    }
)


def _classe_run_capitalizado(token: str) -> str:
    """Classe de um token no run capitalizado da expansão (T025 item 3).

    ``core`` (nome), ``conector`` (``de``/``da``), ``fraco`` (palavra da lista
    negra que pode ser sobrenome, ex.: ``BRASIL``) ou ``limite`` (cargo, título,
    empresa, instituição, palavra de órgão — interrompe o run).
    """
    base = _minusculo(token).strip(".")
    if not base:
        return "limite"
    if (
        base in CARGOS
        or base in TITULOS
        or base in _PREFIXOS_COMANDO
        or base in _SUFIXOS_EMPRESA
        or base in _LIMITES_EXPANSAO
        or _e_partido(token)
    ):
        return "limite"
    if base in CONECTORES:
        return "conector"
    if not token[0].isupper():
        return "limite"
    if len(base) == 1 and not token.endswith("."):
        return "limite"
    if base in PALAVRAS_NAO_PESSOA:
        return "fraco" if base in _FRACOS_PERMITIDOS_EM_NOME else "limite"
    return "core"


def _run_capitalizado_valido(partes: Sequence[str]) -> str | None:
    """Valida/limpa um run capitalizado como nome de pessoa (2+ núcleos).

    Apara conectores das bordas e palavras "fracas" só na ABERTURA
    (``Teresina MARIA…`` → ``MARIA…``: fraco nunca abre nome). No fim o fraco
    é do próprio nome quando o run só existe por causa dele
    (``ENEIAS NAPOLEÃO ARAUJO BRASIL`` é o nome do ouro, não sobrenome solto).
    """
    inicio, fim = 0, len(partes)
    while inicio < fim and _classe_run_capitalizado(partes[inicio]) in (
        "conector",
        "fraco",
    ):
        inicio += 1
    while fim > inicio and _classe_run_capitalizado(partes[fim - 1]) == "conector":
        fim -= 1
    limpos = list(partes[inicio:fim])
    if not 2 <= len(limpos) <= 9:
        return None
    if _classe_run_capitalizado(limpos[0]) != "core":
        return None
    if sum(1 for p in limpos if _classe_run_capitalizado(p) == "core") < 2:
        return None
    return re.sub(r"\s+", " ", " ".join(limpos)).strip().rstrip(".,;:")


def _expandir_run_capitalizado(texto: str, nome: str) -> str:
    """Expande ``nome`` para o run completo de tokens capitalizados da linha.

    Corrige o truncamento por palavra da lista negra no meio do nome
    (``ADMILSON BRASIL LUSTOSA FILHO`` → o extrator emitia só ``LUSTOSA FILHO``).
    Só expande dentro da MESMA linha e para um run que valide como pessoa; na
    dúvida, devolve ``nome``.
    """
    tokens_nome = _TOK.findall(nome)
    if len(tokens_nome) < 2:
        return nome
    padrao = re.compile(r"\s+".join(re.escape(token) for token in tokens_nome))
    melhor = nome
    for achado in padrao.finditer(texto):
        inicio_linha = texto.rfind("\n", 0, achado.start()) + 1
        fim_linha = texto.find("\n", achado.end())
        if fim_linha == -1:
            fim_linha = len(texto)
        linha = texto[inicio_linha:fim_linha]
        a = achado.start() - inicio_linha
        b = achado.end() - inicio_linha
        toks = [(m.start(), m.end(), m.group(0)) for m in _TOK.finditer(linha)]
        indices = [k for k, tok in enumerate(toks) if tok[1] > a and tok[0] < b]
        if not indices:
            continue
        i, j = indices[0], indices[-1] + 1
        # Só expande quando o run foi CORTADO por uma palavra "fraca" da lista
        # negra (``ADMILSON BRASIL LUSTOSA FILHO``): o vizinho imediato precisa
        # ser ``fraco``. Sem isso, ``Atenciosamente Ismael…`` viraria nome.
        vizinho_esq = toks[i - 1][2] if i > 0 else None
        vizinho_dir = toks[j][2] if j < len(toks) else None
        if not (
            (vizinho_esq and _classe_run_capitalizado(vizinho_esq) == "fraco")
            or (vizinho_dir and _classe_run_capitalizado(vizinho_dir) == "fraco")
        ):
            continue
        while i > 0 and _classe_run_capitalizado(toks[i - 1][2]) in (
            "core",
            "fraco",
            "conector",
        ):
            i -= 1
        while j < len(toks) and _classe_run_capitalizado(toks[j][2]) in (
            "core",
            "fraco",
            "conector",
        ):
            j += 1
        candidato = _run_capitalizado_valido([tok[2] for tok in toks[i:j]])
        if candidato and len(candidato) > len(melhor):
            melhor = candidato
    return melhor


def extrair_nomes(texto: str) -> list[str]:
    """Extrai nomes de pessoas por contexto + lista negra (regras determinísticas)."""
    texto = _dehyphen(texto)
    linhas = texto.split("\n")
    tokens = _tokens(texto)
    nomes: list[str] = []

    def _registrar(nome: str | None) -> None:
        if (
            nome
            and 2 <= len(nome.split()) <= 9
            and not em_contexto_lugar(texto, nome)
        ):
            nomes.append(nome)

    cargo_forward = len(GATILHOS_FRENTE) - 1
    for indice, gatilho in enumerate(GATILHOS_FRENTE):
        for casamento in gatilho.finditer(texto):
            if indice == cargo_forward:
                # Cargo: o nome precisa estar colado (senão o gatilho pega nome
                # de outra oração e trunca a superfície).
                _registrar(_nome_colado_ao_cargo(tokens, texto, casamento.end()))
                continue
            janela = JANELAS_FRENTE[indice] if indice < len(JANELAS_FRENTE) else 10
            _registrar(_nome_apos(tokens, texto, casamento.end(), janela=janela))

    for gatilho in GATILHOS_TRAS:
        for casamento in gatilho.finditer(texto):
            if not _gatilho_adjacente(texto, tokens, casamento.start()):
                continue
            _registrar(_nome_antes(tokens, texto, casamento.start()))

    # ``ASSINAM``: todos os runs de nome no BLOCO de assinatura, que pode
    # atravessar a quebra de linha (um signatário fecha a linha e o outro abre a
    # seguinte). Uma janela de tokens cobre o bloco sem engolir o parágrafo todo.
    for casamento in _RE_ASSINAM.finditer(texto):
        inicio = None
        for indice, token in enumerate(tokens):
            if token[0] >= casamento.end():
                inicio = indice
                break
        if inicio is None:
            continue
        proximo = min(len(tokens), inicio + JANELA_ASSINAM)
        for nome in _runs_de_nomes(tokens, texto, inicio, proximo):
            _registrar(nome)

    for nome in _nomes_de_linha_nome(linhas):
        _registrar(nome)

    unicos: list[str] = []
    vistos: set[str] = set()
    for nome in nomes:
        # T025 item 3: expande o run truncado por palavra da lista negra no meio
        # (``ADMILSON BRASIL LUSTOSA FILHO`` não pode virar só ``LUSTOSA FILHO``).
        expandido = _expandir_run_capitalizado(texto, nome)
        chave = chave_nome(expandido)
        if chave and chave not in vistos:
            vistos.add(chave)
            unicos.append(expandido)
    return unicos


# --------------------------------------------------------------------------- #
# CPF / CNPJ                                                                  #
# --------------------------------------------------------------------------- #

#: CPF completo: ``000.000.000-00`` (tolera espaços em volta da pontuação).
_RE_CPF_COMPLETO = re.compile(r"\b\d{3}\s*\.\s*\d{3}\s*\.\s*\d{3}\s*-\s*\d{2}\b")
#: CNPJ completo: ``05.522.917/0001-70``.
_RE_CNPJ_COMPLETO = re.compile(
    r"\b\d{2}\s*\.\s*\d{3}\s*\.\s*\d{3}\s*/\s*\d{4}\s*-\s*\d{2}\b"
)
#: CPF mascarado em qualquer posição: ``***.382.303 -**``, ``048.215.***-**``,
#: ``***.567.**-87``, ``***.260.***-00``.
_RE_CPF_MASCARADO = re.compile(
    r"(?:\*{2,3}|\d{3})\s*\.\s*(?:\*{1,3}|\d{3})\s*\.\s*(?:\*{1,3}|\d{3})"
    r"\s*-?\s*(?:\*{1,3}|\d{2})"
)
#: CNPJ mascarado: ``**.554.***/****-**``.
_RE_CNPJ_MASCARADO = re.compile(
    r"(?:\*{2,3}|\d{2})\s*\.\s*(?:\*{2,3}|\d{3})\s*\.\s*(?:\*{2,3}|\d{3})"
    r"\s*/\s*(?:\*{3,4}|\d{4})\s*-\s*(?:\*{1,2}|\d{2})"
)
#: CPF quebrado: ``898.19187-20`` (grupos com nº de dígitos trocado).
_RE_CPF_QUEBRADO = re.compile(r"\b\d{3}\.\d{5}-\d{2}\b")
#: CPF quebrado com pontos no lugar do hífen: ``517.047.063.00`` / ``062.979.903.28``.
_RE_CPF_PONTOS = re.compile(r"\b\d{3}\.\d{3}\.\d{3}\.\d{2}\b")
#: CNPJ sem pontuação de grupo: ``19590049/0001-70``.
_RE_CNPJ_SEM_GRUPO = re.compile(r"\b\d{8}/\d{4}-\d{2}\b")
#: CNPJ com ponto a mais antes da barra: ``45.517.180./0001-11``.
_RE_CNPJ_PONTO_BARRA = re.compile(r"\b\d{2}\.\d{3}\.\d{3}\./\d{4}-\d{2}\b")
#: CNPJ quebrado com pontos entre raiz e filial: ``18.893.557.0001-64``.
_RE_CNPJ_PONTOS = re.compile(r"\b\d{2}\.\d{3}\.\d{3}\.\d{4}-\d{2}\b")

#: Qualquer CPF/CNPJ (usado como contexto da linha de tabela).
_RE_CPF_QUALQUER = re.compile(
    r"(?:\*{2,3}|\d{3})\s*\.\s*(?:\*{1,3}|\d{3})\s*\.\s*(?:\*{1,3}|\d{3})"
    r"\s*-?\s*(?:\*{1,3}|\d{2})"
    r"|\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"
    r"|\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b"
    r"|\b\d{3}\.\d{5}-\d{2}\b"
)

#: Padrões de CPF/CNPJ na ordem de extração: ``(tipo, regex, exige_asterisco)``.
_PADROES_CPF_CNPJ: tuple[tuple[TipoEntidade, re.Pattern[str], bool], ...] = (
    ("cnpj", _RE_CNPJ_COMPLETO, False),
    ("cnpj", _RE_CNPJ_PONTO_BARRA, False),
    ("cnpj", _RE_CNPJ_SEM_GRUPO, False),
    ("cnpj", _RE_CNPJ_PONTOS, False),
    ("cnpj", _RE_CNPJ_MASCARADO, True),
    ("cpf", _RE_CPF_COMPLETO, False),
    ("cpf", _RE_CPF_MASCARADO, True),
    ("cpf", _RE_CPF_QUEBRADO, False),
    ("cpf", _RE_CPF_PONTOS, False),
)


def extrair_cpf_cnpj(texto: str) -> list[tuple[TipoEntidade, str]]:
    """Extrai CPF/CNPJ nos formatos completo, mascarado e quebrado (D019)."""
    texto = _dehyphen(texto)
    achados: list[tuple[TipoEntidade, str]] = []
    for tipo, padrao, exige_asterisco in _PADROES_CPF_CNPJ:
        for casamento in padrao.finditer(texto):
            valor = re.sub(r"\s+", " ", casamento.group(0)).strip()
            if not valor or (exige_asterisco and "*" not in valor):
                continue
            achados.append((tipo, valor))

    unicos: list[tuple[TipoEntidade, str]] = []
    vistos: set[str] = set()
    for tipo, valor in achados:
        chave = f"{tipo}:{chave_cpf_cnpj(valor)}"
        if chave not in vistos:
            vistos.add(chave)
            unicos.append((tipo, valor))
    return unicos


def _spans_cpf_cnpj(texto: str) -> list[tuple[int, int]]:
    """Intervalos (start, end) de CPF/CNPJ — usados para não extraí-los como número."""
    spans: list[tuple[int, int]] = []
    for _, padrao, exige_asterisco in _PADROES_CPF_CNPJ:
        for casamento in padrao.finditer(texto):
            if exige_asterisco and "*" not in casamento.group(0):
                continue
            spans.append((casamento.start(), casamento.end()))
    spans.sort()
    fundidos: list[tuple[int, int]] = []
    for inicio, fim in spans:
        if fundidos and inicio <= fundidos[-1][1]:
            fundidos[-1] = (fundidos[-1][0], max(fundidos[-1][1], fim))
        else:
            fundidos.append((inicio, fim))
    return fundidos


# --------------------------------------------------------------------------- #
# Números citados (ato / processo / contrato)                                 #
# --------------------------------------------------------------------------- #

_RE_NUMERO_PROCESSO = re.compile(
    r"\d{2,6}(?:[.]\d{2,6})*/\d{4}(?:\s*-\s*\d{2}(?!\d))?"
)
#: Número único CNJ (``0800557-30.2024.8.18.0003``): é processo (D031).
#: O hífen pode quebrar em nova linha na diagramação (``0800557-\n30...``).
_RE_NUMERO_CNJ = re.compile(r"\b\d{7}-\s*\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b")
_RE_NUMERO_BARRA = re.compile(r"\b\d{1,6}/\d{4}(?:-\d{2})?\b")
_RE_NUMERO_MILHAR = re.compile(r"\b\d{1,3}(?:[.]\d{3})+(?:-\d{2})?\b(?!,\d)")
_RE_NUMERO_MARCADOR = re.compile(
    r"(?:n[ºo°.]*|numero|número)\s*([0-9][0-9.,/\-]*)", re.IGNORECASE
)
#: Data em qualquer grafia: não é número de ato (``04/02/2026``, ``23.12.2025``).
_RE_E_DATA = re.compile(
    r"^\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4}$"
    r"|^\d{4}\s*-\s*\d{2}\s*-\s*\d{2}$"
    r"|^\d{1,2}\s*[.]\s*\d{1,2}\s*[.]\s*\d{2,4}$"
)
#: Classificador do tipo do número pelo líder textual mais próximo (D027/D028).
#: Espelha o léxico do gate (D028): **contrato/instrumento** (contrato, ordem de
#: fornecimento, ata de registro de preços, apostilamento, termo aditivo, termo
#: de fomento, termo de colaboração, convênio) e **processo** (processo, SEI,
#: licitatório). Leis/decretos e demais números caem em ``ato_citado`` e não
#: entram no gate (D027).
_RE_CLASSIFICADOR_NUMERO = re.compile(
    r"(?P<contrato>contratos?|ordem\s+de\s+fornecimento|"
    r"ata\s+de\s+registro\s+de\s+pre[çc]os|apostilamento|termo\s+aditivo|"
    r"termo\s+de\s+fomento|termo\s+de\s+colabora[çc][ãa]o|conv[êe]nios?)"
    r"|(?P<processo>processos?|\bsei\b|licitat\w*)",
    re.IGNORECASE,
)
#: Contexto em que um número puro de 4–6 dígitos é ato/lei/decreto.
_RE_CONTEXTO_ATO = re.compile(r"\b(?:lei|decreto|complementar|portaria|resolu)\w*", re.I)
#: Norma CITADA como líder de número (``Lei 8.666/93``, ``Lei nº 14.133/2021``,
#: ``Decreto 24.759/2023``, ``Lei Complementar nº 101/2000``, T028 item 3): quando
#: é o líder mais próximo, o número é ``ato_citado`` — nunca contrato/processo,
#: mesmo com "contrato" antes na frase. ``\bleis?\b`` não casa ``leilão``.
_RE_LIDER_NORMA = re.compile(
    r"\b(?:leis?|decretos?(?:[\s-]+leis?)?|medidas?\s+provis[óo]rias?|"
    r"emendas?\s+constituciona(?:l|is)|instru[çc](?:[ãa]o|[õo]es)\s+normativas?|"
    r"resolu[çc](?:[ãa]o|[õo]es)|portarias?)\b",
    re.IGNORECASE,
)
#: Data ``dd/mm/aaaa`` ou ``dd/mm/aa`` (T028 item 1): o trecho inteiro é proibido
#: para números citados, senão ``02/2026`` de ``02/02/2026`` vira contrato.
_RE_DATA_BARRA = re.compile(
    r"(?<![\d./])(\d{1,2})[ \t]*/[ \t]*(\d{1,2})[ \t]*/[ \t]*(\d{4}|\d{2})(?!\d)"
)
#: Hífens Unicode tratados como ``-`` antes de casar identificadores (T028 item 2):
#: U+2010 hífen, U+2011 não-quebrável, U+2012 figura, U+2013 en, U+2014 em,
#: U+2212 menos. Troca 1:1 (preserva as posições).
_HIFENS_UNICODE = str.maketrans(dict.fromkeys("\u2010\u2011\u2012\u2013\u2014\u2212", "-"))


def _spans_datas(texto: str) -> list[tuple[int, int]]:
    """Intervalos de datas ``dd/mm/aaaa``/``dd/mm/aa`` válidas (dia 1–31, mês 1–12)."""
    return [
        (m.start(), m.end())
        for m in _RE_DATA_BARRA.finditer(texto)
        if 1 <= int(m.group(1)) <= 31 and 1 <= int(m.group(2)) <= 12
    ]


#: Distância máxima (caracteres) entre o líder e o número para que ele seja
#: considerado o introdutor daquele número (``CONTRATO Nº``, ``Processo
#: Administrativo nº``, ``Ordem de Fornecimento GEAFH-DAE``, ``CONVÊNIO DE
#: CESSÃO RECÍPROCA Nº``). Palavras de outra oração afastam o líder e o número
#: cai em ``ato_citado``.
GAP_LEADER_NUMERO = 45

#: Acima deste tamanho o documento é o "gigante" (D012): nos anexos só a coluna
#: NOME interessa — CPF/CNPJ e números citados não são extraídos (evita milhares
#: de entidades inúteis, ex.: 4197#23 com 6.646).
LIMITE_GIGANTE: int = 50_000


def _classificar_numero(texto: str, inicio: int, fim: int) -> str:
    """Tipo do número citado (D027/D028): ``processo``, ``contrato`` ou ``ato_citado``.

    Usa o líder textual **mais próximo antes** do número (``CONTRATO Nº 78/2026``,
    ``SEI nº 00045…``), aceitando até :data:`GAP_LEADER_NUMERO` caracteres de
    preenchimento (``nº``, qualificadores, código do órgão). Sem líder antes,
    ``ato_citado``. Léxico idêntico ao do gate (D028).
    """
    antes = texto[max(0, inicio - 90) : inicio]
    # Norma citada (``Lei nº 14.133/2021``) como líder mais próximo vence
    # qualquer líder de instrumento/processo anterior (T028 item 3).
    lideres = sorted(
        [*_RE_CLASSIFICADOR_NUMERO.finditer(antes), *_RE_LIDER_NORMA.finditer(antes)],
        key=lambda m: m.end(),
    )
    for casamento in reversed(lideres):
        preenchimento = antes[casamento.end() :]
        if len(preenchimento) <= GAP_LEADER_NUMERO and not re.search(
            # Fim de oração (``. Palavra``) ou parágrafo quebra o líder; ponto e
            # vírgula NÃO (é lista de números: ``Contratos nº 01/2026; 02/2026``).
            r"\.\s+[A-ZÀ-Þ]|\n\s*\n",
            preenchimento,
        ):
            if casamento.re is _RE_LIDER_NORMA:
                return "ato_citado"
            return casamento.lastgroup or "ato_citado"
    return "ato_citado"


def _formatar_milhar(digitos: str) -> str:
    """Forma canônica com separador de milhar (``6311`` → ``6.311``)."""
    return f"{int(digitos[:-3]):,}".replace(",", ".") + "." + digitos[-3:]


def extrair_numeros_citados(texto: str) -> list[tuple[str, str]]:
    """Extrai identificadores citados com o tipo do líder textual (D019/D027).

    Cada candidato é isolado na própria posição (sem partir nos ``.`` dos
    processos SEI). Datas e valores monetários são descartados pelo próprio
    formato; CPF/CNPJ nunca viram "número citado" (são entidades próprias).

    O tipo devolvido é ``processo``, ``contrato`` (instrumento) ou
    ``ato_citado`` (lei/decreto/portaria citados) — ver
    :func:`_classificar_numero` e D027.
    """
    texto = _dehyphen(texto).translate(_HIFENS_UNICODE)
    spans_proibidos = _spans_cpf_cnpj(texto) + _spans_datas(texto)
    brutos: list[tuple[int, int, str]] = []
    for padrao in (_RE_NUMERO_PROCESSO, _RE_NUMERO_CNJ, _RE_NUMERO_BARRA, _RE_NUMERO_MILHAR):
        brutos.extend((m.start(), m.end(), m.group(0)) for m in padrao.finditer(texto))
    for casamento in _RE_NUMERO_MARCADOR.finditer(texto):
        valor = casamento.group(1).strip(".,;:-/ ")
        if valor and re.search(r"\d", valor):
            brutos.append((casamento.start(1), casamento.start(1) + len(valor), valor))

    # Remove identificadores contidos em outro maior (``133/2021`` em ``14.133/2021``).
    brutos.sort(key=lambda item: (item[0], -item[1]))
    uteis: list[tuple[int, int, str]] = []
    for inicio, fim, valor in brutos:
        if any(inicio >= i and fim <= f for i, f, _ in uteis):
            continue
        uteis.append((inicio, fim, valor))

    achados: list[tuple[str, str]] = []
    for inicio, fim, valor in uteis:
        if any(inicio < f and fim > i for i, f in spans_proibidos):
            continue
        # Sufixo de processo pode vir separado por espaço (``/2022 -28``): a
        # superfície canônica remove os espaços.
        compacto = re.sub(r"\s+", "", valor)
        if _RE_E_DATA.match(compacto):
            continue
        if _RE_NUMERO_CNJ.fullmatch(compacto):
            tipo = "processo"  # número único CNJ = processo (D031)
        else:
            tipo = _classificar_numero(texto, inicio, fim)
        # ``LEI Nº 6311`` (sem milhar) → ``6.311``, a forma do ouro.
        if re.fullmatch(r"\d{4,6}", compacto) and _RE_CONTEXTO_ATO.search(
            texto[max(0, inicio - 80) : fim + 40]
        ):
            compacto = _formatar_milhar(compacto)
        achados.append((tipo, compacto))

    # Dedup por identificador normalizado, com PREFERÊNCIA pelo tipo mais
    # específico: o mesmo número pode aparecer 1º sem líder (``TO Nº 27/2022``
    # no início do corpo, quebra do título) e depois com líder (``CONTRATO Nº
    # 27/2022``); o tipo do líder é o correto (T022c).
    preferencia = {"contrato": 0, "processo": 0, "ato_citado": 1}
    melhor: dict[str, tuple[int, str, str]] = {}
    for tipo, valor in achados:
        chave = normalizar_numero_citado(valor)
        if not chave:
            continue
        atual = melhor.get(chave)
        if atual is None or preferencia.get(tipo, 1) < atual[0]:
            melhor[chave] = (preferencia.get(tipo, 1), tipo, valor)
    return [(tipo, valor) for _, tipo, valor in melhor.values()]


# --------------------------------------------------------------------------- #
# API pública                                                                 #
# --------------------------------------------------------------------------- #

_TIPO_ENTIDADE_NUMERO: dict[str, TipoEntidade] = {
    "ato_citado": "numero_ato",
    "processo": "processo",
    "contrato": "contrato",
}


def extrair_entidades(
    texto: str,
    *,
    pagina: int = 1,
    incluir_cpf: bool = True,
    incluir_numeros: bool = True,
) -> list[Entidade]:
    """Função pura: texto de um documento (corpo + tabelas) → entidades.

    ``pagina`` é a página atribuída às entidades (a do documento); o medidor não
    usa a página para P/R, mas o contrato exige o campo. ``incluir_cpf``/
    ``incluir_numeros`` permitem ao chamador desligar essas classes (usado no
    documento gigante, D012).
    """
    entidades: list[Entidade] = []
    vistos: set[tuple[str, str]] = set()

    def _adicionar(tipo: TipoEntidade, valor: str, normalizado: str) -> None:
        chave = (tipo, normalizado)
        if not valor or not normalizado or chave in vistos:
            return
        vistos.add(chave)
        entidades.append(
            Entidade(
                tipo=tipo, valor=valor, valor_normalizado=normalizado, pagina=pagina
            )
        )

    for nome in extrair_nomes(texto):
        _adicionar("nome", nome, chave_nome(nome))
    if incluir_cpf:
        for tipo, valor in extrair_cpf_cnpj(texto):
            _adicionar(tipo, valor, chave_cpf_cnpj(valor) or "")
    if incluir_numeros:
        for tipo_numero, valor in extrair_numeros_citados(texto):
            _adicionar(
                _TIPO_ENTIDADE_NUMERO[tipo_numero],
                valor,
                normalizar_numero_citado(valor),
            )
    return entidades


def entidades_do_documento(documento: Documento) -> list[Entidade]:
    """Extrai entidades do corpo + tabelas de um :class:`Documento`.

    No documento **gigante** (> :data:`LIMITE_GIGANTE`, D012) só a coluna NOME
    interessa: CPF/CNPJ e números citados das centenas de páginas de anexo são
    pulados (evitava milhares de entidades inúteis, ex.: 4197#23).
    """
    partes: list[str] = [documento.texto, *documento.tabelas]
    texto = "\n".join(parte for parte in partes if parte)
    gigante = len(texto) > LIMITE_GIGANTE
    return extrair_entidades(
        texto,
        pagina=documento.pagina_inicio,
        incluir_cpf=not gigante,
        incluir_numeros=not gigante,
    )


def com_entidades(documento: Documento) -> Documento:
    """Devolve cópia do documento com ``entidades`` preenchidas (helper de lote)."""
    return documento.model_copy(update={"entidades": entidades_do_documento(documento)})

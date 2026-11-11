# Golden set de avaliação do chat (`harness/eval`)

Documento de referência para o time usar **a mesma convenção** ao consumir o
golden set. O código de geração/verificação é `harness/eval/golden.py`.

## Convenção de `doc_id`

Todo alvo do golden é identificado por:

```
doc_id = "<arquivo_sem_extensao>#<doc_seq>"
```

- `arquivo_sem_extensao` — nome-base do PDF em `diarios/` (o diretório é
  descartado), ex.: `DOM4196-11022026-ASSINADO`.
- `doc_seq` — ordem do documento no PDF (1-based), exatamente como na anotação
  de referência (`dados/anotacao/DOM*.jsonl`).

Exemplos: `DOM4196-11022026-ASSINADO#1`, `DOM4198-13022026-ASSINADO#74`.

> Essa convenção é a mesma do `Documento.doc_id` do contrato
> (`harness/contracts.py`). O `id_dom` de 18 dígitos **não** entra no `doc_id`:
> ele é metadado opcional e pode até repetir (ver `knowledge/regras_diario.md`).

## Formato de cada caso (uma linha JSONL)

```json
{
  "id": "nome_unico-dev-001",
  "categoria": "nome_unico",
  "turnos": ["Quero o documento que cita FULANO DE TAL."],
  "esperado": {
    "doc_ids_alvo": ["DOM4196-11022026-ASSINADO#8"],
    "buscar": true,
    "perguntar": false,
    "opcoes_doc_ids": [],
    "posicao_resultado": null,
    "doc_ids_candidatos": []
  },
  "evidencia": {
    "arquivo": "DOM4196-11022026-ASSINADO.pdf",
    "pagina": 4,
    "trecho_literal": "FULANO DE TAL"
  }
}
```

- `turnos`: mensagens **do usuário**, em ordem. Cada categoria é dividida em
  `harness/eval/dev/golden_dev.jsonl` (dev) e
  `~/escavador_holdout/golden_holdout.jsonl` (holdout — **nunca** copiar para o
  repositório).
- `esperado.doc_ids_alvo`: doc_ids que contam como **acerto** (hit@5) na resposta
  final. Em `ambiguo` fica vazio de propósito (não há alvo único).
- `esperado.buscar`: `true` se a mensagem deve disparar busca; `false` só em
  `conversa_sem_busca`.
- `esperado.perguntar`: `true` quando a resposta correta é **desambiguar**; as
  opções reais estão em `opcoes_doc_ids` (comparar com
  `RespostaChat.pergunta.opcoes[].doc_id` e `precisa_esclarecimento`).
- `esperado.posicao_resultado` + `esperado.doc_ids_candidatos`: exclusivos de
  **follow-up por posição** (ver seção própria). Quando `posicao_resultado` é
  preenchido, `doc_ids_alvo` fica vazio e o alvo é a posição no ranking do turno
  anterior.
- `evidencia`: `null` apenas em `conversa_sem_busca` (não há documento). Nos
  demais, `arquivo`+`pagina`+`trecho_literal` **foram verificados contra o PDF**
  na geração.

## Categorias

| Categoria | Esperado | Observação |
| --- | --- | --- |
| `conversa_sem_busca` | não buscar | saudação, conceito, meta-pergunta; sem evidência |
| `nome_unico` | 1 pessoa | `doc_ids_alvo` = todos os docs anotados que citam o nome |
| `nomes_multiplos` | 2–3 pessoas **distintas** do mesmo doc | alvo = interseção dos docs que citam todas; acerto ≥1 no top-5; rejeita grafias do mesmo nome (`nomes_distintos`) |
| `id_dom` | busca por ID de 18 dígitos | com/sem `ID:` e com espaços |
| `numero` | tipo + número/ano ou processo | alvo **único** no corpus (checado por código); variações de escrita |
| `cpf_cnpj` | CPF/CNPJ com/sem pontuação | mistura mascarado, completo e formatos quebrados reais |
| `contexto` | tema em linguagem natural (paráfrase) | sem nomes/números e sem copiar 3-gramas do documento |
| `nome_com_erro` | nome real com 1–2 erros | evidência usa a **grafia real** do PDF |
| `ambiguo` | perguntar | `perguntar=true`, `doc_ids_alvo=[]`, `opcoes_doc_ids` reais |
| `follow_up` | 2–3 turnos | âncora por número/ID/nome/contexto; foco e **posição** |

### `numero` é sempre único (T009b.1)

O alvo de `numero` precisa ser **único no corpus para aquela consulta**. A
unicidade é conferida por código contra a anotação (`doc_com_numero_unico`):
a chave canônica ignora zeros à esquerda e sufixos (`055/2026/GAB/SEMEC` ≡
`55/2026`); quem não é único vai para `ambiguo`. Número **solto e sem ano**
(``ato-2``) é genérico demais e fica fora (ex.: "extrato número 2" não é
`numero`). As consultas trazem variações: com/sem "nº", zeros à esquerda,
sufixo de órgão presente/ausente, além de decreto/lei ("Lei 6.322").

### `contexto` é paráfrase, não cópia (T009b.2)

As consultas de contexto são **paráfrases em linguagem de cidadão**, verificadas
por código (`parafrase_valida`): **zero 3-gramas em comum** e no máximo **2
palavras de conteúdo** iguais às do documento — comparadas com o `enunciado` e o
`inicio_literal` de **todos** os membros do grupo (a checagem não cobre o corpo
inteiro, que não está na anotação). Ex.:

- ruim: "feriados e declara ponto facultativo nas datas que especifica no órgão CÂMARA"
- bom: "a câmara deu folga em algum feriado?"

Quando o tema legitimamente cobre vários documentos, **todos** entram em
`doc_ids_alvo` e o acerto é **pelo menos um alvo no top-5** (ver Métrica).

### `follow_up` por posição — ordem canônica (R009/B2)

Além de retomada por foco ("abre esse", "quem assina?"), há casos por **posição**
("e o segundo?", "abre o primeiro") após uma busca que retorna vários documentos.
A **ordem canônica de exibição é a do ranking devolvido pelo harness no turno
anterior** (`SaidaBusca.doc_ids()` → `EstadoConversa.ultimos_resultados`, 1-based;
`[0]` é o primeiro). O caso grava:

- `doc_ids_candidatos`: documentos que o turno 1 deveria exibir (auditoria e
  evita repetir doc-alvo entre dev e holdout);
- `posicao_resultado`: índice 1-based; o alvo do follow-up é o resultado que o
  **próprio sistema** exibiu nessa posição.

O runner **executa TODOS os `turnos` em ordem** (inclusive os intermediários de
casos de 3 turnos, como `[âncora, "me mostra o segundo", "quem assina?"]`) e
mantém o `EstadoConversa` a cada turno. A medição:

1. após o **turno 1**, fixa `alvo_pos = estado1.ultimos_resultados[posicao_resultado - 1].doc_id`
   (`estado1` lido de `EstadoConversa.ultimos_resultados`, não de `RespostaChat`);
2. roda os turnos restantes;
3. no **último turno**, apura o documento resolvido/citado (ver §Métrica);
4. acerto se `alvo_pos` for esse documento.

Falhas: se `len(estado1.ultimos_resultados) < posicao_resultado`, ou se nenhum
`doc_ids_candidatos` apareceu no turno 1, o erro é atribuído à **recuperação**
(não ao follow-up) e reportado separadamente.

## Verificação (regra (a) do T009)

O gabarito é gravado **na geração** a partir da anotação e **conferido por
código contra o texto do PDF**: o `trecho_literal` precisa existir na página
indicada. A comparação:

1. remove acentos e caixa (NFKD + casefold);
2. remove **hífen de quebra de linha** (`Es-\ntrutura` → `Estrutura`);
3. colapsa espaços/tabs/quebras em um espaço;
4. tolera hífens não mapeados (fallback) ao casar o trecho sem hífens.

Casos que não passam são **descartados** (o relatório da geração lista categoria
e motivo). O gabarito nunca é derivado da saída do sistema em execução.

Além da evidência, a geração valida por código: **unicidade de `numero`**
(`doc_com_numero_unico`), **ausência de cópia em `contexto`**
(`parafrase_valida`) e a **mistura de formatos de CPF/CNPJ**
(`formato_cpf_cnpj`).

## Divisão dev/holdout (regra (b)/(c) do T009)

- Estratificada por categoria: alvo de **10 dev + 10 holdout** por categoria.
- Dentro de cada categoria, evita que o mesmo **documento-alvo** apareça nos dois
  splits (união de `doc_ids_alvo` ∪ `opcoes_doc_ids` ∪ `doc_ids_candidatos`); o
  relatório traz `conflitos_doc_entre_splits` por categoria. Dois casos do
  **mesmo** split podem compartilhar documento — o que é proibido é o documento
  virar alvo nos DOIS splits (T009e: a colocação só é aceita se os docs do caso
  não colidem com o split oposto).
- Holdout em `~/escavador_holdout/golden_holdout.jsonl`, **fora do repositório**.

## Como regenerar

```bash
# grava dev no repo e holdout em ~/escavador_holdout/
uv run python -m harness.eval.golden

# só dev (sandbox/testes)
uv run python -m harness.eval.golden --dev /tmp/golden_dev.jsonl --sem-holdout

# testes
uv run pytest harness/eval/tests/test_golden.py
```

## Medição do baseline do `entender` (E001d)

A acurácia do decisor no golden dev (campos `tipo`, `modo_busca`, `suficiente`)
é medida por `harness/eval/medir_entender.py` — o código saiu de
`harness/search/entender.py`, que hoje é só o caminho de produção (sem CLI, sem
constante vinda do golden):

```bash
uv run python -m harness.eval.medir_entender
```

Referência no golden dev (100 casos): tipo 1.000 (100/100) · modo_busca 1.000
(100/100) · suficiente 0.990 (99/100), única discordância `contexto-dev-001`
(baseline do E001a/R033 — não é gate, só mede). O T031 repete o MESMO comando
para medir o laya (`definir_decisor(LayaDecisor())`).

## Métrica hit@5 (como o runner deve medir)

Para cada caso, alimentar **todos os `turnos` em ordem** e manter o
`EstadoConversa` a cada turno. A apuração usa o **último turno** e a noção de
"documento entregue", para não gerar falso negativo em retomadas sem nova busca
(R009/B1):

```
entregues = []
if estado_ultimo.documento_em_foco:                 # EstadoConversa
    entregues.append(documento_em_foco.doc_id)
entregues += [c.doc_id for c in resposta_ultima.citacoes]
if ultimo_turno_disparou_busca:                     # intenção busca/follow_up com busca
    entregues += resposta_ultima.doc_ids_recuperados[:5]
```

- `conversa_sem_busca`: acerto se nenhuma busca foi disparada
  (`intencao == "conversa"` e sem `doc_ids_recuperados`).
- `nome_unico`, `nomes_multiplos`, `id_dom`, `numero`, `cpf_cnpj`,
  `nome_com_erro`, `contexto`: acerto se
  `set(entregues) ∩ esperado.doc_ids_alvo ≠ ∅`.
  Como `doc_ids_alvo` pode ter vários documentos (ex.: `contexto` de um tema),
  o acerto é **pelo menos um alvo no top-5**.
- `ambiguo`: acerto se `resposta.precisa_esclarecimento` e
  `{opcao.doc_id for opcao in resposta.pergunta.opcoes
    if opcao.doc_id is not None} == set(esperado.opcoes_doc_ids)` (O1: ignora
  opções sem `doc_id`).
- `follow_up` por conteúdo/foco: acerto se `entregues` contém o `doc_ids_alvo`
  (o alvo é o documento resolvido/citado no último turno; `doc_ids_recuperados`
  entra apenas quando o último turno realmente dispara busca).
- `follow_up` por posição (`posicao_resultado`): `alvo_pos` fixado no turno 1
  (ver seção própria) e acerto se `alvo_pos ∈ entregues` no último turno.

`RespostaChat.doc_ids_recuperados` preserva a ordem do ranking recebido
(`SaidaBusca.doc_ids()`), logo o corte é literalmente nos 5 primeiros. O
`documento_em_foco` vive em `EstadoConversa`, por isso o runner precisa manter o
estado da conversa (não basta olhar a `RespostaChat`).

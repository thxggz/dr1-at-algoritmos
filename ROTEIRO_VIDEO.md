# Roteiro do vídeo — DR1_AT

**Duração alvo:** 7 minutos (a faixa pedida é 5 a 8).
**Ferramenta:** Loom ou OBS, tela + webcam ligada.
**Entrega:** upload no Google Drive da conta Infnet, permissão "qualquer pessoa
com o link", e envia o link.

---

## O que o enunciado exige obrigatoriamente

| # | Exigência | Onde está neste roteiro |
|---|---|---|
| 1 | Um exemplo de instrumentação e como sustenta a análise | **Bloco 2** (1:00–2:45) |
| 2 | Um caso de borda tratado | **Bloco 3** (2:45–4:30) |
| 3 | Uma justificativa técnica de otimização | **Bloco 4** (4:30–6:15) |

Os três blocos centrais são os que valem nota. Abertura e fechamento existem só
para emoldurar.

---

## Antes de gravar — checklist de 3 minutos

- [ ] `pytest -q` rodado, com o resultado visível na tela (860 testes passando)
- [ ] notebook **já executado**, aberto no exercício 2 (é onde o vídeo começa)
- [ ] VS Code com fonte grande (Ctrl+`+` umas 3 vezes) — o professor vai ver isso
      numa tela pequena
- [ ] webcam enquadrada, luz de frente, sem contraluz
- [ ] abas abertas na ordem: notebook → `src/counters.py` → `src/hashtable.py` →
      `src/search_engine.py`
- [ ] celular no silencioso, notificações do sistema desligadas

> **Regra do vídeo:** mostrar **número**, não tela. Toda vez que aparecer uma
> tabela, ler em voz alta a coluna que prova o ponto. Rolar código em silêncio é
> o que faz um vídeo desses perder nota.

---

## Bloco 1 — Abertura (0:00 – 1:00)

**O que mostrar:** a estrutura do projeto no explorador de arquivos do VS Code.

**O que dizer:**

> "Henrique Goldstein Maluhy Mendes Amaral, DR1_AT de Algoritmos e Estruturas de Dados. Os doze
> exercícios estão implementados em `src/`, um módulo por assunto, com 860 testes
> em `tests/`. O notebook não define nenhuma estrutura de dados: ele importa de
> `src/` e demonstra.
>
> A regra que eu segui no trabalho inteiro é esta: **nenhuma afirmação sobre Big O
> entra no notebook sem uma tabela de contagens que a sustente.** Nunca 'é O(n
> log n)' sozinho; sempre 'é O(n log n), e a razão medida ficou entre tanto e
> tanto enquanto n cresceu tantas vezes'.
>
> Vou mostrar três coisas em sete minutos: como a instrumentação funciona, um
> caso de borda que eu tratei, e uma otimização com o ganho medido."

**Tempo:** 1 minuto. Não passar disso.

---

## Bloco 2 — Instrumentação (1:00 – 2:45) ⭐ EXIGÊNCIA 1

**O que mostrar:** `src/counters.py`, depois a tabela do **exercício 2** no
notebook (seção "TRÊS ALGORITMOS × QUATRO PADRÕES").

**O que dizer:**

> "Existe um único contador no projeto, em `src/counters.py`, e os doze exercícios
> usam ele. Quatro campos: comparações de chaves, cópias, chamadas recursivas e
> saltos de ponteiro.
>
> Os três primeiros são o que o enunciado pede. O quarto, `hops`, eu acrescentei —
> e o motivo está no exercício 10: com ponteiro `tail`, `insert_last` faz zero
> comparações e uma cópia, **exatamente igual** à versão sem `tail`. Os contadores
> obrigatórios são cegos para a diferença entre O(1) e O(n) ali. Sem `hops`, a
> evidência não existiria.
>
> E o critério é fixo: **uma troca conta 3 cópias**, um deslocamento conta 1.
> Olha o que isso revela."

**[trocar para o notebook, exercício 2, linha do vetor reverso com n=4.000]**

> "Vetor reverso, quatro mil elementos. Os três algoritmos fazem **exatamente as
> mesmas 7.998.000 comparações** — é o pior caso dos três, n vezes n menos um
> sobre dois.
>
> Mas olha a coluna de cópias: bubble faz 23 milhões e 994 mil; insertion faz 8
> milhões; selection faz **seis mil**. Bubble troca, e cada troca são 3 cópias.
> Insertion desloca, 1 cópia cada. Selection faz no máximo n/2 trocas.
>
> Se eu tivesse instrumentado só comparações, os três pareceriam idênticos no pior
> caso. **É a segunda coluna que separa quem troca de quem desloca** — e é por
> isso que o enunciado pede as duas."

**[rolar até a tabela de sensibilidade: comparações × inversões]**

> "E a instrumentação sustenta uma afirmação exata. A pergunta do enunciado é quem
> mais se beneficia de 'quase ordenado'. A resposta é o insertion, e o mecanismo é
> este: cada comparação bem-sucedida do laço interno desfaz exatamente uma
> inversão. Então o custo é Θ(n + d), com d = número de inversões.
>
> A razão `comparações / (n + d)` medida ficou em **0,9995 nos oito níveis de
> desordem**, com d indo de zero a um milhão. Não é 'aproximadamente linear': é a
> fórmula batendo. E o selection fica em 1.999.000 em todos eles — ele varre o
> sufixo inteiro para achar cada mínimo, então nenhuma ordem prévia ajuda."

**Tempo:** 1 min 45. Se estourar, cortar o parágrafo da sensibilidade.

---

## Bloco 3 — Caso de borda (2:45 – 4:30) ⭐ EXIGÊNCIA 2

**O que mostrar:** o exercício 4 do notebook, tabela "CENÁRIO ADVERSARIAL ×
BENIGNO", e depois a célula que imprime os buckets por capacidade.

**O que dizer:**

> "O caso de borda que eu escolhi é o da hashtable, porque ele desmonta uma frase
> que todo mundo repete: 'hashtable é O(1)'.
>
> É O(1) **em média, condicionado a o hash espalhar**. Eu quebrei essa condição de
> propósito."

**[mostrar a tabela adversarial]**

> "Duas tabelas com a mesma configuração — mesma função de hash, mesmo limiar de
> carga, redimensionamento ligado. A diferença são as chaves.
>
> No cenário benigno, 400 chaves ocupam 332 buckets e cada busca custa 1,08
> comparação. No adversarial, as mesmas 400 chaves ocupam **um bucket** e cada
> busca custa **400 comparações**. A razão comparações sobre n dá exatamente
> 1,0000 — é a assinatura de Θ(n)."

**[mostrar a célula com os buckets por capacidade]**

> "E aqui está o detalhe que fecha o argumento. A tabela **cresceu**: ela começou
> com 8 buckets e foi para 1.024 durante o ataque. Não adiantou.
>
> As chaves foram escolhidas com hash congruente módulo uma potência de dois. Se
> dois números são congruentes módulo 2 elevado a k, eles também são congruentes
> módulo qualquer potência menor. Então elas ficam no mesmo bucket em 8, em 16, em
> 64, em 256 e em 1.024. **Crescer não defende.**
>
> E eu errei isso na primeira vez: eu consultava uma chave ausente qualquer, que
> caía num bucket vazio e media zero comparações. O modelo de ataque correto é o
> adversário controlar **também a consulta** — é assim que um hash flooding
> funciona. Corrigido, o número saiu exato."

**[opcional, se sobrar tempo: mostrar `find_colliding_keys` em `src/hashtable.py`]**

**Tempo:** 1 min 45.

---

## Bloco 4 — Justificativa técnica de otimização (4:30 – 6:15) ⭐ EXIGÊNCIA 3

**O que mostrar:** o exercício 12 do notebook, seção "OTIMIZAÇÃO 1", e o gráfico
log-log.

**O que dizer:**

> "A otimização é do exercício 12, o motor de busca. Uma consulta `a AND b` precisa
> da interseção das listas de documentos dos dois termos.
>
> O jeito ingênuo é comparar cada elemento de uma lista com todos os da outra:
> Θ(m·n). O jeito bom aproveita que as listas estão **ordenadas** e anda um
> ponteiro de cada vez: Θ(m + n).
>
> E o ponto que eu quero fazer é sobre **de onde vem a ordenação**. Ela não custa
> nada. Os documentos são indexados em ordem crescente de id, e cada ocorrência é
> anexada ao fim da lista com `insert_last` — que é O(1) porque a lista encadeada
> do exercício 10 tem ponteiro `tail`. Uma decisão tomada no exercício 10 é o que
> habilita a otimização no exercício 12."

**[mostrar a tabela e o gráfico]**

> "Os números: com listas de 100 elementos, a ordenada faz 366 comparações e a
> ingênua faz 9.406 — 26 vezes. Com listas de 3.200, a diferença vai para **mais
> de 1.600 vezes**.
>
> E o que prova a classe não é a razão entre as duas, é a estabilidade de cada
> uma: a razão da ordenada contra n fica presa em 3,69 nas quatro escalas, e a da
> ingênua contra n ao quadrado fica em 0,95. Uma é linear, a outra é quadrática.
> Por isso o ganho **cresce com n**: é ganho de classe, não de constante.
>
> Uma coisa que eu fiz questão de garantir: há teste provando que as duas versões
> devolvem **exatamente o mesmo resultado** em cem pares de listas sorteadas. Uma
> otimização que muda o resultado não é otimização, é bug."

**Tempo:** 1 min 45.

---

## Bloco 5 — Fechamento (6:15 – 7:00)

**O que mostrar:** o terminal com `pytest -q` e a tabela de rastreabilidade no
fim do notebook.

**O que dizer:**

> "Fechando: 860 testes passando. E vários deles testam algo que o notebook não
> mostra — que `check_invariants` **falha** quando eu corrompo a estrutura de
> propósito. Sem isso, 'a invariante passou' não provaria nada: um método vazio
> também passaria.
>
> A última seção do notebook mapeia os vinte itens da rubrica para a seção onde
> cada um está demonstrado.
>
> E o que eu levo deste trabalho são os casos em que a medição contrariou a
> intuição, e que eu deixei registrados em vez de esconder: a deduplicação lenta é
> Θ(n·u) e não Θ(n²), então com 5% de valores distintos ela parece aceitável; a
> BST degenerada por entrada crescente responde `k_smallest(1)` em zero saltos
> enquanto a degenerada por entrada decrescente gasta n−1, ou seja 'a BST
> degenerou' não é conclusão suficiente; e o quicksort com pivô ingênuo em entrada
> ordenada é Θ(n²) em tempo e **O(log n) em memória** ao mesmo tempo.
>
> Obrigado."

**Tempo:** 45 segundos.

---

## Plano B se o tempo estourar

Cortar, nesta ordem:

1. o parágrafo da sensibilidade no Bloco 2 (economiza ~25 s);
2. o `find_colliding_keys` opcional no Bloco 3 (~20 s);
3. os exemplos do fechamento, deixando só um (~20 s).

**Não cortar:** nenhum dos três blocos obrigatórios, e nenhuma leitura de número
em voz alta. É o número que vale.

---

## Erros a evitar

- **Rolar código em silêncio.** Se aparece na tela, tem de ser dito por quê.
- **Ler o docstring inteiro.** Dizer a conclusão, não recitar.
- **Explicar o que é uma hashtable.** O professor sabe. Ir direto ao que foi
  medido.
- **Pedir desculpa por algo.** Se uma decisão tem trade-off, **declarar o
  trade-off** — isso é a competência sendo avaliada, não uma falha.
- **Passar de 8 minutos.** É limite, não sugestão.

---

## Se o professor perguntar algo fora dos 5 blocos

**"Cadê a deleção da BST?"** — exercício 2, seção *"Deleção na BST — os três
casos estruturais"*. Resposta curta:

> "Nenhum dos doze exercícios pede remoção, mas o item 4.4 da rubrica cobra
> deleção mantendo invariantes, então eu demonstrei os três casos ali. Folha: o
> pai passa a apontar `None`. Um filho: o pai pula direto para o neto. Dois
> filhos: o **sucessor in-order** — a menor chave da subárvore direita — assume a
> vaga, porque é por definição a menor chave maior que a removida, então cabe sem
> violar a ordenação; e como ele tem no máximo um filho, removê-lo recai no caso
> anterior.
>
> E os asserts conferem o **ponteiro**, não só a travessia. Uma implementação que
> reconstruísse a árvore do zero devolveria a mesma in-order ordenada e estaria
> errada."

**"Como você sabe que o PDF está completo?"** — `python verificar_entrega.py`.
São 8 conferências, e uma delas existe porque o PDF estava cortando o lado
direito das tabelas largas em silêncio: ela confere que o final das 285 linhas
mais largas do notebook sobreviveu à impressão.

---

## Números para ter na ponta da língua

| Onde | Número |
|---|---|
| testes | 860 passando |
| ex 2, reverso n=4.000 | mesmas 7.998.000 comparações; cópias 23.994.000 / 8.005.998 / 6.000 |
| ex 2, insertion | razão comparações/(n+d) = **0,9995** em 8 níveis de desordem |
| ex 4, adversarial n=400 | 1 bucket de 1.024 ocupado; 400 comparações por busca; razão/n = 1,0000 |
| ex 12, otimização 1 | 26× → **1.642×**; ordenada/n = 3,69 estável; ingênua/n² = 0,95 estável |
| ex 10, ponteiro `tail` | 1.328× no tempo; saltos/n² → 0,4998 |
| ex 11, ponteiro `prev` | 1.489× no tempo; 0 saltos contra n(n−1)/2 |
| ex 8, memoization | 142.129 → 68 chamadas em amount=24 (**2.090×**) |
| ex 2, deleção na BST | 3 casos: folha · um filho · dois filhos com **sucessor in-order** |
| entrega | 64 páginas, 17 marcadores, 20 gráficos, 8 conferências automáticas |

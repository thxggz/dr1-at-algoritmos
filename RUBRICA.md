# Rastreabilidade da rubrica — TP DR1_AT

Os 20 itens da rubrica oficial, transcritos literalmente, mapeados para onde cada um é
demonstrado. Este arquivo é o documento de trabalho; a tabela que o professor lê é a
`RASTREABILIDADE` no fim do notebook, e `verificar_entrega.py` confere que a evidência
citada aqui existe mesmo no PDF entregue.

Status: ✅ pronto e testado · 🟡 parcial · ⬜ pendente

**Os 20 itens estão ✅.**

---

## Competência 1 — Avaliar a velocidade de algoritmos introdutórios usando a notação Big O

| # | Item da rubrica | Onde | Evidência concreta | Status |
|---|---|---|---|---|
| 1.1 | Implementar busca linear e busca binária com retorno correto e **tratamento de casos não encontrados** | Ex 1 | `linear_search`/`binary_search` devolvendo índice ou `-1`; 5 escalas de 10² a 10⁶ nos 3 cenários; pré-condição verificada em 3 modos, com `UnsortedInputError` em vez de `-1` | ✅ |
| 1.2 | Instrumentar algoritmos para contar **comparações e cópias** e registrar evidências de execução no notebook | Ex 1, 2, 3, 9 (base: `src/counters.py`) | Ex 2: no reverso os três fazem 7.998.000 comparações e 23.994.000 / 8.005.998 / 6.000 cópias — só a coluna de cópias distingue trocar de deslocar. 94 células de código executadas no notebook, 35 tabelas e 20 gráficos log-log | ✅ |
| 1.3 | Relacionar resultados instrumentados com Big O, **distinguindo melhor, médio e pior caso** | Ex 1, 2, 10 | Ex 1: linear 1 / 0,5n / n e binária 1 / ~log₂n / ⌊log₂n⌋+1, com razão contra a curva teórica em 5 escalas; ex 10 idem para lista | ✅ |
| 1.4 | **Otimizar** uma solução de maior custo assintótico para uma de menor, mantendo corretude por testes | Ex 3 | `deduplicate_slow` Θ(n·u) → `deduplicate_fast` Θ(n) e `k_smallest` A Θ(n log n) → B Θ(n+k log k); equivalência provada contra modelo de referência em 200+120 entradas sorteadas | ✅ |
| 1.5 | **Justificar** decisões técnicas de otimização com trade-offs, restrições e evidências | Ex 11, 9, 7, 12 | Ex 11: `prev` — 0 saltos vs n(n−1)/2. Ex 9: mediana de três e recursão na metade menor. Ex 7: filhos em lista vs hashtable. Ex 12: interseção Θ(m+n) (1.642×) e poda por comprimento (2,1×), as duas provadas exatas por teste | ✅ |

## Competência 2 — Utilizar as estruturas de dados hash tables, filas (queues) e pilhas (stacks)

| # | Item da rubrica | Onde | Evidência concreta | Status |
|---|---|---|---|---|
| 2.1 | Implementar **hashtable com colisões por encadeamento** e `put`, `get`, `delete` corretas | Ex 4 | `HashTableChained`; bucket = `HashBucket(SinglyLinkedList)` do ex 10; rehash em 0,75; medido/teórico de `1+α/2` entre 0,99 e 1,23 com α crescendo 32× | ✅ |
| 2.2 | Implementar **pilha e fila com invariantes claros**, incluindo overflow e underflow explícitos | Ex 5 | `Stack`/`Queue` de capacidade fixa; 5 exceções próprias por estrutura; 10.000 operações num array de 8 posições com 8.743 voltas e FIFO preservado | ✅ |
| 2.3 | Aplicar pilha em **parsing e avaliação de expressão**, com detecção de entradas inválidas | Ex 5, 12 | Ex 5: árvore de expressão de pós-fixa, 5 classes de entrada inválida com a posição do token. Ex 12: shunting-yard com parênteses e NOT unário associativo à direita | ✅ |
| 2.4 | Aplicar filas em um **simulador de eventos**, mantendo rastreabilidade e consistência | Ex 6 | 4 filas A–D com a `Queue` do ex 5; snapshot em não alfabético; overflow/underflow com contexto e simulação continua; índice em BST com chave `ClientId` (`B2 < B17`) | ✅ |
| 2.5 | **Documentar e testar** estruturas de dados com casos de borda, garantindo robustez | Ex 4, 5, 10, 11 | Ex 11: `check_invariants` testado falhando em estrutura corrompida de propósito; 106 remoções em deque vazio. Ex 4: cenário adversarial, tipos de chave rejeitados, rehash validado. Ex 5: overflow/underflow numa sequência de 5.000 operações, expressão inválida, divisão por zero | ✅ |

## Competência 3 — Desenvolver algoritmos avançados usando recursão e programação dinâmica

| # | Item da rubrica | Onde | Evidência concreta | Status |
|---|---|---|---|---|
| 3.1 | **Travessia recursiva** correta sobre estrutura hierárquica, com base cases e composição | Ex 7 | `walk(root)` recursivo em pré-ordem com caso base explícito, verificado contra `walk_iterative` que produz a mesma lista | ✅ |
| 3.2 | Identificar **sobreposição de subproblemas** e aplicar memoization para reduzir chamadas | Ex 8 | `min_coins` puro (142.129 chamadas em a=24) vs memoizado (68); árvore de subproblemas com 9 folhas `memo` mostrando onde a poda agiu | ✅ |
| 3.3 | Implementar **QuickSort e QuickSelect** instrumentados e analisar o impacto dos padrões de entrada | Ex 9 | 4 padrões × 3 `short` × 3 estratégias de pivô; `last`+ordenado = n(n−1)/2 e `median3` na mesma entrada = melhor caso; quickselect Θ(n) vs Θ(n log n) | ✅ |
| 3.4 | Expressar **custos de recursão** em Big O e discutir profundidade e consumo de memória | Ex 5, 7, 9 | Ex 5: mesmos 25.601 `calls` nas duas versões, recursiva quebra em altura 3.201. Ex 9: Θ(n²) em tempo com O(log n) de pilha. Ex 7: `walk` recursivo quebra em 3.000 níveis, iterativo não | ✅ |
| 3.5 | **Justificar a escolha de DP** em módulo aplicado, com **evidências quantitativas de ganho** | Ex 8, 12 | Ex 8: redução de 2.090× em a=24 e as 3 granularidades de chave medidas. Ex 12: distância de edição memoizada, 28.531× menos chamadas | ✅ |

## Competência 4 — Utilizar as estruturas de dados listas encadeadas (linked lists) e árvores (trees)

| # | Item da rubrica | Onde | Evidência concreta | Status |
|---|---|---|---|---|
| 4.1 | Implementar **lista encadeada** com leitura, busca, inserção e deleção, tratando casos de borda | Ex 10 | `SinglyLinkedList` completa; exceções próprias; 133 testes | ✅ |
| 4.2 | **Mensurar** a eficiência de operações em lista encadeada e **comparar com array** | Ex 1, 10 | Ex 1: mesmas comparações nas duas, 0 vs n saltos, e 1,8× mais nanossegundos por comparação na lista; Ex 10: `bench_insert_last` e `bench_search` | ✅ |
| 4.3 | Implementar **lista duplamente encadeada** e **deque O(1) nas duas pontas** | Ex 11 | `hops = 0` em todas as operações; µs/op estável (≤1,32× com n crescendo 16×) | ✅ |
| 4.4 | Implementar **BST com inserção, busca e deleção**, mantendo invariantes estruturais | Ex 2, seção "Deleção na BST — os três casos estruturais"; usada em 3, 6, 8, 12 | `delete` nos 3 casos verificados **estruturalmente** (folha, 1 filho, 2 filhos via sucessor in-order); `check_invariants` com limites herdados detecta violação global que a comparação local deixa passar | ✅ |
| 4.5 | **Integrar árvores e listas** em sistema maior, documentando decisões, validando invariantes e com **relatório técnico de desempenho** | Ex 12 | 8 estruturas integradas (teste verifica os tipos); `check_invariants` do índice; `performance_report` separa parsing/avaliação/ranqueamento/sugestão; 2 otimizações medidas (1.642× e 2,1×) | ✅ |

---

## Alertas encontrados na leitura do enunciado oficial

1. **Item 4.4 cobra `delete` na BST, e nenhum exercício pede `delete` no texto da tarefa.**
   O ex 2 pede inserir + in-order; o ex 3, `k_smallest_bst`; o ex 6, busca e listagem; o
   ex 12, listagem ordenada. Nenhum deles remove nada. Se a BST sair sem `delete`, o item
   se perde sem que nada "quebre". Decisão: `src/bst.py` implementa `delete` completo e o
   notebook demonstra os três casos explicitamente na seção do ex 2.

2. **Item 4.5 cobra "relatório técnico de desempenho"**, não apenas análise em Big O solta.
   O ex 12 termina com uma seção nomeada assim, consolidando custo de indexação, consulta,
   ranqueamento e sugestão, mais as duas otimizações com evidência.

3. **Item 1.2 cobra "comparações e cópias"** em conjunto. Buscas contam comparações; para
   as buscas do ex 1 a coluna de cópias fica em 0, e isso deve ser dito explicitamente no
   texto em vez de a coluna sumir — coluna ausente parece instrumentação faltando.

4. **Ex 1 pede escalas até 10⁶.** Em array isso é tranquilo. A variante em
   `SinglyLinkedList` a 10⁶ custa ~56 MB só de nós; a comparação array↔lista roda em
   escala menor, com o motivo declarado no notebook (o número de comparações é idêntico
   por construção; o que muda é o custo de travessia).

5. **O PDF cortava as tabelas largas em silêncio.** O Chrome headless imprime sem
   quebrar linha em bloco de código, e o Jupyter marca as áreas de saída com
   `overflow: auto`, que na impressão vira corte. As colunas perdidas eram as de
   razão contra a curva teórica — a prova de cada afirmação de Big O. Corrigido no
   `build_pdf.py` (folha em paisagem, 7pt, `overflow: visible`) e guardado por
   `verificar_entrega.py`, que confere as 285 linhas mais largas a cada geração.

6. **Operacional:** 2 tentativas permitidas, envio a partir de 22/09/2026 07h00. O vídeo
   sobe no Google Drive da conta Infnet com permissão "qualquer pessoa com o link".

# TP DR1_AT — Algoritmos e Estruturas de Dados (Infnet)

Avaliação individual. 12 exercícios de algoritmos e estruturas de dados.
Entrega: 26/09/2026 23h59. Meta interna: concluir em 3 dias.

## 1. Entregáveis
- `Henrique_Goldstein_DR1_AT.pdf` — evidências organizadas por exercício (1 a 12), títulos visíveis.
- `Henrique_Goldstein_DR1_AT.mp4` — vídeo 5–8 min, tela + webcam.

Operacional (do enunciado oficial): envios aceitos a partir de 22/09/2026 07h00, prazo
26/09/2026 23h59, **2 tentativas permitidas**. O vídeo vai para o Google Drive da conta
Infnet com permissão "qualquer pessoa com o link", e entrega-se o link.

O PDF deve permitir verificar rapidamente: (a) algoritmos implementados como pedido, (b) execuções correspondendo às tarefas, (c) resultados coerentes, (d) fluxo lógico seguido.

O vídeo cobre obrigatoriamente: (1) um exemplo de instrumentação e como sustenta a análise; (2) um caso de borda tratado; (3) uma justificativa técnica de otimização.

## 2. Princípio que rege o projeto
A nota não vem de "funcionar", vem da coerência entre implementação, medição e análise em Big O. A rubrica tem 20 itens e vários são sobre evidência: "registrar evidências de execução", "distinguir melhor/médio/pior caso", "evidências quantitativas de ganho", "documentar decisões e validar invariantes".

Regra dura: nenhuma afirmação assintótica entra no notebook sem tabela de contagens que a sustente. Nunca "é O(n log n)" sozinho; sempre "é O(n log n), e a razão comparações/(n·log₂n) medida ficou entre 1,09 e 1,24 enquanto n cresceu 32×".

## 3. Estrutura do repositório

notebook/DR1_AT.ipynb # notebook principal, executado ponta a ponta
src/counters.py # instrumentação compartilhada
src/linked_list.py # SinglyLinkedList (ex 1,9,10,12)
src/doubly_linked.py # DoublyLinkedList + Deque (ex 11)
src/stack_queue.py # Stack e Queue capacidade fixa (ex 5,6,12)
src/bst.py # BST completa (ex 2,3,6,8,12)
src/hashtable.py # HashTableChained (ex 4,8,12)
src/searching.py # ex 1
src/sorting.py # ex 2 e 9
src/selection.py # ex 3
src/simulator.py # ex 6
src/dirtree.py # ex 7
src/dp.py # ex 8
src/search_engine.py # ex 12
tests/test_*.py # pytest por módulo
figures/ # gráficos png
build_pdf.py # notebook executado -> HTML -> PDF (CSS de impressão + marcadores)
verificar_entrega.py # 8 conferências antes de enviar
ROTEIRO_VIDEO.md # gerado no dia 3

O código real vive em `src/`; o notebook importa e demonstra. Notebook com classes coladas dentro fica ilegível e impossível de defender no vídeo.

## 4. Núcleo reutilizável
Os exercícios NÃO são independentes: o ex 12 exige por enunciado hashtable + pilha + fila + recursão + DP + ordenação eficiente + lista encadeada + árvore. Cada estrutura é definida UMA vez e reutilizada:

| Estrutura | Definida em | Consumida por |
|---|---|---|
| Counters | seção 0 | todos |
| SinglyLinkedList | seção 0 / ex 10 | 1,9,10,12 |
| Stack, Queue | ex 5 | 5,6,12 |
| HashTableChained | ex 4 | 4,8,12 |
| BST | ex 2 | 2,3,6,8,12 |
| DoublyLinkedList, Deque | ex 11 | 11 |
| quicksort, quickselect | ex 9 | 3,9,12 |

Ordem de DESENVOLVIMENTO: 0 → 10 → 11 → 4 → 5 → 1 → 2 → 3 → 9 → 6 → 7 → 8 → 12.
Ordem de APRESENTAÇÃO no PDF: 1 a 12, conforme o enunciado.

## 5. Padrão de instrumentação (uniforme e obrigatório)
```python
@dataclass
class Counters:
    comparisons: int = 0   # comparações de chaves
    copies: int = 0        # movimentações (1 troca = 3 cópias)
    calls: int = 0         # chamadas recursivas
```
- Toda função instrumentada retorna `(resultado, Counters)`.
- Uma troca conta como 3 cópias — critério fixo do enunciado.
- `random.seed(42)` em todos os experimentos: números do notebook, PDF e vídeo idênticos.
- Toda medição vira linha de DataFrame com n, padrão de entrada, comparações, cópias, chamadas e a razão contra a curva teórica (c/n, c/(n·log₂n), c/n²). Razão ~constante é a prova da classe.
- Gráfico log-log quando houver ≥4 pontos.

## 6. O que cada exercício exige

**Ex 1 — Buscas.** `linear_search(arr,target)` e `binary_search(sorted_arr,target)` retornando posição ou −1, contando comparações. Gerador de vetores em 3 cenários (ordenado, reverso, aleatório), 5 escalas de 10² a 10⁶. Verificação explícita de pré-condição de ordenação na binária, com comportamento definido quando falha. Estratégia de falha rápida mais barata que ordenar: amostragem de k pares adjacentes, O(k) vs O(n log n), com discussão honesta do risco de falso positivo. Reescrever busca linear sobre SinglyLinkedList e comparar contagens com array: mesmo nº de comparações, custo prático diferente (localidade de cache, ponteiro por nó).

**Ex 2 — Ordenações quadráticas.** bubble/selection/insertion retornando comparações e cópias. 4 padrões (ordenado, reverso, quase ordenado, aleatório) × ≥4 tamanhos. TETO DE 4.000 elementos (10⁶ em bubble levaria dias). Identificar quem mais se beneficia de "quase ordenado" (insertion, O(n+d) com d inversões) e por quê. Depois: inserir tudo na BST, in-order, verificar ordenação; comparar custo BST (inserções + travessia) vs insertion sort em Big O e evidência. A BST precisa ter **delete completo (folha, 1 filho, 2 filhos com sucessor in-order)** e validação de invariante, ainda que nenhum enunciado peça: o item 4.4 da rubrica cobra "inserção, busca e deleção, mantendo invariantes estruturais" e sem isso o item se perde.

**Ex 3 — Otimização.** `deduplicate_slow` O(n²) vs `deduplicate_fast` O(n) preservando ordem do primeiro aparecimento, ambos instrumentados, testes provando equivalência. `k_smallest` versão A (ordena tudo) e B (quickselect), comparadas em Big O e contagens para vários n e k. `k_smallest_bst(root,k)` por in-order controlada (para em k). Pior caso: BST construída com entrada ordenada degrada para lista encadeada, O(n) por operação — DEMONSTRAR com medição.

**Ex 4 — HashTableChained.** put/get/delete/__len__. Cada bucket é uma LISTA ENCADEADA de pares (key,value). Atualizar chave existente sem duplicar. Redimensionamento automático por fator de carga (limiar 0.75) com rehash; teste provando que todas as chaves seguem acessíveis depois. Contar comparações dentro dos buckets e relacionar com α (custo médio ≈ 1 + α/2). Cenário adversarial: chaves que colidem no mesmo bucket, mostrando degradação para O(n).

**Ex 5 — Stack, Queue e árvore de expressão.** Stack e Queue sobre array de CAPACIDADE FIXA, com exceções próprias (StackOverflowError, StackUnderflowError etc — nunca Exception genérica). Fila com wraparound (buffer circular). Árvore binária de expressão a partir de PÓS-FIXA usando a Stack; detectar entradas inválidas (operador sem operandos suficientes; sobra de operandos no fim). Avaliar de duas formas: pós-ordem recursiva e iterativa com stack; discutir profundidade de recursão e custo de memória.

**Ex 6 — Simulador de eventos.** 4 filas A,B,C,D. Minúscula = chegada, maiúscula = saída. Caractere não alfabético dispara SNAPSHOT do estado de todas as filas. Cliente com identificador sequencial por fila (ex "B17"). Overflow/underflow registram erro COM CONTEXTO e a simulação CONTINUA. Clientes atendidos indexados em BST com chave = identificador completo e metadados (ordem de atendimento, fila, timestamp lógico); busca e listagem in-order, com Big O de cada operação.

**Ex 7 — Recursão em diretórios.** Estrutura EM MEMÓRIA (não tocar no filesystem real — reprodutibilidade). Nós com nome, tipo, filhos. `walk(root)` recursivo devolvendo caminhos completos em pré-ordem. `DirectoryTree` com insert(path,node_type), search(path), delete(path) e validações. POLÍTICA DE DELETE: nó interno com filhos exige `recursive=True`; sem a flag, levanta erro (decisão de projeto a justificar). Caminho atual mantido em LISTA ENCADEADA (push/pop), comparado com lista Python em operações fundamentais.

**Ex 8 — DP e memoization.** Problema: `min_coins(amount, coins)` (sobreposição mais visual que knapsack, árvore de estados mais legível). Versão recursiva pura contando chamadas e subproblemas distintos. Memoization usando a HashTableChained do ex 4 — PROIBIDO dict ou lru_cache, o enunciado pede a estrutura própria. Chave do memo = estado mínimo suficiente (justificar por que basta). Mostrar redução quantitativa de chamadas e mudança de complexidade. Construir a árvore de subproblemas visitados (nó = estado, filhos = transições) e implementar busca por estado, discutindo o custo.

**Ex 9 — QuickSort e QuickSelect.** `quicksort(arr, short)` contando cópias e comparações, usando INSERTION SORT para subarrays ≤ short; 4 padrões × ≥3 valores de short (0, 8, 32). `quickselect(arr, goal_index)` retornando elemento + contagens, comparado com ordenar tudo. Variante em SinglyLinkedList: partição em três listas (menores, iguais, maiores) preservando estabilidade onde possível; discutir por que o custo muda sem acesso por índice (sem pivô aleatório barato, sem partição in-place) e relacionar com Big O.

**Ex 10 — SinglyLinkedList.** insert_first, insert_last, search, delete, __len__, __str__, insert_at(index,value), delete_at(index) com validações e mensagens claras. Tabela de custo por operação em Big O + experimento evidenciando o custo da busca linear. IMPLEMENTAR COM E SEM ponteiro tail e medir insert_last nos dois — a rubrica pede justificativa da decisão com evidência, então a evidência precisa existir.

**Ex 11 — DoublyLinkedList e Deque.** Nós com prev e next; insert_first, insert_last, delete_first, delete_last, is_empty. Deque sobre ela: insert_left, insert_right, remove_left, remove_right, peek_left, peek_right, todas O(1). Bateria de testes com sequências longas alternando as pontas, verificando INVARIANTES após cada sequência: size correto, head.prev is None, tail.next is None, travessia ida/volta produzindo a mesma sequência invertida. Documentar tratamento de deque vazio.

**Ex 12 — Motor de indexação (integrador).**
- Indexação: tokenizar e normalizar; índice invertido em HashTableChained onde cada termo aponta para uma LISTA ENCADEADA de ocorrências (doc_id, posições); BST de termos para listagem ordenada e busca.
- Consulta: parsing com parênteses e AND/OR/NOT, infixa→pós-fixa por shunting-yard com Stack, e avaliação; interseção/união eficiente sobre listas ordenadas de doc_ids; QUEUE para BFS na BST gerando relatório de diagnóstico (níveis, balanceamento aproximado, termos por nível).
- Ranqueamento e DP: score por frequência e posição, ordenado com o quicksort instrumentado do ex 9; sugestão de termo ausente por distância de edição com memoization na hashtable.
- Análise final: custo assintótico de indexação, consulta, ranqueamento e sugestão; mais DUAS otimizações concretas justificadas em Big O e com evidência do notebook.

## 7. Cronograma
- Dia 1: counters, linked_list (ex 10), doubly_linked (ex 11), hashtable (ex 4), stack_queue (ex 5), exercício 1. Testes pytest junto com cada módulo, não depois.
- Dia 2: sorting (ex 2 e 9), bst, selection (ex 3), simulator (ex 6), dirtree (ex 7). Rodar experimentos e gerar tabelas.
- Dia 3: dp (ex 8), search_engine (ex 12), consolidação do notebook na ordem 1→12, tabela de rastreabilidade da rubrica, PDF e ROTEIRO_VIDEO.md.

## 8. Formato de cada seção do notebook
Sempre nesta ordem, porque é a que o professor vai verificar:
Exercício N — Título
Enunciado (resumo em 2 linhas)
Implementação -> import de src/ + docstring com a decisão de projeto
Execução -> chamadas reais, com asserts de corretude
Saída e medições -> DataFrame + gráfico quando fizer sentido
Análise em Big O -> texto conectando os números medidos à classe de complexidade
Casos de borda -> o que quebra, como foi tratado, prova de que foi tratado
No fim do notebook: tabela de rastreabilidade mapeando os 20 itens da rubrica para a seção onde cada um está demonstrado.

## 9. Convenções
- Python 3.11+, só stdlib + pandas, matplotlib, pytest, jupyter.
- PROIBIDO usar estrutura pronta onde o enunciado pede implementação própria: collections.deque, dict como memo, sorted() dentro dos algoritmos. sorted() só como referência de corretude nos testes.
- Type hints nas assinaturas públicas.
- Docstring registrando a decisão de projeto e o custo em Big O.
- Textos de análise e comentários em português; identificadores em inglês.
- Exceções próprias por estrutura, nunca `raise Exception(...)`.
- `pytest -q` verde antes de considerar qualquer exercício pronto.

## 10. Geração do PDF

jupyter nbconvert --to notebook --execute --inplace notebook/DR1_AT.ipynb
python build_pdf.py

PDF navegável, títulos visíveis por exercício, ordem 1 a 12, nomeado conforme a regra.

## 11. Primeira tarefa
Criar a estrutura de pastas, `src/counters.py` e `src/linked_list.py` completo (exercício 10), com `tests/test_linked_list.py` e as duas variantes (com e sem ponteiro tail) para a medição comparativa de insert_last.

Antes de escrever código, confirmar comigo: versão do Python instalada e se prefiro venv ou instalação global.

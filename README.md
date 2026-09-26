# DR1_AT — Algoritmos e Estruturas de Dados

**Henrique Goldstein Maluhy Mendes Amaral** · Infnet · Avaliação individual

Doze exercícios de algoritmos e estruturas de dados. O princípio que rege o
trabalho: **nenhuma afirmação assintótica sem tabela de contagens que a**
**sustente** — nunca "é O(n log n)" sozinho, sempre com a razão medida contra a
curva teórica enquanto n cresce.

## Abrir no Google Colab

Cada notebook roda sozinho: a primeira célula clona este repositório, o resto
importa de `src/` e demonstra. Use **Ambiente de execução → Executar tudo**.

| # | Exercício | Abrir |
|---|---|---|
| 1 | Busca em estruturas lineares | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thxggz/dr1-at-algoritmos/blob/main/colab/ex01_buscas.ipynb) |
| 2 | Ordenações quadráticas | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thxggz/dr1-at-algoritmos/blob/main/colab/ex02_ordenacoes_quadraticas.ipynb) |
| 3 | Otimização orientada a Big O | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thxggz/dr1-at-algoritmos/blob/main/colab/ex03_otimizacao.ipynb) |
| 4 | Hashtable com colisões e encadeamento | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thxggz/dr1-at-algoritmos/blob/main/colab/ex04_hashtable.ipynb) |
| 5 | Pilha, fila e árvore de expressão | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thxggz/dr1-at-algoritmos/blob/main/colab/ex05_pilha_fila_expressao.ipynb) |
| 6 | Simulador de eventos com múltiplas filas | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thxggz/dr1-at-algoritmos/blob/main/colab/ex06_simulador_de_filas.ipynb) |
| 7 | Recursão em árvore de diretórios | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thxggz/dr1-at-algoritmos/blob/main/colab/ex07_recursao_diretorios.ipynb) |
| 8 | Programação dinâmica e memoization | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thxggz/dr1-at-algoritmos/blob/main/colab/ex08_programacao_dinamica.ipynb) |
| 9 | QuickSort e QuickSelect | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thxggz/dr1-at-algoritmos/blob/main/colab/ex09_quicksort_quickselect.ipynb) |
| 10 | Lista encadeada com operações completas | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thxggz/dr1-at-algoritmos/blob/main/colab/ex10_lista_encadeada.ipynb) |
| 11 | Lista duplamente encadeada e Deque | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thxggz/dr1-at-algoritmos/blob/main/colab/ex11_lista_dupla_deque.ipynb) |
| 12 | Motor de indexação e consulta (integrador) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thxggz/dr1-at-algoritmos/blob/main/colab/ex12_motor_de_indexacao.ipynb) |

O notebook único, com os doze exercícios na ordem e a tabela de rastreabilidade
da rubrica, está em [`notebook/DR1_AT.ipynb`](notebook/DR1_AT.ipynb).

## Como o repositório está organizado

```
src/          as estruturas e algoritmos — o código real vive aqui
tests/        pytest por módulo (860 testes)
notebook/     o notebook único, executado ponta a ponta
colab/        um notebook por exercício, para abrir no Colab
figures/      os gráficos em PNG
```

O notebook **não define** nenhuma estrutura de dados: importa de `src/` e
demonstra. Classe colada dentro de notebook fica ilegível e impossível de
defender.

## Rodar na máquina

```bash
pip install pandas matplotlib pytest jupyter nbconvert
pytest -q                      # 860 testes
python verificar_entrega.py    # 8 conferências da entrega
```

## Núcleo reutilizável

Os exercícios não são independentes — o ex 12 exige hashtable, pilha, fila,
recursão, DP, ordenação eficiente, lista encadeada e árvore ao mesmo tempo.
Cada estrutura é definida **uma** vez:

| Estrutura | Definida em | Consumida por |
|---|---|---|
| `Counters` | `src/counters.py` | todos |
| `SinglyLinkedList` | `src/linked_list.py` | 1, 9, 10, 12 |
| `Stack`, `Queue` | `src/stack_queue.py` | 5, 6, 12 |
| `HashTableChained` | `src/hashtable.py` | 4, 8, 12 |
| `BinarySearchTree` | `src/bst.py` | 2, 3, 6, 8, 12 |
| `DoublyLinkedList`, `Deque` | `src/doubly_linked.py` | 11 |
| `quicksort`, `quickselect` | `src/sorting.py` | 3, 9, 12 |

## Instrumentação

Um único contador, em `src/counters.py`, com quatro campos: `comparisons`,
`copies` (uma troca conta 3), `calls` e `hops`. O quarto não é do enunciado e
existe por um motivo medido: com ponteiro `tail`, `insert_last` faz 0
comparações e 1 cópia — **igual** à versão sem `tail`. Os três contadores
obrigatórios são cegos para a diferença entre O(1) e O(n) ali; sem `hops` a
evidência do exercício 10 não existiria.

Todos os experimentos usam `random.seed(42)`, então os números dos notebooks, do
PDF e do Colab são idênticos.

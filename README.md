# DR1_AT — Algoritmos e Estruturas de Dados

**Henrique Goldstein Maluhy Mendes Amaral** · Infnet · Avaliação individual

Doze exercícios de algoritmos e estruturas de dados, cada um num notebook do Google Colab.
Em todos, as conclusões sobre complexidade vêm acompanhadas da medição que as sustenta:
contagem de comparações, cópias ou chamadas, e a razão contra a curva teórica enquanto `n` cresce.

## Os exercícios

| # | Exercício | Colab |
|---|---|---|
| 1 | Busca em estruturas lineares e impacto da organização | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thxggz/dr1-at-algoritmos/blob/main/colab/ex01_buscas.ipynb) |
| 2 | Ordenações quadráticas com instrumentação e diagnóstico | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thxggz/dr1-at-algoritmos/blob/main/colab/ex02_ordenacoes_quadraticas.ipynb) |
| 3 | Otimização orientada a Big O com requisitos de desempenho | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thxggz/dr1-at-algoritmos/blob/main/colab/ex03_otimizacao.ipynb) |
| 4 | Hashtable com colisões e encadeamento por lista | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thxggz/dr1-at-algoritmos/blob/main/colab/ex04_hashtable.ipynb) |
| 5 | Pilha e fila com invariantes e árvore de expressão | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thxggz/dr1-at-algoritmos/blob/main/colab/ex05_pilha_fila_expressao.ipynb) |
| 6 | Simulador de eventos com múltiplas filas e índice em árvore | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thxggz/dr1-at-algoritmos/blob/main/colab/ex06_simulador_de_filas.ipynb) |
| 7 | Recursão para navegação em árvore de diretórios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thxggz/dr1-at-algoritmos/blob/main/colab/ex07_recursao_diretorios.ipynb) |
| 8 | Programação dinâmica e memoization com hashtable e árvore de subproblemas | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thxggz/dr1-at-algoritmos/blob/main/colab/ex08_programacao_dinamica.ipynb) |
| 9 | QuickSort e QuickSelect com contagem de operações e variante em lista encadeada | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thxggz/dr1-at-algoritmos/blob/main/colab/ex09_quicksort_quickselect.ipynb) |
| 10 | Lista encadeada com operações completas e análise de eficiência | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thxggz/dr1-at-algoritmos/blob/main/colab/ex10_lista_encadeada.ipynb) |
| 11 | Lista duplamente encadeada e Deque | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thxggz/dr1-at-algoritmos/blob/main/colab/ex11_lista_dupla_deque.ipynb) |
| 12 | Motor de indexação e consulta com estruturas combinadas | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thxggz/dr1-at-algoritmos/blob/main/colab/ex12_motor_de_indexacao.ipynb) |

## Como rodar

Abra o link do exercício e use **Ambiente de execução → Executar tudo**. Cada notebook é
independente: não depende de nenhum outro arquivo, e usa só a biblioteca padrão do Python,
`pandas` e `matplotlib` (que já vêm no Colab). Os experimentos usam semente fixa
(`random.seed(42)`), então os números saem iguais a cada execução.

Os notebooks já estão salvos com as saídas, então dá para ler os resultados antes mesmo de rodar.

## Arquivos

- `colab/` — os doze notebooks, um por exercício
- `Henrique_Goldstein_DR1_AT.pdf` — os doze notebooks impressos, com os links na primeira página

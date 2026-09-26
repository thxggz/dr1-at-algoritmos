"""Exercícios 2 e 9 — Ordenações instrumentadas.

Exercício 2: as três quadráticas (bubble, selection, insertion).
Exercício 9: ``quicksort`` com corte para insertion sort, ``quickselect`` e a
variante em lista encadeada com partição em três vias. As decisões de projeto
1 a 4 abaixo valem para o módulo inteiro; as 5 a 8 estão na seção do ex 9.

Decisões de projeto
-------------------
1. **Os três algoritmos copiam a entrada e devolvem lista nova.**
   Ordenar no lugar pouparia Θ(n) de memória, mas tornaria impossível rodar os
   três sobre o **mesmo** vetor: o primeiro a rodar entregaria um vetor ordenado
   aos outros dois, e comparar bubble com insertion em entradas diferentes não
   compara nada. A cópia defensiva **não é contada** — ela é custo do arranjo
   experimental, não do algoritmo, e como os três pagam exatamente a mesma,
   contá-la só somaria ``n`` a todas as colunas. Para efeito de análise de
   memória, os três continuam sendo Θ(1) de espaço extra.

2. **Bubble sort tem detecção de passagem sem troca.** Sem ela, o bubble faria
   ``n(n−1)/2`` comparações até num vetor já ordenado, e a coluna "ordenado"
   não distinguiria nada. Com ela, o melhor caso do bubble é O(n) — e a
   pergunta do enunciado ("quem mais se beneficia de quase ordenado?") passa a
   ter uma resposta interessante em vez de óbvia.

3. **Critério de cópias uniforme, o do enunciado.** Uma **troca** conta 3
   cópias (``tmp = a; a = b; b = tmp``); um **deslocamento** conta 1. Isso não é
   detalhe: bubble e insertion fazem o mesmo número de comparações no vetor
   reverso, e é só na coluna de cópias que a diferença entre trocar e deslocar
   aparece — bubble paga ``3·n(n−1)/2``, insertion paga ``n(n−1)/2``.

4. **A contagem de inversões não é instrumentada.** :func:`count_inversions` é
   ferramenta de análise (serve para mostrar que o insertion custa ``≈ n+d``),
   não um dos algoritmos sob teste. Contá-la somaria comparações que nenhum dos
   três executa.

Custo em Big O
--------------
Com ``n`` elementos e ``d`` inversões na entrada:

==============  ==================  ====================  ==================
Algoritmo       Melhor caso         Caso médio            Pior caso
==============  ==================  ====================  ==================
``bubble``      Θ(n) comparações    Θ(n²)                 Θ(n²)
``selection``   Θ(n²) **sempre**    Θ(n²)                 Θ(n²)
``insertion``   Θ(n) comparações    Θ(n²)                 Θ(n²)
==============  ==================  ====================  ==================

A linha do ``selection`` é o ponto: ele é Θ(n²) **em todos os casos**, porque
para achar o mínimo de cada sufixo ele precisa olhar o sufixo inteiro, esteja
ele ordenado ou não. Nenhuma organização prévia da entrada ajuda.

O ``insertion`` é o que mais aproveita a ordem prévia, e dá para ser preciso:
ele faz entre ``d`` e ``d + (n−1)`` comparações, isto é **Θ(n + d)**. Num vetor
quase ordenado ``d`` é pequeno e o algoritmo é quase linear; num vetor aleatório
``d ≈ n²/4`` e ele volta a ser quadrático. :func:`bench_nearly_sorted_sensitivity`
mede exatamente essa curva.

O ``bubble`` também aproveita, mas por um mecanismo mais fraco: o número de
passagens é ``1 +`` a maior distância que algum elemento precisa andar **para a
esquerda**. Um único elemento pequeno lá no fim força ``n`` passagens mesmo com
todo o resto ordenado, enquanto o insertion resolveria esse caso em ``O(n)``.

Espaço: os três são Θ(1) de memória extra. A comparação com a ordenação por BST
(:func:`bench_bst_vs_insertion`) é onde isso pesa: a BST gasta Θ(n) de nós.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Iterable, Sequence, TypeVar

from src.bst import BinarySearchTree
from src.counters import RANDOM_SEED, Counters, make_row, ratio, stopwatch
from src.linked_list import SinglyLinkedList
from src.searching import generate_array

T = TypeVar("T")

#: Padrões de entrada do experimento, na ordem em que o enunciado os cita.
SORT_PATTERNS: tuple[str, ...] = (
    "ordenado",
    "reverso",
    "quase_ordenado",
    "aleatorio",
)

#: Teto de tamanho do exercício 2. Bubble sort em 10⁶ elementos faria 5·10¹¹
#: comparações — dias de execução para produzir um número que a teoria já dá.
MAX_QUADRATIC_SIZE: int = 4_000


# ----------------------------------------------------------------------
# Os três algoritmos
# ----------------------------------------------------------------------
def bubble_sort(arr: Sequence[T], counters: Counters | None = None) -> list[T]:
    """Bubble sort com detecção de passagem sem troca (decisão de projeto 2).

    A cada passagem, o maior elemento restante "borbulha" até o fim, então o
    sufixo já ordenado cresce de 1 em 1 e a passagem seguinte pode ser mais
    curta. Se uma passagem inteira não troca nada, o vetor está ordenado e o
    laço externo para.

    Contagens: 1 comparação por par adjacente examinado; 3 cópias por troca.

    Custo: Θ(n) comparações no melhor caso (já ordenado: uma passagem),
    Θ(n²) no médio e no pior.
    """
    counters = counters if counters is not None else Counters()
    data = list(arr)  # cópia defensiva, não contada (decisão de projeto 1)
    n = len(data)
    for passagem in range(n - 1):
        trocou = False
        for index in range(n - 1 - passagem):
            if counters.gt(data[index], data[index + 1]):
                counters.swap(data, index, index + 1)
                trocou = True
        if not trocou:
            break
    return data


def selection_sort(arr: Sequence[T], counters: Counters | None = None) -> list[T]:
    """Selection sort: acha o mínimo do sufixo e o traz para a posição atual.

    Faz **exatamente** ``n(n−1)/2`` comparações, sempre, em qualquer entrada:
    para saber qual é o mínimo de um sufixo é preciso olhar o sufixo inteiro, e
    nada no vetor permite pular essa varredura.

    Em compensação, é o que menos move dados: no máximo ``n−1`` trocas, isto é
    ``3(n−1)`` cópias — contra ``Θ(n²)`` cópias dos outros dois no pior caso.
    Quando mover um elemento é caro (registros grandes), essa coluna importa
    mais que a das comparações.

    Custo: Θ(n²) comparações em todos os casos; O(n) cópias.
    """
    counters = counters if counters is not None else Counters()
    data = list(arr)
    n = len(data)
    for position in range(n - 1):
        smallest = position
        for index in range(position + 1, n):
            if counters.lt(data[index], data[smallest]):
                smallest = index
        if smallest != position:
            counters.swap(data, position, smallest)
    return data


def insertion_sort(arr: Sequence[T], counters: Counters | None = None) -> list[T]:
    """Insertion sort: insere cada elemento na posição certa do prefixo ordenado.

    É o algoritmo que mais aproveita ordem prévia, e dá para dizer exatamente
    por quê: o laço interno só anda enquanto encontra elementos **maiores** que
    a chave, e cada passo desses desfaz uma inversão. Com ``d`` inversões no
    vetor, o total de comparações fica entre ``d`` e ``d + (n−1)`` — ou seja
    **Θ(n + d)**, e não Θ(n²) incondicional.

    Contagens: 1 comparação por elemento examinado no prefixo; 1 cópia por
    deslocamento, mais 2 por elemento inserido (guardar a chave e escrevê-la no
    lugar). Note que ele **desloca** em vez de trocar: no vetor reverso faz as
    mesmas comparações que o bubble com um terço das cópias.

    Custo: Θ(n) no melhor caso (já ordenado), Θ(n²) no médio e no pior.
    """
    counters = counters if counters is not None else Counters()
    data = list(arr)
    for position in range(1, len(data)):
        key = counters.read(data[position])
        index = position - 1
        while index >= 0 and counters.gt(data[index], key):
            counters.write(data, index + 1, data[index])
            index -= 1
        counters.write(data, index + 1, key)
    return data


def bubble_sort_counted(arr: Sequence[T]) -> tuple[list[T], Counters]:
    """:func:`bubble_sort` no protocolo ``(resultado, Counters)``."""
    counters = Counters()
    return bubble_sort(arr, counters), counters


def selection_sort_counted(arr: Sequence[T]) -> tuple[list[T], Counters]:
    """:func:`selection_sort` no protocolo ``(resultado, Counters)``."""
    counters = Counters()
    return selection_sort(arr, counters), counters


def insertion_sort_counted(arr: Sequence[T]) -> tuple[list[T], Counters]:
    """:func:`insertion_sort` no protocolo ``(resultado, Counters)``."""
    counters = Counters()
    return insertion_sort(arr, counters), counters


#: Os três algoritmos quadráticos, para os experimentos iterarem sobre eles.
QUADRATIC_SORTS: dict[str, Callable[..., list[Any]]] = {
    "bubble": bubble_sort,
    "selection": selection_sort,
    "insertion": insertion_sort,
}


# ----------------------------------------------------------------------
# Ferramentas de análise (não instrumentadas — decisão de projeto 4)
# ----------------------------------------------------------------------
def is_sorted_list(values: Sequence[T]) -> bool:
    """``True`` se ``values`` é não decrescente. Custo: O(n).

    Verificação de corretude própria, sem ``sorted()``: a convenção 9 reserva
    ``sorted()`` para os testes, e esta função é chamada de dentro dos
    experimentos.
    """
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def count_inversions(values: Iterable[T]) -> int:
    """Número de pares ``(i, j)`` com ``i < j`` e ``values[i] > values[j]``.

    É a medida de "quão desordenado" um vetor está, e é o parâmetro ``d`` da
    análise Θ(n + d) do insertion sort.

    Implementada por merge sort, O(n log n), porque a versão ingênua O(n²)
    custaria mais que os próprios algoritmos sob teste em ``n = 4.000``.
    **Não é instrumentada** (decisão de projeto 4): é instrumento de medida, e
    suas comparações não pertencem a nenhum dos algoritmos avaliados.

    Custo: O(n log n) em tempo, O(n) de memória.
    """
    def sort_and_count(items: list[T]) -> tuple[list[T], int]:
        if len(items) <= 1:
            return items, 0
        middle = len(items) // 2
        left, left_inversions = sort_and_count(items[:middle])
        right, right_inversions = sort_and_count(items[middle:])

        merged: list[T] = []
        inversions = left_inversions + right_inversions
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:  # type: ignore[operator]
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
                # left[i:] são todos maiores que right[j]: uma inversão cada
                inversions += len(left) - i
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged, inversions

    return sort_and_count(list(values))[1]


# ----------------------------------------------------------------------
# Tabela de custo (consumida pelo notebook)
# ----------------------------------------------------------------------
#: Custo por algoritmo e caso. ``d`` é o número de inversões da entrada.
COST_TABLE: list[dict[str, str]] = [
    {"algoritmo": "bubble", "melhor": "Θ(n) comp. / 0 cópias",
     "medio": "Θ(n²) / Θ(n²)", "pior": "n(n−1)/2 comp. / 3·n(n−1)/2 cópias",
     "espaco": "Θ(1)",
     "aproveita_ordem": "sim, pela detecção de passagem sem troca — mas o "
                        "ganho depende da maior distância para a ESQUERDA"},
    {"algoritmo": "selection", "melhor": "n(n−1)/2 comp. / 0 cópias",
     "medio": "n(n−1)/2 / ≤3(n−1)", "pior": "n(n−1)/2 comp. / 3(n−1) cópias",
     "espaco": "Θ(1)",
     "aproveita_ordem": "NÃO — varre o sufixo inteiro para achar cada mínimo, "
                        "esteja ele ordenado ou não"},
    {"algoritmo": "insertion", "melhor": "n−1 comp. / 2(n−1) cópias",
     "medio": "Θ(n + d) / Θ(n + d)", "pior": "n(n−1)/2 comp. / n(n−1)/2 + 2(n−1)",
     "espaco": "Θ(1)",
     "aproveita_ordem": "sim, e é o que mais aproveita: custo Θ(n + d), "
                        "proporcional ao número de inversões"},
    {"algoritmo": "BST (inserir + in-order)",
     "melhor": "Θ(n log n) se a ordem de chegada for aleatória",
     "medio": "Θ(n log n)", "pior": "n(n−1)/2 se a entrada já vier ordenada",
     "espaco": "Θ(n) — um nó por elemento",
     "aproveita_ordem": "INVERTE: entrada ordenada é o PIOR caso da BST e o "
                        "MELHOR do insertion"},
]


# ----------------------------------------------------------------------
# Experimentos do exercício 2
# ----------------------------------------------------------------------
def bench_quadratic_sorts(
    sizes: Iterable[int] = (250, 500, 1_000, 2_000, 4_000),
    patterns: Iterable[str] = SORT_PATTERNS,
    *,
    seed: int = RANDOM_SEED,
    algorithms: Iterable[str] = tuple(QUADRATIC_SORTS),
) -> list[dict[str, Any]]:
    """Os três algoritmos × 4 padrões × 5 tamanhos, com comparações e cópias.

    Cada linha traz as razões contra ``n`` e contra ``n²``: razão estável contra
    ``n²`` atesta a classe quadrática, razão estável contra ``n`` atesta a
    linear. Traz também ``d`` (inversões da entrada) e a razão
    ``comparações/(n+d)``, que é o que sustenta a afirmação Θ(n + d) do
    insertion.

    Os três recebem **o mesmo vetor** em cada célula (decisão de projeto 1),
    então as colunas são diretamente comparáveis.

    Tamanho máximo: :data:`MAX_QUADRATIC_SIZE`. A execução completa leva
    dezenas de segundos — é o preço de medir Θ(n²) de verdade em vez de
    extrapolar.

    Custo: O(Σ n²) por algoritmo e padrão.
    """
    rows: list[dict[str, Any]] = []
    for n in sizes:
        if n <= 0:
            raise ValueError(f"tamanho deve ser positivo, recebido {n}")
        if n > MAX_QUADRATIC_SIZE:
            raise ValueError(
                f"tamanho {n} passa do teto de {MAX_QUADRATIC_SIZE} do "
                "exercício 2: ordenação quadrática acima disso leva horas e "
                "não acrescenta evidência"
            )
        for pattern in patterns:
            if pattern not in SORT_PATTERNS:
                raise ValueError(
                    f"padrão desconhecido: {pattern!r}; "
                    f"disponíveis: {list(SORT_PATTERNS)}"
                )
            arr = generate_array(n, pattern, seed=seed)
            inversions = count_inversions(arr)

            for name in algorithms:
                if name not in QUADRATIC_SORTS:
                    raise ValueError(f"algoritmo desconhecido: {name!r}")
                sorter = QUADRATIC_SORTS[name]
                counters = Counters()
                with stopwatch() as elapsed:
                    resultado = sorter(arr, counters)
                if not is_sorted_list(resultado):
                    raise AssertionError(
                        f"{name} não ordenou o padrão {pattern!r} com n={n}"
                    )

                rows.append(
                    make_row(
                        exercise="ex2",
                        algorithm=name,
                        n=n,
                        pattern=pattern,
                        counters=counters,
                        metric="comparisons",
                        curves=("n", "n_log2n", "n2"),
                        elapsed_s=elapsed[0],
                        extra={
                            "inversoes": inversions,
                            "inversoes/n2": inversions / (n * n),
                            "comparacoes/(n+d)": (
                                counters.comparisons / (n + inversions)
                            ),
                            "copies/n2": ratio(counters.copies, n, "n2"),
                            "copies/comparisons": (
                                counters.copies / counters.comparisons
                                if counters.comparisons
                                else 0.0
                            ),
                            "ordenado_ok": True,
                        },
                    )
                )
            del arr
    return rows


def bench_nearly_sorted_sensitivity(
    n: int = 2_000,
    disorders: Iterable[float] = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 1.0),
    *,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Quem mais se beneficia de "quase ordenado" — a curva, não a opinião.

    Fixa ``n`` e varia o grau de desordem da entrada, medindo as comparações dos
    três algoritmos contra o número real de inversões ``d``.

    O que se espera ver, e que é a resposta à pergunta do enunciado:

    * ``selection`` — coluna **constante**. Faz ``n(n−1)/2`` comparações em
      qualquer nível de desordem, porque varrer o sufixo inteiro para achar o
      mínimo não depende de o sufixo estar ordenado;
    * ``insertion`` — cresce **proporcionalmente a d**, com
      ``comparações/(n+d) ≈ 1``. É o que mais aproveita a ordem prévia, e o
      mecanismo é direto: cada comparação bem-sucedida do laço interno desfaz
      exatamente uma inversão;
    * ``bubble`` — aproveita, mas menos e de forma mais irregular: o que conta
      para ele não é o número de inversões e sim a maior distância que algum
      elemento precisa andar para a esquerda.

    Atenção ao alcance do gerador: ``generate_array(..., "quase_ordenado",
    disorder=x)`` faz ``x·n`` trocas de pares **adjacentes**, e cada troca cria
    no máximo uma inversão (algumas se cancelam). Então mesmo ``disorder=1.0``
    produz ``d = O(n)`` — um vetor ainda quase ordenado, longe do ``d ≈ n²/4``
    de um vetor aleatório. A âncora superior da curva não está aqui: são as
    linhas ``aleatorio`` de :func:`bench_quadratic_sorts`, onde o insertion
    volta a ser quadrático.

    Custo: O(|disorders| · n²).
    """
    if n <= 0:
        raise ValueError(f"tamanho deve ser positivo, recebido {n}")
    if n > MAX_QUADRATIC_SIZE:
        raise ValueError(
            f"tamanho {n} passa do teto de {MAX_QUADRATIC_SIZE} do exercício 2"
        )
    rows: list[dict[str, Any]] = []
    maximo = n * (n - 1) // 2
    for disorder in disorders:
        if not 0.0 <= disorder <= 1.0:
            raise ValueError(f"disorder deve estar em [0, 1], recebido {disorder}")
        arr = generate_array(n, "quase_ordenado", seed=seed, disorder=disorder)
        inversions = count_inversions(arr)

        for name, sorter in QUADRATIC_SORTS.items():
            counters = Counters()
            with stopwatch() as elapsed:
                resultado = sorter(arr, counters)
            if not is_sorted_list(resultado):
                raise AssertionError(f"{name} não ordenou com disorder={disorder}")
            rows.append(
                {
                    "exercicio": "ex2",
                    "algoritmo": name,
                    "n": n,
                    "padrao": f"quase_ordenado (disorder={disorder})",
                    "desordem": disorder,
                    "inversoes": inversions,
                    "inversoes_relativas": inversions / maximo if maximo else 0.0,
                    **counters.as_dict(),
                    "comparacoes/(n+d)": counters.comparisons / (n + inversions),
                    "comparacoes/pior_caso": counters.comparisons / maximo,
                    "copies/comparisons": (
                        counters.copies / counters.comparisons
                        if counters.comparisons
                        else 0.0
                    ),
                    "tempo_s": elapsed[0],
                }
            )
    return rows


def bench_bst_vs_insertion(
    sizes: Iterable[int] = (250, 500, 1_000, 2_000),
    patterns: Iterable[str] = SORT_PATTERNS,
    *,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Ordenar por BST × insertion sort: custo total e memória.

    Para cada célula, o custo da BST é **inserções + travessia in-order** — a
    travessia é O(n) e faz 0 comparações, então o que decide é a construção.

    O achado central, e a razão de este experimento existir:
    **o melhor caso de um é o pior caso do outro.**

    * entrada ``ordenado`` — insertion faz ``n−1`` comparações (Θ(n)); a BST
      degenera em espinha e faz ``n(n−1)/2`` (Θ(n²));
    * entrada ``aleatorio`` — insertion faz ``≈ n²/4`` (Θ(n²)); a BST fica
      equilibrada e faz ``≈ 1,39·n·log₂n`` (Θ(n log n)).

    Não existe "a BST é melhor" nem "o insertion é melhor": existe qual dos dois
    casa com a distribuição esperada da entrada. E a BST ainda cobra Θ(n) de
    memória em nós, contra Θ(1) do insertion — o que a torna a escolha errada
    quando o ganho é pequeno.

    Custo: O(Σ n²) no pior padrão de cada um.
    """
    rows: list[dict[str, Any]] = []
    for n in sizes:
        if n <= 0:
            raise ValueError(f"tamanho deve ser positivo, recebido {n}")
        if n > MAX_QUADRATIC_SIZE:
            raise ValueError(
                f"tamanho {n} passa do teto de {MAX_QUADRATIC_SIZE} do exercício 2"
            )
        for pattern in patterns:
            if pattern not in SORT_PATTERNS:
                raise ValueError(f"padrão desconhecido: {pattern!r}")
            arr = generate_array(n, pattern, seed=seed)

            insertion_counters = Counters()
            with stopwatch() as insertion_elapsed:
                por_insertion = insertion_sort(arr, insertion_counters)

            bst_build = Counters()
            tree: BinarySearchTree[int, None] = BinarySearchTree()
            with stopwatch() as bst_elapsed:
                for value in arr:
                    tree.insert(value, None, bst_build)
                bst_traversal = Counters()
                por_bst = list(tree.in_order_keys(bst_traversal))

            if not is_sorted_list(por_insertion) or por_bst != por_insertion:
                raise AssertionError(
                    f"BST e insertion discordaram no padrão {pattern!r}, n={n}"
                )
            tree.check_invariants()
            relatorio = tree.balance_report()

            total_bst = bst_build.comparisons + bst_traversal.comparisons
            rows.append(
                {
                    "exercicio": "ex2",
                    "algoritmo": "BST × insertion sort",
                    "n": n,
                    "padrao": pattern,
                    "insertion_comparacoes": insertion_counters.comparisons,
                    "insertion_copias": insertion_counters.copies,
                    "insertion/n2": ratio(insertion_counters.comparisons, n, "n2"),
                    "insertion/n": ratio(insertion_counters.comparisons, n, "n"),
                    "bst_comparacoes_insercao": bst_build.comparisons,
                    "bst_comparacoes_travessia": bst_traversal.comparisons,
                    "bst_comparacoes_total": total_bst,
                    "bst/n_log2n": ratio(total_bst, n, "n_log2n"),
                    "bst/n2": ratio(total_bst, n, "n2"),
                    "bst_altura": relatorio["altura"],
                    "bst_altura_ideal": relatorio["altura_ideal"],
                    # razao_altura torna visível o caso QUASE degenerado, que a
                    # bandeira estrita (altura == n) deixa passar
                    "bst_razao_altura": relatorio["razao_altura"],
                    "bst_degenerada": relatorio["degenerada"],
                    "razao_bst_sobre_insertion": (
                        total_bst / insertion_counters.comparisons
                        if insertion_counters.comparisons
                        else math.inf
                    ),
                    "vencedor": (
                        "empate"
                        if insertion_counters.comparisons == total_bst
                        else (
                            "insertion"
                            if insertion_counters.comparisons < total_bst
                            else "BST"
                        )
                    ),
                    "memoria_extra_insertion": "Θ(1)",
                    "memoria_extra_bst": f"Θ(n) — {tree.node_count} nós",
                    "tempo_insertion_s": insertion_elapsed[0],
                    "tempo_bst_s": bst_elapsed[0],
                }
            )
            del arr
    return rows


# ======================================================================
# Exercício 9 — QuickSort e QuickSelect
# ======================================================================
#
# Decisões de projeto do exercício 9
# ----------------------------------
# 5. **A escolha do pivô é um parâmetro, não uma constante.** ``pivot="last"``
#    (ingênuo), ``"median3"`` (padrão) e ``"random"``. Sem isso o exercício não
#    conseguiria mostrar o que pede: com ``"last"``, uma entrada **já ordenada**
#    faz o quicksort virar Θ(n²); com ``"median3"``, a mesma entrada vira o
#    melhor caso. O impacto do padrão de entrada não é propriedade do quicksort,
#    é propriedade do par (quicksort, estratégia de pivô).
#
# 6. **Recursão só na metade menor; a maior vira laço.** Isso limita a pilha de
#    chamadas a ⌊log₂n⌋+1 quadros mesmo quando a partição é péssima. É uma
#    otimização de *memória* que não altera o *tempo*: com ``"last"`` em entrada
#    ordenada o algoritmo continua fazendo Θ(n²) comparações, só que sem
#    estourar a pilha. Separar as duas coisas é parte da análise.
#
# 7. **``quickselect`` é iterativo.** Ele desce em um lado só, e um laço faz
#    isso com O(1) de pilha em vez de O(log n).
#
# 8. **A variante em lista encadeada particiona em TRÊS listas**, como o
#    enunciado pede, e é **estável**: cada elemento é anexado ao fim da lista
#    correspondente, preservando a ordem de chegada dentro de cada grupo. A
#    versão em array (Lomuto, duas vias) **não** é estável — trocas movem
#    elementos por cima de iguais. A troca é consciente: estabilidade custa
#    Θ(n) de memória por nível.

#: Estratégias de escolha de pivô aceitas por :func:`quicksort` e
#: :func:`quickselect`.
PIVOT_STRATEGIES: tuple[str, ...] = ("last", "median3", "random")

#: Valores de ``short`` usados no experimento do exercício 9.
SHORT_VALUES: tuple[int, ...] = (0, 8, 32)


def _insertion_sort_range(
    data: list[T], low: int, high: int, counters: Counters
) -> None:
    """Insertion sort **no lugar** sobre ``data[low..high]``. Custo: O(m²).

    Difere do :func:`insertion_sort` público em dois pontos, os dois exigidos
    pelo quicksort: opera sobre uma faixa e não copia a entrada.
    """
    for position in range(low + 1, high + 1):
        key = counters.read(data[position])
        index = position - 1
        while index >= low and counters.gt(data[index], key):
            counters.write(data, index + 1, data[index])
            index -= 1
        counters.write(data, index + 1, key)


def _median_of_three(data: list[T], low: int, high: int, counters: Counters) -> int:
    """Índice da mediana entre ``data[low]``, o do meio e ``data[high]``.

    Custo: O(1) — 2 ou 3 comparações, todas contadas.
    """
    middle = (low + high) // 2
    first, center, last = data[low], data[middle], data[high]
    if counters.gt(first, center):
        if counters.gt(center, last):
            return middle
        return high if counters.gt(first, last) else low
    if counters.gt(first, last):
        return low
    return high if counters.gt(center, last) else middle


def _choose_pivot(
    data: list[T],
    low: int,
    high: int,
    strategy: str,
    rng: random.Random,
    counters: Counters,
) -> int:
    """Índice do pivô conforme a estratégia. Custo: O(1)."""
    if strategy == "last":
        return high
    if strategy == "random":
        return rng.randint(low, high)
    if strategy == "median3":
        return _median_of_three(data, low, high, counters)
    raise ValueError(
        f"estratégia de pivô desconhecida: {strategy!r}; "
        f"disponíveis: {list(PIVOT_STRATEGIES)}"
    )


def _partition(data: list[T], low: int, high: int, counters: Counters) -> int:
    """Partição de Lomuto em torno de ``data[high]``; devolve a posição final.

    O ``if i != index`` evita contar troca de um elemento consigo mesmo: sem
    ele, uma entrada já ordenada apareceria com ``3n`` cópias de nada.

    Custo: ``high − low`` comparações, O(high − low) cópias.
    """
    pivot = data[high]
    i = low - 1
    for index in range(low, high):
        if counters.le(data[index], pivot):
            i += 1
            if i != index:
                counters.swap(data, i, index)
    if i + 1 != high:
        counters.swap(data, i + 1, high)
    return i + 1


def quicksort(
    arr: Sequence[T],
    short: int = 0,
    *,
    pivot: str = "median3",
    seed: int = RANDOM_SEED,
    counters: Counters | None = None,
) -> list[T]:
    """QuickSort com corte para insertion sort em subvetores de até ``short``.

    ``short=0`` desliga o corte (quicksort puro). Valores típicos ficam entre 8
    e 32: abaixo desse tamanho a constante do quicksort (escolher pivô,
    particionar, chamar) supera a do insertion sort, que percorre memória
    contígua e, em subvetor pequeno, já está quase ordenado pelas partições
    anteriores.

    ``pivot`` decide a robustez a padrões de entrada (decisão de projeto 5). A
    recursão desce só na metade menor (decisão 6), então a pilha fica em
    O(log n) mesmo no pior caso de partição — o que é medido por
    :func:`quicksort_depth`.

    Contagens: comparações de chaves, cópias (troca = 3) e ``calls`` = número
    de chamadas recursivas efetivas.

    Custo: Θ(n log n) no caso médio; Θ(n²) no pior. Memória: Θ(log n) de pilha e
    Θ(1) de dados além da cópia defensiva da entrada.
    """
    if not isinstance(short, int) or isinstance(short, bool):
        raise ValueError(f"short deve ser int, recebido {type(short).__name__}")
    if short < 0:
        raise ValueError(f"short deve ser >= 0, recebido {short}")
    if pivot not in PIVOT_STRATEGIES:
        raise ValueError(
            f"estratégia de pivô desconhecida: {pivot!r}; "
            f"disponíveis: {list(PIVOT_STRATEGIES)}"
        )
    counters = counters if counters is not None else Counters()
    data = list(arr)
    rng = random.Random(seed)
    deepest = [0]

    def sort_range(low: int, high: int, depth: int) -> None:
        counters.count_call()
        if depth > deepest[0]:
            deepest[0] = depth
        while low < high:
            if high - low + 1 <= short:
                _insertion_sort_range(data, low, high, counters)
                return
            chosen = _choose_pivot(data, low, high, pivot, rng, counters)
            if chosen != high:
                counters.swap(data, chosen, high)
            split = _partition(data, low, high, counters)
            # recursão na metade MENOR, laço na maior: pilha O(log n)
            if split - low < high - split:
                sort_range(low, split - 1, depth + 1)
                low = split + 1
            else:
                sort_range(split + 1, high, depth + 1)
                high = split - 1

    if data:
        sort_range(0, len(data) - 1, 1)
    quicksort.last_max_depth = deepest[0]  # type: ignore[attr-defined]
    return data


quicksort.last_max_depth = 0  # type: ignore[attr-defined]


def quicksort_counted(
    arr: Sequence[T], short: int = 0, *, pivot: str = "median3"
) -> tuple[list[T], Counters]:
    """:func:`quicksort` no protocolo ``(resultado, Counters)``."""
    counters = Counters()
    return quicksort(arr, short, pivot=pivot, counters=counters), counters


def quicksort_depth(
    arr: Sequence[T], short: int = 0, *, pivot: str = "median3"
) -> tuple[list[T], Counters, int]:
    """:func:`quicksort` devolvendo também a profundidade máxima de recursão.

    É a medida que comprova a decisão de projeto 6: a pilha fica em O(log n)
    mesmo quando as comparações são Θ(n²).

    Custo: o mesmo de :func:`quicksort`.
    """
    counters = Counters()
    resultado = quicksort(arr, short, pivot=pivot, counters=counters)
    return resultado, counters, quicksort.last_max_depth  # type: ignore[attr-defined]


def quickselect_partitioned(
    arr: Sequence[T],
    goal_index: int,
    *,
    pivot: str = "median3",
    seed: int = RANDOM_SEED,
    counters: Counters | None = None,
) -> tuple[T, list[T]]:
    """Como :func:`quickselect`, mas devolve também o vetor **particionado**.

    A garantia do vetor devolvido é a que o exercício 3 aproveita: toda posição
    antes de ``goal_index`` guarda um elemento ``<=`` ao que está em
    ``goal_index``, e toda posição depois guarda um ``>=``. Ou seja, os
    ``goal_index + 1`` menores elementos já estão no prefixo — **sem que o
    vetor tenha sido ordenado**. Pegar os ``k`` menores custa então
    ``Θ(n) + O(k log k)`` em vez de ``Θ(n log n)``.

    Custo: Θ(n) no caso médio, Θ(n²) no pior; O(1) de pilha (é iterativo).
    """
    if not isinstance(goal_index, int) or isinstance(goal_index, bool):
        raise ValueError(
            f"goal_index deve ser int, recebido {type(goal_index).__name__}"
        )
    data = list(arr)
    if not data:
        raise ValueError("quickselect em sequência vazia")
    if not 0 <= goal_index < len(data):
        raise ValueError(f"goal_index {goal_index} fora da faixa 0..{len(data) - 1}")
    if pivot not in PIVOT_STRATEGIES:
        raise ValueError(f"estratégia de pivô desconhecida: {pivot!r}")

    counters = counters if counters is not None else Counters()
    rng = random.Random(seed)
    low, high = 0, len(data) - 1
    while low < high:
        chosen = _choose_pivot(data, low, high, pivot, rng, counters)
        if chosen != high:
            counters.swap(data, chosen, high)
        split = _partition(data, low, high, counters)
        if split == goal_index:
            return data[split], data
        if goal_index < split:
            high = split - 1
        else:
            low = split + 1
    return data[low], data


def quickselect(
    arr: Sequence[T],
    goal_index: int,
    *,
    pivot: str = "median3",
    seed: int = RANDOM_SEED,
    counters: Counters | None = None,
) -> T:
    """Elemento que ocuparia ``goal_index`` se ``arr`` estivesse ordenado.

    Particiona como o quicksort, mas desce em **um** lado só: o que contém o
    índice procurado. Iterativo (decisão de projeto 7), então gasta O(1) de
    pilha.

    Custo: Θ(n) no caso médio — a recorrência ``T(n) = T(n/2) + Θ(n)`` soma uma
    série geométrica, ``n + n/2 + n/4 + … < 2n``, contra os Θ(n log n) de
    ordenar tudo. Θ(n²) no pior caso de partição.
    """
    valor, _ = quickselect_partitioned(
        arr, goal_index, pivot=pivot, seed=seed, counters=counters
    )
    return valor


def quickselect_counted(
    arr: Sequence[T], goal_index: int, *, pivot: str = "median3"
) -> tuple[T, Counters]:
    """:func:`quickselect` no protocolo ``(resultado, Counters)``."""
    counters = Counters()
    return quickselect(arr, goal_index, pivot=pivot, counters=counters), counters


# ----------------------------------------------------------------------
# Variante em lista encadeada (partição em três listas)
# ----------------------------------------------------------------------
def partition_three_way(
    items: SinglyLinkedList[T], pivot: T, counters: Counters | None = None
) -> tuple[SinglyLinkedList[T], SinglyLinkedList[T], SinglyLinkedList[T]]:
    """Separa ``items`` em (menores, iguais, maiores) que ``pivot``.

    **Estável**: cada elemento é anexado ao FIM da lista correspondente, então a
    ordem relativa dentro de cada grupo é a ordem original. O ``insert_last`` é
    O(1) por causa do ponteiro ``tail`` do exercício 10 — sem ele esta partição
    seria Θ(n²).

    Contagens: até 2 comparações por elemento (``< pivô`` e, se falhar,
    ``== pivô``), 1 salto por avanço e 1 cópia por anexação.

    Custo: Θ(n) em tempo e Θ(n) em memória — três listas novas. É a diferença
    estrutural para a versão em array, que particiona **no lugar** com Θ(1).
    """
    counters = counters if counters is not None else Counters()
    less: SinglyLinkedList[T] = SinglyLinkedList()
    equal: SinglyLinkedList[T] = SinglyLinkedList()
    greater: SinglyLinkedList[T] = SinglyLinkedList()
    first = True
    for value in items:
        if not first:
            counters.count_hop()
        first = False
        if counters.lt(value, pivot):
            less.insert_last(value, counters)
        elif counters.eq(value, pivot):
            equal.insert_last(value, counters)
        else:
            greater.insert_last(value, counters)
    return less, equal, greater


def quicksort_linked(
    items: SinglyLinkedList[T],
    counters: Counters | None = None,
    *,
    _depth: int = 1,
    _deepest: list[int] | None = None,
) -> SinglyLinkedList[T]:
    """QuickSort sobre :class:`SinglyLinkedList`, com partição em três listas.

    Por que o custo muda sem acesso por índice:

    * **não há partição no lugar.** O array troca dois elementos por índice em
      O(1); a lista teria de religar ponteiros conhecendo o antecessor. A saída
      é construir listas novas — Θ(n) de memória por nível, Θ(n log n) no total,
      contra Θ(1) do array;
    * **não há pivô aleatório barato.** Sortear um índice e ir até ele custa
      O(n) numa lista. O pivô aqui é o **primeiro** elemento, que é O(1) — e o
      preço dessa escolha é que entrada já ordenada vira o pior caso, Θ(n²),
      exatamente como ``pivot="last"`` no array;
    * **em compensação, sai estabilidade de graça** (decisão de projeto 8), e a
      partição em três vias resolve o caso de muitas chaves iguais em Θ(n) —
      cenário em que a partição de Lomuto do array degenera para Θ(n²).

    A recursão é sobre ``less`` e ``greater``. Em entrada ordenada a
    profundidade é ``n`` e o Python levanta ``RecursionError`` — a mesma lição
    do exercício 5, agora num algoritmo de ordenação.

    Custo: Θ(n log n) no caso médio, Θ(n²) no pior; Θ(n) de memória por nível.
    """
    counters = counters if counters is not None else Counters()
    deepest = _deepest if _deepest is not None else [0]
    counters.count_call()
    if _depth > deepest[0]:
        deepest[0] = _depth

    if len(items) <= 1:
        quicksort_linked.last_max_depth = deepest[0]  # type: ignore[attr-defined]
        return SinglyLinkedList(items)

    pivot = items.peek_first()
    less, equal, greater = partition_three_way(items, pivot, counters)

    ordered: SinglyLinkedList[T] = SinglyLinkedList()
    for part, precisa_ordenar in ((less, True), (equal, False), (greater, True)):
        origem = (
            quicksort_linked(part, counters, _depth=_depth + 1, _deepest=deepest)
            if precisa_ordenar
            else part
        )
        for value in origem:
            ordered.insert_last(value, counters)
    quicksort_linked.last_max_depth = deepest[0]  # type: ignore[attr-defined]
    return ordered


quicksort_linked.last_max_depth = 0  # type: ignore[attr-defined]


def quicksort_linked_counted(
    items: SinglyLinkedList[T],
) -> tuple[SinglyLinkedList[T], Counters]:
    """:func:`quicksort_linked` no protocolo ``(resultado, Counters)``."""
    counters = Counters()
    return quicksort_linked(items, counters), counters


# ----------------------------------------------------------------------
# Experimentos do exercício 9
# ----------------------------------------------------------------------
def bench_quicksort(
    sizes: Iterable[int] = (1_000, 4_000, 16_000),
    patterns: Iterable[str] = SORT_PATTERNS,
    shorts: Iterable[int] = SHORT_VALUES,
    *,
    pivot: str = "median3",
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """QuickSort em 4 padrões × 3 valores de ``short``, com contagens.

    Previsão: ``comparações/(n·log₂n)`` aproximadamente constante em todos os
    padrões quando o pivô é ``median3`` — é isso que torna o quicksort útil na
    prática, e é o contraste com a tabela do exercício 2, onde a mesma coluna
    explodia.

    O efeito do ``short`` é de **constante, não de classe**: ele troca
    comparações do quicksort por comparações do insertion sort em subvetores
    pequenos. A coluna ``calls`` cai bastante (menos chamadas recursivas), e o
    total de comparações se mexe pouco.

    Custo: O(Σ n log n) por combinação.
    """
    rows: list[dict[str, Any]] = []
    for n in sizes:
        if n <= 0:
            raise ValueError(f"tamanho deve ser positivo, recebido {n}")
        for pattern in patterns:
            if pattern not in SORT_PATTERNS:
                raise ValueError(f"padrão desconhecido: {pattern!r}")
            arr = generate_array(n, pattern, seed=seed)
            for short in shorts:
                counters = Counters()
                with stopwatch() as elapsed:
                    resultado = quicksort(
                        arr, short, pivot=pivot, seed=seed, counters=counters
                    )
                if not is_sorted_list(resultado):
                    raise AssertionError(
                        f"quicksort(short={short}) não ordenou {pattern!r}, n={n}"
                    )
                rows.append(
                    make_row(
                        exercise="ex9",
                        algorithm=f"quicksort(short={short}, pivot={pivot})",
                        n=n,
                        pattern=pattern,
                        counters=counters,
                        metric="comparisons",
                        curves=("n", "n_log2n", "n2"),
                        elapsed_s=elapsed[0],
                        extra={
                            "short": short,
                            "pivo": pivot,
                            "profundidade_maxima": quicksort.last_max_depth,  # type: ignore[attr-defined]
                            "log2n": math.log2(n),
                            "copies/n_log2n": ratio(counters.copies, n, "n_log2n"),
                            "calls/n": counters.calls / n,
                        },
                    )
                )
            del arr
    return rows


def bench_pivot_strategies(
    sizes: Iterable[int] = (500, 1_000, 2_000, 4_000),
    patterns: Iterable[str] = ("ordenado", "aleatorio"),
    *,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """O impacto do padrão de entrada depende da estratégia de pivô.

    Mesmo vetor, mesmo código, só o pivô diferente:

    * ``last`` em entrada **ordenada** — toda partição separa 1 de n−1, então
      são ``n(n−1)/2`` comparações: Θ(n²), o mesmo do insertion sort;
    * ``median3`` na **mesma** entrada — a mediana de três num vetor ordenado é
      o elemento do meio, partição perfeita: Θ(n log n), e o melhor caso;
    * em entrada aleatória as três estratégias empatam em ordem de grandeza.

    A coluna ``profundidade_maxima`` mostra a decisão de projeto 6 funcionando:
    mesmo no caso Θ(n²), a pilha fica em O(log n), porque a recursão só desce na
    metade menor. Tempo ruim e memória boa são coisas separadas.

    Custo: O(Σ n²) na combinação ``last`` + ``ordenado``.
    """
    rows: list[dict[str, Any]] = []
    for n in sizes:
        if n <= 0:
            raise ValueError(f"tamanho deve ser positivo, recebido {n}")
        for pattern in patterns:
            arr = generate_array(n, pattern, seed=seed)
            for strategy in PIVOT_STRATEGIES:
                counters = Counters()
                with stopwatch() as elapsed:
                    resultado = quicksort(
                        arr, 0, pivot=strategy, seed=seed, counters=counters
                    )
                if not is_sorted_list(resultado):
                    raise AssertionError(f"quicksort(pivot={strategy}) falhou")
                rows.append(
                    {
                        "exercicio": "ex9",
                        "algoritmo": f"quicksort(pivot={strategy})",
                        "n": n,
                        "padrao": pattern,
                        "pivo": strategy,
                        **counters.as_dict(),
                        "comparacoes/n_log2n": ratio(
                            counters.comparisons, n, "n_log2n"
                        ),
                        "comparacoes/n2": ratio(counters.comparisons, n, "n2"),
                        "profundidade_maxima": quicksort.last_max_depth,  # type: ignore[attr-defined]
                        "log2n_mais_1": math.floor(math.log2(n)) + 1,
                        "tempo_s": elapsed[0],
                    }
                )
            del arr
    return rows


def bench_quickselect_vs_sort(
    sizes: Iterable[int] = (1_000, 4_000, 16_000, 64_000),
    *,
    seed: int = RANDOM_SEED,
    positions: Iterable[float] = (0.0, 0.25, 0.5, 1.0),
) -> list[dict[str, Any]]:
    """Selecionar um elemento × ordenar tudo para pegá-lo.

    Previsão: ``quickselect`` faz Θ(n) comparações — a razão
    ``comparações/n`` fica presa perto de uma constante (entre 2 e 4) enquanto
    ``n`` cresce 64× — contra Θ(n log n) do quicksort, cuja razão
    ``comparações/n`` cresce como ``log₂n``.

    O ganho não é de constante: é de **classe**. E ele some assim que forem
    necessárias muitas estatísticas de ordem do mesmo vetor — aí ordenar uma vez
    e indexar passa a valer, pelo mesmo raciocínio de ponto de equilíbrio do
    exercício 1.

    Custo: O(Σ n log n), dominado pela ordenação de referência.
    """
    rows: list[dict[str, Any]] = []
    for n in sizes:
        if n <= 0:
            raise ValueError(f"tamanho deve ser positivo, recebido {n}")
        arr = generate_array(n, "aleatorio", seed=seed)

        ordenar = Counters()
        with stopwatch() as elapsed_sort:
            ordenado = quicksort(arr, 16, counters=ordenar)

        for fracao in positions:
            if not 0.0 <= fracao <= 1.0:
                raise ValueError(f"posição deve estar em [0, 1], recebido {fracao}")
            alvo = min(n - 1, int(fracao * n))
            selecionar = Counters()
            with stopwatch() as elapsed_select:
                valor = quickselect(arr, alvo, seed=seed, counters=selecionar)
            if valor != ordenado[alvo]:
                raise AssertionError(
                    f"quickselect({alvo}) devolveu {valor}, esperado {ordenado[alvo]}"
                )
            rows.append(
                {
                    "exercicio": "ex9",
                    "algoritmo": "quickselect × ordenar tudo",
                    "n": n,
                    "padrao": "aleatorio",
                    "posicao_relativa": fracao,
                    "goal_index": alvo,
                    "select_comparacoes": selecionar.comparisons,
                    "select_copias": selecionar.copies,
                    "select/n": ratio(selecionar.comparisons, n, "n"),
                    "sort_comparacoes": ordenar.comparisons,
                    "sort/n": ratio(ordenar.comparisons, n, "n"),
                    "sort/n_log2n": ratio(ordenar.comparisons, n, "n_log2n"),
                    "razao_sort_sobre_select": (
                        ordenar.comparisons / selecionar.comparisons
                        if selecionar.comparisons
                        else math.inf
                    ),
                    "tempo_select_s": elapsed_select[0],
                    "tempo_sort_s": elapsed_sort[0],
                }
            )
        del arr
    return rows


def bench_linked_quicksort(
    sizes: Iterable[int] = (250, 500, 1_000, 2_000),
    *,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """QuickSort em array × em lista encadeada, no mesmo vetor aleatório.

    Mede o que a falta de acesso por índice cobra:

    * ``copies`` na lista incluem **uma anexação por elemento por nível** —
      a partição cria listas novas, então a movimentação de dados é Θ(n log n)
      contra Θ(n log n) de trocas do array, mas com constante bem maior e
      alocação de nó junto;
    * ``hops`` existe só na lista: o array não segue ponteiro nenhum;
    * a lista faz **mais comparações** porque a partição é de três vias (até 2
      comparações por elemento contra 1 da de Lomuto) — o preço de tratar
      chaves iguais em Θ(n).

    Custo: O(Σ n log n).
    """
    rows: list[dict[str, Any]] = []
    for n in sizes:
        if n <= 0:
            raise ValueError(f"tamanho deve ser positivo, recebido {n}")
        arr = generate_array(n, "aleatorio", seed=seed)

        array_counters = Counters()
        with stopwatch() as elapsed_array:
            por_array = quicksort(arr, 0, counters=array_counters)

        items: SinglyLinkedList[int] = SinglyLinkedList(arr)
        linked_counters = Counters()
        with stopwatch() as elapsed_linked:
            por_lista = quicksort_linked(items, linked_counters)

        resultado_lista = por_lista.to_list()
        if resultado_lista != por_array:
            raise AssertionError("array e lista discordaram")

        rows.append(
            {
                "exercicio": "ex9",
                "algoritmo": "quicksort: array × lista encadeada",
                "n": n,
                "padrao": "aleatorio",
                "array_comparacoes": array_counters.comparisons,
                "array_copias": array_counters.copies,
                "array_saltos": array_counters.hops,
                "array_chamadas": array_counters.calls,
                "array/n_log2n": ratio(array_counters.comparisons, n, "n_log2n"),
                "lista_comparacoes": linked_counters.comparisons,
                "lista_copias": linked_counters.copies,
                "lista_saltos": linked_counters.hops,
                "lista_chamadas": linked_counters.calls,
                "lista/n_log2n": ratio(linked_counters.comparisons, n, "n_log2n"),
                "lista_profundidade": quicksort_linked.last_max_depth,  # type: ignore[attr-defined]
                "razao_comparacoes": (
                    linked_counters.comparisons / array_counters.comparisons
                ),
                "razao_copias": linked_counters.copies / array_counters.copies,
                "razao_tempo": elapsed_linked[0] / elapsed_array[0],
                "tempo_array_s": elapsed_array[0],
                "tempo_lista_s": elapsed_linked[0],
            }
        )
        del arr, items
    return rows

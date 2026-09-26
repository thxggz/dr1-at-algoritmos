"""Testes do exercício 9 — QuickSort, QuickSelect e a variante em lista.

(O código vive em ``src/sorting.py``, junto com as quadráticas do ex 2; a suíte
é separada só para não misturar dois exercícios num arquivo só.)

O eixo dos testes é a **separação entre tempo e memória**: o quicksort com
``pivot="last"`` sobre entrada ordenada é Θ(n²) em comparações e ainda assim usa
O(1) de pilha, porque a recursão só desce na metade menor. Confundir as duas
coisas é o erro que a decisão de projeto 6 evita, e há teste para cada uma.
"""

from __future__ import annotations

import math
import random
import sys

import pytest

from src.counters import Counters
from src.linked_list import SinglyLinkedList
from src.searching import generate_array
from src.sorting import (
    PIVOT_STRATEGIES,
    SHORT_VALUES,
    SORT_PATTERNS,
    bench_linked_quicksort,
    bench_pivot_strategies,
    bench_quickselect_vs_sort,
    bench_quicksort,
    partition_three_way,
    quickselect,
    quickselect_counted,
    quicksort,
    quicksort_counted,
    quicksort_depth,
    quicksort_linked,
    quicksort_linked_counted,
)


class PorChave:
    """Item comparado só pela chave — para observar estabilidade."""

    __slots__ = ("chave", "ordem")

    def __init__(self, chave: int, ordem: int) -> None:
        self.chave = chave
        self.ordem = ordem

    def __lt__(self, other: "PorChave") -> bool:
        return self.chave < other.chave

    def __gt__(self, other: "PorChave") -> bool:
        return self.chave > other.chave

    def __le__(self, other: "PorChave") -> bool:
        return self.chave <= other.chave

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PorChave) and self.chave == other.chave

    def __repr__(self) -> str:
        return f"({self.chave},{self.ordem})"


# ----------------------------------------------------------------------
# QuickSort — corretude
# ----------------------------------------------------------------------
@pytest.mark.parametrize("pivot", PIVOT_STRATEGIES)
@pytest.mark.parametrize("short", SHORT_VALUES)
@pytest.mark.parametrize("pattern", SORT_PATTERNS)
def test_quicksort_ordena_em_todas_as_combinacoes(
    pivot: str, short: int, pattern: str
) -> None:
    arr = generate_array(300, pattern)
    assert quicksort(arr, short, pivot=pivot) == sorted(arr)


@pytest.mark.parametrize("pivot", PIVOT_STRATEGIES)
def test_quicksort_em_entradas_de_borda(pivot: str) -> None:
    assert quicksort([], pivot=pivot) == []
    assert quicksort([1], pivot=pivot) == [1]
    assert quicksort([2, 1], pivot=pivot) == [1, 2]
    assert quicksort([7] * 50, pivot=pivot) == [7] * 50
    assert quicksort([3, 1, 3, 1, 2], pivot=pivot) == [1, 1, 2, 3, 3]


def test_quicksort_contra_sorted_em_vetores_aleatorios() -> None:
    rng = random.Random(42)
    for _ in range(150):
        arr = [rng.randrange(25) for _ in range(rng.randrange(0, 60))]
        for pivot in PIVOT_STRATEGIES:
            for short in (0, 4, 16):
                assert quicksort(arr, short, pivot=pivot) == sorted(arr)


def test_quicksort_nao_altera_a_entrada() -> None:
    arr = generate_array(100, "aleatorio")
    copia = list(arr)
    quicksort(arr)
    assert arr == copia


def test_quicksort_e_reprodutivel_com_pivo_aleatorio() -> None:
    arr = generate_array(500, "aleatorio")
    _, primeira = quicksort_counted(arr, pivot="random")
    _, segunda = quicksort_counted(arr, pivot="random")
    assert primeira.as_dict() == segunda.as_dict()


def test_quicksort_valida_parametros() -> None:
    with pytest.raises(ValueError, match="short deve ser >= 0"):
        quicksort([1, 2], -1)
    with pytest.raises(ValueError, match="short deve ser int"):
        quicksort([1, 2], 2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="estratégia de pivô"):
        quicksort([1, 2], pivot="primeiro")


# ----------------------------------------------------------------------
# QuickSort — o que o pivô muda (e o que não muda)
# ----------------------------------------------------------------------
def test_pivo_ingenuo_em_entrada_ordenada_e_quadratico() -> None:
    """Θ(n²) exato: toda partição separa 1 de n−1."""
    for n in (100, 500, 1_000):
        arr = generate_array(n, "ordenado")
        _, counters = quicksort_counted(arr, 0, pivot="last")
        assert counters.comparisons == n * (n - 1) // 2


def test_mediana_de_tres_transforma_o_pior_caso_no_melhor() -> None:
    n = 2_000
    arr = generate_array(n, "ordenado")
    _, ingenuo = quicksort_counted(arr, 0, pivot="last")
    _, mediana = quicksort_counted(arr, 0, pivot="median3")
    assert ingenuo.comparisons == n * (n - 1) // 2
    assert mediana.comparisons < 2 * n * math.log2(n)
    assert ingenuo.comparisons > 80 * mediana.comparisons


def test_profundidade_fica_logaritmica_mesmo_no_caso_quadratico() -> None:
    """Decisão de projeto 6: tempo ruim e memória boa são coisas separadas."""
    n = 2_000
    arr = generate_array(n, "ordenado")
    _, counters, profundidade = quicksort_depth(arr, 0, pivot="last")
    assert counters.comparisons == n * (n - 1) // 2  # tempo: Θ(n²)
    assert profundidade <= math.floor(math.log2(n)) + 1  # memória: O(log n)


@pytest.mark.parametrize("pattern", SORT_PATTERNS)
def test_profundidade_nunca_passa_de_log2n_mais_1(pattern: str) -> None:
    n = 4_000
    arr = generate_array(n, pattern)
    for pivot in PIVOT_STRATEGIES:
        _, _, profundidade = quicksort_depth(arr, 0, pivot=pivot)
        assert profundidade <= math.floor(math.log2(n)) + 1


def test_quicksort_nao_estoura_a_pilha_em_entrada_adversarial() -> None:
    """Sem a recursão na metade menor, isto levantaria RecursionError."""
    n = 20 * sys.getrecursionlimit()
    arr = list(range(n))
    resultado, _, profundidade = quicksort_depth(arr, 32, pivot="median3")
    assert resultado == arr
    assert profundidade < sys.getrecursionlimit()


# ----------------------------------------------------------------------
# QuickSort — efeito do short
# ----------------------------------------------------------------------
def test_short_reduz_chamadas_recursivas() -> None:
    arr = generate_array(4_000, "aleatorio")
    chamadas = {}
    for short in SHORT_VALUES:
        _, counters = quicksort_counted(arr, short)
        chamadas[short] = counters.calls
    assert chamadas[32] < chamadas[8] < chamadas[0]
    assert chamadas[32] < chamadas[0] / 8


def test_short_mexe_na_constante_e_nao_na_classe() -> None:
    n = 4_000
    arr = generate_array(n, "aleatorio")
    razoes = []
    for short in SHORT_VALUES:
        _, counters = quicksort_counted(arr, short)
        razoes.append(counters.comparisons / (n * math.log2(n)))
    # todas as variantes continuam na mesma faixa de n·log2n
    assert all(0.7 < razao < 1.6 for razao in razoes)
    assert max(razoes) / min(razoes) < 1.4


def test_short_maior_que_o_vetor_vira_insertion_sort_puro() -> None:
    arr = generate_array(50, "aleatorio")
    resultado, counters = quicksort_counted(arr, 100)
    assert resultado == sorted(arr)
    assert counters.calls == 1  # uma chamada, nenhuma partição


# ----------------------------------------------------------------------
# QuickSelect
# ----------------------------------------------------------------------
def test_quickselect_devolve_a_estatistica_de_ordem_certa() -> None:
    rng = random.Random(42)
    for _ in range(60):
        arr = [rng.randrange(40) for _ in range(rng.randrange(1, 50))]
        ordenado = sorted(arr)
        for k in range(len(arr)):
            assert quickselect(arr, k) == ordenado[k]


@pytest.mark.parametrize("pivot", PIVOT_STRATEGIES)
def test_quickselect_nos_extremos(pivot: str) -> None:
    arr = generate_array(500, "aleatorio")
    assert quickselect(arr, 0, pivot=pivot) == 0
    assert quickselect(arr, 499, pivot=pivot) == 499
    assert quickselect(arr, 250, pivot=pivot) == 250


def test_quickselect_nao_altera_a_entrada() -> None:
    arr = generate_array(100, "aleatorio")
    copia = list(arr)
    quickselect(arr, 50)
    assert arr == copia


def test_quickselect_custa_theta_de_n_e_nao_n_log_n() -> None:
    tamanhos = (1_000, 4_000, 16_000, 64_000)
    razoes_select = []
    razoes_sort = []
    razoes_sort_nlogn = []
    for n in tamanhos:
        arr = generate_array(n, "aleatorio")
        _, select = quickselect_counted(arr, n // 2)
        _, ordenar = quicksort_counted(arr, 16)
        razoes_select.append(select.comparisons / n)
        razoes_sort.append(ordenar.comparisons / n)
        razoes_sort_nlogn.append(ordenar.comparisons / (n * math.log2(n)))

    # select: razão contra n presa numa faixa estreita => Θ(n)
    assert all(1.0 < razao < 6.0 for razao in razoes_select)
    assert max(razoes_select) / min(razoes_select) < 2.5

    # sort: é a razão contra n·log2n que fica estável => Θ(n log n)
    assert max(razoes_sort_nlogn) / min(razoes_sort_nlogn) < 1.3
    # e a razão contra n cresce, acompanhando log2 n. n cresce 64x e log2 n
    # cresce só 1,6x, então o crescimento esperado aqui é ~1,6 e não ~2.
    crescimento = razoes_sort[-1] / razoes_sort[0]
    esperado = math.log2(tamanhos[-1]) / math.log2(tamanhos[0])
    assert crescimento == pytest.approx(esperado, rel=0.20)
    assert crescimento > 1.4


def test_quickselect_valida_parametros() -> None:
    with pytest.raises(ValueError, match="sequência vazia"):
        quickselect([], 0)
    with pytest.raises(ValueError, match="fora da faixa"):
        quickselect([1, 2, 3], 3)
    with pytest.raises(ValueError, match="fora da faixa"):
        quickselect([1, 2, 3], -1)
    with pytest.raises(ValueError, match="goal_index deve ser int"):
        quickselect([1, 2, 3], 1.5)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Variante em lista encadeada
# ----------------------------------------------------------------------
def test_particao_em_tres_separa_menores_iguais_e_maiores() -> None:
    items: SinglyLinkedList[int] = SinglyLinkedList([5, 2, 8, 5, 1, 9, 5])
    less, equal, greater = partition_three_way(items, 5)
    assert less.to_list() == [2, 1]
    assert equal.to_list() == [5, 5, 5]
    assert greater.to_list() == [8, 9]
    assert len(less) + len(equal) + len(greater) == len(items)


def test_particao_em_tres_e_estavel_dentro_de_cada_grupo() -> None:
    """Decisão de projeto 8: anexar no fim preserva a ordem de chegada."""
    entrada = [PorChave(k, i) for i, k in enumerate([3, 1, 3, 1, 5, 3])]
    items: SinglyLinkedList[PorChave] = SinglyLinkedList(entrada)
    less, equal, greater = partition_three_way(items, PorChave(3, -1))
    assert [item.ordem for item in less] == [1, 3]
    assert [item.ordem for item in equal] == [0, 2, 5]
    assert [item.ordem for item in greater] == [4]


def test_particao_conta_saltos_e_comparacoes() -> None:
    items: SinglyLinkedList[int] = SinglyLinkedList(list(range(100)))
    counters = Counters()
    partition_three_way(items, 50, counters)
    assert counters.hops == 99  # um salto por avanço
    assert counters.comparisons >= 100  # ao menos uma comparação por elemento
    assert counters.copies == 100  # uma anexação por elemento


def test_quicksort_linked_ordena() -> None:
    rng = random.Random(42)
    for _ in range(60):
        arr = [rng.randrange(25) for _ in range(rng.randrange(0, 50))]
        items: SinglyLinkedList[int] = SinglyLinkedList(arr)
        assert quicksort_linked(items).to_list() == sorted(arr)


def test_quicksort_linked_e_estavel() -> None:
    """A versão em array (Lomuto) não é; a da lista é, e de graça."""
    entrada = [PorChave(k, i) for i, k in enumerate([3, 1, 3, 2, 1, 3, 2])]
    items: SinglyLinkedList[PorChave] = SinglyLinkedList(entrada)
    resultado = quicksort_linked(items).to_list()
    assert [item.chave for item in resultado] == [1, 1, 2, 2, 3, 3, 3]
    # dentro de cada chave, a ordem original foi preservada
    assert [item.ordem for item in resultado] == [1, 4, 3, 6, 0, 2, 5]


def test_quicksort_linked_nao_altera_a_lista_de_entrada() -> None:
    arr = generate_array(60, "aleatorio")
    items: SinglyLinkedList[int] = SinglyLinkedList(arr)
    quicksort_linked(items)
    assert items.to_list() == arr


def test_quicksort_linked_trata_chaves_iguais_em_tempo_linear() -> None:
    """Partição de três vias: todo mundo cai em `equal` e acaba em um nível."""
    n = 2_000
    items: SinglyLinkedList[int] = SinglyLinkedList([7] * n)
    counters = Counters()
    resultado = quicksort_linked(items, counters)
    assert resultado.to_list() == [7] * n
    # 3 chamadas: a raiz mais duas em listas vazias, que retornam de imediato.
    # Nenhum segundo nível: todos os n elementos caíram em `equal`.
    assert counters.calls == 3
    assert counters.comparisons == 2 * n  # uma tentativa `<` e uma `==` por item


def test_quicksort_linked_protocolo_resultado_counters() -> None:
    items: SinglyLinkedList[int] = SinglyLinkedList([3, 1, 2])
    resultado, counters = quicksort_linked_counted(items)
    assert resultado.to_list() == [1, 2, 3]
    assert counters.calls >= 1


def test_quicksort_linked_estoura_a_pilha_em_entrada_ordenada() -> None:
    """O pivô é o primeiro elemento porque sortear custaria O(n) numa lista.

    O preço: entrada ordenada vira o pior caso, com profundidade n — a mesma
    lição do exercício 5, agora num algoritmo de ordenação.
    """
    n = 3 * sys.getrecursionlimit()
    items: SinglyLinkedList[int] = SinglyLinkedList(range(n))
    with pytest.raises(RecursionError):
        quicksort_linked(items)


# ----------------------------------------------------------------------
# Experimentos
# ----------------------------------------------------------------------
def test_bench_quicksort_cobre_padroes_e_shorts() -> None:
    rows = bench_quicksort([500, 1_000], SORT_PATTERNS, SHORT_VALUES)
    assert len(rows) == 2 * 4 * 3
    for row in rows:
        # com median3 todos os padrões ficam na mesma faixa de n·log2n. A faixa
        # é larga (até 3,0) porque short=32 em n=500 deixa quase todo o trabalho
        # para o insertion sort dentro dos blocos — efeito de constante, não de
        # classe, e é justamente o que test_short_mexe_na_constante verifica.
        assert 0.5 < row["comparisons/n_log2n"] < 3.0
        assert row["profundidade_maxima"] <= math.floor(math.log2(row["n"])) + 1


def test_bench_quicksort_valida_parametros() -> None:
    with pytest.raises(ValueError, match="tamanho deve ser positivo"):
        bench_quicksort([0])
    with pytest.raises(ValueError, match="padrão desconhecido"):
        bench_quicksort([100], ["torto"])


def test_bench_pivot_separa_as_estrategias() -> None:
    rows = bench_pivot_strategies([500, 1_000], ["ordenado", "aleatorio"])
    por_chave = {(row["padrao"], row["pivo"], row["n"]): row for row in rows}

    for n in (500, 1_000):
        ingenuo = por_chave[("ordenado", "last", n)]
        mediana = por_chave[("ordenado", "median3", n)]
        assert ingenuo["comparacoes/n2"] == pytest.approx(0.5, abs=0.01)
        assert mediana["comparacoes/n_log2n"] < 1.5
        # e mesmo o caso Θ(n²) usa pilha O(log n)
        assert ingenuo["profundidade_maxima"] <= ingenuo["log2n_mais_1"]

        # em entrada aleatória as três empatam em ordem de grandeza
        aleatorias = [
            por_chave[("aleatorio", p, n)]["comparisons"] for p in PIVOT_STRATEGIES
        ]
        assert max(aleatorias) / min(aleatorias) < 1.5


def test_bench_pivot_valida_tamanho() -> None:
    with pytest.raises(ValueError, match="tamanho deve ser positivo"):
        bench_pivot_strategies([0])


def test_bench_quickselect_mostra_o_ganho_de_classe() -> None:
    rows = bench_quickselect_vs_sort([1_000, 4_000, 16_000], positions=(0.5,))
    assert len(rows) == 3
    razoes_select = [row["select/n"] for row in rows]
    razoes_sort = [row["sort/n"] for row in rows]
    assert max(razoes_select) / min(razoes_select) < 2.5  # Θ(n)
    assert razoes_sort[-1] > razoes_sort[0]  # Θ(n log n): cresce com log n
    assert all(row["razao_sort_sobre_select"] > 2 for row in rows)


def test_bench_quickselect_valida_parametros() -> None:
    with pytest.raises(ValueError, match="tamanho deve ser positivo"):
        bench_quickselect_vs_sort([0])
    with pytest.raises(ValueError, match="posição deve estar"):
        bench_quickselect_vs_sort([100], positions=(2.0,))


def test_bench_linked_quantifica_o_preco_da_lista() -> None:
    rows = bench_linked_quicksort([250, 500])
    assert len(rows) == 2
    for row in rows:
        assert row["array_saltos"] == 0  # array não segue ponteiro
        assert row["lista_saltos"] > 0
        # a partição de três vias compara mais que a de Lomuto
        assert row["razao_comparacoes"] > 1.0
        # e move muito mais dados, porque cria listas novas a cada nível
        assert row["razao_copias"] > 1.0
        assert row["lista/n_log2n"] > 0


def test_bench_linked_valida_tamanho() -> None:
    with pytest.raises(ValueError, match="tamanho deve ser positivo"):
        bench_linked_quicksort([0])

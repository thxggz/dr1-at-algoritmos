"""Testes do exercício 2 — bubble, selection e insertion instrumentados.

As contagens são verificadas contra **fórmulas fechadas**, não contra valores
observados e congelados: ``n(n−1)/2`` para o selection em qualquer entrada,
``n−1`` para bubble e insertion em entrada ordenada, e os limites ``d`` e
``d + (n−1)`` para o insertion em função do número de inversões. Um teste que
só repete o número que o código produziu hoje não detecta regressão de
contagem — que é justamente o que sustenta toda a análise em Big O do trabalho.
"""

from __future__ import annotations

import random

import pytest

from src.counters import COPIES_PER_SWAP, Counters
from src.searching import generate_array
from src.sorting import (
    COST_TABLE,
    MAX_QUADRATIC_SIZE,
    QUADRATIC_SORTS,
    SORT_PATTERNS,
    bench_bst_vs_insertion,
    bench_nearly_sorted_sensitivity,
    bench_quadratic_sorts,
    bubble_sort,
    bubble_sort_counted,
    count_inversions,
    insertion_sort,
    insertion_sort_counted,
    is_sorted_list,
    selection_sort,
    selection_sort_counted,
)

NOMES = list(QUADRATIC_SORTS)


# ----------------------------------------------------------------------
# Corretude
# ----------------------------------------------------------------------
@pytest.mark.parametrize("nome", NOMES)
@pytest.mark.parametrize("pattern", SORT_PATTERNS)
def test_ordena_todos_os_padroes(nome: str, pattern: str) -> None:
    arr = generate_array(200, pattern)
    resultado = QUADRATIC_SORTS[nome](arr)
    assert resultado == sorted(arr)  # sorted() como referência (convenção 9)
    assert is_sorted_list(resultado) is True


@pytest.mark.parametrize("nome", NOMES)
def test_ordena_entradas_de_borda(nome: str) -> None:
    sorter = QUADRATIC_SORTS[nome]
    assert sorter([]) == []
    assert sorter([7]) == [7]
    assert sorter([2, 1]) == [1, 2]
    assert sorter([5, 5, 5, 5]) == [5, 5, 5, 5]
    assert sorter([3, 1, 3, 1, 2]) == [1, 1, 2, 3, 3]


@pytest.mark.parametrize("nome", NOMES)
def test_ordena_vetores_aleatorios_contra_sorted(nome: str) -> None:
    rng = random.Random(42)
    sorter = QUADRATIC_SORTS[nome]
    for _ in range(100):
        arr = [rng.randrange(30) for _ in range(rng.randrange(0, 40))]
        assert sorter(arr) == sorted(arr)


@pytest.mark.parametrize("nome", NOMES)
def test_nao_altera_a_entrada(nome: str) -> None:
    """Decisão de projeto 1: os três precisam receber o MESMO vetor."""
    arr = generate_array(50, "aleatorio")
    copia = list(arr)
    QUADRATIC_SORTS[nome](arr)
    assert arr == copia


@pytest.mark.parametrize("nome", NOMES)
def test_ordena_strings_tambem(nome: str) -> None:
    palavras = ["pera", "uva", "abacate", "maçã"]
    assert QUADRATIC_SORTS[nome](palavras) == sorted(palavras)


# ----------------------------------------------------------------------
# Contagens contra fórmulas fechadas
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n", [2, 10, 100, 500])
def test_selection_faz_sempre_n_vezes_n_menos_1_sobre_2(n: int) -> None:
    """O ponto do selection: nenhuma organização da entrada ajuda."""
    esperado = n * (n - 1) // 2
    for pattern in SORT_PATTERNS:
        counters = Counters()
        selection_sort(generate_array(n, pattern), counters)
        assert counters.comparisons == esperado


@pytest.mark.parametrize("n", [2, 10, 100, 500])
def test_bubble_e_insertion_fazem_n_menos_1_em_vetor_ordenado(n: int) -> None:
    arr = generate_array(n, "ordenado")

    bubble = Counters()
    bubble_sort(arr, bubble)
    assert bubble.comparisons == n - 1  # uma passagem, sem troca nenhuma
    assert bubble.copies == 0

    insertion = Counters()
    insertion_sort(arr, insertion)
    assert insertion.comparisons == n - 1
    assert insertion.copies == 2 * (n - 1)  # ler a chave e escrevê-la de volta


@pytest.mark.parametrize("n", [2, 10, 100, 400])
def test_vetor_reverso_e_o_pior_caso_dos_tres(n: int) -> None:
    arr = generate_array(n, "reverso")
    pior = n * (n - 1) // 2

    contagens = {}
    for nome, sorter in QUADRATIC_SORTS.items():
        counters = Counters()
        sorter(arr, counters)
        contagens[nome] = counters
        assert counters.comparisons == pior

    # mesmas comparações, movimentação de dados muito diferente:
    assert contagens["bubble"].copies == COPIES_PER_SWAP * pior
    assert contagens["insertion"].copies == pior + 2 * (n - 1)
    assert contagens["selection"].copies == COPIES_PER_SWAP * (n // 2)
    # selection move Θ(n) dados (≈1,5n) e insertion move Θ(n²) (≈n²/2), então a
    # razão entre os dois cresce com n (≈ n/3). Em n=2 as fórmulas coincidem em
    # 3, por isso a desigualdade estrita só vale a partir de n maior.
    assert contagens["selection"].copies <= contagens["insertion"].copies
    if n >= 10:
        razao = contagens["insertion"].copies / contagens["selection"].copies
        assert razao > n / 5


@pytest.mark.parametrize("pattern", SORT_PATTERNS)
@pytest.mark.parametrize("n", [50, 200, 500])
def test_insertion_custa_entre_d_e_d_mais_n_menos_1(pattern: str, n: int) -> None:
    """A afirmação Θ(n + d), verificada pelos dois lados."""
    arr = generate_array(n, pattern)
    d = count_inversions(arr)
    counters = Counters()
    insertion_sort(arr, counters)
    assert d <= counters.comparisons <= d + (n - 1)


def test_insertion_em_quase_ordenado_bate_exatamente_n_menos_1_mais_d() -> None:
    arr = generate_array(500, "quase_ordenado")
    d = count_inversions(arr)
    counters = Counters()
    insertion_sort(arr, counters)
    # nenhum elemento vai até a posição 0, então não há comparação economizada
    assert counters.comparisons == (500 - 1) + d


def test_bubble_para_cedo_quando_nao_ha_troca() -> None:
    """Sem a detecção, bubble faria n(n−1)/2 até num vetor ordenado."""
    n = 200
    counters = Counters()
    bubble_sort(generate_array(n, "ordenado"), counters)
    assert counters.comparisons == n - 1
    assert counters.comparisons < n * (n - 1) // 2 / 50


def test_uma_troca_conta_tres_copias() -> None:
    counters = Counters()
    bubble_sort([2, 1], counters)
    assert counters.comparisons == 1
    assert counters.copies == COPIES_PER_SWAP == 3


@pytest.mark.parametrize(
    ("funcao", "esperado_comparacoes"),
    [(bubble_sort_counted, 1), (selection_sort_counted, 1), (insertion_sort_counted, 1)],
)
def test_protocolo_resultado_counters(
    funcao: object, esperado_comparacoes: int
) -> None:
    resultado, counters = funcao([2, 1])  # type: ignore[operator]
    assert resultado == [1, 2]
    assert counters.comparisons == esperado_comparacoes


def test_contadores_acumulam_no_contador_do_chamador() -> None:
    counters = Counters()
    bubble_sort([2, 1], counters)
    insertion_sort([2, 1], counters)
    assert counters.comparisons == 2


# ----------------------------------------------------------------------
# Ferramentas de análise
# ----------------------------------------------------------------------
def test_is_sorted_list() -> None:
    assert is_sorted_list([]) is True
    assert is_sorted_list([1]) is True
    assert is_sorted_list([1, 1, 2]) is True
    assert is_sorted_list([2, 1]) is False


def test_count_inversions_em_casos_conhecidos() -> None:
    assert count_inversions([]) == 0
    assert count_inversions([1]) == 0
    assert count_inversions([1, 2, 3]) == 0
    assert count_inversions([3, 2, 1]) == 3  # todos os pares
    assert count_inversions([2, 1]) == 1
    assert count_inversions([1, 1, 1]) == 0  # iguais não são inversão


@pytest.mark.parametrize("n", [5, 20, 60])
def test_count_inversions_bate_com_a_contagem_ingenua(n: int) -> None:
    rng = random.Random(42)
    for _ in range(20):
        arr = [rng.randrange(15) for _ in range(n)]
        ingenua = sum(
            1 for i in range(n) for j in range(i + 1, n) if arr[i] > arr[j]
        )
        assert count_inversions(arr) == ingenua


def test_vetor_reverso_tem_o_maximo_de_inversoes() -> None:
    n = 100
    assert count_inversions(generate_array(n, "reverso")) == n * (n - 1) // 2
    assert count_inversions(generate_array(n, "ordenado")) == 0


# ----------------------------------------------------------------------
# Experimento principal
# ----------------------------------------------------------------------
def test_bench_quadratic_cobre_os_quatro_padroes() -> None:
    rows = bench_quadratic_sorts([100, 200], SORT_PATTERNS)
    assert len(rows) == 2 * 4 * 3  # tamanhos x padrões x algoritmos
    assert {row["padrao"] for row in rows} == set(SORT_PATTERNS)
    assert {row["algoritmo"] for row in rows} == set(NOMES)
    assert all(row["ordenado_ok"] is True for row in rows)


def test_bench_quadratic_mostra_a_razao_estavel_contra_n2() -> None:
    rows = bench_quadratic_sorts([100, 200, 400], ["reverso"])
    for nome in NOMES:
        razoes = [
            row["comparisons/n2"] for row in rows if row["algoritmo"] == nome
        ]
        # pior caso dos três: exatamente n(n−1)/2, razão -> 0,5
        assert all(razao == pytest.approx(0.5, abs=0.01) for razao in razoes)


def test_bench_quadratic_mostra_o_selection_indiferente_ao_padrao() -> None:
    rows = bench_quadratic_sorts([200], SORT_PATTERNS, algorithms=["selection"])
    contagens = {row["comparisons"] for row in rows}
    assert len(contagens) == 1  # o mesmo número nos quatro padrões
    assert contagens == {200 * 199 // 2}


def test_bench_quadratic_respeita_o_teto_de_4000() -> None:
    with pytest.raises(ValueError, match="teto de 4000"):
        bench_quadratic_sorts([MAX_QUADRATIC_SIZE + 1])


def test_bench_quadratic_valida_parametros() -> None:
    with pytest.raises(ValueError, match="tamanho deve ser positivo"):
        bench_quadratic_sorts([0])
    with pytest.raises(ValueError, match="padrão desconhecido"):
        bench_quadratic_sorts([10], ["torto"])
    with pytest.raises(ValueError, match="algoritmo desconhecido"):
        bench_quadratic_sorts([10], ["ordenado"], algorithms=["merge"])


# ----------------------------------------------------------------------
# Sensibilidade a "quase ordenado" — a pergunta do enunciado
# ----------------------------------------------------------------------
def test_sensibilidade_responde_quem_mais_se_beneficia() -> None:
    rows = bench_nearly_sorted_sensitivity(400, (0.0, 0.05, 0.25, 1.0))
    por_algoritmo: dict[str, list[dict[str, object]]] = {nome: [] for nome in NOMES}
    for row in rows:
        por_algoritmo[row["algoritmo"]].append(row)

    # selection: constante, indiferente à desordem
    selection = [row["comparisons"] for row in por_algoritmo["selection"]]
    assert len(set(selection)) == 1
    assert selection[0] == 400 * 399 // 2

    # insertion: acompanha d de perto, razão comparações/(n+d) presa perto de 1
    for row in por_algoritmo["insertion"]:
        assert row["comparacoes/(n+d)"] == pytest.approx(1.0, abs=0.05)

    # e é o que menos compara quando a entrada está quase ordenada
    quase = {
        row["algoritmo"]: row["comparisons"]
        for row in rows
        if row["desordem"] == 0.05
    }
    assert quase["insertion"] < quase["bubble"] < quase["selection"]

    # o ganho do insertion sobre o pior caso é enorme com pouca desordem
    pouca = next(
        row
        for row in por_algoritmo["insertion"]
        if row["desordem"] == 0.05
    )
    assert pouca["comparacoes/pior_caso"] < 0.05


def test_sensibilidade_com_desordem_zero_e_o_melhor_caso() -> None:
    rows = bench_nearly_sorted_sensitivity(200, (0.0,))
    por_algoritmo = {row["algoritmo"]: row for row in rows}
    assert por_algoritmo["insertion"]["inversoes"] == 0
    assert por_algoritmo["insertion"]["comparisons"] == 199
    assert por_algoritmo["bubble"]["comparisons"] == 199
    assert por_algoritmo["selection"]["comparisons"] == 200 * 199 // 2


def test_sensibilidade_valida_parametros() -> None:
    with pytest.raises(ValueError, match="tamanho deve ser positivo"):
        bench_nearly_sorted_sensitivity(0)
    with pytest.raises(ValueError, match="teto"):
        bench_nearly_sorted_sensitivity(MAX_QUADRATIC_SIZE + 1)
    with pytest.raises(ValueError, match="disorder"):
        bench_nearly_sorted_sensitivity(100, (1.5,))


# ----------------------------------------------------------------------
# BST × insertion sort
# ----------------------------------------------------------------------
def test_bst_e_insertion_produzem_o_mesmo_resultado() -> None:
    rows = bench_bst_vs_insertion([100, 200], SORT_PATTERNS)
    assert len(rows) == 2 * 4
    # a própria bench levanta se discordarem; aqui confirma-se que rodou
    assert {row["padrao"] for row in rows} == set(SORT_PATTERNS)


def test_o_melhor_caso_de_um_e_o_pior_caso_do_outro() -> None:
    """O achado central do experimento."""
    rows = {
        row["padrao"]: row for row in bench_bst_vs_insertion([400], SORT_PATTERNS)
    }

    ordenado = rows["ordenado"]
    assert ordenado["insertion_comparacoes"] == 399  # Θ(n): melhor caso
    assert ordenado["bst_comparacoes_total"] == 400 * 399 // 2  # Θ(n²): pior caso
    assert ordenado["bst_degenerada"] is True
    assert ordenado["vencedor"] == "insertion"
    assert ordenado["razao_bst_sobre_insertion"] > 100

    aleatorio = rows["aleatorio"]
    assert aleatorio["bst_degenerada"] is False
    assert aleatorio["bst_comparacoes_total"] < aleatorio["insertion_comparacoes"]
    assert aleatorio["vencedor"] == "BST"
    assert aleatorio["bst/n_log2n"] < 1.5
    assert aleatorio["insertion/n2"] == pytest.approx(0.25, abs=0.06)

    # no vetor reverso os dois fazem exatamente n(n−1)/2: é empate, e o campo
    # precisa dizer isso em vez de eleger um vencedor por desempate de código
    reverso = rows["reverso"]
    assert reverso["insertion_comparacoes"] == reverso["bst_comparacoes_total"]
    assert reverso["vencedor"] == "empate"

    # quase_ordenado é QUASE degenerado: a bandeira estrita (altura == n) não
    # pega, mas a razão de altura mostra o problema
    quase = rows["quase_ordenado"]
    assert quase["bst_degenerada"] is False
    assert quase["bst_altura"] > 0.8 * 400  # altura quase igual a n
    assert quase["bst_razao_altura"] > 10  # dezenas de vezes a altura ideal
    assert quase["vencedor"] == "insertion"


def test_travessia_in_order_nao_faz_comparacao_de_chave() -> None:
    """O custo da BST está todo na construção; a leitura é O(n) e sem comparar."""
    for row in bench_bst_vs_insertion([200], ["aleatorio"]):
        assert row["bst_comparacoes_travessia"] == 0
        assert row["bst_comparacoes_total"] == row["bst_comparacoes_insercao"]


def test_bst_cobra_memoria_que_o_insertion_nao_cobra() -> None:
    row = bench_bst_vs_insertion([300], ["aleatorio"])[0]
    assert row["memoria_extra_insertion"] == "Θ(1)"
    assert "300 nós" in row["memoria_extra_bst"]


def test_bench_bst_vs_insertion_valida_parametros() -> None:
    with pytest.raises(ValueError, match="tamanho deve ser positivo"):
        bench_bst_vs_insertion([0])
    with pytest.raises(ValueError, match="teto"):
        bench_bst_vs_insertion([MAX_QUADRATIC_SIZE + 1])
    with pytest.raises(ValueError, match="padrão desconhecido"):
        bench_bst_vs_insertion([10], ["torto"])


# ----------------------------------------------------------------------
# Tabela de custo
# ----------------------------------------------------------------------
def test_tabela_de_custo_cobre_os_tres_mais_a_bst() -> None:
    algoritmos = {row["algoritmo"] for row in COST_TABLE}
    assert {"bubble", "selection", "insertion"} <= algoritmos
    assert any("BST" in nome for nome in algoritmos)
    for row in COST_TABLE:
        assert set(row) == {
            "algoritmo", "melhor", "medio", "pior", "espaco", "aproveita_ordem"
        }
        assert row["espaco"]


def test_tabela_registra_que_o_selection_nao_aproveita_ordem() -> None:
    selection = next(row for row in COST_TABLE if row["algoritmo"] == "selection")
    assert "NÃO" in selection["aproveita_ordem"]
    assert "n(n−1)/2" in selection["melhor"]

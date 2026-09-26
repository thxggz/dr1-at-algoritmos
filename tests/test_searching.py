"""Testes do exercício 1 — busca linear, busca binária e array × lista.

Quatro exigências do enunciado viram teste explícito:

* as duas buscas devolvem a posição ou ``-1`` (tratamento de "não encontrado");
* a busca binária tem verificação **explícita** de pré-condição, com
  comportamento definido quando ela falha;
* a estratégia de falha rápida é mais barata que ordenar, e seu risco de falso
  positivo é quantificado — não apenas mencionado;
* a busca linear sobre :class:`SinglyLinkedList` faz **exatamente** as mesmas
  comparações que sobre array.
"""

from __future__ import annotations

import math

import pytest

from src.counters import Counters
from src.linked_list import SinglyLinkedList
from src.searching import (
    CHECK_MODES,
    DEFAULT_SAMPLE_SIZE,
    PATTERNS,
    InvalidCheckModeError,
    InvalidPatternError,
    SortSample,
    UnsortedInputError,
    bench_array_vs_linked,
    bench_fast_fail,
    bench_search_break_even,
    bench_searches,
    binary_search,
    binary_search_counted,
    demonstrate_precondition,
    find_first_descent,
    generate_array,
    is_sorted,
    linear_search,
    linear_search_counted,
    linear_search_linked,
    linear_search_linked_counted,
    sample_sorted_check,
)


# ----------------------------------------------------------------------
# Gerador de vetores
# ----------------------------------------------------------------------
@pytest.mark.parametrize("pattern", PATTERNS)
def test_gerador_produz_permutacao_de_0_a_n_menos_1(pattern: str) -> None:
    arr = generate_array(200, pattern)
    assert len(arr) == 200
    assert sorted(arr) == list(range(200))  # valores distintos, só a ordem muda


def test_gerador_respeita_cada_padrao() -> None:
    assert generate_array(5, "ordenado") == [0, 1, 2, 3, 4]
    assert generate_array(5, "reverso") == [4, 3, 2, 1, 0]
    assert generate_array(200, "aleatorio") != list(range(200))
    quase = generate_array(200, "quase_ordenado")
    assert quase != list(range(200))
    # "quase" tem de ser bem mais ordenado que "aleatório"
    descents_quase = sum(1 for i in range(199) if quase[i] > quase[i + 1])
    aleatorio = generate_array(200, "aleatorio")
    descents_aleatorio = sum(
        1 for i in range(199) if aleatorio[i] > aleatorio[i + 1]
    )
    assert descents_quase < descents_aleatorio / 3


def test_gerador_e_reprodutivel_com_a_mesma_semente() -> None:
    assert generate_array(100, "aleatorio", seed=42) == generate_array(
        100, "aleatorio", seed=42
    )
    assert generate_array(100, "aleatorio", seed=1) != generate_array(
        100, "aleatorio", seed=2
    )


def test_gerador_aceita_vetor_vazio_e_unitario() -> None:
    for pattern in PATTERNS:
        assert generate_array(0, pattern) == []
        assert generate_array(1, pattern) == [0]


def test_gerador_valida_parametros() -> None:
    with pytest.raises(InvalidPatternError, match="padrão desconhecido"):
        generate_array(10, "bagunçado")
    with pytest.raises(InvalidPatternError, match="n deve ser >= 0"):
        generate_array(-1, "ordenado")
    with pytest.raises(InvalidPatternError, match="n deve ser int"):
        generate_array("10", "ordenado")  # type: ignore[arg-type]
    with pytest.raises(InvalidPatternError, match="disorder"):
        generate_array(10, "quase_ordenado", disorder=2.0)


# ----------------------------------------------------------------------
# Busca linear em array
# ----------------------------------------------------------------------
def test_linear_search_devolve_a_posicao() -> None:
    arr = [10, 20, 30, 40]
    assert [linear_search(arr, v) for v in arr] == [0, 1, 2, 3]


def test_linear_search_devolve_menos_um_quando_ausente() -> None:
    assert linear_search([10, 20, 30], 99) == -1
    assert linear_search([], 1) == -1


def test_linear_search_devolve_a_primeira_ocorrencia() -> None:
    assert linear_search([7, 3, 7, 3], 7) == 0
    assert linear_search([7, 3, 7, 3], 3) == 1


@pytest.mark.parametrize("position", [0, 1, 5, 49])
def test_linear_search_conta_uma_comparacao_por_elemento(position: int) -> None:
    arr = generate_array(50, "ordenado")
    counters = Counters()
    assert linear_search(arr, arr[position], counters) == position
    assert counters.comparisons == position + 1
    assert counters.hops == 0  # array é contíguo: nenhum ponteiro a seguir
    assert counters.copies == 0  # busca não move dados (decisão de projeto 5)


def test_linear_search_pior_caso_custa_n() -> None:
    for n in (1, 10, 1_000):
        counters = Counters()
        assert linear_search(generate_array(n, "aleatorio"), n + 1, counters) == -1
        assert counters.comparisons == n


def test_linear_search_protocolo_resultado_counters() -> None:
    resultado, counters = linear_search_counted([1, 2, 3], 3)
    assert resultado == 2
    assert counters.comparisons == 3


# ----------------------------------------------------------------------
# Busca binária
# ----------------------------------------------------------------------
def test_binary_search_encontra_todos_os_elementos() -> None:
    for n in (1, 2, 3, 17, 64, 1_000):
        arr = generate_array(n, "ordenado")
        for value in arr:
            assert binary_search(arr, value, check="off") == value


def test_binary_search_devolve_menos_um_quando_ausente() -> None:
    arr = generate_array(100, "ordenado")
    assert binary_search(arr, -1, check="off") == -1
    assert binary_search(arr, 100, check="off") == -1
    assert binary_search([], 1, check="off") == -1


@pytest.mark.parametrize(
    ("n", "esperado"), [(1, 1), (16, 5), (17, 5), (100, 7), (1_024, 11)]
)
def test_binary_search_pior_caso_e_log2_mais_um(n: int, esperado: int) -> None:
    """Alvo acima de todos: o pior caso, ⌊log₂n⌋+1 sondagens."""
    arr = generate_array(n, "ordenado")
    counters = Counters()
    assert binary_search(arr, n + 1, counters, check="off") == -1
    assert counters.comparisons == esperado == math.floor(math.log2(n)) + 1
    assert counters.copies == 0


def test_binary_search_melhor_caso_e_uma_comparacao() -> None:
    for n in (1, 15, 16, 1_000):
        arr = generate_array(n, "ordenado")
        counters = Counters()
        assert binary_search(arr, arr[(n - 1) // 2], counters, check="off") >= 0
        assert counters.comparisons == 1


def test_binary_search_nunca_passa_do_pior_caso() -> None:
    n = 500
    arr = generate_array(n, "ordenado")
    teto = math.floor(math.log2(n)) + 1
    for value in arr:
        counters = Counters()
        binary_search(arr, value, counters, check="off")
        assert counters.comparisons <= teto


def test_binary_search_protocolo_resultado_counters() -> None:
    resultado, counters = binary_search_counted(
        generate_array(64, "ordenado"), 40, check="off"
    )
    assert resultado == 40
    assert 0 < counters.comparisons <= 7


# ----------------------------------------------------------------------
# Pré-condição: verificação e comportamento na falha
# ----------------------------------------------------------------------
def test_find_first_descent_e_is_sorted() -> None:
    assert find_first_descent([1, 2, 3]) == -1
    assert find_first_descent([1, 3, 2]) == 1
    assert find_first_descent([3, 1, 2]) == 0
    assert is_sorted(generate_array(100, "ordenado")) is True
    assert is_sorted(generate_array(100, "reverso")) is False


def test_verificacao_completa_custa_n_menos_um() -> None:
    counters = Counters()
    arr = generate_array(1_000, "ordenado")
    assert is_sorted(arr, counters) is True
    assert counters.comparisons == 999  # é o custo O(n) que a amostragem evita


@pytest.mark.parametrize("check", ["sampled", "full"])
def test_pre_condicao_violada_levanta_e_nao_devolve_menos_um(check: str) -> None:
    """Decisão de projeto 3: -1 quer dizer 'não encontrei', não 'vetor errado'."""
    arr = generate_array(1_000, "reverso")
    with pytest.raises(UnsortedInputError, match="pré-condição de ordenação falhou"):
        binary_search(arr, 500, check=check)


def test_mensagem_da_falha_traz_o_indice_e_os_valores() -> None:
    arr = [1, 2, 9, 3, 4]
    with pytest.raises(UnsortedInputError, match=r"arr\[2\]=9 > arr\[3\]=3"):
        binary_search(arr, 4, check="full")


def test_modo_off_devolve_resposta_errada_em_silencio() -> None:
    """A justificativa de a verificação existir: sem ela, o erro não aparece."""
    arr = generate_array(1_000, "reverso")
    alvo = arr[300]
    assert binary_search(arr, alvo, check="off") != arr.index(alvo)


def test_verificacao_nao_reprova_vetor_ordenado() -> None:
    arr = generate_array(10_000, "ordenado")
    for check in CHECK_MODES:
        assert binary_search(arr, 7_777, check=check) == 7_777


def test_custo_de_cada_modo_de_verificacao() -> None:
    n = 10_000
    arr = generate_array(n, "ordenado")
    custos: dict[str, int] = {}
    for check in CHECK_MODES:
        counters = Counters()
        binary_search(arr, 7_777, counters, check=check)
        custos[check] = counters.comparisons

    busca_pura = custos["off"]
    # amostrada: busca + k comparações (constante) -> continua O(log n)
    assert custos["sampled"] == busca_pura + DEFAULT_SAMPLE_SIZE
    # completa: busca + (n-1) -> a chamada inteira vira O(n)
    assert custos["full"] == busca_pura + (n - 1)
    assert custos["full"] > 100 * custos["sampled"]


def test_modo_de_verificacao_invalido_levanta() -> None:
    with pytest.raises(InvalidCheckModeError, match="modo de verificação"):
        binary_search([1, 2, 3], 2, check="talvez")


# ----------------------------------------------------------------------
# Falha rápida: custo e risco de falso positivo
# ----------------------------------------------------------------------
def test_amostragem_custa_k_e_nao_n() -> None:
    arr = generate_array(100_000, "ordenado")
    counters = Counters()
    sample = sample_sorted_check(arr, 16, counters=counters)
    assert sample.looks_sorted is True
    assert sample.sampled_pairs == 16
    assert counters.comparisons == 16  # e não 99.999


def test_amostragem_sempre_detecta_vetor_reverso() -> None:
    """d = n-1: qualquer par olhado está fora de ordem."""
    arr = generate_array(1_000, "reverso")
    for seed in range(50):
        sample = sample_sorted_check(arr, 1, seed=seed)
        assert sample.looks_sorted is False
        assert sample.comparisons == 1  # detecta na primeira comparação


def test_amostragem_quase_sempre_detecta_vetor_aleatorio() -> None:
    arr = generate_array(10_000, "aleatorio")
    detectados = sum(
        1 for seed in range(100) if not sample_sorted_check(arr, 16, seed=seed).looks_sorted
    )
    assert detectados == 100  # 0,5^16 ≈ 1,5e-5 de chance de escapar


def test_amostragem_quase_nunca_detecta_uma_unica_inversao() -> None:
    """O risco de falso positivo, medido em vez de apenas mencionado."""
    n = 10_000
    arr = generate_array(n, "ordenado")
    meio = n // 2
    arr[meio], arr[meio + 1] = arr[meio + 1], arr[meio]
    assert is_sorted(arr) is False  # o vetor realmente está errado

    detectados = sum(
        1 for seed in range(200) if not sample_sorted_check(arr, 16, seed=seed).looks_sorted
    )
    assert detectados <= 2  # previsto: 1 - (1 - 1/9999)^16 ≈ 0,16%


def test_amostragem_limita_k_ao_numero_de_pares() -> None:
    sample = sample_sorted_check([1, 2, 3], 100)
    assert sample.sampled_pairs == 2  # só existem 2 pares adjacentes
    assert sample_sorted_check([], 4).sampled_pairs == 0
    assert sample_sorted_check([1], 4).looks_sorted is True


def test_amostragem_valida_k() -> None:
    with pytest.raises(ValueError, match="sample_size deve ser >= 1"):
        sample_sorted_check([1, 2, 3], 0)


def test_miss_probability_bate_com_a_formula() -> None:
    sample = SortSample(looks_sorted=True, sampled_pairs=16, comparisons=16,
                        first_violation=None)
    assert sample.miss_probability(10_001, 1) == pytest.approx(
        (1 - 1 / 10_000) ** 16
    )
    assert sample.miss_probability(1_001, 1_000) == 0.0  # reverso: nunca escapa
    assert sample.miss_probability(1_000, 0) == 0.0  # sem violação, nada a perder


# ----------------------------------------------------------------------
# Array × lista encadeada
# ----------------------------------------------------------------------
def test_busca_em_lista_encadeada_devolve_a_posicao() -> None:
    items: SinglyLinkedList[int] = SinglyLinkedList([10, 20, 30])
    assert [linear_search_linked(items, v) for v in (10, 20, 30)] == [0, 1, 2]
    assert linear_search_linked(items, 99) == -1
    assert linear_search_linked(SinglyLinkedList(), 1) == -1


def test_busca_na_lista_conta_igual_ao_metodo_do_exercicio_10() -> None:
    """Prova que a reescrita é fiel: mesmo algoritmo, mesmas contagens."""
    arr = generate_array(200, "aleatorio")
    items: SinglyLinkedList[int] = SinglyLinkedList(arr)
    for alvo in (arr[0], arr[99], arr[199], 999):
        reescrita = Counters()
        metodo = Counters()
        assert linear_search_linked(items, alvo, reescrita) == items.search(
            alvo, metodo
        )
        assert reescrita.as_dict() == metodo.as_dict()


def test_array_e_lista_fazem_as_mesmas_comparacoes() -> None:
    """A conclusão central do exercício: mesmo algoritmo, mesmas comparações;
    o que muda é o custo de chegar ao próximo elemento."""
    arr = generate_array(500, "aleatorio")
    items: SinglyLinkedList[int] = SinglyLinkedList(arr)
    for alvo in (arr[0], arr[250], arr[499], 999):
        no_array = Counters()
        na_lista = Counters()
        assert linear_search(arr, alvo, no_array) == linear_search_linked(
            items, alvo, na_lista
        )
        assert no_array.comparisons == na_lista.comparisons
        assert no_array.hops == 0
        assert na_lista.hops == no_array.comparisons - (0 if alvo == 999 else 1)


def test_busca_na_lista_protocolo_resultado_counters() -> None:
    items: SinglyLinkedList[int] = SinglyLinkedList([1, 2, 3])
    resultado, counters = linear_search_linked_counted(items, 3)
    assert resultado == 2
    assert counters.comparisons == 3
    assert counters.hops == 2


# ----------------------------------------------------------------------
# Experimentos
# ----------------------------------------------------------------------
def test_bench_searches_separa_melhor_medio_e_pior() -> None:
    rows = bench_searches([100, 1_000, 10_000], average_samples=8)
    lineares = [row for row in rows if row["algoritmo"].startswith("linear")]
    binarias = [row for row in rows if row["algoritmo"].startswith("binary")]
    assert len(lineares) == 3 * 3 * 3  # 3 tamanhos x 3 padrões x 3 casos
    assert len(binarias) == 3 * 3  # 3 tamanhos x 3 casos

    for row in lineares:
        assert row["copies"] == 0  # decisão de projeto 5: a coluna fica na tabela
        assert row["hops"] == 0
        n = row["n"]
        if row["caso"].startswith("melhor"):
            assert row["comparacoes_por_busca"] == 1
        elif row["caso"].startswith("pior"):
            assert row["comparacoes_por_busca"] == n
            assert row["comparacoes_por_busca/n"] == 1.0
        else:
            # estratificado: média ≈ (n+1)/2, ou seja razão ≈ 0,5
            assert row["comparacoes_por_busca/n"] == pytest.approx(0.5, abs=0.05)

    for row in binarias:
        n = row["n"]
        if row["caso"].startswith("melhor"):
            assert row["comparacoes_por_busca"] == 1
        elif row["caso"].startswith("pior"):
            assert row["comparacoes_por_busca"] == math.floor(math.log2(n)) + 1
            assert row["comparacoes_por_busca/log2n"] == pytest.approx(1.0, abs=0.2)


def test_bench_searches_mostra_a_separacao_entre_as_duas_curvas() -> None:
    rows = bench_searches([100, 10_000], average_samples=8)
    piores = {
        (row["algoritmo"], row["n"]): row["comparacoes_por_busca"]
        for row in rows
        if row["caso"].startswith("pior")
    }
    # a linear cresce 100x quando n cresce 100x; a binária cresce ~1,4x
    linear = piores[("linear_search (array)", 10_000)] / piores[
        ("linear_search (array)", 100)
    ]
    binaria = piores[("binary_search (array ordenado)", 10_000)] / piores[
        ("binary_search (array ordenado)", 100)
    ]
    assert linear == pytest.approx(100.0)
    assert binaria < 2.5


def test_bench_searches_valida_parametros() -> None:
    with pytest.raises(ValueError, match="tamanho deve ser positivo"):
        bench_searches([0])
    with pytest.raises(InvalidPatternError):
        bench_searches([10], patterns=["torto"])


def test_demonstrate_precondition_cobre_os_tres_modos() -> None:
    rows = demonstrate_precondition(500)
    assert len(rows) == 3 * 3  # 3 padrões x 3 modos

    ordenado = [row for row in rows if row["padrao"] == "ordenado"]
    assert all(row["detectou"] is False for row in ordenado)
    assert all(row["indice_devolvido"] == row["indice_correto"] for row in ordenado)
    assert all(row["pares_fora_de_ordem"] == 0 for row in ordenado)

    for padrao in ("reverso", "aleatorio"):
        grupo = {row["modo"]: row for row in rows if row["padrao"] == padrao}
        assert grupo["sampled"]["detectou"] is True
        assert grupo["full"]["detectou"] is True
        # no modo off passa batido e devolve resposta errada sem avisar
        assert grupo["off"]["detectou"] is False
        assert grupo["off"]["resposta_errada_em_silencio"] is True


def test_demonstrate_precondition_valida_n() -> None:
    with pytest.raises(ValueError, match="n deve ser >= 2"):
        demonstrate_precondition(1)


def test_bench_fast_fail_confirma_a_previsao_teorica() -> None:
    rows = bench_fast_fail(n=10_000, sample_sizes=(1, 4, 16), trials=100)
    assert len(rows) == 3 * 3

    for row in rows:
        # custo da amostragem é k, muito abaixo dos n-1 da verificação completa
        assert row["comparacoes_por_verificacao"] <= row["k"]
        assert row["custo_relativo_vs_completa"] < 0.01
        # medido bate com (1 - d/(n-1))^k
        assert row["taxa_deteccao_medida"] == pytest.approx(
            row["taxa_deteccao_prevista"], abs=0.12
        )

    reverso = [row for row in rows if row["padrao"] == "reverso"]
    assert all(row["taxa_deteccao_medida"] == 1.0 for row in reverso)

    uma = {row["k"]: row for row in rows if row["padrao"] == "uma_inversao"}
    assert uma[16]["taxa_deteccao_medida"] < 0.05
    assert uma[16]["risco_falso_positivo_previsto"] > 0.99


def test_bench_fast_fail_valida_parametros() -> None:
    with pytest.raises(ValueError, match="trials deve ser >= 1"):
        bench_fast_fail(trials=0)
    with pytest.raises(ValueError, match="n deve ser >= 4"):
        bench_fast_fail(n=2)
    with pytest.raises(ValueError, match="sample_size deve ser >= 1"):
        bench_fast_fail(n=100, sample_sizes=(0,), trials=1)


def test_bench_array_vs_linked_isola_a_diferenca() -> None:
    rows = bench_array_vs_linked([500, 1_000], repeats=1)
    assert len(rows) == 4

    for n in (500, 1_000):
        do_n = {row["estrutura"]: row for row in rows if row["n"] == n}
        array, lista = do_n["array"], do_n["lista encadeada"]
        # mesmo algoritmo: exatamente as mesmas comparações
        assert array["comparisons"] == lista["comparisons"] == n
        # o que muda: o array não segue ponteiro nenhum
        assert array["hops"] == 0
        assert lista["hops"] == n
        assert array["copies"] == lista["copies"] == 0


def test_bench_array_vs_linked_valida_parametros() -> None:
    with pytest.raises(ValueError, match="repeats deve ser >= 1"):
        bench_array_vs_linked([10], repeats=0)
    with pytest.raises(ValueError, match="tamanho deve ser positivo"):
        bench_array_vs_linked([0])


def test_bench_break_even_mostra_quando_vale_ordenar() -> None:
    rows = bench_search_break_even([1_000, 10_000], (1, 4, 16, 64, 256))
    assert len(rows) == 2 * 5

    for row in rows:
        assert row["comparacoes_binaria_por_busca"] < row[
            "comparacoes_linear_por_busca"
        ]
        # o equilíbrio é ~2·log2(n): dezenas de consultas, não milhares
        assert row["equilibrio_teorico_m"] == pytest.approx(
            2 * math.log2(row["n"]), rel=0.15
        )

    for n in (1_000, 10_000):
        do_n = {row["consultas"]: row for row in rows if row["n"] == n}
        assert do_n[1]["vale_ordenar"] is False  # uma consulta só: não compensa
        assert do_n[256]["vale_ordenar"] is True  # muitas consultas: compensa
        # e a decisão é monotônica em m
        decisoes = [do_n[m]["vale_ordenar"] for m in (1, 4, 16, 64, 256)]
        assert decisoes == sorted(decisoes)


def test_bench_break_even_valida_parametros() -> None:
    with pytest.raises(ValueError, match="tamanho deve ser positivo"):
        bench_search_break_even([0])
    with pytest.raises(ValueError, match="query_counts deve ser >= 1"):
        bench_search_break_even([100], (0,))

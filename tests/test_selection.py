"""Testes do exercício 3 — deduplicação e k menores.

O enunciado pede "testes que provem equivalência de saída" entre a versão lenta
e a rápida. Aqui isso é feito por **modelo de referência** (``dict.fromkeys``
para a ordem do primeiro aparecimento, ``sorted()[:k]`` para os k menores) e
sobre entradas sorteadas, não sobre três exemplos escolhidos a dedo: uma
otimização que funciona nos casos que o autor lembrou não está provada.
"""

from __future__ import annotations

import math
import random

import pytest

from src.bst import BinarySearchTree
from src.counters import Counters
from src.hashtable import find_colliding_keys
from src.searching import generate_array
from src.selection import (
    InvalidKError,
    bench_deduplicate,
    bench_k_smallest,
    bench_k_smallest_bst,
    deduplicate_fast,
    deduplicate_fast_counted,
    deduplicate_slow,
    deduplicate_slow_counted,
    k_smallest_bst,
    k_smallest_bst_counted,
    k_smallest_quickselect,
    k_smallest_quickselect_counted,
    k_smallest_sort,
    k_smallest_sort_counted,
)

DEDUPLICADORES = (deduplicate_slow, deduplicate_fast)


# ----------------------------------------------------------------------
# Deduplicação — equivalência
# ----------------------------------------------------------------------
@pytest.mark.parametrize("dedup", DEDUPLICADORES)
def test_deduplicacao_em_casos_de_borda(dedup: object) -> None:
    assert dedup([]) == []  # type: ignore[operator]
    assert dedup([1]) == [1]  # type: ignore[operator]
    assert dedup([1, 1, 1]) == [1]  # type: ignore[operator]
    assert dedup([1, 2, 3]) == [1, 2, 3]  # type: ignore[operator]


@pytest.mark.parametrize("dedup", DEDUPLICADORES)
def test_deduplicacao_preserva_o_primeiro_aparecimento(dedup: object) -> None:
    """Exigência do enunciado — e sem isso as duas versões não seriam
    comparáveis elemento a elemento."""
    assert dedup([3, 1, 3, 2, 1, 4]) == [3, 1, 2, 4]  # type: ignore[operator]
    assert dedup(["b", "a", "b", "c"]) == ["b", "a", "c"]  # type: ignore[operator]


def test_as_duas_deduplicacoes_produzem_exatamente_a_mesma_saida() -> None:
    rng = random.Random(42)
    for _ in range(200):
        arr = [rng.randrange(12) for _ in range(rng.randrange(0, 50))]
        referencia = list(dict.fromkeys(arr))  # modelo: ordem de 1º aparecimento
        assert deduplicate_slow(arr) == referencia
        assert deduplicate_fast(arr) == referencia


def test_deduplicacao_funciona_com_chaves_de_tipos_variados() -> None:
    arr = ["a", "b", "a", 1, 2, 1, None, None, (1, 2), (1, 2)]
    referencia = list(dict.fromkeys(arr))
    assert deduplicate_slow(arr) == referencia
    assert deduplicate_fast(arr) == referencia


@pytest.mark.parametrize("dedup", DEDUPLICADORES)
def test_deduplicacao_nao_altera_a_entrada(dedup: object) -> None:
    arr = [3, 1, 3, 2]
    copia = list(arr)
    dedup(arr)  # type: ignore[operator]
    assert arr == copia


# ----------------------------------------------------------------------
# Deduplicação — o ganho medido
# ----------------------------------------------------------------------
def test_lenta_e_quadratica_quando_tudo_e_distinto() -> None:
    for n in (50, 200, 500):
        arr = list(range(n))  # u = n
        counters = Counters()
        deduplicate_slow(arr, counters)
        assert counters.comparisons == n * (n - 1) // 2


def test_lenta_custa_n_vezes_u_e_nao_n2_cego() -> None:
    """Com poucos distintos a versão lenta parece aceitável — e é justamente
    por isso que medir só o caso fácil esconderia o problema."""
    n = 1_000
    for u in (2, 10, 100):
        arr = [value % u for value in range(n)]
        counters = Counters()
        deduplicate_slow(arr, counters)
        # ≈ n·u/2, muito abaixo de n²/2 quando u << n
        assert counters.comparisons < n * u
        assert counters.comparisons > n * (u - 1) / 2


def test_rapida_e_linear() -> None:
    razoes = []
    for n in (500, 2_000, 8_000):
        arr = list(range(n))
        counters = Counters()
        deduplicate_fast(arr, counters)
        razoes.append(counters.comparisons / n)
    # razão contra n presa numa faixa estreita => Θ(n)
    assert all(razao < 3.0 for razao in razoes)
    assert max(razoes) / min(razoes) < 1.6


def test_a_rapida_ganha_da_lenta_por_ordens_de_grandeza() -> None:
    n = 2_000
    arr = list(range(n))
    _, lenta = deduplicate_slow_counted(arr)
    _, rapida = deduplicate_fast_counted(arr)
    assert lenta.comparisons / rapida.comparisons > 200


def test_a_rapida_tem_pior_caso_e_ele_tem_nome() -> None:
    """Colisão adversarial: a hashtable degrada e a versão rápida vira Θ(n²)."""
    n = 300
    colidentes = find_colliding_keys(n, modulus=1024)
    counters = Counters()
    resultado = deduplicate_fast(colidentes, counters)
    assert resultado == colidentes  # corretude não se perde
    # todas no mesmo bucket: a i-ésima consulta varre i−1 entradas. A contagem
    # bate com n(n−1)/2 porque put_if_absent faz UMA travessia por elemento.
    assert counters.comparisons == n * (n - 1) // 2


# ----------------------------------------------------------------------
# k menores — equivalência das três versões
# ----------------------------------------------------------------------
def test_as_tres_versoes_concordam_em_entradas_aleatorias() -> None:
    rng = random.Random(42)
    for _ in range(120):
        arr = [rng.randrange(40) for _ in range(rng.randrange(1, 50))]
        esperado_completo = sorted(arr)
        tree: BinarySearchTree[int, None] = BinarySearchTree(arr)
        for k in range(len(arr) + 1):
            esperado = esperado_completo[:k]
            assert k_smallest_sort(arr, k) == esperado
            assert k_smallest_quickselect(arr, k) == esperado
            assert k_smallest_bst(tree.root, k) == esperado


def test_k_zero_e_k_igual_a_n() -> None:
    arr = generate_array(50, "aleatorio")
    tree: BinarySearchTree[int, None] = BinarySearchTree(arr)
    assert k_smallest_sort(arr, 0) == []
    assert k_smallest_quickselect(arr, 0) == []
    assert k_smallest_bst(tree.root, 0) == []
    assert k_smallest_sort(arr, 50) == sorted(arr)
    assert k_smallest_quickselect(arr, 50) == sorted(arr)
    assert k_smallest_bst(tree.root, 50) == sorted(arr)


def test_k_smallest_bst_conta_multiplicidade() -> None:
    tree: BinarySearchTree[int, None] = BinarySearchTree([5, 1, 5, 9, 5])
    assert k_smallest_bst(tree.root, 4) == [1, 5, 5, 5]


def test_k_smallest_bst_em_arvore_vazia() -> None:
    assert k_smallest_bst(None, 5) == []


def test_k_smallest_nao_altera_a_entrada() -> None:
    arr = generate_array(60, "aleatorio")
    copia = list(arr)
    k_smallest_sort(arr, 10)
    k_smallest_quickselect(arr, 10)
    assert arr == copia


def test_k_smallest_valida_k() -> None:
    arr = [1, 2, 3]
    for funcao in (k_smallest_sort, k_smallest_quickselect):
        with pytest.raises(InvalidKError, match="k deve ser >= 0"):
            funcao(arr, -1)
        with pytest.raises(InvalidKError, match="maior que o tamanho"):
            funcao(arr, 4)
        with pytest.raises(InvalidKError, match="k deve ser int"):
            funcao(arr, 1.5)  # type: ignore[arg-type]
    tree: BinarySearchTree[int, None] = BinarySearchTree(arr)
    with pytest.raises(InvalidKError, match="k deve ser >= 0"):
        k_smallest_bst(tree.root, -1)


def test_k_smallest_protocolo_resultado_counters() -> None:
    arr = generate_array(40, "aleatorio")
    tree: BinarySearchTree[int, None] = BinarySearchTree(arr)
    for resultado, counters in (
        k_smallest_sort_counted(arr, 5),
        k_smallest_quickselect_counted(arr, 5),
        k_smallest_bst_counted(tree.root, 5),
    ):
        assert resultado == [0, 1, 2, 3, 4]
        assert counters.total >= 0


# ----------------------------------------------------------------------
# k menores — o ganho medido
# ----------------------------------------------------------------------
def test_versao_A_nao_muda_com_k() -> None:
    """A insensibilidade a k é o defeito: ordena tudo mesmo para k=1."""
    arr = generate_array(4_000, "aleatorio")
    contagens = {
        k: k_smallest_sort_counted(arr, k)[1].comparisons for k in (1, 10, 1_000)
    }
    assert len(set(contagens.values())) == 1


def test_versao_B_e_linear_para_k_pequeno() -> None:
    razoes = []
    for n in (1_000, 4_000, 16_000, 64_000):
        arr = generate_array(n, "aleatorio")
        _, counters = k_smallest_quickselect_counted(arr, 10)
        razoes.append(counters.comparisons / n)
    assert all(1.0 < razao < 6.0 for razao in razoes)
    assert max(razoes) / min(razoes) < 2.5  # Θ(n)


def test_versao_B_ganha_de_A_e_o_ganho_cresce_com_n() -> None:
    ganhos = []
    for n in (1_000, 16_000, 64_000):
        arr = generate_array(n, "aleatorio")
        _, a = k_smallest_sort_counted(arr, 10)
        _, b = k_smallest_quickselect_counted(arr, 10)
        ganhos.append(a.comparisons / b.comparisons)
    assert all(ganho > 2 for ganho in ganhos)
    assert ganhos[-1] > ganhos[0]  # a vantagem cresce com log2 n


def test_versao_B_converge_para_A_quando_k_igual_a_n() -> None:
    n = 2_000
    arr = generate_array(n, "aleatorio")
    _, a = k_smallest_sort_counted(arr, n)
    _, b = k_smallest_quickselect_counted(arr, n)
    assert b.comparisons / a.comparisons < 2.5  # mesma ordem de grandeza


def test_versao_C_custa_h_mais_k() -> None:
    n = 4_000
    arr = generate_array(n, "aleatorio")
    tree: BinarySearchTree[int, None] = BinarySearchTree(arr)
    altura = tree.height()
    for k in (1, 10, 100):
        _, counters = k_smallest_bst_counted(tree.root, k)
        assert counters.hops <= altura + k


# ----------------------------------------------------------------------
# O pior caso da BST (exigência do enunciado)
# ----------------------------------------------------------------------
def test_degeneracao_dói_em_operacoes_diferentes_conforme_a_direcao() -> None:
    """Dizer "a BST degenerou" não basta: é preciso dizer QUAL operação sofre."""
    n = 2_000
    crescente: BinarySearchTree[int, None] = BinarySearchTree(range(n))
    decrescente: BinarySearchTree[int, None] = BinarySearchTree(range(n - 1, -1, -1))
    assert crescente.height() == decrescente.height() == n  # as duas degeneraram

    # espinha à direita: o MENOR elemento é a própria raiz
    subindo = Counters()
    assert k_smallest_bst(crescente.root, 1, subindo) == [0]
    assert subindo.hops == 0

    # espinha à esquerda: o menor está no fundo, e chegar nele custa n−1
    descendo = Counters()
    assert k_smallest_bst(decrescente.root, 1, descendo) == [0]
    assert descendo.hops == n - 1


def test_busca_na_espinha_e_linear_nas_duas_direcoes() -> None:
    """A busca por chave sofre nos dois casos — é a altura que manda."""
    n = 1_000
    for keys in (range(n), range(n - 1, -1, -1)):
        tree: BinarySearchTree[int, None] = BinarySearchTree(keys)
        counters = Counters()
        tree.search(n // 2, counters)
        assert counters.comparisons >= n / 2  # Θ(n), não Θ(log n)
        assert counters.comparisons > 20 * math.log2(n)


# ----------------------------------------------------------------------
# Experimentos
# ----------------------------------------------------------------------
def test_bench_deduplicate_mostra_n_vezes_u_contra_n() -> None:
    rows = bench_deduplicate([500, 1_000], (0.05, 1.0))
    assert len(rows) == 4
    for row in rows:
        # lenta: razão contra n·u presa perto de 0,5
        assert row["lenta/(n*u)"] == pytest.approx(0.5, abs=0.15)
        # rápida: razão contra n presa numa constante pequena
        assert row["rapida/n"] < 3.0
        assert row["razao_lenta_sobre_rapida"] > 1.0

    # com 100% de distintos a lenta é quadrática de verdade
    completos = [row for row in rows if row["fracao_distintos"] == 1.0]
    for row in completos:
        assert row["lenta/n2"] == pytest.approx(0.5, abs=0.02)
        assert row["razao_lenta_sobre_rapida"] > 100

    # com 5% de distintos ela parece aceitável — e é esse o ponto
    poucos = [row for row in rows if row["fracao_distintos"] == 0.05]
    for row in poucos:
        assert row["lenta/n2"] < 0.05


def test_bench_deduplicate_valida_parametros() -> None:
    with pytest.raises(ValueError, match="tamanho deve ser positivo"):
        bench_deduplicate([0])
    with pytest.raises(ValueError, match="fração de distintos"):
        bench_deduplicate([100], (0.0,))


def test_bench_k_smallest_separa_as_duas_classes() -> None:
    rows = bench_k_smallest([1_000, 4_000, 16_000], (1, 10, 1_000))
    por_k = {}
    for row in rows:
        por_k.setdefault(row["k"], []).append(row)

    for k in (1, 10):
        linhas = por_k[k]
        # B: razão contra n estável => Θ(n)
        razoes_b = [row["B/n"] for row in linhas]
        assert max(razoes_b) / min(razoes_b) < 2.5
        # A: razão contra n·log2n estável => Θ(n log n)
        razoes_a = [row["A/n_log2n"] for row in linhas]
        assert max(razoes_a) / min(razoes_a) < 1.3
        # e A/n cresce, porque acompanha log2 n
        assert linhas[-1]["A/n"] > linhas[0]["A/n"]
        assert all(row["razao_A_sobre_B"] > 2 for row in linhas)


def test_bench_k_smallest_valida_tamanho() -> None:
    with pytest.raises(ValueError, match="tamanho deve ser positivo"):
        bench_k_smallest([0])


def test_bench_k_smallest_bst_separa_as_tres_formas() -> None:
    rows = bench_k_smallest_bst([500, 1_000], (1, 10))
    por_forma: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        por_forma.setdefault(row["forma"], []).append(row)
    assert set(por_forma) == {
        "aleatória", "degenerada crescente", "degenerada decrescente"
    }

    for row in por_forma["aleatória"]:
        assert row["saltos"] <= row["saltos_previstos_h_mais_k"]
        assert row["altura/log2n"] < 4

    # crescente: a raiz é o menor, então k=1 não custa salto nenhum
    crescentes = [row for row in por_forma["degenerada crescente"] if row["k"] == 1]
    assert all(row["saltos"] == 0 for row in crescentes)

    # decrescente: o menor está no fundo, e k=1 custa n−1 saltos
    decrescentes = [
        row for row in por_forma["degenerada decrescente"] if row["k"] == 1
    ]
    for row in decrescentes:
        assert row["saltos"] == row["n"] - 1
        assert row["saltos/n"] == pytest.approx(1.0, abs=0.01)
        assert row["degenerada"] is True


def test_bench_k_smallest_bst_valida_tamanho() -> None:
    with pytest.raises(ValueError, match="tamanho deve ser positivo"):
        bench_k_smallest_bst([0])

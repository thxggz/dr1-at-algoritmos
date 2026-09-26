"""Testes de src/counters.py — a instrumentação que sustenta todas as análises.

Se o contador estiver errado, TODA afirmação de Big O do trabalho cai junto.
Por isso ele é testado antes de qualquer estrutura de dados.
"""

from __future__ import annotations

import math

import pytest

from src.counters import COPIES_PER_SWAP, Counters, make_row, ratio


def test_contadores_comecam_em_zero() -> None:
    counters = Counters()
    assert counters.as_dict() == {
        "comparisons": 0,
        "copies": 0,
        "calls": 0,
        "hops": 0,
    }
    assert counters.total == 0


def test_uma_troca_vale_tres_copias() -> None:
    """Critério fixo do enunciado: 1 troca == 3 cópias."""
    assert COPIES_PER_SWAP == 3
    counters = Counters()
    counters.count_swap()
    assert counters.copies == 3


def test_swap_troca_de_fato_e_conta_tres_copias() -> None:
    counters = Counters()
    data = [10, 20, 30]
    counters.swap(data, 0, 2)
    assert data == [30, 20, 10]
    assert counters.copies == 3
    assert counters.comparisons == 0


def test_write_conta_uma_copia() -> None:
    counters = Counters()
    data = [1, 2, 3]
    counters.write(data, 1, 99)
    assert data == [1, 99, 3]
    assert counters.copies == 1


@pytest.mark.parametrize(
    "method,left,right,expected",
    [
        ("lt", 1, 2, True),
        ("lt", 2, 1, False),
        ("le", 2, 2, True),
        ("gt", 3, 2, True),
        ("ge", 1, 2, False),
        ("eq", 5, 5, True),
        ("eq", 5, 6, False),
    ],
)
def test_helpers_de_comparacao_respondem_certo_e_contam_uma_vez(
    method: str, left: int, right: int, expected: bool
) -> None:
    counters = Counters()
    assert getattr(counters, method)(left, right) is expected
    assert counters.comparisons == 1


def test_soma_de_contadores() -> None:
    a = Counters(comparisons=1, copies=2, calls=3, hops=4)
    b = Counters(comparisons=10, copies=20, calls=30, hops=40)
    assert (a + b).as_dict() == {
        "comparisons": 11,
        "copies": 22,
        "calls": 33,
        "hops": 44,
    }
    # a e b não foram mutados pelo operador +
    assert a.comparisons == 1 and b.comparisons == 10


def test_soma_no_lugar_e_absorb() -> None:
    a = Counters(comparisons=1)
    a += Counters(comparisons=2, hops=5)
    assert a.comparisons == 3 and a.hops == 5
    assert a.absorb(Counters(calls=7)) is a
    assert a.calls == 7


def test_ratio_contra_cada_curva() -> None:
    assert ratio(100, 10, "n") == pytest.approx(10.0)
    assert ratio(100, 10, "n2") == pytest.approx(1.0)
    # 10 * log2(10) = 33.219...
    assert ratio(100, 10, "n_log2n") == pytest.approx(100 / (10 * math.log2(10)))


def test_ratio_devolve_nan_quando_curva_nao_e_definida() -> None:
    """n·log2 n não é utilizável para n <= 1; a razão não pode explodir."""
    assert math.isnan(ratio(5, 1, "n_log2n"))
    assert math.isnan(ratio(5, 0, "n_log2n"))


def test_ratio_rejeita_curva_desconhecida() -> None:
    with pytest.raises(ValueError, match="curva desconhecida"):
        ratio(1, 1, "n_cubed")


def test_make_row_tem_as_colunas_do_padrao() -> None:
    counters = Counters(comparisons=45, copies=30, calls=0, hops=9)
    row = make_row(
        exercise="ex10",
        algorithm="search",
        n=10,
        pattern="aleatório",
        counters=counters,
        metric="comparisons",
        curves=("n", "n2"),
        elapsed_s=0.5,
    )
    for column in ("exercicio", "algoritmo", "n", "padrao", "metrica"):
        assert column in row
    for column in ("comparisons", "copies", "calls", "hops"):
        assert column in row
    assert row["comparisons/n"] == pytest.approx(4.5)
    assert row["comparisons/n2"] == pytest.approx(0.45)
    assert row["tempo_s"] == 0.5
    assert row["tempo_us_por_op"] == pytest.approx(0.5 * 1e6 / 10)


def test_make_row_rejeita_metrica_desconhecida() -> None:
    with pytest.raises(ValueError, match="métrica desconhecida"):
        make_row(exercise="ex10", algorithm="x", n=10, metric="cache_misses")


def test_stopwatch_mede_tempo_nao_negativo() -> None:
    from src.counters import stopwatch

    with stopwatch() as elapsed:
        sum(range(10_000))
    assert len(elapsed) == 1
    assert elapsed[0] >= 0.0

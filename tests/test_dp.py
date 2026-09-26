"""Testes do exercício 8 — programação dinâmica e memoization.

O teste que carrega o argumento do exercício é
``test_granularidade_da_chave_muda_tudo``: as três granularidades de chave dão
**a mesma resposta** e custos completamente diferentes. Sem ele, "a chave do
memo deve ser o estado mínimo suficiente" seria uma frase no docstring.
"""

from __future__ import annotations

import math
import random
import sys

import pytest

from src.counters import Counters
from src.dp import (
    DEFAULT_CALL_BUDGET,
    INFEASIBLE,
    CallBudgetExceeded,
    DPResult,
    InvalidAmountError,
    InvalidCoinsError,
    StateNotFoundError,
    SubproblemNode,
    bench_key_granularity,
    bench_memo_scaling,
    bench_memoization,
    bench_subproblem_tree,
    build_subproblem_tree,
    count_tree,
    min_coins_bottom_up,
    min_coins_memo,
    min_coins_memo_counted,
    min_coins_naive,
    min_coins_naive_counted,
    search_subproblem_tree,
    tree_to_string,
    validate_problem,
)

VERSOES = (min_coins_naive, min_coins_memo, min_coins_bottom_up)


def referencia(amount: int, coins: tuple[int, ...]) -> int:
    """Modelo de referência independente, escrito da forma mais direta."""
    melhor = [math.inf] * (amount + 1)
    melhor[0] = 0
    for alvo in range(1, amount + 1):
        for moeda in coins:
            if moeda <= alvo and melhor[alvo - moeda] + 1 < melhor[alvo]:
                melhor[alvo] = melhor[alvo - moeda] + 1
    return INFEASIBLE if math.isinf(melhor[amount]) else int(melhor[amount])


# ----------------------------------------------------------------------
# Corretude
# ----------------------------------------------------------------------
@pytest.mark.parametrize("resolver", VERSOES)
@pytest.mark.parametrize(
    ("amount", "coins", "esperado"),
    [
        (0, (1,), 0),
        (1, (1,), 1),
        (11, (1, 3, 4), 3),  # 3 + 4 + 4
        (6, (1, 3, 4), 2),  # 3 + 3
        (7, (2, 4), INFEASIBLE),  # ímpar com moedas pares
        (3, (5,), INFEASIBLE),
        (10, (10,), 1),
        (30, (1, 5, 10, 25), 2),  # 25 + 5
    ],
)
def test_casos_conhecidos(
    resolver: object, amount: int, coins: tuple[int, ...], esperado: int
) -> None:
    assert resolver(amount, coins).value == esperado  # type: ignore[operator]


def test_as_tres_versoes_concordam_com_a_referencia() -> None:
    rng = random.Random(42)
    for _ in range(150):
        coins = tuple(
            sorted({rng.randrange(1, 9) for _ in range(rng.randrange(1, 4))})
        )
        amount = rng.randrange(0, 22)
        esperado = referencia(amount, coins)
        for resolver in VERSOES:
            assert resolver(amount, coins).value == esperado  # type: ignore[operator]


def test_moeda_de_valor_1_sempre_torna_viavel() -> None:
    for amount in (0, 1, 7, 50):
        assert min_coins_memo(amount, (1, 5)).value == (
            amount // 5 + amount % 5
        )


def test_protocolo_resultado_counters() -> None:
    valor, counters = min_coins_memo_counted(11, (1, 3, 4))
    assert valor == 3
    assert counters.calls > 0
    valor, counters = min_coins_naive_counted(11, (1, 3, 4))
    assert valor == 3


# ----------------------------------------------------------------------
# Validação
# ----------------------------------------------------------------------
def test_validacao_de_entrada() -> None:
    with pytest.raises(InvalidAmountError, match="amount deve ser >= 0"):
        min_coins_memo(-1, (1,))
    with pytest.raises(InvalidAmountError, match="amount deve ser int"):
        min_coins_memo(1.5, (1,))  # type: ignore[arg-type]
    with pytest.raises(InvalidCoinsError, match="ao menos uma moeda"):
        min_coins_memo(10, ())
    with pytest.raises(InvalidCoinsError, match="moeda deve ser > 0"):
        min_coins_memo(10, (1, 0))
    with pytest.raises(InvalidCoinsError, match="moeda deve ser int"):
        min_coins_memo(10, (1, 2.5))  # type: ignore[arg-type]
    with pytest.raises(InvalidCoinsError, match="repetidas"):
        min_coins_memo(10, (1, 3, 3))


def test_validacao_normaliza_e_ordena_as_moedas() -> None:
    assert validate_problem(10, (4, 1, 3)) == (1, 3, 4)


# ----------------------------------------------------------------------
# Sobreposição de subproblemas e o ganho da memoization
# ----------------------------------------------------------------------
def test_ingenua_visita_o_mesmo_estado_muitas_vezes() -> None:
    """A sobreposição, contada: muitas chamadas, poucos estados distintos."""
    resultado = min_coins_naive(16, (1, 3, 4))
    assert resultado.distinct_states <= 17  # só 0..16 são possíveis
    assert resultado.counters.calls > 20 * resultado.distinct_states


def test_memoizada_resolve_cada_estado_uma_vez() -> None:
    amount = 40
    resultado = min_coins_memo(amount, (1, 3, 4))
    assert resultado.memo is not None
    assert len(resultado.memo) == amount  # estados 1..amount
    # k chamadas por estado resolvido, mais as consultas que acertam o memo
    assert resultado.counters.calls < 5 * amount


def test_reducao_de_chamadas_cresce_exponencialmente_com_amount() -> None:
    reducoes = []
    for amount in (10, 14, 18, 22):
        ingenua = min_coins_naive(amount, (1, 3, 4))
        memoizada = min_coins_memo(amount, (1, 3, 4))
        assert ingenua.value == memoizada.value
        reducoes.append(ingenua.counters.calls / memoizada.counters.calls)
    assert reducoes == sorted(reducoes)
    assert reducoes[-1] > 30 * reducoes[0]


def test_memoizada_alcanca_amounts_que_a_ingenua_nao_alcanca() -> None:
    amount = 200
    memoizada = min_coins_memo(amount, (1, 3, 4))
    assert memoizada.value == referencia(amount, (1, 3, 4))
    with pytest.raises(CallBudgetExceeded, match="exponencial"):
        min_coins_naive(amount, (1, 3, 4), call_budget=200_000)


def test_orcamento_de_chamadas_e_respeitado() -> None:
    with pytest.raises(CallBudgetExceeded, match="1,000 chamadas"):
        min_coins_naive(60, (1, 3, 4), call_budget=1_000)


def test_iterativa_nao_tem_teto_de_recursao() -> None:
    """As recursivas quebram; a de baixo para cima não."""
    amount = 3 * sys.getrecursionlimit()
    assert min_coins_bottom_up(amount, (1, 3, 4)).value == referencia(
        amount, (1, 3, 4)
    )
    with pytest.raises(RecursionError):
        min_coins_memo(amount, (1, 3, 4))


def test_profundidade_da_memoizada_e_amount_sobre_menor_moeda() -> None:
    resultado = min_coins_memo(60, (3, 5))
    assert resultado.max_depth == 60 // 3 + 1


# ----------------------------------------------------------------------
# Granularidade da chave do memo
# ----------------------------------------------------------------------
def test_granularidade_da_chave_muda_tudo() -> None:
    """Mesma resposta, custos completamente diferentes (decisão de projeto 2)."""
    rows = bench_key_granularity(18, (1, 3, 4))
    assert len(rows) == 3
    por_chave = {row["chave"]: row for row in rows}

    respostas = {row["resposta"] for row in rows if not row["orcamento_estourado"]}
    assert respostas == {5}  # 18 = 4+4+4+3+3

    minimo = por_chave["restante (mínimo suficiente)"]
    profundidade = por_chave["(restante, profundidade)"]
    ultima = por_chave["(restante, última moeda)"]

    # o estado mínimo guarda uma chave por valor de 1..amount
    assert minimo["chaves_guardadas"] == 18
    assert minimo["chaves_por_amount"] == pytest.approx(1.0)

    # granularidade fina demais: muito mais chaves e muito mais chamadas
    assert profundidade["chaves_guardadas"] > 5 * minimo["chaves_guardadas"]
    assert profundidade["chamadas"] > 5 * minimo["chamadas"]

    # o erro sutil: multiplica o espaço de estados por k sem mudar resposta
    assert ultima["chaves_guardadas"] > minimo["chaves_guardadas"]
    assert ultima["chamadas"] > minimo["chamadas"]


# ----------------------------------------------------------------------
# Árvore de subproblemas
# ----------------------------------------------------------------------
def test_arvore_memoizada_e_muito_menor_que_a_completa() -> None:
    memoizada, _ = build_subproblem_tree(8, (1, 3, 4), memoize=True)
    completa, _ = build_subproblem_tree(8, (1, 3, 4), memoize=False)
    com_memo = count_tree(memoizada)
    sem_memo = count_tree(completa)
    assert com_memo["total"] == 20
    assert sem_memo["total"] == 64
    assert sem_memo["memo"] == 0  # sem memo não existe folha de memo
    assert com_memo["memo"] == 9  # 9 expansões evitadas


def test_arvore_registra_o_motivo_de_cada_folha() -> None:
    """Decisão de projeto 5: sem o motivo, a árvore só pareceria menor."""
    raiz, _ = build_subproblem_tree(8, (1, 3, 4), memoize=True)
    contagem = count_tree(raiz)
    assert contagem["total"] == (
        contagem["expandido"] + contagem["base"]
        + contagem["memo"] + contagem["inviavel"]
    )
    assert contagem["base"] > 0
    assert contagem["memo"] > 0


def test_arvore_marca_estado_inviavel() -> None:
    raiz, resultado = build_subproblem_tree(3, (5,), memoize=True)
    assert resultado.value == INFEASIBLE
    assert raiz.status == "inviavel"
    assert raiz.is_leaf()


def test_arvore_da_a_mesma_resposta_que_as_resolucoes() -> None:
    for amount in (0, 5, 11, 16):
        _, resultado = build_subproblem_tree(amount, (1, 3, 4))
        assert resultado.value == min_coins_memo(amount, (1, 3, 4)).value


def test_arvore_respeita_o_orcamento() -> None:
    with pytest.raises(CallBudgetExceeded):
        build_subproblem_tree(40, (1, 3, 4), memoize=False, call_budget=5_000)


def test_tree_to_string_mostra_a_estrutura_e_corta() -> None:
    raiz, _ = build_subproblem_tree(8, (1, 3, 4), memoize=True)
    texto = tree_to_string(raiz, max_lines=5)
    assert texto.startswith("f(8)")
    assert "omitida" in texto
    completo = tree_to_string(raiz, max_lines=100)
    assert "memo" in completo
    assert "base" in completo
    # sem caracteres fora do cp1252, que quebrariam o console e o PDF
    completo.encode("cp1252")


# ----------------------------------------------------------------------
# Busca na árvore por estado
# ----------------------------------------------------------------------
def test_busca_na_arvore_acha_todas_as_ocorrencias() -> None:
    """A multiplicidade é a evidência da sobreposição; esconder a 1ª apagaria."""
    memoizada, _ = build_subproblem_tree(8, (1, 3, 4), memoize=True)
    completa, _ = build_subproblem_tree(8, (1, 3, 4), memoize=False)
    assert len(search_subproblem_tree(memoizada, 4)) == 3
    assert len(search_subproblem_tree(completa, 4)) == 4
    assert all(node.state == 4 for node in search_subproblem_tree(completa, 4))


def test_busca_na_arvore_devolve_a_ocorrencia_mais_rasa_primeiro() -> None:
    raiz, _ = build_subproblem_tree(10, (1, 3, 4), memoize=False)
    ocorrencias = search_subproblem_tree(raiz, 6)
    assert len(ocorrencias) > 1
    assert ocorrencias[0].state == 6


def test_busca_de_estado_ausente_levanta() -> None:
    raiz, _ = build_subproblem_tree(8, (1, 3, 4))
    with pytest.raises(StateNotFoundError, match="f\\(99\\)"):
        search_subproblem_tree(raiz, 99)


def test_buscar_na_arvore_custa_muito_mais_que_consultar_o_memo() -> None:
    """A discussão que o enunciado pede: Θ(V) contra O(1)."""
    raiz, resultado = build_subproblem_tree(14, (1, 3, 4), memoize=True)
    total = count_tree(raiz)["total"]

    na_arvore = Counters()
    search_subproblem_tree(raiz, 7, na_arvore)
    assert na_arvore.comparisons == total  # visita todos os nós

    assert resultado.memo is not None
    no_memo = Counters()
    resultado.memo.get_or(7, None, no_memo)
    assert no_memo.comparisons <= 2  # uma cadeia curtíssima
    assert na_arvore.comparisons > 10 * max(1, no_memo.comparisons)


# ----------------------------------------------------------------------
# Experimentos
# ----------------------------------------------------------------------
def test_bench_memoization_mostra_a_mudanca_de_classe() -> None:
    rows = bench_memoization((8, 12, 16, 20))
    assert len(rows) == 4
    for row in rows:
        assert row["orcamento_estourado"] is False
        assert row["estados_distintos"] == row["amount"]
        assert row["reducao_de_chamadas"] > 1
        # memo: chamadas por (a·k) presa numa faixa estreita => Θ(a·k)
        assert 0.5 < row["memo/(a*k)"] < 3.0

    # ingênua: cresce exponencialmente; a razão contra 1,8^a fica estável
    razoes = [row["ingenua/1.8^a"] for row in rows]
    assert max(razoes) / min(razoes) < 6.0
    # e a redução cresce muito com amount
    reducoes = [row["reducao_de_chamadas"] for row in rows]
    assert reducoes == sorted(reducoes)
    assert reducoes[-1] > 10 * reducoes[0]


def test_bench_memo_scaling_confirma_theta_a_vezes_k() -> None:
    rows = bench_memo_scaling((100, 200, 400, 800))
    assert len(rows) == 4
    razoes = [row["memo/(a*k)"] for row in rows if not row["memo_quebrou"]]
    assert len(razoes) == 4  # com m=1, amount até 800 cabe na pilha
    assert max(razoes) / min(razoes) < 1.3
    for row in rows:
        assert row["iterativa/(a*k)"] > 0
        assert row["resposta"] >= 0


def test_bench_memo_scaling_padrao_mostra_o_teto_da_pilha() -> None:
    """A linha que quebra é evidência, não falha: o teto na mesma tabela."""
    rows = bench_memo_scaling()
    quebraram = [row for row in rows if row["memo_quebrou"]]
    inteiros = [row for row in rows if not row["memo_quebrou"]]
    assert len(inteiros) >= 3
    assert len(quebraram) >= 1
    for row in quebraram:
        assert row["amount"] > sys.getrecursionlimit()
        assert row["chamadas_iterativa"] == row["amount"]  # a iterativa terminou


def test_bench_memo_scaling_registra_a_quebra_da_recursiva() -> None:
    grande = 3 * sys.getrecursionlimit()
    rows = bench_memo_scaling((grande,), coins=(1, 3))
    row = rows[0]
    assert row["memo_quebrou"] is True
    assert row["chamadas_memo"] is None
    assert row["chamadas_iterativa"] == grande  # a iterativa terminou


def test_bench_memo_scaling_valida_amount() -> None:
    with pytest.raises(ValueError, match="amount deve ser positivo"):
        bench_memo_scaling((0,))


def test_bench_subproblem_tree_quantifica_a_poda() -> None:
    rows = bench_subproblem_tree((8, 12, 16))
    assert len(rows) == 3
    for row in rows:
        assert row["nos_com_memo"] < row["nos_sem_memo"]
        assert row["folhas_memo"] > 0
        assert row["ocorrencias_na_arvore"] >= 1
        # buscar na árvore custa Θ(V); consultar o memo custa O(1)
        assert row["comparacoes_busca_arvore"] == row["nos_com_memo"]
        assert row["vantagem_do_memo"] > 5
    reducoes = [row["reducao_de_nos"] for row in rows]
    assert reducoes == sorted(reducoes)  # a poda rende mais conforme amount cresce

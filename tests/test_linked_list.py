"""Testes do exercício 10 — SinglyLinkedList com e sem ponteiro tail.

A suíte é parametrizada sobre as DUAS implementações. Isso não é só economia de
código: provar que as duas respondem exatamente igual em toda a API observável é
o que permite atribuir a diferença medida em ``insert_last`` unicamente ao
ponteiro ``tail``, e não a alguma outra divergência de implementação.

Os testes de corretude usam uma ``list`` do Python como modelo de referência
(permitido pela convenção 9: estrutura pronta só como referência nos testes).
"""

from __future__ import annotations

import random

import pytest

from src.counters import Counters
from src.linked_list import (
    COST_TABLE,
    EmptyListError,
    InvalidIndexError,
    ListInvariantError,
    Node,
    SinglyLinkedList,
    SinglyLinkedListNoTail,
    ValueNotFoundError,
    bench_insert_last,
    bench_search,
)

IMPLEMENTATIONS = [SinglyLinkedList, SinglyLinkedListNoTail]


@pytest.fixture(params=IMPLEMENTATIONS, ids=["com_tail", "sem_tail"])
def impl(request: pytest.FixtureRequest) -> type:
    """Classe sob teste: uma vez com ponteiro tail, uma vez sem."""
    return request.param


# ----------------------------------------------------------------------
# Nó
# ----------------------------------------------------------------------
def test_no_guarda_valor_e_proximo() -> None:
    second: Node[int] = Node(2)
    first: Node[int] = Node(1, second)
    assert first.value == 1
    assert first.next is second
    assert second.next is None


def test_no_usa_slots_e_nao_aceita_atributo_novo() -> None:
    """Sem __dict__ por nó: sustenta o argumento de memória do exercício 1."""
    node: Node[int] = Node(1)
    with pytest.raises(AttributeError):
        node.peso = 10  # type: ignore[attr-defined]


# ----------------------------------------------------------------------
# Estado inicial e leitura
# ----------------------------------------------------------------------
def test_lista_nova_esta_vazia(impl: type) -> None:
    target = impl()
    assert len(target) == 0
    assert target.is_empty() is True
    assert target.to_list() == []
    assert str(target) == "[]"
    target.check_invariants()


def test_construcao_a_partir_de_iteravel_preserva_ordem(impl: type) -> None:
    target = impl([3, 1, 4, 1, 5])
    assert target.to_list() == [3, 1, 4, 1, 5]
    assert len(target) == 5
    assert target.is_empty() is False
    target.check_invariants()


def test_str_mostra_encadeamento(impl: type) -> None:
    target = impl(["a", "b"])
    assert str(target) == "['a' -> 'b']"


def test_repr_inclui_tamanho_e_classe(impl: type) -> None:
    target = impl([1, 2])
    assert impl.__name__ in repr(target)
    assert "size=2" in repr(target)


def test_iteracao_percorre_do_inicio_ao_fim(impl: type) -> None:
    target = impl([10, 20, 30])
    assert [value for value in target] == [10, 20, 30]


def test_peek_first_e_peek_last(impl: type) -> None:
    target = impl([7, 8, 9])
    assert target.peek_first() == 7
    assert target.peek_last() == 9


def test_peek_em_lista_vazia_levanta_com_mensagem(impl: type) -> None:
    target = impl()
    with pytest.raises(EmptyListError, match="peek_first em lista vazia"):
        target.peek_first()
    with pytest.raises(EmptyListError, match="peek_last em lista vazia"):
        target.peek_last()


# ----------------------------------------------------------------------
# Inserção
# ----------------------------------------------------------------------
def test_insert_first_inverte_a_ordem_de_chegada(impl: type) -> None:
    target = impl()
    for value in [1, 2, 3]:
        target.insert_first(value)
    assert target.to_list() == [3, 2, 1]
    target.check_invariants()


def test_insert_last_preserva_a_ordem_de_chegada(impl: type) -> None:
    target = impl()
    for value in [1, 2, 3]:
        target.insert_last(value)
    assert target.to_list() == [1, 2, 3]
    assert target.peek_last() == 3
    target.check_invariants()


def test_insert_first_e_insert_last_intercalados(impl: type) -> None:
    target = impl()
    target.insert_last(2)
    target.insert_first(1)
    target.insert_last(3)
    target.insert_first(0)
    assert target.to_list() == [0, 1, 2, 3]
    target.check_invariants()


def test_insert_at_no_inicio_no_meio_e_no_fim(impl: type) -> None:
    target = impl([1, 4])
    target.insert_at(1, 2)          # meio
    target.insert_at(2, 3)          # meio
    target.insert_at(0, 0)          # início
    target.insert_at(len(target), 5)  # fim, índice == len é permitido
    assert target.to_list() == [0, 1, 2, 3, 4, 5]
    target.check_invariants()


def test_insert_at_no_indice_len_equivale_a_insert_last(impl: type) -> None:
    a, b = impl([1, 2]), impl([1, 2])
    a.insert_at(len(a), 3)
    b.insert_last(3)
    assert a.to_list() == b.to_list() == [1, 2, 3]
    assert a.peek_last() == b.peek_last() == 3


def test_insert_at_em_lista_vazia_aceita_apenas_indice_zero(impl: type) -> None:
    target = impl()
    target.insert_at(0, "x")
    assert target.to_list() == ["x"]
    vazia = impl()
    with pytest.raises(InvalidIndexError, match=r"faixa permitida é 0\.\.0"):
        vazia.insert_at(1, "x")


@pytest.mark.parametrize("bad_index", [-1, -10, 4, 100])
def test_insert_at_rejeita_indice_fora_da_faixa(impl: type, bad_index: int) -> None:
    target = impl([1, 2, 3])
    with pytest.raises(InvalidIndexError, match="faixa permitida"):
        target.insert_at(bad_index, 99)
    assert target.to_list() == [1, 2, 3]  # nada foi alterado


@pytest.mark.parametrize("bad_index", ["1", 1.0, None, True])
def test_insert_at_rejeita_indice_nao_inteiro(impl: type, bad_index: object) -> None:
    """`True` é rejeitado de propósito: bool é int em Python, e aceitar
    `insert_at(True, x)` como posição 1 esconderia erro do chamador."""
    target = impl([1, 2, 3])
    with pytest.raises(InvalidIndexError, match="índice deve ser int"):
        target.insert_at(bad_index, 99)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Busca
# ----------------------------------------------------------------------
def test_search_devolve_indice_da_primeira_ocorrencia(impl: type) -> None:
    target = impl([5, 7, 5, 9])
    assert target.search(5) == 0
    assert target.search(7) == 1
    assert target.search(9) == 3


def test_search_ausente_devolve_menos_um(impl: type) -> None:
    target = impl([1, 2, 3])
    assert target.search(42) == -1
    assert impl().search(42) == -1  # lista vazia também devolve -1


def test_contains_e_operador_in(impl: type) -> None:
    target = impl([1, 2, 3])
    assert target.contains(2) is True
    assert target.contains(99) is False
    assert 2 in target
    assert 99 not in target


@pytest.mark.parametrize(
    "value,expected_index,expected_comparisons,expected_hops",
    [(10, 0, 1, 0), (20, 1, 2, 1), (30, 2, 3, 2), (99, -1, 3, 3)],
)
def test_search_contagens_batem_com_a_posicao(
    impl: type,
    value: int,
    expected_index: int,
    expected_comparisons: int,
    expected_hops: int,
) -> None:
    """Valor em i custa i+1 comparações e i saltos; ausente custa n e n.

    É esta contagem que o notebook usa para afirmar melhor caso O(1) e pior
    caso O(n) — se ela mudar, a análise do exercício 10 fica sem base.
    """
    target = impl([10, 20, 30])
    index, counters = target.search_counted(value)
    assert index == expected_index
    assert counters.comparisons == expected_comparisons
    assert counters.hops == expected_hops


def test_search_acumula_em_contador_do_chamador(impl: type) -> None:
    target = impl([1, 2, 3])
    counters = Counters()
    target.search(3, counters)
    target.search(3, counters)
    assert counters.comparisons == 6  # 3 por busca, duas buscas


# ----------------------------------------------------------------------
# Remoção por valor
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected_index,expected_rest",
    [(1, 0, [2, 3]), (2, 1, [1, 3]), (3, 2, [1, 2])],
)
def test_delete_remove_do_inicio_do_meio_e_do_fim(
    impl: type, value: int, expected_index: int, expected_rest: list[int]
) -> None:
    target = impl([1, 2, 3])
    assert target.delete(value) == expected_index
    assert target.to_list() == expected_rest
    assert len(target) == 2
    target.check_invariants()


def test_delete_remove_apenas_a_primeira_ocorrencia(impl: type) -> None:
    target = impl([5, 7, 5])
    target.delete(5)
    assert target.to_list() == [7, 5]


def test_delete_ate_esvaziar_restaura_estado_de_lista_vazia(impl: type) -> None:
    target = impl([1])
    target.delete(1)
    assert target.is_empty() is True
    assert str(target) == "[]"
    target.check_invariants()


def test_delete_de_valor_ausente_levanta(impl: type) -> None:
    target = impl([1, 2, 3])
    with pytest.raises(ValueNotFoundError, match="valor não encontrado"):
        target.delete(99)
    assert target.to_list() == [1, 2, 3]  # a lista não foi corrompida
    target.check_invariants()


def test_delete_em_lista_vazia_levanta_erro_especifico(impl: type) -> None:
    with pytest.raises(EmptyListError, match="em lista vazia"):
        impl().delete(1)


def test_remover_o_ultimo_e_inserir_de_novo_nao_perde_elemento(impl: type) -> None:
    """Regressão do bug clássico de lista com tail.

    Ao remover o último nó é preciso recuar o ponteiro de fim; quem esquece
    passa a anexar a um nó já desligado e o novo elemento desaparece.
    """
    target = impl([1, 2, 3])
    target.delete(3)
    target.insert_last(4)
    assert target.to_list() == [1, 2, 4]
    assert target.peek_last() == 4
    assert len(target) == 3
    target.check_invariants()


def test_esvaziar_por_completo_e_reconstruir(impl: type) -> None:
    target = impl([1, 2])
    target.delete(1)
    target.delete(2)
    assert target.is_empty()
    target.insert_last(9)
    target.insert_last(10)
    assert target.to_list() == [9, 10]
    assert target.peek_first() == 9 and target.peek_last() == 10
    target.check_invariants()


# ----------------------------------------------------------------------
# Remoção por índice
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "index,expected_value,expected_rest",
    [(0, "a", ["b", "c"]), (1, "b", ["a", "c"]), (2, "c", ["a", "b"])],
)
def test_delete_at_devolve_o_valor_removido(
    impl: type, index: int, expected_value: str, expected_rest: list[str]
) -> None:
    target = impl(["a", "b", "c"])
    assert target.delete_at(index) == expected_value
    assert target.to_list() == expected_rest
    target.check_invariants()


@pytest.mark.parametrize("bad_index", [-1, 3, 99])
def test_delete_at_rejeita_indice_fora_da_faixa(impl: type, bad_index: int) -> None:
    target = impl([1, 2, 3])
    with pytest.raises(InvalidIndexError, match=r"faixa permitida é 0\.\.2"):
        target.delete_at(bad_index)
    assert target.to_list() == [1, 2, 3]


def test_delete_at_nao_aceita_o_indice_len(impl: type) -> None:
    """Assimetria proposital com insert_at: em `len` não existe elemento."""
    target = impl([1, 2, 3])
    with pytest.raises(InvalidIndexError):
        target.delete_at(len(target))


def test_delete_at_em_lista_vazia_levanta_erro_especifico(impl: type) -> None:
    with pytest.raises(EmptyListError, match="em lista vazia"):
        impl().delete_at(0)


def test_delete_at_contagens_sao_proporcionais_ao_indice(impl: type) -> None:
    target = impl(list(range(10)))
    _, counters = target.delete_at_counted(7)
    assert counters.hops == 7
    assert counters.comparisons == 0  # remoção por índice não compara chaves


# ----------------------------------------------------------------------
# Invariantes
# ----------------------------------------------------------------------
def test_check_invariants_detecta_size_corrompido(impl: type) -> None:
    """A verificação precisa realmente falhar quando a estrutura está errada,
    senão passar no teste não prova nada."""
    target = impl([1, 2, 3])
    target._size = 99  # corrupção deliberada
    with pytest.raises(ListInvariantError, match="_size"):
        target.check_invariants()


def test_check_invariants_detecta_tail_desatualizado() -> None:
    target: SinglyLinkedList[int] = SinglyLinkedList([1, 2, 3])
    target._tail = target._head  # aponta para o primeiro nó, não o último
    with pytest.raises(ListInvariantError, match="_tail"):
        target.check_invariants()


def test_invariantes_resistem_a_sequencia_longa_e_aleatoria(impl: type) -> None:
    """Modelo de referência: a cada passo a lista tem de ser igual a uma
    ``list`` do Python submetida às mesmas operações."""
    rng = random.Random(42)
    target = impl()
    model: list[int] = []

    for step in range(800):
        operation = rng.choice(
            ["insert_first", "insert_last", "insert_at", "delete", "delete_at"]
        )
        if operation == "insert_first":
            target.insert_first(step)
            model.insert(0, step)
        elif operation == "insert_last":
            target.insert_last(step)
            model.append(step)
        elif operation == "insert_at":
            index = rng.randint(0, len(model))
            target.insert_at(index, step)
            model.insert(index, step)
        elif operation == "delete":
            if not model:
                with pytest.raises(EmptyListError):
                    target.delete(step)
            else:
                value = rng.choice(model)
                assert target.delete(value) == model.index(value)
                model.remove(value)
        else:  # delete_at
            if not model:
                with pytest.raises(EmptyListError):
                    target.delete_at(0)
            else:
                index = rng.randrange(len(model))
                assert target.delete_at(index) == model[index]
                del model[index]

        assert len(target) == len(model)
        target.check_invariants()

    assert target.to_list() == model
    assert len(model) > 0  # o teste de fato exercitou a estrutura


def test_as_duas_implementacoes_respondem_igual(  # noqa: C901
) -> None:
    """Prova de que a única diferença entre as variantes é o custo, não o
    comportamento — condição para o experimento de insert_last ser válido."""
    rng = random.Random(7)
    with_tail: SinglyLinkedList[int] = SinglyLinkedList()
    without_tail: SinglyLinkedListNoTail[int] = SinglyLinkedListNoTail()

    for step in range(400):
        operation = rng.choice(
            ["insert_first", "insert_last", "insert_at", "search", "delete_at"]
        )
        if operation == "insert_first":
            with_tail.insert_first(step)
            without_tail.insert_first(step)
        elif operation == "insert_last":
            with_tail.insert_last(step)
            without_tail.insert_last(step)
        elif operation == "insert_at":
            index = rng.randint(0, len(with_tail))
            with_tail.insert_at(index, step)
            without_tail.insert_at(index, step)
        elif operation == "search":
            probe = rng.randint(0, step + 1)
            assert with_tail.search(probe) == without_tail.search(probe)
        elif len(with_tail) > 0:
            index = rng.randrange(len(with_tail))
            assert with_tail.delete_at(index) == without_tail.delete_at(index)

        assert with_tail.to_list() == without_tail.to_list()
        assert len(with_tail) == len(without_tail)
        assert str(with_tail) == str(without_tail)
        with_tail.check_invariants()
        without_tail.check_invariants()


# ----------------------------------------------------------------------
# O experimento do ponteiro tail
# ----------------------------------------------------------------------
def test_insert_last_com_tail_nao_da_nenhum_salto() -> None:
    target: SinglyLinkedList[int] = SinglyLinkedList(range(1_000))
    _, counters = target.insert_last_counted(1_000)
    assert counters.hops == 0
    assert counters.comparisons == 0


@pytest.mark.parametrize("size", [1, 2, 10, 100])
def test_insert_last_sem_tail_da_n_menos_um_saltos(size: int) -> None:
    target: SinglyLinkedListNoTail[int] = SinglyLinkedListNoTail(range(size))
    _, counters = target.insert_last_counted(size)
    assert counters.hops == size - 1


def test_insert_last_em_lista_vazia_nao_da_salto_em_nenhuma_variante(
    impl: type,
) -> None:
    target = impl()
    _, counters = target.insert_last_counted(1)
    assert counters.hops == 0


def test_bench_insert_last_confirma_as_duas_classes_de_complexidade() -> None:
    """Θ(n) com tail contra Θ(n²) sem tail, com os números fechando na conta.

    Sem tail, construir n elementos custa exatamente (n-1)(n-2)/2 saltos, então
    saltos/n² tende a 0,5. Com tail, custa 0 saltos para qualquer n.
    """
    sizes = [100, 200, 400, 800]
    rows = bench_insert_last(sizes)
    assert len(rows) == 2 * len(sizes)

    with_tail = [row for row in rows if row["variante"] == "com tail"]
    without_tail = [row for row in rows if row["variante"] == "sem tail"]

    for row in with_tail:
        assert row["hops"] == 0
        assert row["hops/n"] == 0

    for row in without_tail:
        n = row["n"]
        assert row["hops"] == (n - 1) * (n - 2) // 2 == row["hops_teorico"]
        assert row["hops/n2"] == pytest.approx(0.5, abs=0.02)
        # hops/n é o número de saltos por inserção; cresce com n, o que descarta
        # a hipótese de custo constante por inserção
        assert row["hops/n"] == pytest.approx((n - 1) * (n - 2) / 2 / n)

    crescimento = [row["hops/n"] for row in without_tail]
    assert crescimento == sorted(crescimento)
    assert crescimento[-1] > 3 * crescimento[0]  # dobrou n 3x => ~8x o custo/op


def test_bench_insert_last_valida_parametros() -> None:
    with pytest.raises(ValueError, match="repeats deve ser >= 1"):
        bench_insert_last([10], repeats=0)
    with pytest.raises(ValueError, match="tamanho deve ser positivo"):
        bench_insert_last([0])


def test_bench_insert_last_contagens_nao_dependem_de_repeats() -> None:
    """As contagens são determinísticas; repetir só melhora a medida de tempo."""
    uma = bench_insert_last([200], repeats=1)
    tres = bench_insert_last([200], repeats=3)
    assert [row["hops"] for row in uma] == [row["hops"] for row in tres]


def test_bench_search_separa_melhor_medio_e_pior_caso() -> None:
    rows = bench_search([200, 400], average_samples=20)
    for n in (200, 400):
        do_n = [row for row in rows if row["n"] == n]
        melhor = next(r for r in do_n if "melhor" in r["padrao"])
        medio = next(r for r in do_n if "médio" in r["padrao"])
        pior = next(r for r in do_n if "pior" in r["padrao"])

        assert melhor["comparacoes_por_busca"] == 1.0
        assert pior["comparacoes_por_busca"] == float(n)
        assert pior["encontradas"] == 0  # chave ausente, de fato
        assert medio["encontradas"] == medio["buscas"]
        assert 1.0 < medio["comparacoes_por_busca"] < float(n)
        # caso médio ~ n/2: a razão contra n fica em torno de 0,5
        assert medio["comparacoes_por_busca/n"] == pytest.approx(0.5, abs=0.15)


def test_tabela_de_custo_cobre_todas_as_operacoes_do_enunciado() -> None:
    operacoes = {row["operacao"] for row in COST_TABLE}
    for required in (
        "insert_first",
        "insert_last",
        "search",
        "delete (por valor)",
        "insert_at(i)",
        "delete_at(i)",
        "__len__",
        "__str__",
    ):
        assert required in operacoes
    for row in COST_TABLE:
        assert row["por_que"], f"custo de {row['operacao']} sem justificativa"

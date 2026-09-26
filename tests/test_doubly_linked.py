"""Testes do exercício 11 — DoublyLinkedList e Deque.

O enunciado pede explicitamente "bateria de testes com sequências longas
alternando as pontas, verificando invariantes após cada sequência". Esta suíte
faz isso em três camadas:

1. testes unitários por operação, com ``list`` do Python como modelo de
   referência (permitido pela convenção 9: estrutura pronta só nos testes);
2. sequências longas aleatórias, conferindo invariante e conteúdo no meio do
   caminho — :func:`run_alternating_ends_stress`;
3. testes que **corrompem a estrutura de propósito** e exigem que
   ``check_invariants`` detecte. Sem eles, "check_invariants passou" não
   provaria nada: um método vazio também passaria.
"""

from __future__ import annotations

import random

import pytest

from src.counters import Counters
from src.doubly_linked import (
    COST_TABLE,
    DEQUE_COST_TABLE,
    END_OPERATIONS,
    Deque,
    DequeError,
    DoublyLinkedList,
    DoublyLinkedListError,
    DoublyLinkedListInvariantError,
    DoublyNode,
    EmptyDequeError,
    EmptyDoublyLinkedListError,
    bench_delete_last_vs_singly,
    bench_deque_operations,
    run_alternating_ends_stress,
)


# ----------------------------------------------------------------------
# Nó
# ----------------------------------------------------------------------
def test_no_guarda_valor_e_os_dois_ponteiros() -> None:
    middle: DoublyNode[int] = DoublyNode(2)
    first: DoublyNode[int] = DoublyNode(1, None, middle)
    middle.prev = first
    assert first.value == 1
    assert first.prev is None
    assert first.next is middle
    assert middle.prev is first
    assert middle.next is None


def test_no_usa_slots_e_nao_aceita_atributo_novo() -> None:
    """Sem __dict__ por nó: o custo de memória do prev tem de ser o custo real."""
    node: DoublyNode[int] = DoublyNode(1)
    with pytest.raises(AttributeError):
        node.peso = 10  # type: ignore[attr-defined]


def test_repr_do_no_nao_percorre_os_vizinhos() -> None:
    """Repr recursivo em lista duplamente encadeada entraria em laço infinito."""
    first: DoublyNode[int] = DoublyNode(1)
    second: DoublyNode[int] = DoublyNode(2, first)
    first.next = second
    assert repr(first) == "DoublyNode(1)"
    assert repr(second) == "DoublyNode(2)"


# ----------------------------------------------------------------------
# Estado inicial e leitura
# ----------------------------------------------------------------------
def test_lista_nova_esta_vazia() -> None:
    target: DoublyLinkedList[int] = DoublyLinkedList()
    assert len(target) == 0
    assert target.is_empty() is True
    assert target.to_list() == []
    assert target.to_list_reversed() == []
    assert str(target) == "[]"
    target.check_invariants()


def test_construcao_a_partir_de_iteravel_preserva_ordem() -> None:
    target: DoublyLinkedList[int] = DoublyLinkedList([3, 1, 4, 1, 5])
    assert target.to_list() == [3, 1, 4, 1, 5]
    assert target.to_list_reversed() == [5, 1, 4, 1, 3]
    assert len(target) == 5
    assert target.is_empty() is False
    target.check_invariants()


def test_str_mostra_encadeamento_duplo() -> None:
    target: DoublyLinkedList[str] = DoublyLinkedList(["a", "b"])
    assert str(target) == "['a' <-> 'b']"


def test_repr_inclui_tamanho_e_classe() -> None:
    target: DoublyLinkedList[int] = DoublyLinkedList([1, 2])
    assert "DoublyLinkedList" in repr(target)
    assert "size=2" in repr(target)


def test_travessia_nos_dois_sentidos_e_uma_o_inverso_da_outra() -> None:
    target: DoublyLinkedList[int] = DoublyLinkedList(range(10))
    frente = list(target)
    tras = list(reversed(target))
    assert frente == list(range(10))
    assert tras == list(reversed(frente))


def test_peek_first_e_peek_last() -> None:
    target: DoublyLinkedList[int] = DoublyLinkedList([7, 8, 9])
    assert target.peek_first() == 7
    assert target.peek_last() == 9
    assert len(target) == 3  # peek não remove


@pytest.mark.parametrize("operation", ["peek_first", "peek_last"])
def test_peek_em_lista_vazia_levanta_com_mensagem(operation: str) -> None:
    target: DoublyLinkedList[int] = DoublyLinkedList()
    with pytest.raises(EmptyDoublyLinkedListError, match="lista vazia"):
        getattr(target, operation)()


# ----------------------------------------------------------------------
# Inserção
# ----------------------------------------------------------------------
def test_insert_first_inverte_a_ordem_de_chegada() -> None:
    target: DoublyLinkedList[int] = DoublyLinkedList()
    for value in [1, 2, 3]:
        target.insert_first(value)
    assert target.to_list() == [3, 2, 1]
    assert target.to_list_reversed() == [1, 2, 3]
    target.check_invariants()


def test_insert_last_preserva_a_ordem_de_chegada() -> None:
    target: DoublyLinkedList[int] = DoublyLinkedList()
    for value in [1, 2, 3]:
        target.insert_last(value)
    assert target.to_list() == [1, 2, 3]
    assert target.to_list_reversed() == [3, 2, 1]
    target.check_invariants()


def test_insert_first_e_insert_last_intercalados() -> None:
    target: DoublyLinkedList[int] = DoublyLinkedList()
    target.insert_first(2)
    target.insert_last(3)
    target.insert_first(1)
    target.insert_last(4)
    assert target.to_list() == [1, 2, 3, 4]
    assert target.to_list_reversed() == [4, 3, 2, 1]
    assert target.peek_first() == 1
    assert target.peek_last() == 4
    target.check_invariants()


def test_insercao_em_lista_vazia_faz_head_e_tail_apontarem_para_o_mesmo_no() -> None:
    for operation in ("insert_first", "insert_last"):
        target: DoublyLinkedList[int] = DoublyLinkedList()
        getattr(target, operation)(42)
        assert len(target) == 1
        assert target.peek_first() == 42
        assert target.peek_last() == 42
        target.check_invariants()


# ----------------------------------------------------------------------
# Remoção
# ----------------------------------------------------------------------
def test_delete_first_devolve_e_remove_da_frente() -> None:
    target: DoublyLinkedList[int] = DoublyLinkedList([1, 2, 3])
    assert target.delete_first() == 1
    assert target.to_list() == [2, 3]
    assert target.to_list_reversed() == [3, 2]
    target.check_invariants()


def test_delete_last_devolve_e_remove_do_fim() -> None:
    target: DoublyLinkedList[int] = DoublyLinkedList([1, 2, 3])
    assert target.delete_last() == 3
    assert target.to_list() == [1, 2]
    assert target.to_list_reversed() == [2, 1]
    target.check_invariants()


@pytest.mark.parametrize("operation", ["delete_first", "delete_last"])
def test_remover_o_unico_elemento_restaura_estado_de_lista_vazia(
    operation: str,
) -> None:
    target: DoublyLinkedList[int] = DoublyLinkedList([42])
    assert getattr(target, operation)() == 42
    assert target.is_empty() is True
    assert target.to_list() == []
    assert target.to_list_reversed() == []
    target.check_invariants()
    with pytest.raises(EmptyDoublyLinkedListError):
        target.peek_first()


@pytest.mark.parametrize("operation", ["delete_first", "delete_last"])
def test_delete_em_lista_vazia_levanta_com_mensagem(operation: str) -> None:
    target: DoublyLinkedList[int] = DoublyLinkedList()
    with pytest.raises(EmptyDoublyLinkedListError, match="lista vazia"):
        getattr(target, operation)()


def test_esvaziar_pela_frente_e_reconstruir() -> None:
    target: DoublyLinkedList[int] = DoublyLinkedList([1, 2, 3])
    while not target.is_empty():
        target.delete_first()
    target.check_invariants()
    target.insert_last(9)
    target.insert_last(10)
    assert target.to_list() == [9, 10]
    assert target.to_list_reversed() == [10, 9]
    target.check_invariants()


def test_esvaziar_pelo_fim_e_reconstruir() -> None:
    """O bug clássico: esquecer de recuar _head/_tail ao esvaziar pelo fim."""
    target: DoublyLinkedList[int] = DoublyLinkedList([1, 2, 3])
    assert [target.delete_last() for _ in range(3)] == [3, 2, 1]
    target.check_invariants()
    target.insert_first(9)
    assert target.to_list() == [9]
    assert target.peek_first() == 9
    assert target.peek_last() == 9
    target.check_invariants()


def test_alternar_remocoes_pelas_duas_pontas() -> None:
    target: DoublyLinkedList[int] = DoublyLinkedList([1, 2, 3, 4, 5])
    assert target.delete_first() == 1
    assert target.delete_last() == 5
    assert target.delete_first() == 2
    assert target.delete_last() == 4
    assert target.to_list() == [3]
    target.check_invariants()


def test_no_removido_solta_os_ponteiros() -> None:
    """Nó removido não pode continuar apontando para dentro da lista: manteria
    viva uma referência à estrutura inteira e esconderia bug de religação."""
    target: DoublyLinkedList[int] = DoublyLinkedList([1, 2, 3])
    removed_head = target._head
    removed_tail = target._tail
    target.delete_first()
    target.delete_last()
    assert removed_head is not None and removed_head.next is None
    assert removed_tail is not None and removed_tail.prev is None


# ----------------------------------------------------------------------
# Contagens
# ----------------------------------------------------------------------
@pytest.mark.parametrize("operation", ["insert_first", "insert_last"])
def test_insercao_conta_uma_copia_e_nenhum_salto(operation: str) -> None:
    target: DoublyLinkedList[int] = DoublyLinkedList()
    counters = Counters()
    for value in range(100):
        getattr(target, operation)(value, counters)
    assert counters.copies == 100
    assert counters.hops == 0
    assert counters.comparisons == 0


@pytest.mark.parametrize("operation", ["delete_first", "delete_last"])
def test_remocao_nao_da_nenhum_salto(operation: str) -> None:
    """O ponto central do exercício: remover de qualquer ponta é O(1)."""
    target: DoublyLinkedList[int] = DoublyLinkedList(range(500))
    counters = Counters()
    for _ in range(500):
        getattr(target, operation)(counters)
    assert counters.hops == 0
    assert counters.copies == 0
    assert target.is_empty() is True


def test_protocolo_resultado_counters() -> None:
    target: DoublyLinkedList[int] = DoublyLinkedList()
    resultado, counters = target.insert_last_counted(1)
    assert resultado is None
    assert counters.copies == 1
    resultado, counters = target.insert_first_counted(0)
    assert resultado is None
    assert counters.copies == 1
    value, counters = target.delete_first_counted()
    assert value == 0
    assert counters.hops == 0
    value, counters = target.delete_last_counted()
    assert value == 1
    assert counters.hops == 0


# ----------------------------------------------------------------------
# check_invariants tem de FALHAR em estrutura corrompida
# ----------------------------------------------------------------------
def test_check_invariants_detecta_size_maior_que_a_lista() -> None:
    target: DoublyLinkedList[int] = DoublyLinkedList([1, 2, 3])
    target._size = 5
    with pytest.raises(DoublyLinkedListInvariantError, match="nó\\(s\\) alcançáveis"):
        target.check_invariants()


def test_check_invariants_detecta_size_menor_que_a_lista() -> None:
    target: DoublyLinkedList[int] = DoublyLinkedList([1, 2, 3])
    target._size = 2
    with pytest.raises(DoublyLinkedListInvariantError, match="excede _size"):
        target.check_invariants()


def test_check_invariants_detecta_head_prev_nao_nulo() -> None:
    target: DoublyLinkedList[int] = DoublyLinkedList([1, 2])
    assert target._head is not None
    target._head.prev = DoublyNode(99)
    with pytest.raises(DoublyLinkedListInvariantError, match="_head.prev"):
        target.check_invariants()


def test_check_invariants_detecta_tail_next_nao_nulo() -> None:
    target: DoublyLinkedList[int] = DoublyLinkedList([1, 2])
    assert target._tail is not None
    target._tail.next = DoublyNode(99)
    with pytest.raises(DoublyLinkedListInvariantError, match="_tail.next"):
        target.check_invariants()


def test_check_invariants_detecta_elo_de_volta_quebrado() -> None:
    """O bug característico: encadeamento certo para frente, quebrado para trás."""
    target: DoublyLinkedList[int] = DoublyLinkedList([1, 2, 3])
    assert target._head is not None and target._head.next is not None
    target._head.next.prev = None
    with pytest.raises(DoublyLinkedListInvariantError, match="elo quebrado"):
        target.check_invariants()
    # e, antes da corrupção, a travessia para frente continuaria "funcionando":
    assert target.to_list() == [1, 2, 3]


def test_check_invariants_detecta_lista_vazia_com_ponteiro_sujo() -> None:
    target: DoublyLinkedList[int] = DoublyLinkedList()
    target._head = DoublyNode(1)
    with pytest.raises(DoublyLinkedListInvariantError, match="lista vazia"):
        target.check_invariants()


def test_check_invariants_detecta_size_negativo() -> None:
    target: DoublyLinkedList[int] = DoublyLinkedList()
    target._size = -1
    with pytest.raises(DoublyLinkedListInvariantError, match="negativo"):
        target.check_invariants()


# ----------------------------------------------------------------------
# Sequência longa alternando as pontas (exigência do enunciado)
# ----------------------------------------------------------------------
def test_invariantes_resistem_a_sequencia_longa_na_lista() -> None:
    rng = random.Random(42)
    target: DoublyLinkedList[int] = DoublyLinkedList()
    model: list[int] = []
    for step in range(3_000):
        operation = rng.choice(
            ["insert_first", "insert_last", "delete_first", "delete_last"]
        )
        if operation == "insert_first":
            value = rng.randrange(1000)
            target.insert_first(value)
            model.insert(0, value)
        elif operation == "insert_last":
            value = rng.randrange(1000)
            target.insert_last(value)
            model.append(value)
        elif operation == "delete_first":
            if model:
                assert target.delete_first() == model.pop(0)
            else:
                with pytest.raises(EmptyDoublyLinkedListError):
                    target.delete_first()
        else:
            if model:
                assert target.delete_last() == model.pop()
            else:
                with pytest.raises(EmptyDoublyLinkedListError):
                    target.delete_last()

        if step % 100 == 0:
            target.check_invariants()
            assert target.to_list() == model
            assert target.to_list_reversed() == list(reversed(model))

    target.check_invariants()
    assert target.to_list() == model


# ----------------------------------------------------------------------
# Deque
# ----------------------------------------------------------------------
def test_deque_novo_esta_vazio() -> None:
    target: Deque[int] = Deque()
    assert len(target) == 0
    assert target.is_empty() is True
    assert target.to_list() == []
    target.check_invariants()


def test_deque_construido_a_partir_de_iteravel_entra_pela_direita() -> None:
    target: Deque[int] = Deque([1, 2, 3])
    assert target.to_list() == [1, 2, 3]
    assert target.peek_left() == 1
    assert target.peek_right() == 3


def test_deque_str_identifica_as_pontas() -> None:
    target: Deque[int] = Deque([1, 2])
    assert str(target) == "esquerda [1 <-> 2] direita"
    assert "size=2" in repr(target)


def test_deque_insere_e_remove_nas_duas_pontas() -> None:
    target: Deque[int] = Deque()
    target.insert_right(2)
    target.insert_right(3)
    target.insert_left(1)
    target.insert_left(0)
    assert target.to_list() == [0, 1, 2, 3]
    assert target.remove_left() == 0
    assert target.remove_right() == 3
    assert target.to_list() == [1, 2]
    target.check_invariants()


def test_deque_peek_nao_remove() -> None:
    target: Deque[int] = Deque([1, 2, 3])
    assert target.peek_left() == 1
    assert target.peek_right() == 3
    assert len(target) == 3


def test_deque_com_um_elemento_tem_as_duas_pontas_iguais() -> None:
    target: Deque[str] = Deque(["único"])
    assert target.peek_left() == "único"
    assert target.peek_right() == "único"
    assert target.remove_right() == "único"
    assert target.is_empty() is True
    target.check_invariants()


def test_deque_fifo_pela_esquerda_e_lifo_pela_direita() -> None:
    """O deque tem de servir como fila e como pilha — é o que o ex 12 vai usar."""
    fila: Deque[int] = Deque()
    for value in range(5):
        fila.insert_right(value)
    assert [fila.remove_left() for _ in range(5)] == [0, 1, 2, 3, 4]

    pilha: Deque[int] = Deque()
    for value in range(5):
        pilha.insert_right(value)
    assert [pilha.remove_right() for _ in range(5)] == [4, 3, 2, 1, 0]


@pytest.mark.parametrize(
    "operation", ["remove_left", "remove_right", "peek_left", "peek_right"]
)
def test_deque_vazio_levanta_excecao_propria(operation: str) -> None:
    """Tratamento de deque vazio: exceção própria, nunca None nem Exception crua."""
    target: Deque[int] = Deque()
    with pytest.raises(EmptyDequeError, match="deque vazio"):
        getattr(target, operation)()


def test_excecao_do_deque_nao_vaza_a_da_estrutura_interna() -> None:
    """Decisão de projeto 2: quem usa o Deque não captura exceção da lista."""
    assert issubclass(EmptyDequeError, DequeError)
    assert not issubclass(EmptyDequeError, DoublyLinkedListError)
    assert not issubclass(DequeError, DoublyLinkedListError)

    target: Deque[int] = Deque()
    with pytest.raises(EmptyDequeError):
        target.remove_left()
    # e a exceção da lista NÃO é capturável como erro de deque:
    assert not isinstance(EmptyDoublyLinkedListError(), DequeError)


def test_deque_nao_expoe_operacoes_de_custo_linear() -> None:
    """Composição, não herança: a promessa 'tudo O(1)' vale para o tipo inteiro."""
    target: Deque[int] = Deque([1, 2, 3])
    for forbidden in ("insert_at", "delete_at", "search"):
        assert not hasattr(target, forbidden)


def test_deque_conta_como_a_lista() -> None:
    target: Deque[int] = Deque()
    counters = Counters()
    for value in range(50):
        target.insert_left(value, counters)
        target.insert_right(value, counters)
    for _ in range(50):
        target.remove_left(counters)
        target.remove_right(counters)
    assert counters.copies == 100  # só as inserções contam cópia
    assert counters.hops == 0  # nenhuma operação percorre a estrutura
    assert target.is_empty() is True


def test_deque_remove_counted_segue_o_protocolo() -> None:
    target: Deque[int] = Deque([1, 2])
    value, counters = target.remove_left_counted()
    assert (value, counters.hops) == (1, 0)
    value, counters = target.remove_right_counted()
    assert (value, counters.hops) == (2, 0)


# ----------------------------------------------------------------------
# Bateria longa via src (a evidência que vai para o notebook)
# ----------------------------------------------------------------------
def test_stress_alternando_pontas_mantem_invariantes() -> None:
    row = run_alternating_ends_stress(operations=4_000, check_every=250)
    assert row["n"] == 4_000
    assert row["verificacoes_de_invariante"] == 16
    assert row["hops"] == 0  # nenhuma operação percorreu a estrutura
    assert sum(row[f"op_{name}"] for name in END_OPERATIONS) == 4_000
    # o caso de borda "deque vazio" foi exercitado de verdade, não uma vez só:
    assert row["tentativas_em_vazio"] > 0
    assert row["tamanho_maximo"] > 0
    inserts = row["op_insert_left"] + row["op_insert_right"]
    removes = row["op_remove_left"] + row["op_remove_right"]
    assert row["tamanho_final"] == inserts - (removes - row["tentativas_em_vazio"])
    assert row["copies"] == inserts


def test_stress_e_reprodutivel_com_a_mesma_semente() -> None:
    primeira = run_alternating_ends_stress(operations=1_000, seed=42, check_every=500)
    segunda = run_alternating_ends_stress(operations=1_000, seed=42, check_every=500)
    for key in ("tamanho_final", "tentativas_em_vazio", "copies", "hops"):
        assert primeira[key] == segunda[key]


def test_stress_valida_parametros() -> None:
    with pytest.raises(ValueError, match="operations deve ser >= 1"):
        run_alternating_ends_stress(operations=0)
    with pytest.raises(ValueError, match="check_every deve ser >= 1"):
        run_alternating_ends_stress(operations=10, check_every=0)


# ----------------------------------------------------------------------
# Experimentos
# ----------------------------------------------------------------------
def test_bench_deque_operations_mostra_custo_constante() -> None:
    rows = bench_deque_operations([100, 200, 400], repeats=1)
    assert len(rows) == 3 * len(END_OPERATIONS)
    for row in rows:
        assert row["hops"] == 0 == row["hops_teorico"]
        assert row["hops/n"] == 0
        assert row["exercicio"] == "ex11"
        assert row["tempo_us_por_op"] > 0

    # cada operação aparece uma vez por tamanho
    for operation in END_OPERATIONS:
        medidas = [row for row in rows if row["operacao"] == operation]
        assert [row["n"] for row in medidas] == [100, 200, 400]


def test_bench_deque_operations_valida_parametros() -> None:
    with pytest.raises(ValueError, match="repeats deve ser >= 1"):
        bench_deque_operations([10], repeats=0)
    with pytest.raises(ValueError, match="tamanho deve ser positivo"):
        bench_deque_operations([0])


def test_bench_delete_last_confirma_as_duas_classes_de_complexidade() -> None:
    sizes = [50, 100, 200, 400]
    rows = bench_delete_last_vs_singly(sizes, repeats=1)
    duplas = [row for row in rows if row["estrutura"] == "duplamente encadeada"]
    simples = [row for row in rows if row["estrutura"] == "simplesmente encadeada"]
    assert len(duplas) == len(simples) == len(sizes)

    for row in duplas:
        assert row["hops"] == 0 == row["hops_teorico"]
        assert row["hops/n"] == 0

    for row in simples:
        n = row["n"]
        # contagem fecha com a fórmula fechada, dígito a dígito
        assert row["hops"] == n * (n - 1) // 2 == row["hops_teorico"]
        # razão contra n² aproximadamente constante => Θ(n²)
        assert row["hops/n2"] == pytest.approx(0.5, abs=0.02)
        # saltos por remoção cresce com n => não é O(1) por operação
        assert row["hops/n"] == pytest.approx((n - 1) / 2)

    por_remocao = [row["hops/n"] for row in simples]
    assert por_remocao == sorted(por_remocao)
    assert por_remocao[-1] > 7 * por_remocao[0]  # n dobrou 3x => ~8x


def test_bench_delete_last_valida_parametros() -> None:
    with pytest.raises(ValueError, match="repeats deve ser >= 1"):
        bench_delete_last_vs_singly([10], repeats=0)
    with pytest.raises(ValueError, match="tamanho deve ser positivo"):
        bench_delete_last_vs_singly([-1])


def test_contagens_nao_dependem_de_repeats() -> None:
    uma = bench_deque_operations([200], repeats=1)
    tres = bench_deque_operations([200], repeats=3)
    assert [row["copies"] for row in uma] == [row["copies"] for row in tres]
    assert [row["hops"] for row in uma] == [row["hops"] for row in tres]


# ----------------------------------------------------------------------
# Tabelas de custo
# ----------------------------------------------------------------------
def test_tabela_de_custo_cobre_as_operacoes_do_enunciado() -> None:
    operacoes = {row["operacao"] for row in COST_TABLE}
    for required in (
        "insert_first",
        "insert_last",
        "delete_first",
        "delete_last",
        "is_empty",
    ):
        assert required in operacoes
    for row in COST_TABLE:
        assert set(row) == {"operacao", "duplamente", "simplesmente", "por_que"}
        assert row["por_que"]


def test_tabela_de_custo_registra_a_unica_divergencia_relevante() -> None:
    """delete_last é a linha que justifica a estrutura; tem de estar registrada."""
    linha = next(row for row in COST_TABLE if row["operacao"] == "delete_last")
    assert linha["duplamente"] == "O(1)"
    assert linha["simplesmente"] == "O(n)"


def test_tabela_do_deque_promete_o_um_em_todas_as_seis_operacoes() -> None:
    operacoes = {row["operacao"] for row in DEQUE_COST_TABLE}
    assert operacoes == {
        "insert_left",
        "insert_right",
        "remove_left",
        "remove_right",
        "peek_left",
        "peek_right",
    }
    for row in DEQUE_COST_TABLE:
        assert row["custo"] == "O(1)"
        assert row["saltos"] == "0"

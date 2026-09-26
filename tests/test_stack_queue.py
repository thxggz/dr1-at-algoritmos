"""Testes do exercício 5 — Stack, Queue circular e árvore de expressão.

O enunciado pede quatro coisas que viram teste explícito:

* exceções **específicas** para push em pilha cheia e pop em pilha vazia
  (e o par correspondente na fila);
* fila com **wraparound**, provado por contador, não por suposição;
* detecção de entradas inválidas na montagem da árvore — operador sem
  operandos e sobra de operandos no fim;
* as duas avaliações (pós-ordem recursiva e iterativa com stack) chegando ao
  mesmo resultado, com a diferença de memória medida.
"""

from __future__ import annotations

import random
import sys

import pytest

from src.counters import Counters
from src.stack_queue import (
    OPERATORS,
    DivisionByZeroError,
    EmptyExpressionError,
    Evaluation,
    ExpressionNode,
    InsufficientOperandsError,
    LeftoverOperandsError,
    MalformedTreeError,
    Queue,
    QueueConfigError,
    QueueInvariantError,
    QueueOverflowError,
    QueueUnderflowError,
    Stack,
    StackConfigError,
    StackInvariantError,
    StackOverflowError,
    StackUnderflowError,
    UnknownTokenError,
    bench_queue_wraparound,
    bench_recursion_depth,
    bench_stack_queue_operations,
    build_expression_tree,
    chained_postfix,
    check_expression_tree,
    evaluate_iterative,
    evaluate_iterative_counted,
    evaluate_recursive,
    evaluate_recursive_counted,
    to_infix,
    tree_height,
    tree_size,
)


# ----------------------------------------------------------------------
# Stack
# ----------------------------------------------------------------------
def test_pilha_nova_esta_vazia() -> None:
    stack: Stack[int] = Stack(4)
    assert len(stack) == 0
    assert stack.capacity == 4
    assert stack.is_empty() is True
    assert stack.is_full() is False
    assert stack.to_list() == []
    assert str(stack) == "fundo [] topo"
    stack.check_invariants()


def test_pilha_e_lifo() -> None:
    stack: Stack[int] = Stack(4, [1, 2, 3])
    assert stack.to_list() == [1, 2, 3]  # do fundo para o topo
    assert stack.peek() == 3
    assert [stack.pop() for _ in range(3)] == [3, 2, 1]
    assert stack.is_empty() is True
    stack.check_invariants()


def test_pilha_cheia_levanta_overflow_com_contexto() -> None:
    stack: Stack[int] = Stack(3, [1, 2, 3])
    assert stack.is_full() is True
    with pytest.raises(StackOverflowError, match="capacidade 3 esgotada"):
        stack.push(4)
    assert len(stack) == 3  # a pilha não foi corrompida pela tentativa
    stack.check_invariants()


def test_pilha_vazia_levanta_underflow() -> None:
    stack: Stack[int] = Stack(3)
    with pytest.raises(StackUnderflowError, match="pop em pilha vazia"):
        stack.pop()
    with pytest.raises(StackUnderflowError, match="peek em pilha vazia"):
        stack.peek()


def test_pilha_libera_a_posicao_ao_desempilhar() -> None:
    """Decisão de projeto 3: sem isso o array seguraria o objeto removido."""
    stack: Stack[str] = Stack(3, ["a", "b"])
    stack.pop()
    assert stack._items == ["a", None, None]
    stack.check_invariants()


def test_pilha_registra_o_pico_de_ocupacao() -> None:
    stack: Stack[int] = Stack(10)
    for value in range(7):
        stack.push(value)
    for _ in range(7):
        stack.pop()
    assert len(stack) == 0
    assert stack.peak_size == 7  # a memória de fato usada, não a alocada
    assert stack.capacity == 10


def test_pilha_conta_uma_copia_por_operacao() -> None:
    stack: Stack[int] = Stack(100)
    counters = Counters()
    for value in range(100):
        stack.push(value, counters)
    for _ in range(100):
        stack.pop(counters)
    assert counters.copies == 200
    assert counters.hops == 0
    assert counters.comparisons == 0


def test_pilha_protocolo_resultado_counters() -> None:
    stack: Stack[int] = Stack(4)
    resultado, counters = stack.push_counted(1)
    assert resultado is None and counters.copies == 1
    value, counters = stack.pop_counted()
    assert value == 1 and counters.copies == 1


@pytest.mark.parametrize("bad_capacity", [0, -1])
def test_pilha_com_capacidade_invalida_levanta(bad_capacity: int) -> None:
    with pytest.raises(StackConfigError, match=">= 1"):
        Stack(bad_capacity)


@pytest.mark.parametrize("bad_capacity", ["4", 4.0, None, True])
def test_pilha_com_capacidade_nao_inteira_levanta(bad_capacity: object) -> None:
    with pytest.raises(StackConfigError, match="int"):
        Stack(bad_capacity)  # type: ignore[arg-type]


def test_check_invariants_da_pilha_detecta_lixo_acima_do_topo() -> None:
    stack: Stack[int] = Stack(3, [1])
    stack._items[2] = 99
    with pytest.raises(StackInvariantError, match="acima do topo"):
        stack.check_invariants()


def test_check_invariants_da_pilha_detecta_size_fora_da_faixa() -> None:
    stack: Stack[int] = Stack(3, [1])
    stack._size = 9
    with pytest.raises(StackInvariantError, match="fora da faixa"):
        stack.check_invariants()


# ----------------------------------------------------------------------
# Queue
# ----------------------------------------------------------------------
def test_fila_nova_esta_vazia() -> None:
    queue: Queue[int] = Queue(4)
    assert len(queue) == 0
    assert queue.capacity == 4
    assert queue.is_empty() is True
    assert queue.to_list() == []
    assert str(queue) == "frente [] fundo"
    queue.check_invariants()


def test_fila_e_fifo() -> None:
    queue: Queue[int] = Queue(4, [1, 2, 3])
    assert queue.to_list() == [1, 2, 3]
    assert queue.peek() == 1
    assert [queue.dequeue() for _ in range(3)] == [1, 2, 3]
    assert queue.is_empty() is True
    queue.check_invariants()


def test_fila_cheia_levanta_overflow_com_contexto() -> None:
    queue: Queue[int] = Queue(3, [1, 2, 3])
    assert queue.is_full() is True
    with pytest.raises(QueueOverflowError, match="capacidade 3 esgotada"):
        queue.enqueue(4)
    assert len(queue) == 3
    queue.check_invariants()


def test_fila_vazia_levanta_underflow() -> None:
    queue: Queue[int] = Queue(3)
    with pytest.raises(QueueUnderflowError, match="dequeue em fila vazia"):
        queue.dequeue()
    with pytest.raises(QueueUnderflowError, match="peek em fila vazia"):
        queue.peek()


def test_fila_usa_toda_a_capacidade_sem_sacrificar_posicao() -> None:
    """Decisão de projeto 2: guardar _size em vez de _tail evita desperdiçar
    uma posição para distinguir cheia de vazia."""
    queue: Queue[int] = Queue(4)
    for value in range(4):
        queue.enqueue(value)
    assert len(queue) == 4 == queue.capacity
    assert queue.is_full() is True
    queue.check_invariants()


def test_fila_da_a_volta_no_array() -> None:
    """Wraparound: a cauda volta para a posição 0 e reaproveita slots livres."""
    queue: Queue[int] = Queue(4, [0, 1, 2])
    assert queue.wraparounds == 0
    queue.dequeue()  # head passa para 1, posição 0 fica livre
    queue.dequeue()  # head passa para 2
    queue.enqueue(3)  # cauda em 3, ainda sem dar a volta
    assert queue.wraparounds == 0
    queue.enqueue(4)  # cauda daria em 4 -> volta para 0
    assert queue.wraparounds == 1
    assert queue.to_list() == [2, 3, 4]
    assert queue._head == 2
    queue.check_invariants()


def test_fila_libera_a_posicao_ao_desenfileirar() -> None:
    queue: Queue[str] = Queue(3, ["a", "b"])
    queue.dequeue()
    assert queue._items == [None, "b", None]
    queue.check_invariants()


def test_fila_registra_o_pico_de_ocupacao() -> None:
    queue: Queue[int] = Queue(10)
    for value in range(6):
        queue.enqueue(value)
    for _ in range(6):
        queue.dequeue()
    assert queue.peak_size == 6
    assert len(queue) == 0


def test_fila_conta_uma_copia_por_operacao() -> None:
    queue: Queue[int] = Queue(100)
    counters = Counters()
    for value in range(100):
        queue.enqueue(value, counters)
    for _ in range(100):
        queue.dequeue(counters)
    assert counters.copies == 200
    assert counters.hops == 0


def test_fila_protocolo_resultado_counters() -> None:
    queue: Queue[int] = Queue(4)
    resultado, counters = queue.enqueue_counted(1)
    assert resultado is None and counters.copies == 1
    value, counters = queue.dequeue_counted()
    assert value == 1 and counters.copies == 1


@pytest.mark.parametrize("bad_capacity", [0, -1])
def test_fila_com_capacidade_invalida_levanta(bad_capacity: int) -> None:
    with pytest.raises(QueueConfigError, match=">= 1"):
        Queue(bad_capacity)


@pytest.mark.parametrize("bad_capacity", ["4", 4.0, None, True])
def test_fila_com_capacidade_nao_inteira_levanta(bad_capacity: object) -> None:
    with pytest.raises(QueueConfigError, match="int"):
        Queue(bad_capacity)  # type: ignore[arg-type]


def test_check_invariants_da_fila_detecta_head_fora_da_faixa() -> None:
    queue: Queue[int] = Queue(4, [1, 2])
    queue._head = 9
    with pytest.raises(QueueInvariantError, match="_head"):
        queue.check_invariants()


def test_check_invariants_da_fila_detecta_posicao_suja() -> None:
    queue: Queue[int] = Queue(4, [1, 2])
    queue._items[3] = 99
    with pytest.raises(QueueInvariantError, match="deveria estar livre"):
        queue.check_invariants()


def test_fila_resiste_a_sequencia_longa_alternada() -> None:
    """Modelo de referência em list, permitido pela convenção 9."""
    rng = random.Random(42)
    capacidade = 16
    queue: Queue[int] = Queue(capacidade)
    model: list[int] = []
    cheias = vazias = 0

    for step in range(5_000):
        if rng.random() < 0.5:
            value = rng.randrange(1_000)
            if len(model) == capacidade:
                with pytest.raises(QueueOverflowError):
                    queue.enqueue(value)
                cheias += 1
            else:
                queue.enqueue(value)
                model.append(value)
        else:
            if not model:
                with pytest.raises(QueueUnderflowError):
                    queue.dequeue()
                vazias += 1
            else:
                assert queue.dequeue() == model.pop(0)
        assert len(queue) == len(model)
        if step % 200 == 0:
            queue.check_invariants()
            assert queue.to_list() == model

    queue.check_invariants()
    assert queue.to_list() == model
    assert queue.wraparounds > 0  # o buffer circular foi mesmo exercitado
    assert cheias > 0 and vazias > 0  # os dois casos de borda aconteceram


# ----------------------------------------------------------------------
# Árvore de expressão — construção
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("postfix", "infix", "valor"),
    [
        ("3 4 +", "(3 + 4)", 7),
        ("3 4 + 2 *", "((3 + 4) * 2)", 14),
        ("5 1 2 + 4 * + 3 -", "((5 + ((1 + 2) * 4)) - 3)", 14),
        ("2 3 ^", "(2 ^ 3)", 8),
        ("8 2 /", "(8 / 2)", 4.0),
        ("7", "7", 7),
        ("-3 5 +", "(-3 + 5)", 2),
        ("2.5 1.5 +", "(2.5 + 1.5)", 4.0),
    ],
)
def test_monta_e_avalia_expressoes_validas(
    postfix: str, infix: str, valor: float
) -> None:
    root = build_expression_tree(postfix)
    check_expression_tree(root)
    assert to_infix(root) == infix
    assert evaluate_recursive(root).value == valor
    assert evaluate_iterative(root).value == valor


def test_a_ordem_dos_operandos_nao_e_invertida() -> None:
    """O primeiro desempilhado é o operando da DIREITA; errar isso passa em
    soma e multiplicação e quebra em subtração e divisão."""
    assert evaluate_recursive(build_expression_tree("10 3 -")).value == 7
    assert evaluate_recursive(build_expression_tree("10 2 /")).value == 5.0
    assert to_infix(build_expression_tree("10 3 -")) == "(10 - 3)"


def test_expressao_vazia_levanta() -> None:
    for entrada in ("", "   ", "\t\n"):
        with pytest.raises(EmptyExpressionError, match="vazia"):
            build_expression_tree(entrada)


def test_operador_sem_operandos_suficientes_levanta_com_a_posicao() -> None:
    with pytest.raises(InsufficientOperandsError, match="posição 2"):
        build_expression_tree("3 +")
    with pytest.raises(InsufficientOperandsError, match="posição 1"):
        build_expression_tree("+ 3 4")
    with pytest.raises(InsufficientOperandsError, match="posição 4"):
        build_expression_tree("1 2 + *")


def test_sobra_de_operandos_levanta_listando_o_que_sobrou() -> None:
    with pytest.raises(LeftoverOperandsError, match="2 valores"):
        build_expression_tree("3 4")
    with pytest.raises(LeftoverOperandsError, match="3 valores"):
        build_expression_tree("1 2 3")
    with pytest.raises(LeftoverOperandsError, match="2 valores"):
        build_expression_tree("1 2 + 9")


def test_token_desconhecido_levanta() -> None:
    with pytest.raises(UnknownTokenError, match="não é operando"):
        build_expression_tree("3 4 %")
    with pytest.raises(UnknownTokenError):
        build_expression_tree("3 abc +")


def test_estouro_da_pilha_na_montagem_e_reportado() -> None:
    """Capacidade fixa de verdade: com pilha pequena, montar estoura."""
    with pytest.raises(StackOverflowError, match="capacidade 2 esgotada"):
        build_expression_tree("1 2 3 + +", capacity=2)
    # e com a capacidade padrão a mesma expressão monta sem problema
    assert evaluate_recursive(build_expression_tree("1 2 3 + +")).value == 6


def test_divisao_por_zero_levanta_na_avaliacao() -> None:
    root = build_expression_tree("5 0 /")
    with pytest.raises(DivisionByZeroError, match="divisão por zero"):
        evaluate_recursive(root)
    with pytest.raises(DivisionByZeroError):
        evaluate_iterative(root)


def test_check_expression_tree_detecta_arvore_malformada() -> None:
    operador_sem_filhos = ExpressionNode("+")
    with pytest.raises(MalformedTreeError, match="dois filhos"):
        check_expression_tree(operador_sem_filhos)

    operando_com_filho = ExpressionNode("3", ExpressionNode("1"))
    with pytest.raises(MalformedTreeError, match="não pode ter filhos"):
        check_expression_tree(operando_com_filho)


def test_avaliar_arvore_malformada_tambem_falha() -> None:
    with pytest.raises(MalformedTreeError):
        evaluate_recursive(ExpressionNode("+"))


def test_todos_os_operadores_sao_reconhecidos() -> None:
    assert OPERATORS == {"+", "-", "*", "/", "^"}
    for operador in sorted(OPERATORS):
        root = build_expression_tree(f"6 3 {operador}")
        assert root.is_operator() and root.token == operador
        assert root.left is not None and root.right is not None
        assert evaluate_recursive(root).value == evaluate_iterative(root).value


# ----------------------------------------------------------------------
# Árvore de expressão — forma e travessia
# ----------------------------------------------------------------------
def test_tamanho_e_altura_da_arvore() -> None:
    root = build_expression_tree("3 4 + 2 *")
    assert tree_size(root) == 5  # 3, 4, +, 2, *
    assert tree_height(root) == 3


@pytest.mark.parametrize("lean", ["left", "right"])
@pytest.mark.parametrize("operators", [1, 2, 5, 20])
def test_expressao_encadeada_tem_a_altura_prevista(lean: str, operators: int) -> None:
    root = build_expression_tree(chained_postfix(operators, lean=lean))
    assert tree_size(root) == 2 * operators + 1
    assert tree_height(root) == operators + 1
    assert evaluate_iterative(root).value == operators + 1


def test_chained_postfix_valida_parametros() -> None:
    with pytest.raises(ValueError, match="operators deve ser >= 1"):
        chained_postfix(0)
    with pytest.raises(ValueError, match="lean deve ser"):
        chained_postfix(3, lean="meio")


def test_formato_da_arvore_muda_a_pilha_de_operandos_mas_nao_a_profundidade() -> None:
    """A separação que o experimento usa: mesma altura, picos diferentes."""
    operators = 30
    esquerda = evaluate_iterative(
        build_expression_tree(chained_postfix(operators, lean="left"))
    )
    direita = evaluate_iterative(
        build_expression_tree(chained_postfix(operators, lean="right"))
    )
    assert esquerda.max_depth == direita.max_depth == operators + 1
    assert esquerda.peak_operand_stack == 2
    assert direita.peak_operand_stack == operators + 1


# ----------------------------------------------------------------------
# Avaliação: recursiva × iterativa
# ----------------------------------------------------------------------
def test_as_duas_avaliacoes_fazem_o_mesmo_trabalho() -> None:
    root = build_expression_tree("5 1 2 + 4 * + 3 -")
    recursiva = evaluate_recursive(root)
    iterativa = evaluate_iterative(root)

    assert recursiva.value == iterativa.value == 14
    assert recursiva.counters.calls == iterativa.counters.calls == tree_size(root)
    assert recursiva.counters.copies == iterativa.counters.copies
    assert recursiva.max_depth == tree_height(root)
    assert iterativa.max_depth == tree_height(root)


def test_avaliacoes_seguem_o_protocolo_resultado_counters() -> None:
    root = build_expression_tree("3 4 +")
    for chamada in (evaluate_recursive_counted, evaluate_iterative_counted):
        value, counters = chamada(root)
        assert value == 7
        assert counters.calls == 3

    avaliacao = evaluate_recursive(root)
    assert isinstance(avaliacao, Evaluation)
    assert avaliacao.as_tuple() == (avaliacao.value, avaliacao.counters)


def test_avaliacoes_acumulam_em_contador_do_chamador() -> None:
    root = build_expression_tree("3 4 +")
    counters = Counters()
    evaluate_recursive(root, counters)
    evaluate_iterative(root, counters)
    assert counters.calls == 6  # 3 nós, duas vezes


def test_recursiva_quebra_onde_a_iterativa_nao_quebra() -> None:
    """O caso de borda central do exercício: o teto é do intérprete, não do
    algoritmo. A iterativa faz o mesmo trabalho com memória no heap."""
    operators = 10 * sys.getrecursionlimit()
    root = build_expression_tree(chained_postfix(operators, lean="left"))

    with pytest.raises(RecursionError):
        evaluate_recursive(root)

    iterativa = evaluate_iterative(root)
    assert iterativa.value == operators + 1
    assert iterativa.counters.calls == tree_size(root)
    assert iterativa.max_depth == operators + 1


# ----------------------------------------------------------------------
# Experimentos
# ----------------------------------------------------------------------
def test_bench_operacoes_mostra_custo_constante() -> None:
    rows = bench_stack_queue_operations([500, 1_000, 2_000], repeats=1)
    assert len(rows) == 3 * 4  # 3 tamanhos x (push, pop, enqueue, dequeue)
    for row in rows:
        assert row["hops"] == 0 == row["hops_teorico"]
        assert row["copies"] == row["n"]
        assert row["tempo_us_por_op"] > 0

    for operacao in ("push", "pop", "enqueue", "dequeue"):
        medidas = [row for row in rows if row["operacao"] == operacao]
        assert [row["n"] for row in medidas] == [500, 1_000, 2_000]


def test_bench_operacoes_valida_parametros() -> None:
    with pytest.raises(ValueError, match="repeats deve ser >= 1"):
        bench_stack_queue_operations([10], repeats=0)
    with pytest.raises(ValueError, match="tamanho deve ser positivo"):
        bench_stack_queue_operations([0])


def test_bench_wraparound_prova_o_reaproveitamento() -> None:
    row = bench_queue_wraparound(capacity=8, operations=5_000)
    assert row["fifo_preservado"] is True
    assert row["posicoes_alocadas"] == 8
    assert row["ocupacao_maxima"] == 8
    assert row["restantes_na_fila"] == 7
    # 5.000 inserções em um array de 8 posições só cabem com wraparound
    assert row["voltas_no_array"] > 500
    assert row["copies"] == 5_000 + (5_000 - 7)


def test_bench_wraparound_valida_parametros() -> None:
    with pytest.raises(ValueError, match="capacity deve ser >= 2"):
        bench_queue_wraparound(capacity=1)
    with pytest.raises(ValueError, match="deve ser >= capacity"):
        bench_queue_wraparound(capacity=8, operations=4)


def test_bench_recursion_depth_separa_trabalho_de_memoria() -> None:
    rows = bench_recursion_depth([10, 50, 200])
    assert len(rows) == 6  # 3 tamanhos x 2 formatos
    for row in rows:
        assert row["recursiva_quebrou"] is False
        assert row["valor_recursiva"] == row["valor_iterativa"]
        # mesmo trabalho: as duas visitam todos os nós uma vez
        assert row["calls_recursiva"] == row["calls_iterativa"] == row["n"]
        # mesma profundidade: a altura da árvore
        assert row["profundidade_recursiva"] == row["altura"]
        assert row["profundidade_iterativa"] == row["altura"]

    esquerda = [row for row in rows if "esquerda" in row["padrao"]]
    direita = [row for row in rows if "direita" in row["padrao"]]
    # sem estes dois, os `all(...)` abaixo passariam por vácuo se o filtro
    # deixasse de casar com o rótulo do padrão
    assert len(esquerda) == 3
    assert len(direita) == 3
    assert all(row["pico_operandos"] == 2 for row in esquerda)
    assert all(row["pico_operandos"] == row["altura"] for row in direita)


def test_bench_recursion_depth_registra_a_quebra_da_recursiva() -> None:
    grande = 5 * sys.getrecursionlimit()
    rows = bench_recursion_depth([grande], leans=("left",))
    row = rows[0]
    assert row["recursiva_quebrou"] is True
    assert row["valor_recursiva"] is None
    assert row["valor_iterativa"] == grande + 1  # a iterativa terminou
    assert row["limite_de_recursao"] == sys.getrecursionlimit()


def test_bench_recursion_depth_valida_parametros() -> None:
    with pytest.raises(ValueError, match="tamanho deve ser positivo"):
        bench_recursion_depth([0])

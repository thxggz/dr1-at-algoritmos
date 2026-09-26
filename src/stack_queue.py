"""Exercício 5 — Pilha, fila de capacidade fixa e árvore de expressão.

A :class:`Stack` é reutilizada pelo exercício 12 (conversão infixa→pós-fixa por
shunting-yard); a :class:`Queue` é a base do simulador de eventos do exercício 6
e do percurso em largura da BST no exercício 12.

Decisões de projeto
-------------------
1. **Capacidade fixa de verdade.** O array interno é alocado uma vez, no
   construtor, e nunca cresce. Encher dispara :class:`StackOverflowError` /
   :class:`QueueOverflowError` em vez de realocar. É o que o enunciado pede, e
   é o que torna o overflow um caso de borda observável — com array que cresce
   sozinho, "overflow" nunca acontece e não há o que testar.
2. **A fila guarda ``_head`` e ``_size``, não ``_head`` e ``_tail``.**
   Com dois índices apenas, ``head == tail`` é ambíguo: significa fila vazia e
   fila cheia ao mesmo tempo. As saídas clássicas são sacrificar uma posição do
   array (capacidade útil vira ``capacity - 1``) ou guardar um bit extra.
   Guardar ``_size`` resolve os dois casos sem desperdiçar posição e ainda deixa
   ``__len__`` em O(1). O índice da cauda é derivado:
   ``(_head + _size) % capacity``.
3. **Slots liberados na remoção.** ``pop``/``dequeue`` escrevem ``None`` na
   posição liberada. Sem isso, o array continuaria segurando uma referência ao
   objeto removido e a estrutura seguraria memória que o chamador acha que
   soltou. Custo: uma escrita a mais por remoção.
4. **A árvore de expressão valida na construção, não na avaliação.** Operador
   sem operandos suficientes e sobra de operandos no fim são detectados ao
   montar a árvore, com a posição do token na mensagem. Detectar só na avaliação
   deixaria passar uma árvore malformada para o resto do sistema.
5. **Duas avaliações, mesma árvore.** :func:`evaluate_recursive` e
   :func:`evaluate_iterative` percorrem a MESMA árvore em pós-ordem e contam o
   mesmo número de visitas a nó. A diferença medida é só de memória: a recursiva
   usa a pilha de chamadas do Python (limitada por
   ``sys.getrecursionlimit()``), a iterativa usa uma :class:`Stack` no heap.
   :func:`bench_recursion_depth` mostra o tamanho em que a recursiva quebra e a
   iterativa não.

Custo em Big O
--------------
=========================  ==========  =================================
Operação                   Custo       Observação
=========================  ==========  =================================
``Stack.push``/``pop``     O(1)        índice direto, sem travessia
``Stack.peek``             O(1)
``Queue.enqueue``          O(1)        ``(head + size) % capacity``
``Queue.dequeue``          O(1)        avança ``head`` com wraparound
``__len__``/``is_empty``   O(1)
``build_expression_tree``  O(t)        t = número de tokens
``evaluate_*``             O(n)        n = nós; cada nó é visitado uma vez
=========================  ==========  =================================

Memória: as duas estruturas ocupam Θ(capacity) fixos, independentemente de
quantos elementos estão guardados — é o preço da capacidade fixa. A avaliação
recursiva ocupa Θ(h) de pilha de chamadas (h = altura da árvore) e a iterativa
Θ(h) de pilha de nós mais Θ(p) de pilha de operandos, onde p depende do formato
da árvore: 2 para uma árvore que pende à esquerda, h para uma que pende à
direita. As três quantidades são medidas em :func:`bench_recursion_depth`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Generic, Iterable, Iterator, TypeVar

from src.counters import Counters, make_row, stopwatch

T = TypeVar("T")

#: Tipo dos valores que a árvore de expressão manipula.
Number = int | float


# ----------------------------------------------------------------------
# Exceções próprias (convenção 9: nunca `raise Exception(...)`)
# ----------------------------------------------------------------------
class StackError(Exception):
    """Erro base de :class:`Stack`."""


class StackOverflowError(StackError):
    """``push`` em pilha cheia."""


class StackUnderflowError(StackError):
    """``pop`` ou ``peek`` em pilha vazia."""


class StackConfigError(StackError):
    """Capacidade inválida na construção da pilha."""


class StackInvariantError(StackError):
    """Invariante estrutural da pilha violada."""


class QueueError(Exception):
    """Erro base de :class:`Queue`."""


class QueueOverflowError(QueueError):
    """``enqueue`` em fila cheia."""


class QueueUnderflowError(QueueError):
    """``dequeue`` ou ``peek`` em fila vazia."""


class QueueConfigError(QueueError):
    """Capacidade inválida na construção da fila."""


class QueueInvariantError(QueueError):
    """Invariante estrutural da fila violada."""


class ExpressionError(Exception):
    """Erro base da árvore de expressão."""


class EmptyExpressionError(ExpressionError):
    """A expressão não tem nenhum token."""


class UnknownTokenError(ExpressionError):
    """Token que não é operando numérico nem operador conhecido."""


class InsufficientOperandsError(ExpressionError):
    """Operador encontrado sem os dois operandos de que precisa."""


class LeftoverOperandsError(ExpressionError):
    """A expressão terminou com mais de um valor na pilha."""


class DivisionByZeroError(ExpressionError):
    """Divisão por zero durante a avaliação."""


class MalformedTreeError(ExpressionError):
    """Nó da árvore com aridade incoerente com o token que carrega."""


# ----------------------------------------------------------------------
# Pilha
# ----------------------------------------------------------------------
class Stack(Generic[T]):
    """Pilha LIFO sobre array de capacidade fixa.

    Invariantes (ver :meth:`check_invariants`):

    * ``0 <= _size <= capacity``;
    * as posições ``[0, _size)`` guardam os elementos, do fundo para o topo;
    * as posições ``[_size, capacity)`` valem ``None`` (decisão de projeto 3);
    * ``_peak >= _size`` sempre.
    """

    __slots__ = ("_items", "_size", "_peak")

    def __init__(self, capacity: int, values: Iterable[T] | None = None) -> None:
        """Aloca o array de uma vez. Custo: O(capacity)."""
        if not isinstance(capacity, int) or isinstance(capacity, bool):
            raise StackConfigError(
                f"capacidade deve ser int, recebido {type(capacity).__name__}"
            )
        if capacity < 1:
            raise StackConfigError(f"capacidade deve ser >= 1, recebido {capacity}")
        self._items: list[T | None] = [None] * capacity
        self._size = 0
        self._peak = 0
        if values is not None:
            for value in values:
                self.push(value)

    # -- leitura -------------------------------------------------------
    def __len__(self) -> int:
        """Elementos guardados. Custo: O(1)."""
        return self._size

    @property
    def capacity(self) -> int:
        """Capacidade fixa. Custo: O(1)."""
        return len(self._items)

    @property
    def peak_size(self) -> int:
        """Maior tamanho já atingido — a memória de fato usada. Custo: O(1).

        É esta a medida que sustenta a comparação de custo de memória entre a
        avaliação recursiva e a iterativa: a capacidade alocada é um limite
        superior escolhido pelo chamador, o pico é o que o algoritmo precisou.
        """
        return self._peak

    def is_empty(self) -> bool:
        """Custo: O(1)."""
        return self._size == 0

    def is_full(self) -> bool:
        """Custo: O(1)."""
        return self._size == len(self._items)

    def __iter__(self) -> Iterator[T]:
        """Do fundo para o topo. Custo: O(n)."""
        for index in range(self._size):
            yield self._items[index]  # type: ignore[misc]

    def to_list(self) -> list[T]:
        """Cópia em ``list``, do fundo para o topo. Custo: O(n)."""
        return list(self)

    def __str__(self) -> str:
        """``"fundo [a, b, c] topo"``. Custo: O(n)."""
        inner = ", ".join(repr(value) for value in self)
        return f"fundo [{inner}] topo"

    def __repr__(self) -> str:
        return f"Stack(size={self._size}, capacity={self.capacity})"

    # -- operações -----------------------------------------------------
    def push(self, value: T, counters: Counters | None = None) -> None:
        """Empilha. Custo: O(1).

        Levanta :class:`StackOverflowError` se a pilha está cheia — a
        capacidade é fixa por decisão de projeto 1, não por limitação.
        """
        if self._size == len(self._items):
            raise StackOverflowError(
                f"push em pilha cheia: capacidade {self.capacity} esgotada "
                f"(topo atual: {self._items[self._size - 1]!r})"
            )
        self._items[self._size] = value
        self._size += 1
        if self._size > self._peak:
            self._peak = self._size
        if counters is not None:
            counters.count_copy()

    def pop(self, counters: Counters | None = None) -> T:
        """Desempilha e devolve o topo. Custo: O(1).

        Levanta :class:`StackUnderflowError` em pilha vazia.
        """
        if self._size == 0:
            raise StackUnderflowError(
                f"pop em pilha vazia (capacidade {self.capacity})"
            )
        self._size -= 1
        value = self._items[self._size]
        self._items[self._size] = None  # decisão de projeto 3
        if counters is not None:
            counters.count_copy()
        return value  # type: ignore[return-value]

    def peek(self) -> T:
        """Topo, sem desempilhar. Custo: O(1)."""
        if self._size == 0:
            raise StackUnderflowError(
                f"peek em pilha vazia (capacidade {self.capacity})"
            )
        return self._items[self._size - 1]  # type: ignore[return-value]

    def push_counted(self, value: T) -> tuple[None, Counters]:
        """:meth:`push` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        self.push(value, counters)
        return None, counters

    def pop_counted(self) -> tuple[T, Counters]:
        """:meth:`pop` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        return self.pop(counters), counters

    # -- verificação ---------------------------------------------------
    def check_invariants(self) -> None:
        """Valida as invariantes; levanta :class:`StackInvariantError`. Custo: O(capacity)."""
        capacity = len(self._items)
        if not 0 <= self._size <= capacity:
            raise StackInvariantError(
                f"_size={self._size} fora da faixa 0..{capacity}"
            )
        if self._peak < self._size:
            raise StackInvariantError(
                f"_peak={self._peak} menor que _size={self._size}"
            )
        for index in range(self._size, capacity):
            if self._items[index] is not None:
                raise StackInvariantError(
                    f"posição {index} acima do topo deveria estar liberada, "
                    f"mas guarda {self._items[index]!r}"
                )


# ----------------------------------------------------------------------
# Fila circular
# ----------------------------------------------------------------------
class Queue(Generic[T]):
    """Fila FIFO sobre array de capacidade fixa, com wraparound.

    O array é tratado como circular: quando a cauda chega ao fim, ela volta para
    a posição 0 e reaproveita os slots já liberados pelos ``dequeue``. Sem isso,
    uma fila que só recebe e entrega alternadamente esgotaria a capacidade
    mesmo nunca guardando mais de um elemento.

    Invariantes (ver :meth:`check_invariants`):

    * ``0 <= _size <= capacity`` e ``0 <= _head < capacity``;
    * os ``_size`` elementos ocupam ``(_head + i) % capacity``, ``i < _size``;
    * toda posição fora desse intervalo vale ``None``.
    """

    __slots__ = ("_items", "_head", "_size", "_peak", "_wraparounds")

    def __init__(self, capacity: int, values: Iterable[T] | None = None) -> None:
        """Aloca o array de uma vez. Custo: O(capacity)."""
        if not isinstance(capacity, int) or isinstance(capacity, bool):
            raise QueueConfigError(
                f"capacidade deve ser int, recebido {type(capacity).__name__}"
            )
        if capacity < 1:
            raise QueueConfigError(f"capacidade deve ser >= 1, recebido {capacity}")
        self._items: list[T | None] = [None] * capacity
        self._head = 0
        self._size = 0
        self._peak = 0
        self._wraparounds = 0
        if values is not None:
            for value in values:
                self.enqueue(value)

    # -- leitura -------------------------------------------------------
    def __len__(self) -> int:
        """Elementos na fila. Custo: O(1) — é para isso que ``_size`` existe."""
        return self._size

    @property
    def capacity(self) -> int:
        """Capacidade fixa. Custo: O(1)."""
        return len(self._items)

    @property
    def peak_size(self) -> int:
        """Maior tamanho já atingido. Custo: O(1)."""
        return self._peak

    @property
    def wraparounds(self) -> int:
        """Quantas vezes a cauda deu a volta no array. Custo: O(1).

        Existe como evidência: um teste que enfileira e desenfileira muito mais
        do que a capacidade só prova o wraparound se o contador for maior que
        zero. Sem ele, a fila poderia estar crescendo escondido e o teste passaria.
        """
        return self._wraparounds

    def is_empty(self) -> bool:
        """Custo: O(1)."""
        return self._size == 0

    def is_full(self) -> bool:
        """Custo: O(1)."""
        return self._size == len(self._items)

    def __iter__(self) -> Iterator[T]:
        """Da frente para o fundo. Custo: O(n)."""
        capacity = len(self._items)
        for offset in range(self._size):
            yield self._items[(self._head + offset) % capacity]  # type: ignore[misc]

    def to_list(self) -> list[T]:
        """Cópia em ``list``, da frente para o fundo. Custo: O(n)."""
        return list(self)

    def __str__(self) -> str:
        """``"frente [a, b, c] fundo"``. Custo: O(n)."""
        inner = ", ".join(repr(value) for value in self)
        return f"frente [{inner}] fundo"

    def __repr__(self) -> str:
        return (
            f"Queue(size={self._size}, capacity={self.capacity}, "
            f"head={self._head})"
        )

    # -- operações -----------------------------------------------------
    def enqueue(self, value: T, counters: Counters | None = None) -> None:
        """Enfileira no fim. Custo: O(1).

        Levanta :class:`QueueOverflowError` se a fila está cheia.
        """
        capacity = len(self._items)
        if self._size == capacity:
            raise QueueOverflowError(
                f"enqueue em fila cheia: capacidade {capacity} esgotada "
                f"(frente: {self._items[self._head]!r})"
            )
        tail = self._head + self._size
        if tail >= capacity:
            self._wraparounds += 1
            tail -= capacity
        self._items[tail] = value
        self._size += 1
        if self._size > self._peak:
            self._peak = self._size
        if counters is not None:
            counters.count_copy()

    def dequeue(self, counters: Counters | None = None) -> T:
        """Desenfileira da frente. Custo: O(1).

        Levanta :class:`QueueUnderflowError` em fila vazia.
        """
        if self._size == 0:
            raise QueueUnderflowError(
                f"dequeue em fila vazia (capacidade {self.capacity})"
            )
        value = self._items[self._head]
        self._items[self._head] = None  # decisão de projeto 3
        self._head = (self._head + 1) % len(self._items)
        self._size -= 1
        if counters is not None:
            counters.count_copy()
        return value  # type: ignore[return-value]

    def peek(self) -> T:
        """Elemento da frente, sem remover. Custo: O(1)."""
        if self._size == 0:
            raise QueueUnderflowError(
                f"peek em fila vazia (capacidade {self.capacity})"
            )
        return self._items[self._head]  # type: ignore[return-value]

    def enqueue_counted(self, value: T) -> tuple[None, Counters]:
        """:meth:`enqueue` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        self.enqueue(value, counters)
        return None, counters

    def dequeue_counted(self) -> tuple[T, Counters]:
        """:meth:`dequeue` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        return self.dequeue(counters), counters

    # -- verificação ---------------------------------------------------
    def check_invariants(self) -> None:
        """Valida as invariantes; levanta :class:`QueueInvariantError`. Custo: O(capacity)."""
        capacity = len(self._items)
        if not 0 <= self._size <= capacity:
            raise QueueInvariantError(
                f"_size={self._size} fora da faixa 0..{capacity}"
            )
        if not 0 <= self._head < capacity:
            raise QueueInvariantError(
                f"_head={self._head} fora da faixa 0..{capacity - 1}"
            )
        if self._peak < self._size:
            raise QueueInvariantError(
                f"_peak={self._peak} menor que _size={self._size}"
            )
        ocupadas = {(self._head + offset) % capacity for offset in range(self._size)}
        for index in range(capacity):
            vazia = self._items[index] is None
            if index in ocupadas and vazia:
                raise QueueInvariantError(
                    f"posição {index} deveria estar ocupada e está vazia"
                )
            if index not in ocupadas and not vazia:
                raise QueueInvariantError(
                    f"posição {index} deveria estar livre, mas guarda "
                    f"{self._items[index]!r}"
                )


# ----------------------------------------------------------------------
# Árvore binária de expressão
# ----------------------------------------------------------------------
#: Operadores reconhecidos e sua aridade (todos binários).
OPERATORS: frozenset[str] = frozenset({"+", "-", "*", "/", "^"})


@dataclass(slots=True)
class ExpressionNode:
    """Nó da árvore de expressão: um operando ou um operador binário.

    Operando tem ``left is right is None``; operador tem os dois preenchidos.
    Não existe estado intermediário válido — :func:`check_expression_tree`
    verifica isso.
    """

    token: str
    left: "ExpressionNode | None" = None
    right: "ExpressionNode | None" = None

    def is_operator(self) -> bool:
        """Custo: O(1)."""
        return self.token in OPERATORS

    def is_operand(self) -> bool:
        """Custo: O(1)."""
        return not self.is_operator()

    def __repr__(self) -> str:
        return f"ExpressionNode({self.token!r})"


def parse_operand(token: str) -> Number:
    """Converte o token em ``int`` ou ``float``. Custo: O(len(token)).

    Levanta :class:`UnknownTokenError` se o token não é número nem operador.
    """
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError as error:
        raise UnknownTokenError(
            f"token {token!r} não é operando numérico nem operador "
            f"({', '.join(sorted(OPERATORS))})"
        ) from error


def build_expression_tree(
    postfix: str,
    *,
    capacity: int | None = None,
    counters: Counters | None = None,
) -> ExpressionNode:
    """Monta a árvore binária de expressão a partir de notação pós-fixa.

    O algoritmo é o clássico com pilha: operando vira folha e é empilhado;
    operador desempilha **dois** nós (o primeiro desempilhado é o operando da
    direita, porque foi o último a entrar) e empilha a subárvore.

    ``capacity`` é a capacidade da :class:`Stack` usada. Por padrão é o número
    de tokens, que nunca estoura; passar um valor menor é o jeito de demonstrar
    o :class:`StackOverflowError` com uma expressão real.

    Validações (decisão de projeto 4), todas com a posição do token na mensagem:

    * expressão vazia → :class:`EmptyExpressionError`;
    * token desconhecido → :class:`UnknownTokenError`;
    * operador com menos de dois operandos → :class:`InsufficientOperandsError`;
    * mais de um valor na pilha ao final → :class:`LeftoverOperandsError`.

    Custo: O(t) em tempo, com t tokens; O(t) de memória na pilha no pior caso
    (expressão que pende à direita).
    """
    tokens = postfix.split()
    if not tokens:
        raise EmptyExpressionError(
            "expressão pós-fixa vazia: nada a montar"
        )
    counters = counters if counters is not None else Counters()
    stack: Stack[ExpressionNode] = Stack(
        capacity if capacity is not None else len(tokens)
    )

    for position, token in enumerate(tokens, start=1):
        if token in OPERATORS:
            if len(stack) < 2:
                raise InsufficientOperandsError(
                    f"operador {token!r} na posição {position} precisa de 2 "
                    f"operandos, mas a pilha tem {len(stack)}"
                )
            right = stack.pop(counters)
            left = stack.pop(counters)
            stack.push(ExpressionNode(token, left, right), counters)
        else:
            parse_operand(token)  # valida o token antes de virar folha
            stack.push(ExpressionNode(token), counters)

    if len(stack) != 1:
        raise LeftoverOperandsError(
            f"expressão terminou com {len(stack)} valores na pilha, esperado 1. "
            f"Sobraram operandos sem operador: "
            f"{[node.token for node in stack.to_list()]}"
        )
    return stack.pop(counters)


def check_expression_tree(node: ExpressionNode) -> None:
    """Valida a aridade de todos os nós. Custo: O(n).

    Levanta :class:`MalformedTreeError`. Existe para que a árvore possa ser
    verificada depois de qualquer manipulação, não só logo após a construção.

    Iterativa de propósito: uma validação que só funciona em árvore rasa não
    serve para validar justamente a árvore funda do experimento de recursão.
    """
    pending: list[ExpressionNode] = [node]
    while pending:
        current = pending.pop()
        if current.is_operator():
            if current.left is None or current.right is None:
                raise MalformedTreeError(
                    f"operador {current.token!r} deveria ter dois filhos, tem "
                    f"left={current.left!r} e right={current.right!r}"
                )
            pending.append(current.left)
            pending.append(current.right)
        elif current.left is not None or current.right is not None:
            raise MalformedTreeError(
                f"operando {current.token!r} não pode ter filhos"
            )
        else:
            parse_operand(current.token)


def tree_size(node: ExpressionNode) -> int:
    """Número de nós. Custo: O(n).

    Iterativa pela mesma razão de :func:`tree_height`: ela é chamada por
    :func:`evaluate_iterative` para dimensionar as pilhas, e a versão iterativa
    não pode depender de uma contagem recursiva que estoura antes.
    """
    total = 0
    pending: list[ExpressionNode] = [node]
    while pending:
        current = pending.pop()
        total += 1
        if current.left is not None:
            pending.append(current.left)
        if current.right is not None:
            pending.append(current.right)
    return total


def tree_height(node: ExpressionNode) -> int:
    """Altura em nós (folha = 1). Custo: O(n).

    É exatamente a profundidade que :func:`evaluate_recursive` vai atingir na
    pilha de chamadas do Python — por isso a função é iterativa aqui, senão
    medir a altura de uma árvore funda quebraria antes de medir.
    """
    maximum = 0
    stack: list[tuple[ExpressionNode, int]] = [(node, 1)]
    while stack:
        current, depth = stack.pop()
        if depth > maximum:
            maximum = depth
        if current.left is not None:
            stack.append((current.left, depth + 1))
        if current.right is not None:
            stack.append((current.right, depth + 1))
    return maximum


def _apply(operator: str, left: Number, right: Number) -> Number:
    """Aplica o operador binário. Custo: O(1)."""
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if operator == "/":
        if right == 0:
            raise DivisionByZeroError(
                f"divisão por zero ao avaliar {left!r} / {right!r}"
            )
        return left / right
    if operator == "^":
        return left**right
    raise UnknownTokenError(f"operador desconhecido: {operator!r}")


@dataclass(slots=True)
class Evaluation:
    """Resultado de uma avaliação, com as medidas que o exercício pede.

    Estende o protocolo ``(resultado, Counters)`` da seção 5 do CLAUDE.md em vez
    de substituí-lo: ``.value`` e ``.counters`` são os dois campos do protocolo,
    e os demais são as medidas de memória que a comparação recursiva×iterativa
    exige.
    """

    value: Number
    counters: Counters
    strategy: str
    max_depth: int
    peak_operand_stack: int

    def as_tuple(self) -> tuple[Number, Counters]:
        """O par ``(resultado, Counters)`` do protocolo. Custo: O(1)."""
        return self.value, self.counters


def evaluate_recursive(
    node: ExpressionNode, counters: Counters | None = None
) -> Evaluation:
    """Avalia por travessia pós-ordem **recursiva**.

    Cada nó é visitado uma vez e conta 1 em ``calls``; cada valor produzido
    conta 1 em ``copies``. ``max_depth`` é a profundidade máxima da pilha de
    chamadas, que é igual à altura da árvore.

    Custo: O(n) em tempo, **Θ(h) na pilha de chamadas do Python** — e essa pilha
    tem teto (``sys.getrecursionlimit()``). Uma expressão encadeada com mais de
    ~1000 operadores levanta ``RecursionError`` aqui e não em
    :func:`evaluate_iterative`; é o que :func:`bench_recursion_depth` mede.
    """
    counters = counters if counters is not None else Counters()
    deepest = [0]

    def visit(current: ExpressionNode, depth: int) -> Number:
        counters.count_call()
        if depth > deepest[0]:
            deepest[0] = depth
        if current.is_operand():
            counters.count_copy()
            return parse_operand(current.token)
        if current.left is None or current.right is None:
            raise MalformedTreeError(
                f"operador {current.token!r} sem os dois filhos"
            )
        left = visit(current.left, depth + 1)
        right = visit(current.right, depth + 1)
        counters.count_copy()
        return _apply(current.token, left, right)

    value = visit(node, 1)
    return Evaluation(
        value=value,
        counters=counters,
        strategy="pós-ordem recursiva",
        max_depth=deepest[0],
        peak_operand_stack=0,  # não há pilha de operandos: o valor volta no return
    )


def evaluate_iterative(
    node: ExpressionNode, counters: Counters | None = None
) -> Evaluation:
    """Avalia em pós-ordem **iterativa**, com :class:`Stack` explícita.

    Usa duas pilhas: a de nós, que guia a travessia, e a de operandos, que
    acumula os valores já calculados. O ponteiro ``last`` é o que distingue
    "ainda vou descer à direita" de "já voltei da direita" — sem ele a travessia
    pós-ordem entraria em laço.

    Conta exatamente os mesmos ``calls`` e ``copies`` que
    :func:`evaluate_recursive`: o trabalho é o mesmo, O(n). O que muda é onde a
    memória mora — heap em vez da pilha de chamadas —, e por isso esta versão
    não tem teto de profundidade.

    ``peak_operand_stack`` depende do formato: 2 para árvore que pende à
    esquerda, h para árvore que pende à direita.

    Custo: O(n) em tempo, Θ(h) + Θ(p) de memória no heap.
    """
    counters = counters if counters is not None else Counters()
    total = tree_size(node)
    nodes: Stack[ExpressionNode] = Stack(total)
    operands: Stack[Number] = Stack(total)

    current: ExpressionNode | None = node
    last: ExpressionNode | None = None

    while current is not None or not nodes.is_empty():
        while current is not None:
            nodes.push(current)
            current = current.left
        peeked = nodes.peek()
        if peeked.right is not None and last is not peeked.right:
            current = peeked.right
            continue
        visited = nodes.pop()
        counters.count_call()
        if visited.is_operand():
            counters.count_copy()
            operands.push(parse_operand(visited.token))
        else:
            right = operands.pop()
            left = operands.pop()
            counters.count_copy()
            operands.push(_apply(visited.token, left, right))
        last = visited

    if len(operands) != 1:
        raise MalformedTreeError(
            f"avaliação terminou com {len(operands)} operandos, esperado 1"
        )
    return Evaluation(
        value=operands.pop(),
        counters=counters,
        strategy="pós-ordem iterativa com Stack",
        max_depth=nodes.peak_size,
        peak_operand_stack=operands.peak_size,
    )


def evaluate_recursive_counted(node: ExpressionNode) -> tuple[Number, Counters]:
    """:func:`evaluate_recursive` no protocolo ``(resultado, Counters)``."""
    return evaluate_recursive(node).as_tuple()


def evaluate_iterative_counted(node: ExpressionNode) -> tuple[Number, Counters]:
    """:func:`evaluate_iterative` no protocolo ``(resultado, Counters)``."""
    return evaluate_iterative(node).as_tuple()


def to_infix(node: ExpressionNode) -> str:
    """Expressão em notação infixa, totalmente parentizada. Custo: O(n).

    Serve de evidência de que a árvore montada é a árvore certa: comparar o
    texto com o esperado é mais legível do que inspecionar ponteiros.

    Recursiva, e aqui isso é aceitável: a função só existe para exibir
    expressões pequenas no notebook. Θ(h) de pilha de chamadas, então não a use
    nas árvores fundas de :func:`bench_recursion_depth` — que é exatamente o
    ponto que o experimento está medindo.
    """
    if node.is_operand():
        return node.token
    return (
        f"({to_infix(node.left)} {node.token} "  # type: ignore[arg-type]
        f"{to_infix(node.right)})"  # type: ignore[arg-type]
    )


# ----------------------------------------------------------------------
# Geradores de expressão para os experimentos
# ----------------------------------------------------------------------
def chained_postfix(
    operators: int,
    *,
    lean: str = "left",
    operator: str = "+",
    operand: str = "1",
) -> str:
    """Expressão pós-fixa degenerada, com ``operators`` operadores.

    * ``lean="left"`` → ``"1 1 + 1 + 1 +"``: a árvore pende à esquerda, e a
      pilha de operandos da avaliação iterativa nunca passa de 2;
    * ``lean="right"`` → ``"1 1 1 1 + + +"``: a árvore pende à direita, e a
      pilha de operandos chega a ``operators + 1``.

    Nos dois casos a altura é ``operators + 1``, então a profundidade de
    recursão é a mesma. É essa separação — mesma profundidade de chamada,
    pilhas de operando diferentes — que o experimento usa.

    Custo: O(operators).
    """
    if operators < 1:
        raise ValueError(f"operators deve ser >= 1, recebido {operators}")
    if lean == "left":
        return operand + "".join(f" {operand} {operator}" for _ in range(operators))
    if lean == "right":
        operandos = " ".join([operand] * (operators + 1))
        return operandos + " " + " ".join([operator] * operators)
    raise ValueError(f"lean deve ser 'left' ou 'right', recebido {lean!r}")


# ----------------------------------------------------------------------
# Experimentos do exercício 5
# ----------------------------------------------------------------------
def bench_stack_queue_operations(
    sizes: Iterable[int] = (1_000, 2_000, 4_000, 8_000, 16_000),
    *,
    repeats: int = 3,
) -> list[dict[str, Any]]:
    """Mede push/pop e enqueue/dequeue: O(1) por operação, 0 saltos.

    Cada linha é ``n`` chamadas da mesma operação. Previsão: ``hops == 0``
    sempre e ``tempo_us_por_op`` praticamente constante com n crescendo 16×.

    Custo: O(repeats · Σ n).
    """
    if repeats < 1:
        raise ValueError(f"repeats deve ser >= 1, recebido {repeats}")
    rows: list[dict[str, Any]] = []
    for n in sizes:
        if n <= 0:
            raise ValueError(f"tamanho deve ser positivo, recebido {n}")
        for estrutura, entrada, saida in (
            ("Stack", "push", "pop"),
            ("Queue", "enqueue", "dequeue"),
        ):
            for operation in (entrada, saida):
                counters = Counters()
                best = float("inf")
                for attempt in range(repeats):
                    run_counters = counters if attempt == 0 else Counters()
                    target: Any = Stack(n) if estrutura == "Stack" else Queue(n)
                    if operation == saida:
                        preencher = getattr(target, entrada)
                        for value in range(n):
                            preencher(value)
                    method = getattr(target, operation)
                    with stopwatch() as elapsed:
                        if operation == entrada:
                            for value in range(n):
                                method(value, run_counters)
                        else:
                            for _ in range(n):
                                method(run_counters)
                    best = min(best, elapsed[0])
                    target.check_invariants()
                rows.append(
                    make_row(
                        exercise="ex5",
                        algorithm=f"{estrutura}.{operation}",
                        n=n,
                        pattern=f"{n} operações seguidas",
                        counters=counters,
                        metric="hops",
                        curves=("n", "n2"),
                        elapsed_s=best,
                        extra={
                            "estrutura": estrutura,
                            "operacao": operation,
                            "hops_teorico": 0,
                            "repeticoes": repeats,
                        },
                    )
                )
    return rows


def bench_queue_wraparound(
    capacity: int = 8,
    operations: int = 10_000,
) -> dict[str, Any]:
    """Prova que a fila circular reaproveita as posições do array.

    Alterna ``enqueue`` e ``dequeue`` mantendo a fila quase cheia, por muito
    mais operações do que a capacidade. Se o buffer não fosse circular, o
    array de ``capacity`` posições se esgotaria depois de ``capacity``
    inserções; com wraparound, ``operations`` inserções cabem no mesmo array e
    o contador :attr:`Queue.wraparounds` fica maior que zero.

    Custo: O(operations).
    """
    if capacity < 2:
        raise ValueError(f"capacity deve ser >= 2, recebido {capacity}")
    if operations < capacity:
        raise ValueError(
            f"operations ({operations}) deve ser >= capacity ({capacity})"
        )

    queue: Queue[int] = Queue(capacity)
    counters = Counters()
    ordem_saida: list[int] = []
    for value in range(capacity - 1):
        queue.enqueue(value, counters)

    with stopwatch() as elapsed:
        for value in range(capacity - 1, operations):
            queue.enqueue(value, counters)
            ordem_saida.append(queue.dequeue(counters))
    queue.check_invariants()

    esperado = list(range(operations - capacity + 1))
    return {
        "exercicio": "ex5",
        "algoritmo": "Queue — buffer circular",
        "n": operations,
        "padrao": f"enqueue/dequeue alternados com capacidade {capacity}",
        **counters.as_dict(),
        "capacidade": capacity,
        "voltas_no_array": queue.wraparounds,
        "posicoes_alocadas": capacity,
        "ocupacao_maxima": queue.peak_size,
        "fifo_preservado": ordem_saida == esperado,
        "restantes_na_fila": len(queue),
        "tempo_s": elapsed[0],
        "tempo_us_por_op": elapsed[0] * 1e6 / (2 * (operations - capacity + 1)),
    }


def bench_recursion_depth(
    sizes: Iterable[int] = (50, 200, 800, 3_200, 12_800),
    *,
    leans: Iterable[str] = ("left", "right"),
) -> list[dict[str, Any]]:
    """Recursiva × iterativa: mesmo trabalho, memória em lugares diferentes.

    Para cada tamanho e cada formato de árvore, avalia a MESMA árvore das duas
    formas e registra:

    * ``calls`` e ``copies`` — idênticos nas duas, porque o trabalho é O(n) nos
      dois casos;
    * ``profundidade`` — altura da árvore, que é a profundidade da pilha de
      chamadas na recursiva e o pico da pilha de nós na iterativa;
    * ``pico_operandos`` — 2 na árvore que pende à esquerda, ``h`` na que pende
      à direita;
    * ``recursiva_quebrou`` — se a recursiva levantou ``RecursionError``.

    O teto da recursão é ``sys.getrecursionlimit()`` (registrado em cada linha),
    e é um limite do intérprete, não do algoritmo: a iterativa faz o mesmo
    trabalho com a mesma ordem de memória, só que no heap.

    Custo: O(Σ n) por estratégia.
    """
    limite = sys.getrecursionlimit()
    rows: list[dict[str, Any]] = []
    for lean in leans:
        for operators in sizes:
            if operators <= 0:
                raise ValueError(f"tamanho deve ser positivo, recebido {operators}")
            lado = {"left": "esquerda", "right": "direita"}[lean]
            expressao = chained_postfix(operators, lean=lean)
            root = build_expression_tree(expressao)
            altura = tree_height(root)
            nos = tree_size(root)

            iterativa = evaluate_iterative(root)
            try:
                recursiva: Evaluation | None = evaluate_recursive(root)
                quebrou = False
            except RecursionError:
                recursiva = None
                quebrou = True

            rows.append(
                {
                    "exercicio": "ex5",
                    "algoritmo": "avaliação recursiva × iterativa",
                    "n": nos,
                    "padrao": f"expressão encadeada à {lado}",
                    "operadores": operators,
                    "altura": altura,
                    "limite_de_recursao": limite,
                    "recursiva_quebrou": quebrou,
                    "valor_iterativa": iterativa.value,
                    "valor_recursiva": None if recursiva is None else recursiva.value,
                    "calls_iterativa": iterativa.counters.calls,
                    "calls_recursiva": (
                        None if recursiva is None else recursiva.counters.calls
                    ),
                    "copies_iterativa": iterativa.counters.copies,
                    "profundidade_iterativa": iterativa.max_depth,
                    "profundidade_recursiva": (
                        None if recursiva is None else recursiva.max_depth
                    ),
                    "pico_operandos": iterativa.peak_operand_stack,
                }
            )
    return rows

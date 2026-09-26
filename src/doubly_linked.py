"""Exercício 11 — Lista duplamente encadeada e Deque.

O exercício 10 terminou com um limite explícito: em lista simplesmente
encadeada, ``delete`` é O(n) **mesmo com ponteiro tail**, porque desligar um nó
exige o antecessor e não há ponteiro ``prev`` para chegar até ele. O exercício
11 é exatamente a resposta a esse limite — e é por isso que este módulo mede a
si mesmo contra :class:`~src.linked_list.SinglyLinkedList` em
:func:`bench_delete_last_vs_singly`, em vez de só afirmar que ficou O(1).

Estruturas
----------
:class:`DoublyLinkedList`
    Nós com ``prev`` e ``next``, ponteiros ``_head`` e ``_tail``. As quatro
    operações de ponta (``insert_first``, ``insert_last``, ``delete_first``,
    ``delete_last``) são O(1) com **zero saltos de ponteiro**.
:class:`Deque`
    Fila de duas pontas construída **sobre** a lista duplamente encadeada.

Decisões de projeto
-------------------
1. **Deque compõe a lista, não herda dela.** Por herança, o ``Deque`` exporia
   também ``insert_at``/``delete_at`` e a promessa "todas as operações são O(1)"
   deixaria de ser verdadeira para o tipo — um chamador poderia usar uma
   operação O(n) sem perceber. Por composição, a API do ``Deque`` é exatamente
   as seis operações que o enunciado pede, e a garantia O(1) vale para o tipo
   inteiro, não só para parte dele.
2. **Exceções separadas por estrutura.** ``Deque`` verifica ``is_empty()`` antes
   de operar e levanta :class:`EmptyDequeError`; nunca deixa vazar a
   :class:`EmptyDoublyLinkedListError` da estrutura interna. Quem usa o
   ``Deque`` não precisa saber sobre o que ele é construído.
3. **Critério de contagem idêntico ao do exercício 10.** Inserção conta 1 cópia,
   remoção conta 0 cópias, travessia conta ``hops``. Isso não é detalhe de
   estilo: a tabela que compara ``delete_last`` aqui com ``delete_at(n-1)`` na
   lista simples só é válida se os dois lados contarem a mesma coisa. Qualquer
   divergência de critério apareceria como diferença de custo que não existe.
4. **A invariante é verificada, não assumida.** :meth:`DoublyLinkedList.check_invariants`
   confere os dois sentidos do encadeamento (``no.next.prev is no`` e
   ``no.prev.next is no``), e não apenas o tamanho. Um encadeamento que só está
   certo em um sentido passa em qualquer teste de ``to_list()`` e quebra na
   primeira remoção pelo outro lado — é o bug característico desta estrutura.

Tratamento de estrutura vazia
-----------------------------
Todas as operações que exigem ao menos um elemento (``peek_*``, ``delete_*``,
``remove_*``) levantam exceção própria com mensagem nomeando a operação, em vez
de devolver ``None``. Devolver ``None`` tornaria indistinguíveis "o deque está
vazio" e "o elemento armazenado é ``None``". O caminho sem exceção é
``is_empty()``, que é O(1) — o chamador sempre pode perguntar antes.

Custo por operação (ver :data:`COST_TABLE` e :data:`DEQUE_COST_TABLE`)
----------------------------------------------------------------------
====================  ======================  ==========================
Operação              DoublyLinkedList        SinglyLinkedList (ex 10)
====================  ======================  ==========================
``insert_first``      O(1)                    O(1)
``insert_last``       O(1)                    O(1) (com tail)
``delete_first``      O(1)                    O(1)
``delete_last``       O(1)                    O(n) — precisa do antecessor
``peek_first``        O(1)                    O(1)
``peek_last``         O(1)                    O(1) (com tail)
``is_empty``/``len``  O(1)                    O(1)
travessia             O(n) nos dois sentidos  O(n) só para frente
====================  ======================  ==========================

O preço do ``prev`` é um ponteiro a mais por nó (memória Θ(n)) e uma religação a
mais por operação (constante). O ganho é a coluna ``delete_last``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Generic, Iterable, Iterator, TypeVar

from src.counters import RANDOM_SEED, Counters, make_row, stopwatch
from src.linked_list import SinglyLinkedList

T = TypeVar("T")


# ----------------------------------------------------------------------
# Exceções próprias (convenção 9: nunca `raise Exception(...)`)
# ----------------------------------------------------------------------
class DoublyLinkedListError(Exception):
    """Erro base de :class:`DoublyLinkedList`."""


class EmptyDoublyLinkedListError(DoublyLinkedListError):
    """Operação que exige ao menos um elemento foi chamada em lista vazia."""


class DoublyLinkedListInvariantError(DoublyLinkedListError):
    """Invariante estrutural violada (bug de implementação, não de uso)."""


class DequeError(Exception):
    """Erro base de :class:`Deque`.

    Deliberadamente **não** herda de :class:`DoublyLinkedListError`: o ``Deque``
    é um tipo próprio e quem o usa não deve precisar capturar exceções da
    estrutura interna sobre a qual ele foi construído (decisão de projeto 2).
    """


class EmptyDequeError(DequeError):
    """``remove_*`` ou ``peek_*`` foi chamado em deque vazio."""


# ----------------------------------------------------------------------
# Nó
# ----------------------------------------------------------------------
@dataclass(slots=True)
class DoublyNode(Generic[T]):
    """Nó de lista duplamente encadeada.

    ``slots=True`` pelo mesmo motivo do ex 10: o custo de memória desta
    estrutura é um argumento do trabalho (dois ponteiros por elemento contra um
    da lista simples), e ele precisa ser o custo real, não inflado por um
    ``__dict__`` por nó.
    """

    value: T
    prev: "DoublyNode[T] | None" = None
    next: "DoublyNode[T] | None" = None

    def __repr__(self) -> str:
        """Repr curto: imprimir ``prev``/``next`` recursivamente percorreria a
        lista inteira (e entraria em laço infinito na exibição)."""
        return f"DoublyNode({self.value!r})"


# ----------------------------------------------------------------------
# Lista duplamente encadeada
# ----------------------------------------------------------------------
class DoublyLinkedList(Generic[T]):
    """Lista duplamente encadeada com acesso O(1) às duas pontas.

    Invariantes mantidas por todos os métodos (ver :meth:`check_invariants`):

    * ``_size`` é igual ao número de nós alcançáveis a partir de ``_head``;
    * lista vazia ⇔ ``_head is None`` ⇔ ``_tail is None``;
    * ``_head.prev is None`` e ``_tail.next is None``;
    * para todo nó ``x``: ``x.next.prev is x`` e ``x.prev.next is x``;
    * a travessia de trás para frente produz a travessia da frente invertida.
    """

    __slots__ = ("_head", "_tail", "_size")

    def __init__(self, values: Iterable[T] | None = None) -> None:
        """Cria a lista, opcionalmente preenchendo na ordem de ``values``.

        Custo: O(k) para k valores — ``insert_last`` é O(1).
        """
        self._head: DoublyNode[T] | None = None
        self._tail: DoublyNode[T] | None = None
        self._size = 0
        if values is not None:
            for value in values:
                self.insert_last(value)

    # -- leitura -------------------------------------------------------
    def __len__(self) -> int:
        """Número de elementos. Custo: O(1) (``_size`` é incremental)."""
        return self._size

    def is_empty(self) -> bool:
        """``True`` se a lista não tem elementos. Custo: O(1)."""
        return self._size == 0

    def __iter__(self) -> Iterator[T]:
        """Percorre os valores do início ao fim. Custo: O(n)."""
        current = self._head
        while current is not None:
            yield current.value
            current = current.next

    def __reversed__(self) -> Iterator[T]:
        """Percorre os valores do fim ao início. Custo: O(n).

        Esta é a operação que a lista simplesmente encadeada não oferece: sem
        ``prev``, percorrer de trás para frente custaria O(n) por passo (ou
        exigiria copiar a lista inteira antes).
        """
        current = self._tail
        while current is not None:
            yield current.value
            current = current.prev

    def to_list(self) -> list[T]:
        """Cópia em ``list`` do Python, do início ao fim. Custo: O(n)."""
        return list(self)

    def to_list_reversed(self) -> list[T]:
        """Cópia em ``list`` do Python, do fim ao início. Custo: O(n).

        Usada na verificação de invariante: ``to_list_reversed()`` tem de ser
        igual a ``list(reversed(to_list()))``. Se os dois discordarem, o
        encadeamento está correto em um sentido e quebrado no outro.
        """
        return list(reversed(self))

    def __str__(self) -> str:
        """``"[a <-> b <-> c]"``, ``"[]"`` quando vazia. Custo: O(n)."""
        if self._head is None:
            return "[]"
        return "[" + " <-> ".join(repr(value) for value in self) + "]"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self._size}, {self})"

    def peek_first(self) -> T:
        """Primeiro valor. Custo: O(1).

        Levanta :class:`EmptyDoublyLinkedListError` se a lista está vazia.
        """
        if self._head is None:
            raise EmptyDoublyLinkedListError(
                "peek_first em lista vazia: não há primeiro elemento"
            )
        return self._head.value

    def peek_last(self) -> T:
        """Último valor. Custo: O(1).

        Levanta :class:`EmptyDoublyLinkedListError` se a lista está vazia.
        """
        if self._tail is None:
            raise EmptyDoublyLinkedListError(
                "peek_last em lista vazia: não há último elemento"
            )
        return self._tail.value

    # -- inserção ------------------------------------------------------
    def insert_first(self, value: T, counters: Counters | None = None) -> None:
        """Insere no início. Custo: O(1) — 0 comparações, 0 saltos, 1 cópia."""
        node: DoublyNode[T] = DoublyNode(value, None, self._head)
        if self._head is None:
            self._tail = node
        else:
            self._head.prev = node
        self._head = node
        self._size += 1
        if counters is not None:
            counters.count_copy()

    def insert_last(self, value: T, counters: Counters | None = None) -> None:
        """Insere no fim. Custo: O(1) — 0 comparações, 0 saltos, 1 cópia."""
        node: DoublyNode[T] = DoublyNode(value, self._tail, None)
        if self._tail is None:
            self._head = node
        else:
            self._tail.next = node
        self._tail = node
        self._size += 1
        if counters is not None:
            counters.count_copy()

    def insert_first_counted(self, value: T) -> tuple[None, Counters]:
        """:meth:`insert_first` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        self.insert_first(value, counters)
        return None, counters

    def insert_last_counted(self, value: T) -> tuple[None, Counters]:
        """:meth:`insert_last` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        self.insert_last(value, counters)
        return None, counters

    # -- remoção -------------------------------------------------------
    def delete_first(self, counters: Counters | None = None) -> T:
        """Remove e devolve o primeiro valor. Custo: O(1) — 0 saltos.

        Levanta :class:`EmptyDoublyLinkedListError` se a lista está vazia.
        """
        node = self._head
        if node is None:
            raise EmptyDoublyLinkedListError(
                "delete_first em lista vazia: não há o que remover"
            )
        self._head = node.next
        if self._head is None:
            self._tail = None
        else:
            self._head.prev = None
        node.next = None
        self._size -= 1
        # ``counters`` é recebido e deliberadamente não incrementado: pelo
        # critério do ex 10 a remoção não conta cópia, e aqui não há salto de
        # ponteiro nenhum. O parâmetro existe para o chamador acumular a
        # AUSÊNCIA de saltos ao longo de n remoções — que é a evidência medida.
        return node.value

    def delete_last(self, counters: Counters | None = None) -> T:
        """Remove e devolve o último valor. Custo: O(1) — 0 saltos.

        **Esta é a operação que justifica a estrutura inteira.** Na lista
        simplesmente encadeada do ex 10 a mesma remoção custa O(n), porque o
        antecessor do último nó só é alcançável percorrendo a lista desde o
        início; aqui ele é ``_tail.prev``, acesso direto.

        Levanta :class:`EmptyDoublyLinkedListError` se a lista está vazia.
        """
        node = self._tail
        if node is None:
            raise EmptyDoublyLinkedListError(
                "delete_last em lista vazia: não há o que remover"
            )
        self._tail = node.prev
        if self._tail is None:
            self._head = None
        else:
            self._tail.next = None
        node.prev = None
        self._size -= 1
        # Ver o comentário em :meth:`delete_first`: 0 cópias e 0 saltos.
        return node.value

    def delete_first_counted(self) -> tuple[T, Counters]:
        """:meth:`delete_first` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        value = self.delete_first(counters)
        return value, counters

    def delete_last_counted(self) -> tuple[T, Counters]:
        """:meth:`delete_last` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        value = self.delete_last(counters)
        return value, counters

    # -- verificação ---------------------------------------------------
    def check_invariants(self) -> None:
        """Valida as invariantes; levanta :class:`DoublyLinkedListInvariantError`.

        Verifica os **dois** sentidos do encadeamento. Uma lista em que só o
        sentido ``next`` está correto passa em ``to_list()`` e em qualquer teste
        de tamanho, e só quebra na primeira ``delete_last`` — este método existe
        para que o bug apareça na hora em que é criado, não depois.

        Custo: O(n).
        """
        if self._size < 0:
            raise DoublyLinkedListInvariantError(f"_size negativo: {self._size}")

        if self._size == 0:
            if self._head is not None or self._tail is not None:
                raise DoublyLinkedListInvariantError(
                    "lista vazia deve ter _head e _tail iguais a None, "
                    f"mas _head={self._head!r} e _tail={self._tail!r}"
                )
            return

        if self._head is None or self._tail is None:
            raise DoublyLinkedListInvariantError(
                f"_size={self._size} mas _head ou _tail é None"
            )
        if self._head.prev is not None:
            raise DoublyLinkedListInvariantError("_head.prev deveria ser None")
        if self._tail.next is not None:
            raise DoublyLinkedListInvariantError("_tail.next deveria ser None")

        # Travessia para frente, conferindo o elo de volta de cada nó.
        forward: list[T] = []
        current = self._head
        last: DoublyNode[T] | None = None
        while current is not None:
            forward.append(current.value)
            if len(forward) > self._size:
                raise DoublyLinkedListInvariantError(
                    f"travessia para frente excede _size={self._size} "
                    "(possível ciclo)"
                )
            if current.next is not None and current.next.prev is not current:
                raise DoublyLinkedListInvariantError(
                    f"elo quebrado: no({current.value!r}).next.prev não aponta de volta"
                )
            if current.prev is not None and current.prev.next is not current:
                raise DoublyLinkedListInvariantError(
                    f"elo quebrado: no({current.value!r}).prev.next não aponta de volta"
                )
            last = current
            current = current.next

        if len(forward) != self._size:
            raise DoublyLinkedListInvariantError(
                f"_size={self._size} mas há {len(forward)} nó(s) alcançáveis"
            )
        if last is not self._tail:
            raise DoublyLinkedListInvariantError(
                "_tail não aponta para o último nó da travessia"
            )

        # Travessia para trás: tem de ser a de frente invertida.
        backward: list[T] = []
        current = self._tail
        while current is not None:
            backward.append(current.value)
            if len(backward) > self._size:
                raise DoublyLinkedListInvariantError(
                    f"travessia para trás excede _size={self._size} "
                    "(possível ciclo)"
                )
            current = current.prev
        if backward != list(reversed(forward)):
            raise DoublyLinkedListInvariantError(
                "travessia para trás não é a travessia para frente invertida: "
                f"frente={forward!r}, trás={backward!r}"
            )


# ----------------------------------------------------------------------
# Deque sobre a lista duplamente encadeada
# ----------------------------------------------------------------------
class Deque(Generic[T]):
    """Fila de duas pontas. Todas as operações são O(1).

    Construída por composição sobre :class:`DoublyLinkedList` (decisão de
    projeto 1): a API é exatamente as seis operações do enunciado mais as de
    leitura, e nenhuma delas percorre a estrutura.

    ``collections.deque`` é proibido pela convenção 9 — o enunciado pede a
    implementação própria, e é esta.
    """

    __slots__ = ("_items",)

    def __init__(self, values: Iterable[T] | None = None) -> None:
        """Cria o deque; ``values`` entram da esquerda para a direita.

        Custo: O(k) para k valores.
        """
        self._items: DoublyLinkedList[T] = DoublyLinkedList()
        if values is not None:
            for value in values:
                self.insert_right(value)

    # -- leitura -------------------------------------------------------
    def __len__(self) -> int:
        """Número de elementos. Custo: O(1)."""
        return len(self._items)

    def is_empty(self) -> bool:
        """``True`` se o deque não tem elementos. Custo: O(1)."""
        return self._items.is_empty()

    def __iter__(self) -> Iterator[T]:
        """Percorre da esquerda para a direita. Custo: O(n)."""
        return iter(self._items)

    def __reversed__(self) -> Iterator[T]:
        """Percorre da direita para a esquerda. Custo: O(n)."""
        return reversed(self._items)

    def to_list(self) -> list[T]:
        """Cópia em ``list`` do Python, da esquerda para a direita. Custo: O(n)."""
        return self._items.to_list()

    def __str__(self) -> str:
        """``"esquerda [a <-> b] direita"``. Custo: O(n)."""
        return f"esquerda {self._items} direita"

    def __repr__(self) -> str:
        return f"Deque(size={len(self)}, {self})"

    # -- as seis operações do enunciado --------------------------------
    def insert_left(self, value: T, counters: Counters | None = None) -> None:
        """Insere na ponta esquerda. Custo: O(1)."""
        self._items.insert_first(value, counters)

    def insert_right(self, value: T, counters: Counters | None = None) -> None:
        """Insere na ponta direita. Custo: O(1)."""
        self._items.insert_last(value, counters)

    def remove_left(self, counters: Counters | None = None) -> T:
        """Remove e devolve o elemento da esquerda. Custo: O(1).

        Levanta :class:`EmptyDequeError` em deque vazio.
        """
        if self._items.is_empty():
            raise EmptyDequeError("remove_left em deque vazio: não há o que remover")
        return self._items.delete_first(counters)

    def remove_right(self, counters: Counters | None = None) -> T:
        """Remove e devolve o elemento da direita. Custo: O(1).

        Levanta :class:`EmptyDequeError` em deque vazio.
        """
        if self._items.is_empty():
            raise EmptyDequeError("remove_right em deque vazio: não há o que remover")
        return self._items.delete_last(counters)

    def peek_left(self) -> T:
        """Elemento da esquerda, sem remover. Custo: O(1).

        Levanta :class:`EmptyDequeError` em deque vazio.
        """
        if self._items.is_empty():
            raise EmptyDequeError("peek_left em deque vazio: não há o que espiar")
        return self._items.peek_first()

    def peek_right(self) -> T:
        """Elemento da direita, sem remover. Custo: O(1).

        Levanta :class:`EmptyDequeError` em deque vazio.
        """
        if self._items.is_empty():
            raise EmptyDequeError("peek_right em deque vazio: não há o que espiar")
        return self._items.peek_last()

    # -- protocolo (resultado, Counters) -------------------------------
    def remove_left_counted(self) -> tuple[T, Counters]:
        """:meth:`remove_left` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        return self.remove_left(counters), counters

    def remove_right_counted(self) -> tuple[T, Counters]:
        """:meth:`remove_right` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        return self.remove_right(counters), counters

    # -- verificação ---------------------------------------------------
    def check_invariants(self) -> None:
        """Valida as invariantes da estrutura interna. Custo: O(n)."""
        self._items.check_invariants()


# ----------------------------------------------------------------------
# Tabelas de custo (consumidas pelo notebook)
# ----------------------------------------------------------------------
#: Custo por operação, com a lista simplesmente encadeada do ex 10 como
#: referência. A única linha em que as duas divergem é ``delete_last`` — e é
#: essa linha que :func:`bench_delete_last_vs_singly` mede.
COST_TABLE: list[dict[str, str]] = [
    {"operacao": "insert_first", "duplamente": "O(1)", "simplesmente": "O(1)",
     "por_que": "religa _head nas duas; a dupla ainda ajusta um prev"},
    {"operacao": "insert_last", "duplamente": "O(1)", "simplesmente": "O(1)",
     "por_que": "as duas têm _tail; acesso direto ao fim"},
    {"operacao": "delete_first", "duplamente": "O(1)", "simplesmente": "O(1)",
     "por_que": "o antecessor do primeiro nó não existe, então não precisa ser buscado"},
    {"operacao": "delete_last", "duplamente": "O(1)", "simplesmente": "O(n)",
     "por_que": "antecessor do último: _tail.prev na dupla, travessia inteira na simples"},
    {"operacao": "peek_first", "duplamente": "O(1)", "simplesmente": "O(1)",
     "por_que": "_head é acesso direto"},
    {"operacao": "peek_last", "duplamente": "O(1)", "simplesmente": "O(1)",
     "por_que": "_tail é acesso direto"},
    {"operacao": "is_empty", "duplamente": "O(1)", "simplesmente": "O(1)",
     "por_que": "_size é mantido incrementalmente"},
    {"operacao": "__len__", "duplamente": "O(1)", "simplesmente": "O(1)",
     "por_que": "_size é mantido incrementalmente"},
    {"operacao": "travessia para frente", "duplamente": "O(n)", "simplesmente": "O(n)",
     "por_que": "n saltos de ponteiro nas duas"},
    {"operacao": "travessia para trás", "duplamente": "O(n)", "simplesmente": "O(n²)",
     "por_que": "sem prev, cada passo para trás recomeça do _head"},
    {"operacao": "memória por nó", "duplamente": "2 ponteiros", "simplesmente": "1 ponteiro",
     "por_que": "é o preço pago pela coluna delete_last"},
]

#: Custo das seis operações do :class:`Deque`. Todas O(1), todas com 0 saltos —
#: é o que :func:`bench_deque_operations` confirma experimentalmente.
DEQUE_COST_TABLE: list[dict[str, str]] = [
    {"operacao": "insert_left", "custo": "O(1)", "saltos": "0",
     "por_que": "delega a insert_first: religa _head"},
    {"operacao": "insert_right", "custo": "O(1)", "saltos": "0",
     "por_que": "delega a insert_last: religa _tail"},
    {"operacao": "remove_left", "custo": "O(1)", "saltos": "0",
     "por_que": "delega a delete_first: _head.next é acesso direto"},
    {"operacao": "remove_right", "custo": "O(1)", "saltos": "0",
     "por_que": "delega a delete_last: _tail.prev é acesso direto"},
    {"operacao": "peek_left", "custo": "O(1)", "saltos": "0",
     "por_que": "lê _head.value"},
    {"operacao": "peek_right", "custo": "O(1)", "saltos": "0",
     "por_que": "lê _tail.value"},
]

#: Nomes das quatro operações que mexem nas pontas, usados pelos experimentos.
END_OPERATIONS: tuple[str, str, str, str] = (
    "insert_left",
    "insert_right",
    "remove_left",
    "remove_right",
)


# ----------------------------------------------------------------------
# Experimentos do exercício 11
# ----------------------------------------------------------------------
def bench_deque_operations(
    sizes: Iterable[int] = (1_000, 2_000, 4_000, 8_000, 16_000),
    *,
    repeats: int = 3,
) -> list[dict[str, Any]]:
    """Mede as quatro operações de ponta do :class:`Deque` isoladamente.

    Para cada ``n`` e cada operação, executa ``n`` chamadas **da mesma
    operação** e registra saltos e tempo. Nas remoções, o deque é preenchido
    antes, fora do cronômetro, para que a medição seja só da remoção.

    Previsão teórica: ``hops == 0`` sempre, e ``tempo_us_por_op``
    aproximadamente constante quando ``n`` cresce 16×. Custo total Θ(n) para n
    operações ⇒ Θ(1) amortizado e de pior caso por operação (não há
    redimensionamento, então não há caso amortizado pior).

    ``repeats``: o tempo reportado é o mínimo de ``repeats`` execuções, mesmo
    critério do ex 10 — ruído de máquina só soma tempo, nunca subtrai.

    Custo: O(repeats · Σ n) por operação medida.
    """
    if repeats < 1:
        raise ValueError(f"repeats deve ser >= 1, recebido {repeats}")
    rows: list[dict[str, Any]] = []
    for n in sizes:
        if n <= 0:
            raise ValueError(f"tamanho deve ser positivo, recebido {n}")
        for operation in END_OPERATIONS:
            is_removal = operation.startswith("remove")
            counters = Counters()
            best_elapsed = float("inf")
            for attempt in range(repeats):
                # Só a primeira execução alimenta o contador; as demais existem
                # apenas para descartar ruído de tempo.
                run_counters = counters if attempt == 0 else Counters()
                target: Deque[int] = Deque()
                if is_removal:
                    # Preparação fora do cronômetro e fora da contagem.
                    for value in range(n):
                        target.insert_right(value)
                method = getattr(target, operation)
                with stopwatch() as elapsed:
                    if is_removal:
                        for _ in range(n):
                            method(run_counters)
                    else:
                        for value in range(n):
                            method(value, run_counters)
                best_elapsed = min(best_elapsed, elapsed[0])
                assert len(target) == (0 if is_removal else n)
            rows.append(
                make_row(
                    exercise="ex11",
                    algorithm=f"Deque.{operation}",
                    n=n,
                    pattern="n operações na mesma ponta",
                    counters=counters,
                    metric="hops",
                    curves=("n", "n2"),
                    elapsed_s=best_elapsed,
                    extra={
                        "operacao": operation,
                        "hops_teorico": 0,
                        "repeticoes": repeats,
                    },
                )
            )
    return rows


def bench_delete_last_vs_singly(
    sizes: Iterable[int] = (250, 500, 1_000, 2_000, 4_000),
    *,
    repeats: int = 1,
) -> list[dict[str, Any]]:
    """Esvazia a lista **pelo fim**, na estrutura dupla e na simples do ex 10.

    É a medição que justifica o ponteiro ``prev``:

    * :class:`DoublyLinkedList` — ``n`` chamadas de ``delete_last``, 0 saltos no
      total, Θ(n) para esvaziar;
    * :class:`~src.linked_list.SinglyLinkedList` — a mesma tarefa só é possível
      via ``delete_at(len-1)``, que percorre ``k-1`` nós quando restam ``k``
      elementos. Total exato: ``n(n-1)/2`` saltos, Θ(n²).

    A coluna ``hops_teorico`` traz o valor fechado para conferência dígito a
    dígito; ``hops/n2`` deve convergir para 0,5 no lado simples e ficar em 0 no
    lado duplo.

    ``repeats`` é 1 por padrão, ao contrário do ex 10. Lá a variante rápida era
    sub-milissegundo e o tempo precisava de melhor-de-k para não ser engolido
    pelo ruído; aqui a diferença entre os dois lados é de ordens de grandeza, e
    repetir o lado Θ(n²) triplicaria o custo do notebook sem mudar a conclusão.
    As contagens são determinísticas e não dependem de ``repeats``.

    Custo: O(repeats · Σ n²), dominado pela variante simplesmente encadeada.
    """
    if repeats < 1:
        raise ValueError(f"repeats deve ser >= 1, recebido {repeats}")
    rows: list[dict[str, Any]] = []
    for n in sizes:
        if n <= 0:
            raise ValueError(f"tamanho deve ser positivo, recebido {n}")

        for label in ("duplamente encadeada", "simplesmente encadeada"):
            counters = Counters()
            best_elapsed = float("inf")
            for attempt in range(repeats):
                run_counters = counters if attempt == 0 else Counters()
                if label == "duplamente encadeada":
                    doubly: DoublyLinkedList[int] = DoublyLinkedList(range(n))
                    with stopwatch() as elapsed:
                        for _ in range(n):
                            doubly.delete_last(run_counters)
                    remaining = len(doubly)
                else:
                    singly: SinglyLinkedList[int] = SinglyLinkedList(range(n))
                    with stopwatch() as elapsed:
                        for _ in range(n):
                            singly.delete_at(len(singly) - 1, run_counters)
                    remaining = len(singly)
                best_elapsed = min(best_elapsed, elapsed[0])
                assert remaining == 0

            theoretical = 0 if label == "duplamente encadeada" else n * (n - 1) // 2
            rows.append(
                make_row(
                    exercise="ex11",
                    algorithm=f"esvaziar pelo fim ({label})",
                    n=n,
                    pattern="n remoções no fim",
                    counters=counters,
                    metric="hops",
                    curves=("n", "n2"),
                    elapsed_s=best_elapsed,
                    extra={
                        "estrutura": label,
                        "hops_teorico": theoretical,
                        "repeticoes": repeats,
                    },
                )
            )
    return rows


def run_alternating_ends_stress(
    operations: int = 20_000,
    *,
    seed: int = RANDOM_SEED,
    check_every: int = 1_000,
) -> dict[str, Any]:
    """Bateria longa alternando as duas pontas, verificando invariantes no meio.

    Sorteia ``operations`` operações entre as quatro pontas do :class:`Deque`.
    A cada ``check_every`` operações, e obrigatoriamente ao final, confere:

    1. :meth:`Deque.check_invariants` — tamanho, ``head.prev is None``,
       ``tail.next is None``, elos coerentes nos dois sentidos e travessia de
       volta igual à de ida invertida;
    2. o conteúdo contra um modelo de referência em ``list`` do Python.

    O modelo em ``list`` é permitido pela convenção 9 (estrutura pronta só como
    referência de corretude, nunca dentro do algoritmo): ele não participa de
    nenhuma operação do deque, só diz qual deveria ser o resultado.

    Remoções sorteadas com o deque vazio não são silenciadas — são executadas,
    a :class:`EmptyDequeError` é capturada e contabilizada em
    ``tentativas_em_vazio``. É o caso de borda do enunciado sendo exercitado
    milhares de vezes, não uma vez.

    Devolve uma linha pronta para o DataFrame do notebook.

    Custo: O(operations) nas operações + O(n) por verificação, isto é
    O(operations · n / check_every) no total das verificações.
    """
    if operations < 1:
        raise ValueError(f"operations deve ser >= 1, recebido {operations}")
    if check_every < 1:
        raise ValueError(f"check_every deve ser >= 1, recebido {check_every}")

    rng = random.Random(seed)
    target: Deque[int] = Deque()
    model: list[int] = []
    counters = Counters()
    histogram: dict[str, int] = {name: 0 for name in END_OPERATIONS}
    empty_attempts = 0
    checks = 0
    max_size = 0

    with stopwatch() as elapsed:
        for step in range(1, operations + 1):
            operation = rng.choice(END_OPERATIONS)
            value = rng.randrange(1_000_000)
            histogram[operation] += 1

            if operation == "insert_left":
                target.insert_left(value, counters)
                model.insert(0, value)
            elif operation == "insert_right":
                target.insert_right(value, counters)
                model.append(value)
            elif operation == "remove_left":
                try:
                    removed = target.remove_left(counters)
                except EmptyDequeError:
                    empty_attempts += 1
                else:
                    expected = model.pop(0)
                    if removed != expected:
                        raise DoublyLinkedListInvariantError(
                            f"passo {step}: remove_left devolveu {removed!r}, "
                            f"esperado {expected!r}"
                        )
            else:  # remove_right
                try:
                    removed = target.remove_right(counters)
                except EmptyDequeError:
                    empty_attempts += 1
                else:
                    expected = model.pop()
                    if removed != expected:
                        raise DoublyLinkedListInvariantError(
                            f"passo {step}: remove_right devolveu {removed!r}, "
                            f"esperado {expected!r}"
                        )

            max_size = max(max_size, len(target))
            if step % check_every == 0 or step == operations:
                _assert_matches_model(target, model, step)
                checks += 1

    return {
        "exercicio": "ex11",
        "algoritmo": "Deque — sequência longa alternando as pontas",
        "n": operations,
        "padrao": f"operações sorteadas (seed={seed})",
        **counters.as_dict(),
        "verificacoes_de_invariante": checks,
        "tentativas_em_vazio": empty_attempts,
        "tamanho_final": len(target),
        "tamanho_maximo": max_size,
        **{f"op_{name}": count for name, count in histogram.items()},
        "tempo_s": elapsed[0],
        "hops_teorico": 0,
    }


def _assert_matches_model(target: "Deque[T]", model: list[T], step: int) -> None:
    """Confere invariantes e conteúdo contra o modelo. Custo: O(n)."""
    target.check_invariants()
    if len(target) != len(model):
        raise DoublyLinkedListInvariantError(
            f"passo {step}: len(deque)={len(target)} mas o modelo tem {len(model)}"
        )
    forward = target.to_list()
    if forward != model:
        raise DoublyLinkedListInvariantError(
            f"passo {step}: conteúdo divergente do modelo"
        )
    backward = list(reversed(target))
    if backward != list(reversed(model)):
        raise DoublyLinkedListInvariantError(
            f"passo {step}: travessia para trás divergente do modelo"
        )

"""Exercício 10 — Lista simplesmente encadeada.

Reutilizado pelos exercícios 1 (busca linear sobre lista), 9 (quicksort com
partição em três listas) e 12 (listas de ocorrências do índice invertido).

Duas implementações
-------------------
:class:`SinglyLinkedList`
    Mantém ponteiro ``_tail``. É a implementação usada pelo resto do projeto.
:class:`SinglyLinkedListNoTail`
    Não mantém ``_tail``. Existe como **grupo de controle** do experimento de
    ``insert_last``.

A duplicação é deliberada. A alternativa — uma classe com uma flag
``usa_tail`` — mediria o custo do ``if`` da flag, não o custo de não ter
``tail``: os outros métodos continuariam atualizando o ponteiro e o
experimento perderia o sentido. Com duas classes independentes, a variante sem
``tail`` realmente não tem o atributo, e a suíte de testes roda parametrizada
sobre as duas para provar que a API observável é idêntica — o que isola a
diferença de custo como única diferença entre elas.

Decisões de projeto
-------------------
1. ``search`` devolve o índice ou ``-1``; ``delete`` levanta
   :class:`ValueNotFoundError`. A assimetria é intencional: ``search`` é uma
   *consulta* e o enunciado do ex 1 exige o sentinela ``-1``, enquanto
   ``delete`` é um *comando* — falhar em silêncio esconderia erro do chamador.
2. Métodos que percorrem a lista aceitam ``counters: Counters | None``, para
   acumular em um contador do chamador, e têm um irmão ``*_counted`` que segue o
   protocolo ``(resultado, Counters)`` da seção 5 do CLAUDE.md. Assim o mesmo
   código serve para uso normal e para medição, sem caminho duplicado que possa
   divergir.
3. ``_size`` é mantido incrementalmente, então ``__len__`` é O(1). Contar nós a
   cada ``len()`` tornaria O(n) uma operação que aparece dentro de laços.

Custo por operação (ver também :data:`COST_TABLE`)
--------------------------------------------------
======================  ==============  =================
Operação                com ``tail``    sem ``tail``
======================  ==============  =================
``insert_first``        O(1)            O(1)
``insert_last``         O(1)            O(n)
``search``              O(n)            O(n)
``delete`` (por valor)  O(n)            O(n)
``insert_at(i)``        O(min(i, n))    O(i)
``delete_at(i)``        O(i)            O(i)
``__len__``             O(1)            O(1)
``__str__``             O(n)            O(n)
======================  ==============  =================

``delete`` é O(n) mesmo com ``tail``: remover o último nó exige o
**antecessor**, e em lista simplesmente encadeada não há ponteiro ``prev``.
Esse é exatamente o limite que o exercício 11 resolve com lista duplamente
encadeada.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Generic, Iterable, Iterator, TypeVar

from src.counters import RANDOM_SEED, Counters, make_row, ratio, stopwatch

T = TypeVar("T")


# ----------------------------------------------------------------------
# Exceções próprias (convenção 9: nunca `raise Exception(...)`)
# ----------------------------------------------------------------------
class LinkedListError(Exception):
    """Erro base de :class:`SinglyLinkedList` e variantes."""


class EmptyListError(LinkedListError):
    """Operação que exige ao menos um elemento foi chamada em lista vazia."""


class InvalidIndexError(LinkedListError):
    """Índice fora da faixa permitida pela operação."""


class ValueNotFoundError(LinkedListError):
    """O valor pedido não existe na lista."""


class ListInvariantError(LinkedListError):
    """Invariante estrutural violada (bug de implementação, não de uso)."""


# ----------------------------------------------------------------------
# Nó
# ----------------------------------------------------------------------
@dataclass(slots=True)
class Node(Generic[T]):
    """Nó de lista simplesmente encadeada.

    ``slots=True`` não é enfeite: no ex 1 o argumento é que a busca linear faz o
    mesmo número de comparações em array e em lista encadeada, mas paga um nó
    por elemento. ``__slots__`` deixa esse custo de memória ser o menor
    possível, então a comparação não fica inflada por ``__dict__`` por nó.
    """

    value: T
    next: "Node[T] | None" = None


# ----------------------------------------------------------------------
# Implementação COM ponteiro tail
# ----------------------------------------------------------------------
class SinglyLinkedList(Generic[T]):
    """Lista simplesmente encadeada com ponteiro para o último nó.

    Invariantes mantidas por todos os métodos:

    * ``_size`` é igual ao número de nós alcançáveis a partir de ``_head``;
    * lista vazia ⇔ ``_head is None`` ⇔ ``_tail is None``;
    * lista não vazia ⇒ ``_tail`` é o último nó e ``_tail.next is None``.

    Use :meth:`check_invariants` para verificá-las em testes.
    """

    __slots__ = ("_head", "_tail", "_size")

    def __init__(self, values: Iterable[T] | None = None) -> None:
        """Cria a lista, opcionalmente preenchendo na ordem de ``values``.

        Custo: O(k) para k valores, porque ``insert_last`` é O(1) aqui.
        """
        self._head: Node[T] | None = None
        self._tail: Node[T] | None = None
        self._size: int = 0
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

    def to_list(self) -> list[T]:
        """Cópia em ``list`` do Python, usada nos testes. Custo: O(n)."""
        return list(self)

    def __str__(self) -> str:
        """``"[a -> b -> c]"``, ``"[]"`` quando vazia. Custo: O(n)."""
        if self._head is None:
            return "[]"
        return "[" + " -> ".join(repr(value) for value in self) + "]"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self._size}, {self})"

    def peek_first(self) -> T:
        """Primeiro valor. Levanta :class:`EmptyListError` se vazia. Custo: O(1)."""
        if self._head is None:
            raise EmptyListError("peek_first em lista vazia: não há primeiro elemento")
        return self._head.value

    def peek_last(self) -> T:
        """Último valor. Custo: O(1) — é para isso que o ``tail`` existe."""
        if self._tail is None:
            raise EmptyListError("peek_last em lista vazia: não há último elemento")
        return self._tail.value

    # -- inserção ------------------------------------------------------
    def insert_first(self, value: T, counters: Counters | None = None) -> None:
        """Insere no início. Custo: O(1) — 0 comparações, 0 saltos."""
        node: Node[T] = Node(value, self._head)
        self._head = node
        if self._tail is None:
            self._tail = node
        self._size += 1
        if counters is not None:
            counters.count_copy()

    def insert_last(self, value: T, counters: Counters | None = None) -> None:
        """Insere no fim. Custo: O(1) porque ``_tail`` dá acesso direto ao fim.

        Nenhum salto de ponteiro é executado, e é essa a diferença medida contra
        :meth:`SinglyLinkedListNoTail.insert_last`.
        """
        node: Node[T] = Node(value)
        if self._tail is None:
            self._head = node
        else:
            self._tail.next = node
        self._tail = node
        self._size += 1
        if counters is not None:
            counters.count_copy()

    def insert_last_counted(self, value: T) -> tuple[None, Counters]:
        """:meth:`insert_last` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        self.insert_last(value, counters)
        return None, counters

    def insert_at(self, index: int, value: T, counters: Counters | None = None) -> None:
        """Insere em ``index``, deslocando o resto para a direita.

        ``index`` válido: ``0 <= index <= len(self)``. O limite superior inclui
        ``len(self)`` de propósito — inserir na posição ``n`` é o mesmo que
        anexar no fim, e proibir isso obrigaria o chamador a tratar o append
        como caso especial.

        Custo: O(1) nas pontas, O(index) no meio.
        """
        if not isinstance(index, int) or isinstance(index, bool):
            raise InvalidIndexError(
                f"índice deve ser int, recebido {type(index).__name__}: {index!r}"
            )
        if index < 0 or index > self._size:
            raise InvalidIndexError(
                f"insert_at({index}) inválido: faixa permitida é 0..{self._size} "
                f"(lista tem {self._size} elemento(s))"
            )
        if index == 0:
            self.insert_first(value, counters)
            return
        if index == self._size:
            self.insert_last(value, counters)
            return
        previous = self._node_at(index - 1, counters)
        previous.next = Node(value, previous.next)
        self._size += 1
        if counters is not None:
            counters.count_copy()

    # -- busca ---------------------------------------------------------
    def search(self, value: T, counters: Counters | None = None) -> int:
        """Índice da primeira ocorrência de ``value``, ou ``-1`` se ausente.

        Contagens: ``i+1`` comparações e ``i`` saltos quando o valor está em
        ``i``; ``n`` comparações e ``n`` saltos quando está ausente — ou seja,
        melhor caso O(1), pior e médio caso O(n).

        Custo: O(n).
        """
        counters = counters if counters is not None else Counters()
        current = self._head
        index = 0
        while current is not None:
            if counters.eq(current.value, value):
                return index
            current = current.next
            counters.count_hop()
            index += 1
        return -1

    def search_counted(self, value: T) -> tuple[int, Counters]:
        """:meth:`search` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        index = self.search(value, counters)
        return index, counters

    def contains(self, value: T, counters: Counters | None = None) -> bool:
        """``True`` se ``value`` está na lista. Custo: O(n)."""
        return self.search(value, counters) != -1

    def __contains__(self, value: object) -> bool:
        return self.search(value) != -1  # type: ignore[arg-type]

    # -- remoção -------------------------------------------------------
    def delete(self, value: T, counters: Counters | None = None) -> int:
        """Remove a primeira ocorrência de ``value`` e devolve o índice removido.

        Levanta :class:`ValueNotFoundError` se o valor não existe, e
        :class:`EmptyListError` se a lista está vazia — ver decisão de projeto 1
        no topo do módulo.

        Custo: O(n). Mesmo com ``tail``, remover exige o antecessor do nó.
        """
        counters = counters if counters is not None else Counters()
        if self._head is None:
            raise EmptyListError(f"delete({value!r}) em lista vazia")

        previous: Node[T] | None = None
        current: Node[T] | None = self._head
        index = 0
        while current is not None:
            if counters.eq(current.value, value):
                self._unlink(previous, current)
                return index
            previous = current
            current = current.next
            counters.count_hop()
            index += 1
        raise ValueNotFoundError(
            f"delete({value!r}): valor não encontrado na lista de "
            f"{self._size} elemento(s)"
        )

    def delete_counted(self, value: T) -> tuple[int, Counters]:
        """:meth:`delete` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        index = self.delete(value, counters)
        return index, counters

    def delete_at(self, index: int, counters: Counters | None = None) -> T:
        """Remove o elemento de ``index`` e devolve seu valor.

        ``index`` válido: ``0 <= index < len(self)``. Diferente de
        :meth:`insert_at`, ``len(self)`` **não** é aceito: não há elemento lá.

        Custo: O(index).
        """
        if not isinstance(index, int) or isinstance(index, bool):
            raise InvalidIndexError(
                f"índice deve ser int, recebido {type(index).__name__}: {index!r}"
            )
        if self._size == 0:
            raise EmptyListError(f"delete_at({index}) em lista vazia")
        if index < 0 or index >= self._size:
            raise InvalidIndexError(
                f"delete_at({index}) inválido: faixa permitida é "
                f"0..{self._size - 1} (lista tem {self._size} elemento(s))"
            )
        counters = counters if counters is not None else Counters()
        previous: Node[T] | None = None
        current = self._head
        for _ in range(index):
            previous = current
            current = current.next  # type: ignore[union-attr]
            counters.count_hop()
        assert current is not None  # garantido pela validação de faixa
        value = current.value
        self._unlink(previous, current)
        return value

    def delete_at_counted(self, index: int) -> tuple[T, Counters]:
        """:meth:`delete_at` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        value = self.delete_at(index, counters)
        return value, counters

    # -- internos ------------------------------------------------------
    def _node_at(self, index: int, counters: Counters | None = None) -> Node[T]:
        """Nó da posição ``index`` (já validada pelo chamador). Custo: O(index)."""
        current = self._head
        for _ in range(index):
            current = current.next  # type: ignore[union-attr]
            if counters is not None:
                counters.count_hop()
        assert current is not None
        return current

    def _unlink(self, previous: Node[T] | None, current: Node[T]) -> None:
        """Desliga ``current``, mantendo ``_head``, ``_tail`` e ``_size``.

        O caso que quebra implementações descuidadas: remover o último nó exige
        recuar ``_tail`` para ``previous``; senão ``insert_last`` posterior
        anexaria a um nó já removido e a lista "perderia" elementos.
        """
        if previous is None:
            self._head = current.next
        else:
            previous.next = current.next
        if current is self._tail:
            self._tail = previous
        current.next = None
        self._size -= 1

    # -- verificação ---------------------------------------------------
    def check_invariants(self) -> None:
        """Valida as invariantes da classe; levanta :class:`ListInvariantError`.

        Custo: O(n). Chamada pelos testes depois de sequências longas de
        operações, que é onde os bugs de ponteiro aparecem.
        """
        count = 0
        current = self._head
        last: Node[T] | None = None
        while current is not None:
            count += 1
            last = current
            current = current.next
            if count > self._size + 1:
                raise ListInvariantError(
                    f"lista mais longa que _size={self._size} (possível ciclo)"
                )
        if count != self._size:
            raise ListInvariantError(
                f"_size={self._size} mas há {count} nó(s) alcançáveis"
            )
        if self._size == 0:
            if self._head is not None or self._tail is not None:
                raise ListInvariantError(
                    "lista vazia deve ter _head e _tail iguais a None"
                )
            return
        if self._tail is not last:
            raise ListInvariantError("_tail não aponta para o último nó")
        if self._tail.next is not None:
            raise ListInvariantError("_tail.next deveria ser None")


# ----------------------------------------------------------------------
# Implementação SEM ponteiro tail (grupo de controle do experimento)
# ----------------------------------------------------------------------
class SinglyLinkedListNoTail(Generic[T]):
    """Igual a :class:`SinglyLinkedList`, mas sem ponteiro para o último nó.

    Consequência única e medida: ``insert_last`` precisa percorrer a lista até o
    fim, gastando ``n-1`` saltos de ponteiro. Construir uma lista de ``n``
    elementos com ``insert_last`` sai de Θ(n) para Θ(n²) — ver
    :func:`bench_insert_last`.

    ``peek_last`` sofre o mesmo destino: O(n) em vez de O(1).
    """

    __slots__ = ("_head", "_size")

    def __init__(self, values: Iterable[T] | None = None) -> None:
        """Cria a lista. Custo: O(k²) para k valores, justamente por não ter tail."""
        self._head: Node[T] | None = None
        self._size: int = 0
        if values is not None:
            for value in values:
                self.insert_last(value)

    # -- leitura -------------------------------------------------------
    def __len__(self) -> int:
        """Custo: O(1)."""
        return self._size

    def is_empty(self) -> bool:
        """Custo: O(1)."""
        return self._size == 0

    def __iter__(self) -> Iterator[T]:
        """Custo: O(n)."""
        current = self._head
        while current is not None:
            yield current.value
            current = current.next

    def to_list(self) -> list[T]:
        """Custo: O(n)."""
        return list(self)

    def __str__(self) -> str:
        """Custo: O(n)."""
        if self._head is None:
            return "[]"
        return "[" + " -> ".join(repr(value) for value in self) + "]"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self._size}, {self})"

    def peek_first(self) -> T:
        """Custo: O(1)."""
        if self._head is None:
            raise EmptyListError("peek_first em lista vazia: não há primeiro elemento")
        return self._head.value

    def peek_last(self) -> T:
        """Último valor. Custo: O(n) — sem ``tail`` é preciso percorrer tudo."""
        if self._head is None:
            raise EmptyListError("peek_last em lista vazia: não há último elemento")
        current = self._head
        while current.next is not None:
            current = current.next
        return current.value

    # -- inserção ------------------------------------------------------
    def insert_first(self, value: T, counters: Counters | None = None) -> None:
        """Insere no início. Custo: O(1) — idêntico à variante com ``tail``."""
        self._head = Node(value, self._head)
        self._size += 1
        if counters is not None:
            counters.count_copy()

    def insert_last(self, value: T, counters: Counters | None = None) -> None:
        """Insere no fim percorrendo até o último nó.

        Contagens: ``n-1`` saltos para lista de ``n`` elementos (0 quando vazia),
        0 comparações de chave, 1 cópia. Custo: O(n).
        """
        node: Node[T] = Node(value)
        if self._head is None:
            self._head = node
        else:
            current = self._head
            while current.next is not None:
                current = current.next
                if counters is not None:
                    counters.count_hop()
            current.next = node
        self._size += 1
        if counters is not None:
            counters.count_copy()

    def insert_last_counted(self, value: T) -> tuple[None, Counters]:
        """:meth:`insert_last` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        self.insert_last(value, counters)
        return None, counters

    def insert_at(self, index: int, value: T, counters: Counters | None = None) -> None:
        """Insere em ``index`` (``0 <= index <= len(self)``). Custo: O(index)."""
        if not isinstance(index, int) or isinstance(index, bool):
            raise InvalidIndexError(
                f"índice deve ser int, recebido {type(index).__name__}: {index!r}"
            )
        if index < 0 or index > self._size:
            raise InvalidIndexError(
                f"insert_at({index}) inválido: faixa permitida é 0..{self._size} "
                f"(lista tem {self._size} elemento(s))"
            )
        if index == 0:
            self.insert_first(value, counters)
            return
        previous = self._node_at(index - 1, counters)
        previous.next = Node(value, previous.next)
        self._size += 1
        if counters is not None:
            counters.count_copy()

    # -- busca ---------------------------------------------------------
    def search(self, value: T, counters: Counters | None = None) -> int:
        """Índice da primeira ocorrência, ou ``-1``. Custo: O(n)."""
        counters = counters if counters is not None else Counters()
        current = self._head
        index = 0
        while current is not None:
            if counters.eq(current.value, value):
                return index
            current = current.next
            counters.count_hop()
            index += 1
        return -1

    def search_counted(self, value: T) -> tuple[int, Counters]:
        """:meth:`search` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        index = self.search(value, counters)
        return index, counters

    def contains(self, value: T, counters: Counters | None = None) -> bool:
        """Custo: O(n)."""
        return self.search(value, counters) != -1

    def __contains__(self, value: object) -> bool:
        return self.search(value) != -1  # type: ignore[arg-type]

    # -- remoção -------------------------------------------------------
    def delete(self, value: T, counters: Counters | None = None) -> int:
        """Remove a primeira ocorrência e devolve o índice. Custo: O(n)."""
        counters = counters if counters is not None else Counters()
        if self._head is None:
            raise EmptyListError(f"delete({value!r}) em lista vazia")
        previous: Node[T] | None = None
        current: Node[T] | None = self._head
        index = 0
        while current is not None:
            if counters.eq(current.value, value):
                self._unlink(previous, current)
                return index
            previous = current
            current = current.next
            counters.count_hop()
            index += 1
        raise ValueNotFoundError(
            f"delete({value!r}): valor não encontrado na lista de "
            f"{self._size} elemento(s)"
        )

    def delete_counted(self, value: T) -> tuple[int, Counters]:
        """:meth:`delete` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        index = self.delete(value, counters)
        return index, counters

    def delete_at(self, index: int, counters: Counters | None = None) -> T:
        """Remove o elemento de ``index`` (``0 <= index < len``). Custo: O(index)."""
        if not isinstance(index, int) or isinstance(index, bool):
            raise InvalidIndexError(
                f"índice deve ser int, recebido {type(index).__name__}: {index!r}"
            )
        if self._size == 0:
            raise EmptyListError(f"delete_at({index}) em lista vazia")
        if index < 0 or index >= self._size:
            raise InvalidIndexError(
                f"delete_at({index}) inválido: faixa permitida é "
                f"0..{self._size - 1} (lista tem {self._size} elemento(s))"
            )
        counters = counters if counters is not None else Counters()
        previous: Node[T] | None = None
        current = self._head
        for _ in range(index):
            previous = current
            current = current.next  # type: ignore[union-attr]
            counters.count_hop()
        assert current is not None
        value = current.value
        self._unlink(previous, current)
        return value

    def delete_at_counted(self, index: int) -> tuple[T, Counters]:
        """:meth:`delete_at` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        value = self.delete_at(index, counters)
        return value, counters

    # -- internos ------------------------------------------------------
    def _node_at(self, index: int, counters: Counters | None = None) -> Node[T]:
        """Custo: O(index)."""
        current = self._head
        for _ in range(index):
            current = current.next  # type: ignore[union-attr]
            if counters is not None:
                counters.count_hop()
        assert current is not None
        return current

    def _unlink(self, previous: Node[T] | None, current: Node[T]) -> None:
        """Desliga ``current``. Sem ``_tail``, não há ponteiro extra a corrigir."""
        if previous is None:
            self._head = current.next
        else:
            previous.next = current.next
        current.next = None
        self._size -= 1

    # -- verificação ---------------------------------------------------
    def check_invariants(self) -> None:
        """Valida ``_size`` e ausência de ciclo. Custo: O(n)."""
        count = 0
        current = self._head
        while current is not None:
            count += 1
            current = current.next
            if count > self._size + 1:
                raise ListInvariantError(
                    f"lista mais longa que _size={self._size} (possível ciclo)"
                )
        if count != self._size:
            raise ListInvariantError(
                f"_size={self._size} mas há {count} nó(s) alcançáveis"
            )
        if self._size == 0 and self._head is not None:
            raise ListInvariantError("lista vazia deve ter _head igual a None")


# ----------------------------------------------------------------------
# Tabela de custo (consumida pelo notebook)
# ----------------------------------------------------------------------
#: Custo assintótico por operação nas duas variantes. O notebook transforma isso
#: em DataFrame para a seção "Análise em Big O" do exercício 10.
COST_TABLE: list[dict[str, str]] = [
    {"operacao": "insert_first", "com_tail": "O(1)", "sem_tail": "O(1)",
     "por_que": "só religa _head; nenhuma travessia"},
    {"operacao": "insert_last", "com_tail": "O(1)", "sem_tail": "O(n)",
     "por_que": "com tail há acesso direto ao fim; sem tail percorre n-1 nós"},
    {"operacao": "search", "com_tail": "O(n)", "sem_tail": "O(n)",
     "por_que": "sem acesso por índice, só travessia sequencial"},
    {"operacao": "delete (por valor)", "com_tail": "O(n)", "sem_tail": "O(n)",
     "por_que": "precisa do antecessor; lista simples não tem prev"},
    {"operacao": "insert_at(i)", "com_tail": "O(min(i, n))", "sem_tail": "O(i)",
     "por_que": "com tail, i == n cai no atalho O(1) do insert_last"},
    {"operacao": "delete_at(i)", "com_tail": "O(i)", "sem_tail": "O(i)",
     "por_que": "travessia até o antecessor do índice"},
    {"operacao": "peek_first", "com_tail": "O(1)", "sem_tail": "O(1)",
     "por_que": "_head é acesso direto"},
    {"operacao": "peek_last", "com_tail": "O(1)", "sem_tail": "O(n)",
     "por_que": "mesma razão do insert_last"},
    {"operacao": "__len__", "com_tail": "O(1)", "sem_tail": "O(1)",
     "por_que": "_size é mantido incrementalmente"},
    {"operacao": "__str__", "com_tail": "O(n)", "sem_tail": "O(n)",
     "por_que": "formata todos os elementos"},
]


# ----------------------------------------------------------------------
# Experimentos do exercício 10
# ----------------------------------------------------------------------
def bench_insert_last(
    sizes: Iterable[int] = (500, 1_000, 2_000, 4_000, 8_000),
    *,
    repeats: int = 3,
) -> list[dict[str, Any]]:
    """Mede o custo de construir uma lista de ``n`` elementos com ``insert_last``.

    Para cada ``n``, executa ``n`` chamadas de ``insert_last`` nas duas variantes,
    acumulando os saltos de ponteiro em um único :class:`Counters` e cronometrando
    o tempo de parede.

    Previsão teórica, para confrontar com o medido:

    * com ``tail``: 0 saltos, total Θ(n) — ``hops/n`` deve ser 0;
    * sem ``tail``: ``(n-1)(n-2)/2`` saltos, total Θ(n²) — ``hops/n²`` deve
      convergir para 0,5 por cima, e ``hops/n`` (que é exatamente o número de
      saltos por inserção) deve dobrar quando ``n`` dobra, o que é o que descarta
      a hipótese O(1) por inserção.

    ``repeats``: o tempo reportado é o **mínimo** de ``repeats`` execuções, não a
    média. Na variante com ``tail`` a construção é tão rápida que uma coleta de
    lixo no meio da medição dobra o tempo; o mínimo é o estimador padrão para
    esse caso porque o ruído de máquina só pode somar tempo, nunca subtrair. As
    contagens não precisam de repetição: são determinísticas.

    Custo: O(repeats · Σ n) na variante com ``tail``, O(repeats · Σ n²) na sem.
    """
    if repeats < 1:
        raise ValueError(f"repeats deve ser >= 1, recebido {repeats}")
    rows: list[dict[str, Any]] = []
    for n in sizes:
        if n <= 0:
            raise ValueError(f"tamanho deve ser positivo, recebido {n}")
        for label, factory in (
            ("com tail", SinglyLinkedList),
            ("sem tail", SinglyLinkedListNoTail),
        ):
            counters = Counters()
            best_elapsed = float("inf")
            for attempt in range(repeats):
                # Só a primeira execução alimenta o contador; as demais existem
                # apenas para descartar ruído de tempo.
                run_counters = counters if attempt == 0 else Counters()
                target: Any = factory()
                with stopwatch() as elapsed:
                    for value in range(n):
                        target.insert_last(value, run_counters)
                best_elapsed = min(best_elapsed, elapsed[0])
                assert len(target) == n
            row = make_row(
                exercise="ex10",
                algorithm=f"insert_last ({label})",
                n=n,
                pattern="construção sequencial",
                counters=counters,
                metric="hops",
                curves=("n", "n2"),
                elapsed_s=best_elapsed,
                extra={
                    "variante": label,
                    "repeticoes": repeats,
                    "hops_teorico": 0 if label == "com tail" else (n - 1) * (n - 2) // 2,
                },
            )
            rows.append(row)
    return rows


def bench_search(
    sizes: Iterable[int] = (500, 1_000, 2_000, 4_000, 8_000),
    *,
    seed: int = RANDOM_SEED,
    average_samples: int = 30,
) -> list[dict[str, Any]]:
    """Mede a busca linear em :class:`SinglyLinkedList` nos três casos.

    * **melhor caso**: chave na posição 0 → 1 comparação, independente de ``n``;
    * **caso médio**: ``average_samples`` chaves presentes sorteadas → média
      esperada ``(n+1)/2`` comparações, isto é ``comparações/n ≈ 0,5``;
    * **pior caso**: chave ausente → ``n`` comparações, ``comparações/n = 1``.

    A razão ``comparações_por_busca / n`` aproximadamente constante é a evidência
    de que a busca é Θ(n) no médio e no pior caso.

    Custo: O(Σ n · average_samples).
    """
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for n in sizes:
        if n <= 0:
            raise ValueError(f"tamanho deve ser positivo, recebido {n}")
        values = list(range(n))
        rng.shuffle(values)
        target: SinglyLinkedList[int] = SinglyLinkedList(values)

        cases: list[tuple[str, list[int], int]] = [
            ("melhor caso (1a posição)", [values[0]], 1),
            (
                "caso médio (chave presente sorteada)",
                [rng.choice(values) for _ in range(average_samples)],
                average_samples,
            ),
            ("pior caso (chave ausente)", [n + 1], 1),
        ]
        for pattern, keys, searches in cases:
            counters = Counters()
            found = 0
            with stopwatch() as elapsed:
                for key in keys:
                    if target.search(key, counters) != -1:
                        found += 1
            per_search = counters.comparisons / searches
            rows.append(
                {
                    "exercicio": "ex10",
                    "algoritmo": "SinglyLinkedList.search",
                    "n": n,
                    "padrao": pattern,
                    **counters.as_dict(),
                    "buscas": searches,
                    "encontradas": found,
                    "comparacoes_por_busca": per_search,
                    "comparacoes_por_busca/n": ratio(per_search, n, "n"),
                    "tempo_s": elapsed[0],
                }
            )
    return rows

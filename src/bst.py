"""Árvore binária de busca — definida no exercício 2, usada em 2, 3, 6, 8 e 12.

Consumidores
------------
* **ex 2** — inserir o vetor inteiro, travessia in-order, verificar ordenação e
  comparar o custo com o insertion sort;
* **ex 3** — :meth:`BinarySearchTree.k_smallest` por travessia controlada, e a
  demonstração medida da degeneração para lista encadeada;
* **ex 6** — índice dos clientes atendidos, chave ``"B17"``, valor com os
  metadados do atendimento;
* **ex 8** — árvore de subproblemas visitados, com busca por estado;
* **ex 12** — árvore de termos para listagem ordenada, busca, e o relatório de
  diagnóstico por :meth:`level_order` (BFS com a ``Queue`` do ex 5).

Decisões de projeto
-------------------
1. **Todas as operações são iterativas, inclusive as travessias.**
   O exercício 3 exige *demonstrar com medição* que uma BST construída a partir
   de entrada ordenada degenera em lista encadeada. Nesse cenário a altura é
   ``n``, e qualquer operação recursiva levantaria ``RecursionError`` antes de
   produzir a medição — o experimento mediria o limite do intérprete, não o
   algoritmo. Implementação iterativa mede o que o exercício quer medir.
   (A discussão sobre profundidade de recursão fica no ex 5, que é onde ela é o
   assunto, com a mesma árvore avaliada das duas formas.)

2. **A árvore é um mapa com multiplicidade.** Cada nó guarda ``key``, ``value``
   e ``multiplicity``. Inserir uma chave que já existe **não** cria nó novo:
   incrementa a multiplicidade e atualiza o valor. Isso serve aos dois usos:
   como *mapa* (ex 6, 8 e 12: chave → metadados) e como *multiconjunto* (ex 2:
   ordenar um vetor que pode ter repetições — a travessia devolve cada chave
   tantas vezes quantas ela entrou, então ``bst_sort`` funciona em geral e não
   só quando os valores são distintos).

3. **Uma sondagem = uma comparação de chaves**, o mesmo critério do exercício 1.
   Cada nó visitado na descida conta 1 comparação de três vias, não 2. Sem esse
   critério comum, comparar o custo da BST com o do insertion sort (ex 2) ou com
   o da busca binária (ex 1) mediria unidades diferentes.

4. **``delete`` existe e é completo, embora nenhum enunciado o peça.**
   O item 4.4 da rubrica cobra "inserção, busca e **deleção**, mantendo
   invariantes estruturais". Os três casos estão implementados — folha, um filho
   e dois filhos por sucessor in-order — e :meth:`check_invariants` é chamada
   depois de sequências longas de remoção nos testes.

Custo em Big O
--------------
Com ``n`` nós e altura ``h``:

=====================  =================  ==================  =============
Operação               Árvore equilibrada  Caso médio aleatório  Degenerada
=====================  =================  ==================  =============
``insert``             O(log n)           O(log n)            O(n)
``search``             O(log n)           O(log n)            O(n)
``delete``             O(log n)           O(log n)            O(n)
``min_key``/``max_key`` O(log n)          O(log n)            O(n)
``in_order`` completa  O(n)               O(n)                O(n)
``k_smallest``         O(h + k)           O(log n + k)        O(n)
``level_order``        O(n)               O(n)                O(n)
=====================  =================  ==================  =============

Todas as linhas são O(h); o que muda entre as colunas é quanto vale ``h``.

Com ``n`` chaves em ordem **aleatória**, o comprimento de caminho interno
esperado é ``≈ 2n·ln n − 2,85n``, então construir a árvore custa
``≈ 2n·ln n − 1,85n = 1,386·n·log₂n − 1,85n`` comparações, e a altura esperada é
``≈ 4,31·ln n ≈ 2,99·log₂n``. O termo ``−1,85n`` não é detalhe nesses tamanhos:
para ``n ≤ 10⁴`` ele puxa a razão ``comparações/(n·log₂n)`` medida para a faixa
de 1,0 a 1,2, que só sobe para 1,386 muito devagar. O que prova a classe
Θ(n log n) é a razão ficar **presa numa faixa estreita** enquanto ``n`` cresce
dezenas de vezes — não bater num número específico.

Com chaves **já ordenadas**, ``h = n`` e o custo é exatamente ``n(n−1)/2``.
:func:`bench_insert_shapes` mede os dois lados; :func:`bench_degenerate` isola
a degeneração.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Generic, Iterable, Iterator, TypeVar

from src.counters import RANDOM_SEED, Counters, make_row, ratio, stopwatch
from src.stack_queue import Queue

K = TypeVar("K")
V = TypeVar("V")


# ----------------------------------------------------------------------
# Exceções próprias (convenção 9: nunca `raise Exception(...)`)
# ----------------------------------------------------------------------
class BSTError(Exception):
    """Erro base de :class:`BinarySearchTree`."""


class BSTKeyNotFoundError(BSTError):
    """A chave pedida não está na árvore."""


class EmptyTreeError(BSTError):
    """Operação que exige ao menos um nó foi chamada em árvore vazia."""


class BSTInvariantError(BSTError):
    """Invariante estrutural violada (bug de implementação, não de uso)."""


# ----------------------------------------------------------------------
# Nó
# ----------------------------------------------------------------------
@dataclass(slots=True)
class BSTNode(Generic[K, V]):
    """Nó da árvore: chave, valor, multiplicidade e os dois filhos.

    ``multiplicity`` é o que permite à mesma estrutura servir de mapa e de
    multiconjunto (decisão de projeto 2).
    """

    key: K
    value: V | None = None
    multiplicity: int = 1
    left: "BSTNode[K, V] | None" = None
    right: "BSTNode[K, V] | None" = None

    def is_leaf(self) -> bool:
        """Custo: O(1)."""
        return self.left is None and self.right is None

    def child_count(self) -> int:
        """Quantos filhos o nó tem: 0, 1 ou 2. Custo: O(1)."""
        return (self.left is not None) + (self.right is not None)

    def __repr__(self) -> str:
        sufixo = f"x{self.multiplicity}" if self.multiplicity > 1 else ""
        return f"BSTNode({self.key!r}{sufixo})"


# ----------------------------------------------------------------------
# Árvore
# ----------------------------------------------------------------------
class BinarySearchTree(Generic[K, V]):
    """Árvore binária de busca, sem rebalanceamento.

    **Não** é auto-balanceada de propósito: o exercício 3 pede para demonstrar
    a degeneração, e uma AVL ou rubro-negra tornaria o experimento impossível.
    O preço dessa escolha é medido, não escondido.

    Invariantes (ver :meth:`check_invariants`):

    * para todo nó ``x``: toda chave da subárvore esquerda é ``< x.key`` e toda
      chave da subárvore direita é ``> x.key`` (estrito: repetição vira
      multiplicidade, não nó novo);
    * ``_nodes`` é o número de nós alcançáveis a partir da raiz;
    * ``_size`` é a soma das multiplicidades;
    * nenhuma chave aparece em dois nós, e não há ciclo.
    """

    __slots__ = ("_root", "_nodes", "_size")

    def __init__(self, keys: Iterable[K] | None = None) -> None:
        """Cria a árvore, opcionalmente inserindo ``keys`` na ordem dada.

        A ordem importa e muito: ``BinarySearchTree(range(n))`` produz uma lista
        encadeada disfarçada de árvore. É justamente o cenário do ex 3.

        Custo: O(Σ h) — de O(n log n) a O(n²), conforme a ordem de chegada.
        """
        self._root: BSTNode[K, V] | None = None
        self._nodes = 0
        self._size = 0
        if keys is not None:
            for key in keys:
                self.insert(key)

    # -- leitura -------------------------------------------------------
    def __len__(self) -> int:
        """Total de inserções (soma das multiplicidades). Custo: O(1)."""
        return self._size

    @property
    def node_count(self) -> int:
        """Número de chaves distintas. Custo: O(1)."""
        return self._nodes

    @property
    def root(self) -> BSTNode[K, V] | None:
        """Raiz, para os exercícios que precisam navegar a estrutura (ex 3 e 8)."""
        return self._root

    def is_empty(self) -> bool:
        """Custo: O(1)."""
        return self._nodes == 0

    def __iter__(self) -> Iterator[K]:
        """Chaves em ordem crescente, cada uma uma vez. Custo: O(n)."""
        for node in self.in_order_nodes():
            yield node.key

    def __str__(self) -> str:
        """Chaves em ordem, entre colchetes. Custo: O(n)."""
        return "[" + ", ".join(repr(key) for key in self) + "]"

    def __repr__(self) -> str:
        return (
            f"BinarySearchTree(nos={self._nodes}, insercoes={self._size}, "
            f"altura={self.height()})"
        )

    # -- inserção ------------------------------------------------------
    def insert(
        self, key: K, value: V | None = None, counters: Counters | None = None
    ) -> bool:
        """Insere ``key``; devolve ``True`` se criou nó novo.

        Chave repetida incrementa a multiplicidade e atualiza o valor, sem criar
        nó (decisão de projeto 2).

        Custo: O(h) — O(log n) em árvore equilibrada, O(n) na degenerada.
        """
        counters = counters if counters is not None else Counters()
        if self._root is None:
            self._root = BSTNode(key, value)
            self._nodes = 1
            self._size = 1
            counters.count_copy()
            return True

        current = self._root
        while True:
            counters.count_comparison()  # sondagem de três vias
            if key == current.key:
                current.multiplicity += 1
                current.value = value
                self._size += 1
                counters.count_copy()
                return False
            if key < current.key:  # type: ignore[operator]
                if current.left is None:
                    current.left = BSTNode(key, value)
                    break
                current = current.left
            else:
                if current.right is None:
                    current.right = BSTNode(key, value)
                    break
                current = current.right
            counters.count_hop()
        self._nodes += 1
        self._size += 1
        counters.count_copy()
        return True

    def insert_counted(self, key: K, value: V | None = None) -> tuple[bool, Counters]:
        """:meth:`insert` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        return self.insert(key, value, counters), counters

    # -- busca ---------------------------------------------------------
    def find_node(
        self, key: K, counters: Counters | None = None
    ) -> BSTNode[K, V] | None:
        """Nó com a chave, ou ``None``. Custo: O(h)."""
        counters = counters if counters is not None else Counters()
        current = self._root
        while current is not None:
            counters.count_comparison()
            if key == current.key:
                return current
            current = (
                current.left if key < current.key else current.right  # type: ignore[operator]
            )
            if current is not None:
                counters.count_hop()
        return None

    def search(self, key: K, counters: Counters | None = None) -> V | None:
        """Valor da chave. Levanta :class:`BSTKeyNotFoundError` se ausente.

        Custo: O(h).
        """
        node = self.find_node(key, counters)
        if node is None:
            raise BSTKeyNotFoundError(
                f"chave {key!r} não está na árvore ({self._nodes} nó(s), "
                f"altura {self.height()})"
            )
        return node.value

    def search_counted(self, key: K) -> tuple[V | None, Counters]:
        """:meth:`search` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        return self.search(key, counters), counters

    def get_or(
        self, key: K, default: Any = None, counters: Counters | None = None
    ) -> Any:
        """Valor da chave, ou ``default`` — uma só descida. Custo: O(h)."""
        node = self.find_node(key, counters)
        return default if node is None else node.value

    def contains(self, key: K, counters: Counters | None = None) -> bool:
        """Custo: O(h)."""
        return self.find_node(key, counters) is not None

    def __contains__(self, key: object) -> bool:
        return self.contains(key)  # type: ignore[arg-type]

    def min_key(self, counters: Counters | None = None) -> K:
        """Menor chave. Custo: O(h) — desce sempre à esquerda."""
        if self._root is None:
            raise EmptyTreeError("min_key em árvore vazia")
        counters = counters if counters is not None else Counters()
        current = self._root
        while current.left is not None:
            current = current.left
            counters.count_hop()
        return current.key

    def max_key(self, counters: Counters | None = None) -> K:
        """Maior chave. Custo: O(h) — desce sempre à direita.

        Em árvore degenerada por entrada ordenada, esta é a operação mais cara
        possível: percorre os ``n`` nós da espinha.
        """
        if self._root is None:
            raise EmptyTreeError("max_key em árvore vazia")
        counters = counters if counters is not None else Counters()
        current = self._root
        while current.right is not None:
            current = current.right
            counters.count_hop()
        return current.key

    # -- remoção -------------------------------------------------------
    def delete(
        self, key: K, counters: Counters | None = None, *, all_copies: bool = False
    ) -> V | None:
        """Remove ``key`` e devolve o valor que estava guardado.

        Com multiplicidade > 1, remove **uma** cópia (decrementa) a menos que
        ``all_copies=True``. Os três casos estruturais estão cobertos:

        * **folha** — o pai passa a apontar para ``None``;
        * **um filho** — o pai passa a apontar direto para o neto;
        * **dois filhos** — o nó recebe a chave, o valor e a multiplicidade do
          **sucessor in-order** (o menor da subárvore direita), e o sucessor é
          removido no lugar. O sucessor tem no máximo um filho à direita por
          construção, o que reduz este caso ao anterior.

        Levanta :class:`BSTKeyNotFoundError` se a chave não existe — remover
        algo ausente é erro do chamador, não silêncio.

        Custo: O(h).
        """
        counters = counters if counters is not None else Counters()
        parent: BSTNode[K, V] | None = None
        current = self._root
        while current is not None:
            counters.count_comparison()
            if key == current.key:
                break
            parent = current
            current = (
                current.left if key < current.key else current.right  # type: ignore[operator]
            )
            if current is not None:
                counters.count_hop()
        if current is None:
            raise BSTKeyNotFoundError(
                f"delete({key!r}): chave não está na árvore "
                f"({self._nodes} nó(s))"
            )

        value = current.value
        if current.multiplicity > 1 and not all_copies:
            current.multiplicity -= 1
            self._size -= 1
            return value

        removed_copies = current.multiplicity

        if current.left is not None and current.right is not None:
            # dois filhos: assume a identidade do sucessor in-order
            successor_parent = current
            successor = current.right
            counters.count_hop()
            while successor.left is not None:
                successor_parent = successor
                successor = successor.left
                counters.count_hop()
            current.key = successor.key
            current.value = successor.value
            current.multiplicity = successor.multiplicity
            counters.count_copy()
            parent, current = successor_parent, successor

        child = current.left if current.left is not None else current.right
        if parent is None:
            self._root = child
        elif parent.left is current:
            parent.left = child
        else:
            parent.right = child
        current.left = current.right = None

        self._nodes -= 1
        self._size -= removed_copies
        return value

    def delete_counted(self, key: K) -> tuple[V | None, Counters]:
        """:meth:`delete` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        return self.delete(key, counters), counters

    # -- travessias (iterativas: decisão de projeto 1) ------------------
    def in_order_nodes(
        self, counters: Counters | None = None
    ) -> Iterator[BSTNode[K, V]]:
        """Nós em ordem crescente de chave. Custo: O(n) em tempo, O(h) de pilha.

        Gerador: quem consome parcialmente paga só pelo que consumiu — é o que
        torna :meth:`k_smallest` O(h + k) em vez de O(n).
        """
        counters = counters if counters is not None else Counters()
        stack: list[BSTNode[K, V]] = []
        current = self._root
        while stack or current is not None:
            while current is not None:
                stack.append(current)
                current = current.left
                if current is not None:
                    counters.count_hop()
            node = stack.pop()
            yield node
            current = node.right
            if current is not None:
                counters.count_hop()

    def in_order_items(
        self, counters: Counters | None = None
    ) -> Iterator[tuple[K, V | None]]:
        """Pares (chave, valor) em ordem crescente. Custo: O(n)."""
        for node in self.in_order_nodes(counters):
            yield node.key, node.value

    def in_order_keys(self, counters: Counters | None = None) -> Iterator[K]:
        """Chaves em ordem, **repetidas pela multiplicidade**. Custo: O(n + Σm).

        É esta travessia que faz :func:`bst_sort` reproduzir o multiconjunto de
        entrada, e não só o conjunto de chaves distintas.
        """
        for node in self.in_order_nodes(counters):
            for _ in range(node.multiplicity):
                yield node.key

    def pre_order_nodes(
        self, counters: Counters | None = None
    ) -> Iterator[BSTNode[K, V]]:
        """Nós em pré-ordem (raiz, esquerda, direita). Custo: O(n)."""
        counters = counters if counters is not None else Counters()
        if self._root is None:
            return
        stack: list[BSTNode[K, V]] = [self._root]
        while stack:
            node = stack.pop()
            yield node
            if node.right is not None:
                stack.append(node.right)
                counters.count_hop()
            if node.left is not None:
                stack.append(node.left)
                counters.count_hop()

    def level_order_nodes(
        self, counters: Counters | None = None
    ) -> Iterator[tuple[int, BSTNode[K, V]]]:
        """Pares (nível, nó) em largura, usando a :class:`Queue` do exercício 5.

        A raiz está no nível 0. É a travessia que o ex 12 usa para o relatório
        de diagnóstico (níveis, termos por nível, balanceamento aproximado).

        A fila tem capacidade fixa: ``_nodes`` é um limite superior seguro para
        a largura máxima de qualquer nível.

        Custo: O(n) em tempo, O(largura máxima) de memória.
        """
        counters = counters if counters is not None else Counters()
        if self._root is None:
            return
        queue: Queue[tuple[int, BSTNode[K, V]]] = Queue(self._nodes)
        queue.enqueue((0, self._root))
        while not queue.is_empty():
            level, node = queue.dequeue()
            yield level, node
            if node.left is not None:
                queue.enqueue((level + 1, node.left))
                counters.count_hop()
            if node.right is not None:
                queue.enqueue((level + 1, node.right))
                counters.count_hop()

    # -- consultas derivadas -------------------------------------------
    def k_smallest(
        self, k: int, counters: Counters | None = None
    ) -> list[K]:
        """As ``k`` menores chaves, em ordem — **travessia controlada**.

        Para de percorrer assim que junta ``k`` chaves, em vez de percorrer a
        árvore inteira e cortar. A diferença é assintótica, não de constante:
        O(h + k) contra O(n).

        Conta as multiplicidades: uma chave inserida três vezes ocupa três das
        ``k`` posições, coerente com :meth:`in_order_keys`.

        Custo: O(h + k).
        """
        if not isinstance(k, int) or isinstance(k, bool):
            raise ValueError(f"k deve ser int, recebido {type(k).__name__}")
        if k < 0:
            raise ValueError(f"k deve ser >= 0, recebido {k}")
        counters = counters if counters is not None else Counters()
        resultado: list[K] = []
        if k == 0:
            return resultado
        for key in self.in_order_keys(counters):
            resultado.append(key)
            if len(resultado) == k:
                break
        return resultado

    def k_smallest_counted(self, k: int) -> tuple[list[K], Counters]:
        """:meth:`k_smallest` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        return self.k_smallest(k, counters), counters

    def height(self) -> int:
        """Altura em nós (árvore vazia = 0, só raiz = 1). Custo: O(n).

        Iterativa: em árvore degenerada a altura é ``n``, e é exatamente essa
        árvore que os experimentos precisam medir.
        """
        if self._root is None:
            return 0
        maior = 0
        stack: list[tuple[BSTNode[K, V], int]] = [(self._root, 1)]
        while stack:
            node, depth = stack.pop()
            if depth > maior:
                maior = depth
            if node.left is not None:
                stack.append((node.left, depth + 1))
            if node.right is not None:
                stack.append((node.right, depth + 1))
        return maior

    def min_depth(self) -> int:
        """Profundidade da folha mais rasa. Custo: O(n).

        Junto com :meth:`height`, dá a medida grosseira de desequilíbrio que o
        relatório do ex 12 usa.
        """
        if self._root is None:
            return 0
        menor = math.inf
        stack: list[tuple[BSTNode[K, V], int]] = [(self._root, 1)]
        while stack:
            node, depth = stack.pop()
            if node.is_leaf():
                menor = min(menor, depth)
                continue
            if node.left is not None:
                stack.append((node.left, depth + 1))
            if node.right is not None:
                stack.append((node.right, depth + 1))
        return int(menor)

    def nodes_per_level(self) -> list[int]:
        """Quantos nós há em cada nível, da raiz para baixo. Custo: O(n)."""
        contagem: list[int] = []
        for level, _ in self.level_order_nodes():
            while len(contagem) <= level:
                contagem.append(0)
            contagem[level] += 1
        return contagem

    def balance_report(self) -> dict[str, Any]:
        """Diagnóstico de forma da árvore — a base do relatório do ex 12.

        ``altura_ideal`` é ``⌈log₂(n+1)⌉``, a altura de uma árvore completa com
        ``n`` nós. ``razao_altura = altura / altura_ideal`` é a medida de
        desequilíbrio: 1,0 é perfeito; inserção aleatória tende a
        ``4,31·ln n / log₂n ≈ 3,0`` (aproximado por baixo nos tamanhos deste
        trabalho); a degeneração dá ``n / log₂n``, que cresce sem limite.

        Custo: O(n).
        """
        n = self._nodes
        altura = self.height()
        ideal = math.ceil(math.log2(n + 1)) if n > 0 else 0
        niveis = self.nodes_per_level()
        folhas = sum(1 for node in self.pre_order_nodes() if node.is_leaf())
        return {
            "nos": n,
            "insercoes": self._size,
            "altura": altura,
            "altura_ideal": ideal,
            "razao_altura": (altura / ideal) if ideal else 0.0,
            "profundidade_minima": self.min_depth(),
            "folhas": folhas,
            "niveis": len(niveis),
            "nos_por_nivel": niveis,
            "nivel_mais_largo": max(niveis) if niveis else 0,
            "degenerada": altura == n and n > 1,
        }

    # -- verificação ---------------------------------------------------
    def check_invariants(self) -> None:
        """Valida as invariantes; levanta :class:`BSTInvariantError`. Custo: O(n).

        Verifica a propriedade de BST com **limites herdados** (min/max), e não
        só comparando cada nó com os filhos diretos: a comparação local passa em
        árvores que violam a ordenação global, que é o bug clássico desta
        estrutura.
        """
        if self._root is None:
            if self._nodes != 0 or self._size != 0:
                raise BSTInvariantError(
                    f"árvore sem raiz deveria ter 0 nós e 0 inserções, "
                    f"tem {self._nodes} e {self._size}"
                )
            return

        vistos: set[int] = set()
        chaves: set[Any] = set()
        total_nos = 0
        total_insercoes = 0
        pilha: list[tuple[BSTNode[K, V], Any, Any]] = [(self._root, None, None)]
        while pilha:
            node, menor, maior = pilha.pop()
            if id(node) in vistos:
                raise BSTInvariantError(
                    f"nó {node.key!r} alcançado duas vezes (ciclo ou nó compartilhado)"
                )
            vistos.add(id(node))

            if node.multiplicity < 1:
                raise BSTInvariantError(
                    f"multiplicidade de {node.key!r} deveria ser >= 1, "
                    f"é {node.multiplicity}"
                )
            if node.key in chaves:
                raise BSTInvariantError(
                    f"chave {node.key!r} aparece em mais de um nó"
                )
            chaves.add(node.key)

            if menor is not None and not node.key > menor:
                raise BSTInvariantError(
                    f"chave {node.key!r} deveria ser maior que {menor!r} "
                    "(limite herdado do ancestral)"
                )
            if maior is not None and not node.key < maior:
                raise BSTInvariantError(
                    f"chave {node.key!r} deveria ser menor que {maior!r} "
                    "(limite herdado do ancestral)"
                )

            total_nos += 1
            total_insercoes += node.multiplicity
            if node.left is not None:
                pilha.append((node.left, menor, node.key))
            if node.right is not None:
                pilha.append((node.right, node.key, maior))

        if total_nos != self._nodes:
            raise BSTInvariantError(
                f"_nodes={self._nodes} mas há {total_nos} nó(s) alcançáveis"
            )
        if total_insercoes != self._size:
            raise BSTInvariantError(
                f"_size={self._size} mas as multiplicidades somam {total_insercoes}"
            )


# ----------------------------------------------------------------------
# Ordenação por BST (consumida pelo exercício 2)
# ----------------------------------------------------------------------
def bst_sort(
    values: Iterable[K], counters: Counters | None = None
) -> tuple[list[K], BinarySearchTree[K, None]]:
    """Ordena inserindo tudo na BST e lendo em in-order.

    Devolve a lista ordenada **e** a árvore, para que o chamador possa medir a
    forma que a entrada produziu sem reconstruí-la.

    Custo: O(Σ h) para inserir mais O(n) para percorrer. Com entrada aleatória,
    ``≈ 1,39·n·log₂n`` comparações; com entrada já ordenada, exatamente
    ``n(n−1)/2`` — pior que o insertion sort no mesmo cenário, que faz ``n−1``.

    Memória: Θ(n) de nós, contra Θ(1) do insertion sort, que ordena no lugar.
    É o trade-off que o exercício 2 pede para discutir.
    """
    counters = counters if counters is not None else Counters()
    tree: BinarySearchTree[K, None] = BinarySearchTree()
    for value in values:
        tree.insert(value, None, counters)
    return list(tree.in_order_keys(counters)), tree


# ----------------------------------------------------------------------
# Experimentos
# ----------------------------------------------------------------------
def bench_insert_shapes(
    sizes: Iterable[int] = (125, 250, 500, 1_000, 2_000),
    patterns: Iterable[str] = ("aleatorio", "ordenado", "reverso"),
    *,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Custo de construir a árvore conforme a ORDEM de chegada das chaves.

    Previsões a conferir:

    * ``aleatorio`` — ``≈ 1,386·n·log₂n − 1,85n`` comparações e altura
      ``≈ 4,31·ln n ≈ 2,99·log₂n``. Por causa do termo ``−1,85n``, a razão
      ``comparacoes/(n·log₂n)`` medida fica em torno de 1,0–1,2 nestes tamanhos
      e sobe devagar rumo a 1,386; o que atesta Θ(n log n) é ela ficar **presa
      numa faixa estreita**, não bater num valor;
    * ``ordenado`` e ``reverso`` — a árvore vira uma espinha: altura ``n`` e
      exatamente ``n(n−1)/2`` comparações, então ``comparacoes/n²`` converge
      para 0,5.

    A mesma estrutura, os mesmos dados, só a ordem de inserção diferente — e a
    classe de complexidade muda de Θ(n log n) para Θ(n²).

    Custo: O(Σ n²) no pior padrão.
    """
    rows: list[dict[str, Any]] = []
    for n in sizes:
        if n <= 0:
            raise ValueError(f"tamanho deve ser positivo, recebido {n}")
        for pattern in patterns:
            if pattern == "ordenado":
                keys = list(range(n))
                teorico: int | None = n * (n - 1) // 2
            elif pattern == "reverso":
                keys = list(range(n - 1, -1, -1))
                teorico = n * (n - 1) // 2
            elif pattern == "aleatorio":
                keys = list(range(n))
                random.Random(seed).shuffle(keys)
                teorico = None
            else:
                raise ValueError(f"padrão desconhecido: {pattern!r}")

            counters = Counters()
            tree: BinarySearchTree[int, None] = BinarySearchTree()
            with stopwatch() as elapsed:
                for key in keys:
                    tree.insert(key, None, counters)
            tree.check_invariants()
            relatorio = tree.balance_report()

            rows.append(
                make_row(
                    exercise="ex2/ex3",
                    algorithm="BST — construção",
                    n=n,
                    pattern=pattern,
                    counters=counters,
                    metric="comparisons",
                    curves=("n", "n_log2n", "n2"),
                    elapsed_s=elapsed[0],
                    extra={
                        "altura": relatorio["altura"],
                        "altura_ideal": relatorio["altura_ideal"],
                        "razao_altura": relatorio["razao_altura"],
                        "degenerada": relatorio["degenerada"],
                        "comparacoes_teoricas": teorico,
                        "altura/log2n": (
                            relatorio["altura"] / math.log2(n) if n > 1 else float("nan")
                        ),
                    },
                )
            )
    return rows


def bench_degenerate(
    sizes: Iterable[int] = (125, 250, 500, 1_000, 2_000),
    *,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """A degeneração do exercício 3, operação por operação.

    Duas árvores com as MESMAS chaves, construídas em ordens diferentes, e três
    medições em cada: construção, busca pela maior chave (o pior caso de uma
    descida) e travessia in-order completa.

    A linha que fecha o argumento é a busca: na árvore aleatória custa
    ``≈ log₂n`` comparações; na degenerada custa exatamente ``n``, que é o custo
    de uma busca em **lista encadeada** — a árvore virou uma lista com sintaxe
    de árvore.

    A travessia in-order custa O(n) nas duas, e isso também é informação: nem
    toda operação degrada, só as que dependem da altura.

    Custo: O(Σ n²).
    """
    rows: list[dict[str, Any]] = []
    for n in sizes:
        if n <= 0:
            raise ValueError(f"tamanho deve ser positivo, recebido {n}")
        embaralhado = list(range(n))
        random.Random(seed).shuffle(embaralhado)

        for forma, keys in (
            ("aleatória", embaralhado),
            ("degenerada (entrada ordenada)", list(range(n))),
        ):
            construcao = Counters()
            tree: BinarySearchTree[int, None] = BinarySearchTree()
            for key in keys:
                tree.insert(key, None, construcao)
            tree.check_invariants()

            busca = Counters()
            tree.search(n - 1, busca)  # a maior chave: pior caso da descida

            travessia = Counters()
            with stopwatch() as elapsed:
                total = sum(1 for _ in tree.in_order_nodes(travessia))
            assert total == n

            altura = tree.height()
            rows.append(
                {
                    "exercicio": "ex3",
                    "algoritmo": "BST — degeneração",
                    "n": n,
                    "padrao": forma,
                    "forma": forma,
                    "altura": altura,
                    "altura/log2n": altura / math.log2(n) if n > 1 else float("nan"),
                    "altura/n": altura / n,
                    "comparacoes_construcao": construcao.comparisons,
                    "construcao/n_log2n": ratio(construcao.comparisons, n, "n_log2n"),
                    "construcao/n2": ratio(construcao.comparisons, n, "n2"),
                    "comparacoes_busca_maior": busca.comparisons,
                    "busca/log2n": ratio(busca.comparisons, n, "log2n"),
                    "busca/n": ratio(busca.comparisons, n, "n"),
                    "saltos_travessia": travessia.hops,
                    "tempo_travessia_s": elapsed[0],
                }
            )
    return rows


def bench_k_smallest(
    n: int = 10_000,
    ks: Iterable[int] = (1, 10, 100, 1_000, 10_000),
    *,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Travessia controlada: parar em ``k`` custa O(h + k), não O(n).

    Compara a travessia que para em ``k`` com a alternativa ingênua de percorrer
    a árvore inteira e cortar a lista depois. Os resultados são idênticos; o
    custo, não: a coluna ``saltos`` cresce com ``k`` em um caso e fica fixa em
    ``n`` no outro.

    Custo: O(Σ (h + k)).
    """
    if n <= 0:
        raise ValueError(f"tamanho deve ser positivo, recebido {n}")
    keys = list(range(n))
    random.Random(seed).shuffle(keys)
    tree: BinarySearchTree[int, None] = BinarySearchTree(keys)
    altura = tree.height()

    rows: list[dict[str, Any]] = []
    for k in ks:
        if k < 0:
            raise ValueError(f"k deve ser >= 0, recebido {k}")
        controlada = Counters()
        with stopwatch() as elapsed_controlada:
            parcial = tree.k_smallest(k, controlada)

        completa = Counters()
        with stopwatch() as elapsed_completa:
            tudo = list(tree.in_order_keys(completa))[:k]

        assert parcial == tudo == list(range(min(k, n)))
        rows.append(
            {
                "exercicio": "ex3",
                "algoritmo": "k_smallest_bst",
                "n": n,
                "padrao": "árvore aleatória",
                "k": k,
                "altura": altura,
                "saltos_controlada": controlada.hops,
                "saltos_completa": completa.hops,
                "saltos_teoricos_controlada": None,
                "razao_saltos": (
                    completa.hops / controlada.hops if controlada.hops else float("nan")
                ),
                "tempo_controlada_s": elapsed_controlada[0],
                "tempo_completa_s": elapsed_completa[0],
                "economia": 1 - (controlada.hops / completa.hops)
                if completa.hops
                else 0.0,
            }
        )
    return rows

"""Exercício 7 — Recursão sobre árvore de diretórios em memória.

Decisões de projeto
-------------------
1. **Nada toca o sistema de arquivos real.** A árvore é construída em memória,
   como o enunciado exige, e o motivo é reprodutibilidade: um notebook que lê
   ``C:\\Users`` produz saída diferente em cada máquina e em cada dia, e a
   seção 5 do CLAUDE.md exige que notebook, PDF e vídeo mostrem os mesmos
   números.

2. **Os filhos ficam numa ``SinglyLinkedList``, não num dicionário.**
   Diretório tem poucos filhos, então a busca linear O(k) por nível é aceitável,
   e a lista **preserva a ordem de inserção** — o que faz ``walk`` ser
   determinístico sem precisar ordenar nada. Com ``HashTableChained`` a busca
   por nível cairia para O(1) e a ordem se perderia (viraria ordem de bucket).
   A otimização fica registrada com seu preço: o gargalo desta estrutura é a
   **profundidade**, não a largura, e :func:`bench_search_by_width` mede a
   largura para mostrar quando a troca passaria a valer.

3. **``delete`` de nó interno com filhos exige ``recursive=True``.**
   Apagar em cascata por padrão transforma um erro de digitação em perda
   irreversível de uma subárvore inteira. Exigir a bandeira obriga o chamador a
   declarar a intenção — é a mesma razão de ``rm -r`` existir separado de ``rm``.
   O erro levantado diz **quantos** descendentes seriam removidos, para que a
   decisão de repetir com a bandeira seja informada.

4. **``walk`` é recursivo, como o enunciado pede** — com caso base explícito
   (nó sem filhos) e composição dos resultados dos filhos. Isso custa Θ(d) de
   pilha de chamadas, ``d`` = profundidade, e portanto tem teto: uma hierarquia
   com mais de ~1000 níveis levanta ``RecursionError``. :func:`walk_iterative`
   existe como contraponto medido, e a comparação é a mesma do exercício 5.

5. **O caminho atual é mantido numa lista encadeada com push/pop na CABEÇA.**
   É exigência do enunciado, e a escolha da ponta não é arbitrária: numa lista
   simplesmente encadeada só a cabeça tem remoção O(1) (a lição do exercício
   11). O efeito colateral é que o caminho fica guardado de trás para frente e
   montar a string exige percorrer e inverter — O(d), que é o mesmo custo que a
   ``list`` do Python pagaria para montar a string, então nesse ponto empatam.
   :class:`ListPathStack` é o grupo de controle, e
   :func:`bench_path_stack` mede as duas.

Custo em Big O
--------------
Com ``n`` nós, profundidade ``d`` e largura máxima de diretório ``k``:

=========================  ===============  ==================================
Operação                   Custo            Observação
=========================  ===============  ==================================
``insert(path)``           O(d · k)         desce d níveis, busca linear em cada
``search(path)``           O(d · k)         idem
``delete`` (folha)         O(d · k)         achar o pai e desligar
``delete(recursive)``      O(d·k + m)       m = nós da subárvore removida
``walk``                   Θ(n · d)         cada caminho completo tem O(d) chars
``PathStack.push/pop``     O(1)             nas duas implementações
``PathStack.to_path``      O(d)             nas duas implementações
=========================  ===============  ==================================

``walk`` é Θ(n·d) e não Θ(n) porque o resultado são **strings de caminho
completo**: cada um dos ``n`` caminhos tem comprimento O(d). Montar a saída é
mais caro que visitar a árvore, e isso costuma passar despercebido.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from src.counters import Counters, ratio, stopwatch
from src.linked_list import SinglyLinkedList

#: Separador de caminho. Barra normal mesmo no Windows: a árvore é uma estrutura
#: em memória, não um caminho do sistema operacional.
SEPARATOR: str = "/"

#: Tipos de nó aceitos.
DIRECTORY: str = "dir"
FILE: str = "file"
NODE_TYPES: tuple[str, str] = (DIRECTORY, FILE)


# ----------------------------------------------------------------------
# Exceções próprias (convenção 9)
# ----------------------------------------------------------------------
class DirectoryTreeError(Exception):
    """Erro base da árvore de diretórios."""


class InvalidPathError(DirectoryTreeError):
    """Caminho malformado (vazio, sem barra inicial, com segmento vazio…)."""


class InvalidNodeTypeError(DirectoryTreeError):
    """Tipo de nó fora de ``("dir", "file")``."""


class PathNotFoundError(DirectoryTreeError):
    """O caminho pedido não existe na árvore."""


class PathExistsError(DirectoryTreeError):
    """Já existe um nó nesse caminho."""


class ParentNotFoundError(DirectoryTreeError):
    """O diretório pai do caminho não existe."""


class NotADirectoryTreeError(DirectoryTreeError):
    """Um segmento intermediário do caminho é arquivo, não diretório.

    Nome com sufixo ``Tree`` de propósito: ``NotADirectoryError`` é embutido do
    Python e sombreá-lo esconderia erros reais do sistema operacional.
    """


class DirectoryNotEmptyError(DirectoryTreeError):
    """Remoção de diretório com filhos sem ``recursive=True`` (decisão 3)."""


class DirectoryTreeInvariantError(DirectoryTreeError):
    """Invariante estrutural violada (bug de implementação, não de uso)."""


# ----------------------------------------------------------------------
# Nó
# ----------------------------------------------------------------------
class DirNode:
    """Nó da árvore: nome, tipo e filhos.

    Os filhos ficam numa :class:`~src.linked_list.SinglyLinkedList` (decisão de
    projeto 2), o que preserva a ordem de inserção e faz a busca por nome custar
    O(k) no número de filhos.
    """

    __slots__ = ("name", "node_type", "children")

    def __init__(self, name: str, node_type: str = DIRECTORY) -> None:
        """Custo: O(1)."""
        if node_type not in NODE_TYPES:
            raise InvalidNodeTypeError(
                f"tipo deve ser um de {NODE_TYPES}, recebido {node_type!r}"
            )
        self.name = name
        self.node_type = node_type
        self.children: SinglyLinkedList[DirNode] = SinglyLinkedList()

    def is_dir(self) -> bool:
        """Custo: O(1)."""
        return self.node_type == DIRECTORY

    def is_file(self) -> bool:
        """Custo: O(1)."""
        return self.node_type == FILE

    def is_leaf(self) -> bool:
        """Sem filhos. Um diretório vazio também é folha. Custo: O(1)."""
        return len(self.children) == 0

    def child_count(self) -> int:
        """Custo: O(1)."""
        return len(self.children)

    def find_child(
        self, name: str, counters: Counters | None = None
    ) -> "DirNode | None":
        """Filho de nome ``name``, ou ``None``. Custo: O(k).

        Conta 1 comparação por filho examinado e 1 salto por avanço — as mesmas
        unidades do exercício 10, para que o custo por nível seja comparável
        com o de qualquer outra travessia do trabalho.
        """
        counters = counters if counters is not None else Counters()
        primeiro = True
        for child in self.children:
            if not primeiro:
                counters.count_hop()
            primeiro = False
            if counters.eq(child.name, name):
                return child
        return None

    def __repr__(self) -> str:
        marca = "/" if self.is_dir() else ""
        return f"DirNode({self.name!r}{marca}, filhos={len(self.children)})"


# ----------------------------------------------------------------------
# Caminho atual: duas implementações de pilha
# ----------------------------------------------------------------------
class LinkedPathStack:
    """Caminho atual mantido numa **lista encadeada** (exigência do enunciado).

    ``push`` e ``pop`` operam na **cabeça**, que é a única ponta com remoção
    O(1) numa lista simplesmente encadeada (decisão de projeto 5). Em troca, os
    segmentos ficam guardados de trás para frente e :meth:`to_path` precisa
    inverter — O(d), que é o mesmo custo de montar a string de qualquer jeito.
    """

    __slots__ = ("_items",)

    def __init__(self, segments: Iterable[str] | None = None) -> None:
        """Custo: O(k) para k segmentos."""
        self._items: SinglyLinkedList[str] = SinglyLinkedList()
        if segments is not None:
            for segment in segments:
                self.push(segment)

    def push(self, segment: str, counters: Counters | None = None) -> None:
        """Entra num nível. Custo: O(1) — insere na cabeça."""
        self._items.insert_first(segment, counters)

    def pop(self, counters: Counters | None = None) -> str:
        """Sai do nível atual. Custo: O(1) — remove da cabeça."""
        return self._items.delete_at(0, counters)

    def __len__(self) -> int:
        """Profundidade atual. Custo: O(1)."""
        return len(self._items)

    def is_empty(self) -> bool:
        """Custo: O(1)."""
        return self._items.is_empty()

    def segments(self, counters: Counters | None = None) -> list[str]:
        """Segmentos da raiz para baixo. Custo: O(d) — inverte a lista."""
        counters = counters if counters is not None else Counters()
        invertido: list[str] = []
        primeiro = True
        for segment in self._items:
            if not primeiro:
                counters.count_hop()
            primeiro = False
            invertido.append(segment)
        invertido.reverse()
        return invertido

    def to_path(self, counters: Counters | None = None) -> str:
        """Caminho completo. Custo: O(d)."""
        return SEPARATOR + SEPARATOR.join(self.segments(counters))

    def check_invariants(self) -> None:
        """Valida a lista encadeada subjacente. Custo: O(d)."""
        self._items.check_invariants()


class ListPathStack:
    """Mesma API sobre a ``list`` do Python — o grupo de controle.

    ``append``/``pop`` no fim são O(1) **amortizado** (a lista realoca de vez em
    quando), contra O(1) de pior caso da versão encadeada. Em compensação, a
    memória é contígua e não há um nó alocado por segmento.
    """

    __slots__ = ("_items",)

    def __init__(self, segments: Iterable[str] | None = None) -> None:
        """Custo: O(k)."""
        self._items: list[str] = list(segments) if segments is not None else []

    def push(self, segment: str, counters: Counters | None = None) -> None:
        """Custo: O(1) amortizado."""
        self._items.append(segment)
        if counters is not None:
            counters.count_copy()

    def pop(self, counters: Counters | None = None) -> str:
        """Custo: O(1).

        Não conta cópia, pelo mesmo critério do exercício 10 que
        :class:`LinkedPathStack` herda: inserção conta 1, remoção conta 0. Se as
        duas implementações contassem diferente, a coluna de cópias do
        experimento compararia critérios em vez de estruturas.
        """
        return self._items.pop()

    def __len__(self) -> int:
        """Custo: O(1)."""
        return len(self._items)

    def is_empty(self) -> bool:
        """Custo: O(1)."""
        return not self._items

    def segments(self, counters: Counters | None = None) -> list[str]:
        """Custo: O(d) — já está na ordem certa, só copia."""
        return list(self._items)

    def to_path(self, counters: Counters | None = None) -> str:
        """Custo: O(d)."""
        return SEPARATOR + SEPARATOR.join(self._items)

    def check_invariants(self) -> None:
        """Nada a validar: a ``list`` do Python cuida de si. Custo: O(1)."""
        return None


#: As duas implementações de pilha de caminho, para os experimentos.
PATH_STACKS: dict[str, Any] = {
    "lista encadeada": LinkedPathStack,
    "list do Python": ListPathStack,
}


# ----------------------------------------------------------------------
# Caminhos
# ----------------------------------------------------------------------
def split_path(path: str) -> list[str]:
    """``"/a/b/c"`` → ``["a", "b", "c"]``; ``"/"`` → ``[]``. Custo: O(len(path)).

    Levanta :class:`InvalidPathError` para caminho vazio, sem barra inicial ou
    com segmento vazio no meio (``"/a//b"``). Validar aqui, uma vez, evita que
    cada operação repita a checagem e discorde da outra.
    """
    if not isinstance(path, str):
        raise InvalidPathError(
            f"caminho deve ser str, recebido {type(path).__name__}"
        )
    if not path:
        raise InvalidPathError("caminho vazio")
    if not path.startswith(SEPARATOR):
        raise InvalidPathError(
            f"caminho deve começar com {SEPARATOR!r}, recebido {path!r}"
        )
    if path == SEPARATOR:
        return []
    corpo = path[1:]
    if corpo.endswith(SEPARATOR):
        corpo = corpo[:-1]
    segments = corpo.split(SEPARATOR)
    for segment in segments:
        if not segment:
            raise InvalidPathError(f"segmento vazio em {path!r}")
        if segment in (".", ".."):
            raise InvalidPathError(
                f"segmento {segment!r} não é suportado: a árvore é literal, "
                "sem resolução de caminho relativo"
            )
    return segments


def _child_path(parent: str, name: str) -> str:
    """Caminho do filho ``name`` sob ``parent``. Custo: O(len(parent))."""
    return f"{parent}{name}" if parent == SEPARATOR else f"{parent}{SEPARATOR}{name}"


def join_path(segments: Iterable[str]) -> str:
    """``["a", "b"]`` → ``"/a/b"``; lista vazia → ``"/"``. Custo: O(d)."""
    partes = list(segments)
    return SEPARATOR + SEPARATOR.join(partes) if partes else SEPARATOR


# ----------------------------------------------------------------------
# walk recursivo (exigência do enunciado)
# ----------------------------------------------------------------------
def walk(
    root: DirNode,
    counters: Counters | None = None,
    *,
    path_stack_class: Any = LinkedPathStack,
) -> list[str]:
    """Caminhos completos de toda a árvore, em **pré-ordem**, recursivamente.

    Caso base: um nó sem filhos contribui só com o próprio caminho. Caso
    recursivo: o caminho do nó, seguido da concatenação dos resultados de cada
    filho, na ordem de inserção.

    O caminho corrente é mantido numa pilha (decisão de projeto 5): ``push``
    antes de descer, ``pop`` ao voltar. ``path_stack_class`` permite trocar a
    implementação — é assim que :func:`bench_path_stack` compara a lista
    encadeada com a ``list`` do Python sem duplicar o algoritmo.

    Contagens: ``calls`` = nós visitados; ``copies`` e ``hops`` vêm das
    operações de pilha e da montagem dos caminhos.

    Custo: Θ(n · d) — n nós, cada caminho completo com O(d) caracteres.
    Memória: Θ(d) de pilha de chamadas, e é aí que está o teto (decisão 4).
    """
    counters = counters if counters is not None else Counters()
    stack = path_stack_class()
    resultado: list[str] = []

    def visit(node: DirNode) -> None:
        counters.count_call()
        # A raiz tem nome vazio e caminho "/": empilhá-la produziria "//projeto".
        # Só ela pode ter nome vazio (a invariante garante), então o próprio
        # nome serve de teste, sem precisar de uma bandeira "sou a raiz".
        empilhou = bool(node.name)
        if empilhou:
            stack.push(node.name, counters)
        resultado.append(stack.to_path(counters))
        if node.is_leaf():  # caso base explícito
            if empilhou:
                stack.pop(counters)
            return
        for child in node.children:
            visit(child)
        if empilhou:
            stack.pop(counters)

    visit(root)
    return resultado


def walk_iterative(root: DirNode, counters: Counters | None = None) -> list[str]:
    """Mesma pré-ordem, sem recursão — o contraponto medido da decisão 4.

    Produz exatamente a mesma lista que :func:`walk`, e a igualdade é testada.
    A diferença é onde a memória mora: aqui numa pilha explícita no heap, lá na
    pilha de chamadas do Python, que tem teto de ``sys.getrecursionlimit()``.

    Custo: Θ(n · d) em tempo, Θ(d) de memória no heap.
    """
    counters = counters if counters is not None else Counters()
    resultado: list[str] = []
    pendentes: list[tuple[DirNode, str]] = [(root, "")]
    while pendentes:
        node, prefixo = pendentes.pop()
        counters.count_call()
        caminho = f"{prefixo}{SEPARATOR}{node.name}" if node.name else SEPARATOR
        resultado.append(caminho if caminho else SEPARATOR)
        filhos = node.children.to_list()
        for child in reversed(filhos):  # pilha inverte: empilha ao contrário
            pendentes.append((child, "" if caminho == SEPARATOR else caminho))
    return resultado


# ----------------------------------------------------------------------
# Árvore de diretórios
# ----------------------------------------------------------------------
class DirectoryTree:
    """Árvore de diretórios e arquivos em memória.

    A raiz é um diretório de nome vazio, cujo caminho é ``"/"``.

    Invariantes (ver :meth:`check_invariants`):

    * a raiz é diretório e tem nome vazio;
    * nenhum diretório tem dois filhos com o mesmo nome;
    * arquivo nunca tem filhos;
    * nenhum nome de nó contém o separador;
    * ``_size`` é o número de nós **fora** da raiz.
    """

    __slots__ = ("_root", "_size")

    def __init__(self) -> None:
        """Custo: O(1)."""
        self._root = DirNode("", DIRECTORY)
        self._size = 0

    # -- leitura -------------------------------------------------------
    @property
    def root(self) -> DirNode:
        """Nó raiz. Custo: O(1)."""
        return self._root

    def __len__(self) -> int:
        """Nós fora a raiz. Custo: O(1)."""
        return self._size

    def is_empty(self) -> bool:
        """Custo: O(1)."""
        return self._size == 0

    def __contains__(self, path: object) -> bool:
        try:
            self.search(str(path))
        except DirectoryTreeError:
            return False
        return True

    def __str__(self) -> str:
        """Árvore em texto indentado. Custo: Θ(n · d)."""
        return self.to_tree_string()

    def __repr__(self) -> str:
        return f"DirectoryTree(nos={self._size}, profundidade={self.depth()})"

    # -- navegação -----------------------------------------------------
    def _descend(
        self, segments: list[str], path: str, counters: Counters
    ) -> DirNode:
        """Desce pelos segmentos, validando cada nível. Custo: O(d · k)."""
        node = self._root
        for posicao, segment in enumerate(segments):
            if node.is_file():
                parcial = join_path(segments[:posicao])
                raise NotADirectoryTreeError(
                    f"{parcial!r} é arquivo, então {path!r} não pode existir "
                    "abaixo dele"
                )
            filho = node.find_child(segment, counters)
            if filho is None:
                raise PathNotFoundError(
                    f"{path!r} não existe: o segmento {segment!r} não está em "
                    f"{join_path(segments[:posicao])!r}"
                )
            node = filho
        return node

    def search(self, path: str, counters: Counters | None = None) -> DirNode:
        """Nó do caminho. Levanta :class:`PathNotFoundError` se não existir.

        Custo: O(d · k) — desce ``d`` níveis e faz busca linear de O(k) em cada
        (decisão de projeto 2).
        """
        counters = counters if counters is not None else Counters()
        return self._descend(split_path(path), path, counters)

    def search_counted(self, path: str) -> tuple[DirNode, Counters]:
        """:meth:`search` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        return self.search(path, counters), counters

    def exists(self, path: str, counters: Counters | None = None) -> bool:
        """``True`` se o caminho existe. Custo: O(d · k)."""
        try:
            self.search(path, counters)
        except DirectoryTreeError:
            return False
        return True

    def depth(self) -> int:
        """Profundidade máxima em níveis abaixo da raiz. Custo: Θ(n).

        Iterativa: a árvore funda do experimento de recursão precisa ser
        mensurável sem estourar a pilha.
        """
        maior = 0
        pendentes: list[tuple[DirNode, int]] = [(self._root, 0)]
        while pendentes:
            node, nivel = pendentes.pop()
            if nivel > maior:
                maior = nivel
            for child in node.children:
                pendentes.append((child, nivel + 1))
        return maior

    # -- inserção ------------------------------------------------------
    def insert(
        self,
        path: str,
        node_type: str = DIRECTORY,
        counters: Counters | None = None,
        *,
        create_parents: bool = False,
    ) -> DirNode:
        """Cria o nó em ``path``.

        Validações de consistência, todas com mensagem que nomeia o problema:

        * caminho malformado → :class:`InvalidPathError`;
        * tipo inválido → :class:`InvalidNodeTypeError`;
        * já existe → :class:`PathExistsError`;
        * pai inexistente → :class:`ParentNotFoundError` (a menos que
          ``create_parents=True``, que cria os diretórios faltantes);
        * um segmento intermediário é arquivo → :class:`NotADirectoryTreeError`.

        Custo: O(d · k).
        """
        if node_type not in NODE_TYPES:
            raise InvalidNodeTypeError(
                f"tipo deve ser um de {NODE_TYPES}, recebido {node_type!r}"
            )
        segments = split_path(path)
        if not segments:
            raise PathExistsError("a raiz já existe e não pode ser inserida")
        counters = counters if counters is not None else Counters()

        node = self._root
        for posicao, segment in enumerate(segments[:-1]):
            if node.is_file():
                raise NotADirectoryTreeError(
                    f"{join_path(segments[:posicao])!r} é arquivo; "
                    f"{path!r} não pode ser criado abaixo dele"
                )
            filho = node.find_child(segment, counters)
            if filho is None:
                if not create_parents:
                    raise ParentNotFoundError(
                        f"não é possível criar {path!r}: o diretório "
                        f"{join_path(segments[: posicao + 1])!r} não existe. "
                        "Use create_parents=True para criá-lo."
                    )
                filho = DirNode(segment, DIRECTORY)
                node.children.insert_last(filho, counters)
                self._size += 1
            node = filho

        if node.is_file():
            raise NotADirectoryTreeError(
                f"{join_path(segments[:-1])!r} é arquivo; "
                f"{path!r} não pode ser criado abaixo dele"
            )
        ultimo = segments[-1]
        if node.find_child(ultimo, counters) is not None:
            raise PathExistsError(f"{path!r} já existe")

        novo = DirNode(ultimo, node_type)
        node.children.insert_last(novo, counters)
        self._size += 1
        return novo

    def insert_counted(
        self, path: str, node_type: str = DIRECTORY
    ) -> tuple[DirNode, Counters]:
        """:meth:`insert` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        return self.insert(path, node_type, counters), counters

    # -- remoção -------------------------------------------------------
    def delete(
        self,
        path: str,
        counters: Counters | None = None,
        *,
        recursive: bool = False,
    ) -> int:
        """Remove o nó de ``path``; devolve quantos nós saíram.

        **Política de remoção (decisão de projeto 3):** um diretório com filhos
        só é removido com ``recursive=True``. Sem a bandeira, levanta
        :class:`DirectoryNotEmptyError` informando quantos descendentes seriam
        apagados — a ideia é que a decisão de insistir seja tomada com o número
        na frente, não no escuro.

        Arquivo e diretório vazio são removidos sem bandeira. A raiz nunca é
        removida.

        Custo: O(d · k) para achar o pai; mais O(m) para contar e soltar a
        subárvore de ``m`` nós quando ``recursive=True``.
        """
        segments = split_path(path)
        if not segments:
            raise DirectoryTreeError("a raiz não pode ser removida")
        counters = counters if counters is not None else Counters()

        pai = self._descend(segments[:-1], join_path(segments[:-1]), counters)
        if pai.is_file():
            raise NotADirectoryTreeError(
                f"{join_path(segments[:-1])!r} é arquivo e não tem filhos"
            )
        alvo = pai.find_child(segments[-1], counters)
        if alvo is None:
            raise PathNotFoundError(f"{path!r} não existe")

        descendentes = _count_subtree(alvo)
        if not alvo.is_leaf() and not recursive:
            raise DirectoryNotEmptyError(
                f"{path!r} tem {alvo.child_count()} filho(s) diretos e "
                f"{descendentes - 1} descendente(s) no total. Remover em "
                "cascata apaga todos eles; passe recursive=True se é isso "
                "mesmo que você quer."
            )

        pai.children.delete(alvo, counters)
        self._size -= descendentes
        return descendentes

    def delete_counted(
        self, path: str, *, recursive: bool = False
    ) -> tuple[int, Counters]:
        """:meth:`delete` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        return self.delete(path, counters, recursive=recursive), counters

    # -- saídas --------------------------------------------------------
    def walk(
        self, counters: Counters | None = None, *, path_stack_class: Any = LinkedPathStack
    ) -> list[str]:
        """Todos os caminhos em pré-ordem. Custo: Θ(n · d)."""
        caminhos = walk(self._root, counters, path_stack_class=path_stack_class)
        return caminhos

    def to_tree_string(self) -> str:
        """Árvore indentada, para exibição no notebook. Custo: Θ(n · d)."""
        linhas: list[str] = []
        pendentes: list[tuple[DirNode, int]] = [(self._root, 0)]
        while pendentes:
            node, nivel = pendentes.pop()
            nome = SEPARATOR if not node.name else node.name
            marca = SEPARATOR if node.is_dir() and node.name else ""
            linhas.append("  " * nivel + nome + marca)
            for child in reversed(node.children.to_list()):
                pendentes.append((child, nivel + 1))
        return "\n".join(linhas)

    # -- verificação ---------------------------------------------------
    def check_invariants(self) -> None:
        """Valida as invariantes; levanta :class:`DirectoryTreeInvariantError`.

        Custo: Θ(n · k).
        """
        if not self._root.is_dir():
            raise DirectoryTreeInvariantError("a raiz precisa ser diretório")
        if self._root.name != "":
            raise DirectoryTreeInvariantError(
                f"a raiz precisa ter nome vazio, tem {self._root.name!r}"
            )

        total = 0
        vistos: set[int] = set()
        pendentes: list[DirNode] = [self._root]
        while pendentes:
            node = pendentes.pop()
            if id(node) in vistos:
                raise DirectoryTreeInvariantError(
                    f"nó {node.name!r} alcançado duas vezes (ciclo)"
                )
            vistos.add(id(node))
            node.children.check_invariants()

            if node.is_file() and not node.is_leaf():
                raise DirectoryTreeInvariantError(
                    f"arquivo {node.name!r} não pode ter filhos"
                )
            nomes: set[str] = set()
            for child in node.children:
                if SEPARATOR in child.name:
                    raise DirectoryTreeInvariantError(
                        f"nome de nó não pode conter {SEPARATOR!r}: "
                        f"{child.name!r}"
                    )
                if not child.name:
                    raise DirectoryTreeInvariantError(
                        f"filho de {node.name!r} com nome vazio"
                    )
                if child.name in nomes:
                    raise DirectoryTreeInvariantError(
                        f"{node.name!r} tem dois filhos chamados {child.name!r}"
                    )
                nomes.add(child.name)
                pendentes.append(child)
                total += 1

        if total != self._size:
            raise DirectoryTreeInvariantError(
                f"_size={self._size} mas há {total} nó(s) fora da raiz"
            )


def _count_subtree(node: DirNode) -> int:
    """Nós da subárvore, incluindo ``node``. Custo: Θ(m), iterativa."""
    total = 0
    pendentes: list[DirNode] = [node]
    while pendentes:
        atual = pendentes.pop()
        total += 1
        for child in atual.children:
            pendentes.append(child)
    return total


# ----------------------------------------------------------------------
# Construtores para os experimentos
# ----------------------------------------------------------------------
def build_balanced_tree(
    depth: int, branching: int, *, files_per_dir: int = 0
) -> DirectoryTree:
    """Árvore com ``branching`` subdiretórios por nível, até ``depth`` níveis.

    Total de diretórios: ``(b^(d+1) − 1)/(b − 1) − 1`` fora a raiz. Cresce
    rápido — ``depth=6, branching=3`` já dá mais de mil nós.

    Custo: Θ(n · d).
    """
    if depth < 0:
        raise ValueError(f"depth deve ser >= 0, recebido {depth}")
    if branching < 1:
        raise ValueError(f"branching deve ser >= 1, recebido {branching}")
    if files_per_dir < 0:
        raise ValueError(f"files_per_dir deve ser >= 0, recebido {files_per_dir}")

    tree = DirectoryTree()
    niveis: list[str] = [SEPARATOR]
    for nivel in range(depth):
        proximos: list[str] = []
        for pai in niveis:
            for indice in range(branching):
                caminho = (
                    f"{pai}d{nivel}_{indice}"
                    if pai == SEPARATOR
                    else f"{pai}{SEPARATOR}d{nivel}_{indice}"
                )
                tree.insert(caminho, DIRECTORY)
                proximos.append(caminho)
        niveis = proximos
    if files_per_dir:
        for caminho in niveis:
            for indice in range(files_per_dir):
                tree.insert(f"{caminho}{SEPARATOR}f{indice}.txt", FILE)
    return tree


def build_deep_tree(depth: int) -> DirectoryTree:
    """Corrente de ``depth`` diretórios aninhados — ``/n1/n2/.../nd``.

    É a árvore que quebra o :func:`walk` recursivo quando ``depth`` passa do
    limite de recursão do Python. Custo: Θ(d²) por causa da montagem dos
    caminhos.
    """
    if depth < 0:
        raise ValueError(f"depth deve ser >= 0, recebido {depth}")
    tree = DirectoryTree()
    caminho = ""
    for nivel in range(1, depth + 1):
        caminho = f"{caminho}{SEPARATOR}n{nivel}"
        tree.insert(caminho, DIRECTORY)
    return tree


# ----------------------------------------------------------------------
# Experimentos do exercício 7
# ----------------------------------------------------------------------
def bench_walk(
    depths: Iterable[int] = (2, 3, 4, 5, 6),
    *,
    branching: int = 3,
) -> list[dict[str, Any]]:
    """Custo do ``walk`` recursivo em árvores de tamanho crescente.

    Previsão: ``calls`` é exatamente o número de nós (cada um é visitado uma
    vez), e o **tempo** cresce como Θ(n·d) e não Θ(n), porque montar cada
    caminho completo custa O(d). A coluna ``caracteres_gerados`` deixa isso
    explícito: é ela, e não ``calls``, que acompanha o tempo.

    Custo: Θ(Σ n·d).
    """
    rows: list[dict[str, Any]] = []
    for depth in depths:
        if depth < 0:
            raise ValueError(f"depth deve ser >= 0, recebido {depth}")
        tree = build_balanced_tree(depth, branching)
        tree.check_invariants()
        counters = Counters()
        with stopwatch() as elapsed:
            caminhos = tree.walk(counters)
        nos = len(tree) + 1
        caracteres = sum(len(caminho) for caminho in caminhos)
        rows.append(
            {
                "exercicio": "ex7",
                "algoritmo": "walk recursivo",
                "n": nos,
                "padrao": f"árvore balanceada, ramificação {branching}",
                "profundidade": depth,
                "ramificacao": branching,
                "caminhos": len(caminhos),
                **counters.as_dict(),
                "calls_igual_a_nos": counters.calls == nos,
                "caracteres_gerados": caracteres,
                "caracteres_por_no": caracteres / nos,
                "tempo_s": elapsed[0],
                "tempo_us_por_no": elapsed[0] * 1e6 / nos,
                "tempo_us_por_caractere": elapsed[0] * 1e6 / caracteres,
            }
        )
    return rows


def bench_path_stack(
    depths: Iterable[int] = (100, 400, 1_600, 6_400),
    *,
    repeats: int = 3,
) -> list[dict[str, Any]]:
    """Caminho em lista encadeada × em ``list`` do Python.

    Exercita ``d`` ``push`` seguidos de ``d`` ``pop``, mais uma montagem de
    caminho no ponto mais fundo. As duas implementações são O(1) por operação e
    O(d) para montar o caminho — a diferença é de constante, não de classe, e o
    experimento existe para dizer **de quanto** é essa constante em vez de
    afirmar "lista encadeada é mais lenta".

    A coluna ``hops`` só existe do lado encadeado: a ``list`` não segue ponteiro
    nenhum para montar o caminho, porque os segmentos já estão contíguos e na
    ordem certa.

    Custo: O(repeats · Σ d).
    """
    if repeats < 1:
        raise ValueError(f"repeats deve ser >= 1, recebido {repeats}")
    rows: list[dict[str, Any]] = []
    for depth in depths:
        if depth <= 0:
            raise ValueError(f"profundidade deve ser positiva, recebido {depth}")
        segmentos = [f"n{nivel}" for nivel in range(depth)]
        for nome, classe in PATH_STACKS.items():
            counters = Counters()
            melhor = float("inf")
            for tentativa in range(repeats):
                run_counters = counters if tentativa == 0 else Counters()
                pilha = classe()
                with stopwatch() as elapsed:
                    for segmento in segmentos:
                        pilha.push(segmento, run_counters)
                    caminho = pilha.to_path(run_counters)
                    for _ in range(depth):
                        pilha.pop(run_counters)
                melhor = min(melhor, elapsed[0])
                assert len(pilha) == 0
                assert caminho == SEPARATOR + SEPARATOR.join(segmentos)
            pilha_final = classe(segmentos)
            pilha_final.check_invariants()
            rows.append(
                {
                    "exercicio": "ex7",
                    "algoritmo": f"pilha de caminho ({nome})",
                    "n": depth,
                    "padrao": f"{depth} push, 1 to_path, {depth} pop",
                    "implementacao": nome,
                    **counters.as_dict(),
                    "copies/n": ratio(counters.copies, depth, "n"),
                    "hops/n": ratio(counters.hops, depth, "n"),
                    "tempo_s": melhor,
                    "tempo_us_por_operacao": melhor * 1e6 / (2 * depth + 1),
                    "repeticoes": repeats,
                }
            )
    return rows


def bench_search_by_width(
    widths: Iterable[int] = (4, 16, 64, 256, 1_024),
    *,
    depth: int = 4,
) -> list[dict[str, Any]]:
    """Busca por caminho conforme a **largura** dos diretórios.

    Cada nível tem ``width`` entradas e a busca procura a última delas, que é o
    pior caso da varredura linear da :class:`~src.linked_list.SinglyLinkedList`
    (decisão de projeto 2).

    Previsão: ``comparações ≈ d · k``, e a razão ``comparações/(d·k)`` fica
    presa perto de 1. É a evidência de que trocar os filhos por uma
    ``HashTableChained`` mudaria a busca de O(d·k) para O(d) — otimização que
    só compensa com diretórios largos, e esta tabela diz a partir de qual
    largura.

    Custo: O(Σ d·k).
    """
    rows: list[dict[str, Any]] = []
    for width in widths:
        if width < 1:
            raise ValueError(f"largura deve ser >= 1, recebido {width}")
        tree = DirectoryTree()
        caminho_alvo_partes: list[str] = []
        pai = SEPARATOR
        for nivel in range(depth):
            for indice in range(width):
                tree.insert(_child_path(pai, f"e{indice}"), DIRECTORY)
            ultimo = f"e{width - 1}"
            caminho_alvo_partes.append(ultimo)
            pai = _child_path(pai, ultimo)

        alvo = join_path(caminho_alvo_partes)
        counters = Counters()
        with stopwatch() as elapsed:
            tree.search(alvo, counters)
        previsto = depth * width
        rows.append(
            {
                "exercicio": "ex7",
                "algoritmo": "search por caminho",
                "n": len(tree),
                "padrao": f"{depth} níveis × {width} entradas",
                "profundidade": depth,
                "largura": width,
                **counters.as_dict(),
                "comparacoes_previstas_d_vezes_k": previsto,
                "comparacoes/(d*k)": counters.comparisons / previsto,
                "comparacoes_se_fosse_hash": depth,
                "ganho_potencial": counters.comparisons / depth,
                "tempo_s": elapsed[0],
            }
        )
    return rows

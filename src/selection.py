"""Exercício 3 — Otimização orientada a Big O.

Dois pares "ingênuo × otimizado", cada um com a corretude provada por teste e o
ganho provado por contagem:

* ``deduplicate_slow`` O(n²) → ``deduplicate_fast`` O(n);
* ``k_smallest`` versão A (ordenar tudo) O(n log n) → versão B (quickselect)
  O(n + k log k), mais a versão C em BST, O(h + k).

Decisões de projeto
-------------------
1. **A deduplicação rápida usa a ``HashTableChained`` do exercício 4, não um
   ``set``.** A convenção 9 proíbe estrutura pronta onde o enunciado pede
   implementação própria, e há um ganho colateral: as comparações feitas dentro
   dos buckets são **contadas**, então o "O(n)" da versão rápida é medido no
   mesmo contador que o "O(n²)" da lenta. Com ``set``, o custo das buscas
   sumiria da tabela e a comparação viraria O(n²) contra "zero", que não é honesto.

2. **As duas deduplicações preservam a ordem do primeiro aparecimento.** É
   exigência do enunciado para a rápida, mas vale para a lenta também: se as
   duas não produzissem exatamente a mesma saída, os testes de equivalência não
   provariam nada sobre a otimização.

3. **A versão B devolve os ``k`` menores ordenados**, como a versão A, para que
   a equivalência seja verificável elemento a elemento. O quickselect por si só
   entrega os ``k`` menores **desordenados** no prefixo; ordenar esses ``k``
   custa O(k log k), e é esse termo que aparece na análise. Sem ordenar, a
   versão B seria ainda mais barata — e incomparável com a A.

4. **``k_smallest_bst`` recebe o nó raiz**, como o enunciado escreve, e não a
   árvore. É iterativa pelo mesmo motivo do módulo ``bst``: o pior caso que o
   exercício manda demonstrar é a árvore degenerada, de altura ``n``, e uma
   travessia recursiva estouraria a pilha antes de produzir a medição.

Custo em Big O
--------------
======================================  ==================  ==================
Operação                                Caso médio          Pior caso
======================================  ==================  ==================
``deduplicate_slow``                    Θ(n·u)              Θ(n²)
``deduplicate_fast``                    Θ(n)                Θ(n²) (colisão)
``k_smallest`` A (ordenar)              Θ(n log n)          Θ(n²)
``k_smallest`` B (quickselect)          Θ(n + k log k)      Θ(n²)
``k_smallest_bst`` C                    Θ(log n + k)        Θ(n)
======================================  ==================  ==================

``u`` é o número de valores distintos: a versão lenta compara cada elemento com
todos os já aceitos, então o custo real é ``Θ(n·u)`` e só vira ``Θ(n²)`` quando
quase tudo é distinto. Vale registrar porque é o caso em que a otimização mais
rende — e também o caso em que a versão lenta parece aceitável em teste com
muitas repetições.

O pior caso de cada linha tem causa diferente, e isso é parte da análise:
a hashtable degrada por colisão adversarial (ex 4), o quickselect por partição
desbalanceada (ex 9), e a BST por ordem de inserção (ex 2). Nenhum dos três é
"lento às vezes": cada um tem uma condição nomeável que o derruba.
"""

from __future__ import annotations

import math
import random
from typing import Any, Iterable, Sequence, TypeVar

from src.bst import BinarySearchTree, BSTNode
from src.counters import RANDOM_SEED, Counters, ratio, stopwatch
from src.hashtable import HashTableChained
from src.searching import generate_array
from src.sorting import quickselect_partitioned, quicksort

T = TypeVar("T")


# ----------------------------------------------------------------------
# Exceções próprias (convenção 9)
# ----------------------------------------------------------------------
class SelectionError(Exception):
    """Erro base deste módulo."""


class InvalidKError(SelectionError):
    """``k`` fora da faixa ``0 <= k <= len(arr)``."""


# ----------------------------------------------------------------------
# Deduplicação
# ----------------------------------------------------------------------
def deduplicate_slow(
    arr: Sequence[T], counters: Counters | None = None
) -> list[T]:
    """Deduplicação ingênua: para cada elemento, varre os já aceitos.

    Preserva a ordem do primeiro aparecimento (decisão de projeto 2).

    Contagens: uma comparação por elemento já aceito que for examinado. Com
    ``u`` valores distintos no resultado, o total é ``Θ(n·u)`` — no máximo
    ``n(n−1)/2`` quando tudo é distinto.

    Custo: Θ(n²) no pior caso, Θ(1) de memória além da saída.
    """
    counters = counters if counters is not None else Counters()
    resultado: list[T] = []
    for value in arr:
        ja_existe = False
        for aceito in resultado:
            if counters.eq(aceito, value):
                ja_existe = True
                break
        if not ja_existe:
            resultado.append(value)
            counters.count_copy()
    return resultado


def deduplicate_fast(
    arr: Sequence[T], counters: Counters | None = None
) -> list[T]:
    """Deduplicação em tempo linear, com a ``HashTableChained`` do exercício 4.

    Troca a varredura linear por uma consulta de custo ``O(1 + α)`` na tabela
    hash — e o redimensionamento mantém ``α ≤ 0,75``, então a consulta é O(1)
    amortizada e o total é Θ(n).

    Preserva a ordem do primeiro aparecimento: a tabela só responde "já vi?",
    quem define a ordem é a lista de saída.

    Contagens: as comparações feitas **dentro dos buckets** entram no mesmo
    contador da versão lenta (decisão de projeto 1), então as duas colunas são
    diretamente comparáveis.

    Custo: Θ(n) no caso médio, Θ(n) de memória extra (a tabela). O pior caso é
    Θ(n²), e tem nome: chaves que colidem no mesmo bucket — o cenário
    adversarial medido no exercício 4.
    """
    counters = counters if counters is not None else Counters()
    vistos: HashTableChained[Any, bool] = HashTableChained()
    resultado: list[T] = []
    for value in arr:
        # put_if_absent faz UMA travessia do bucket. Com `contains` seguido de
        # `put` seriam duas, e o número medido dobraria sem que o algoritmo
        # ficasse pior de verdade — instrumentação errada vira análise errada.
        if vistos.put_if_absent(value, True, counters):
            resultado.append(value)
            counters.count_copy()
    return resultado


def deduplicate_slow_counted(arr: Sequence[T]) -> tuple[list[T], Counters]:
    """:func:`deduplicate_slow` no protocolo ``(resultado, Counters)``."""
    counters = Counters()
    return deduplicate_slow(arr, counters), counters


def deduplicate_fast_counted(arr: Sequence[T]) -> tuple[list[T], Counters]:
    """:func:`deduplicate_fast` no protocolo ``(resultado, Counters)``."""
    counters = Counters()
    return deduplicate_fast(arr, counters), counters


# ----------------------------------------------------------------------
# k menores — três versões
# ----------------------------------------------------------------------
def _validate_k(k: int, n: int) -> None:
    """Valida ``k`` contra o tamanho da entrada. Custo: O(1)."""
    if not isinstance(k, int) or isinstance(k, bool):
        raise InvalidKError(f"k deve ser int, recebido {type(k).__name__}")
    if k < 0:
        raise InvalidKError(f"k deve ser >= 0, recebido {k}")
    if k > n:
        raise InvalidKError(f"k={k} maior que o tamanho da entrada ({n})")


def k_smallest_sort(
    arr: Sequence[T], k: int, counters: Counters | None = None
) -> list[T]:
    """Versão A: ordena tudo e devolve os ``k`` primeiros.

    Faz trabalho que ninguém pediu — ordenar os ``n−k`` elementos que não vão
    ser usados. É a solução correta e ineficiente que o exercício manda otimizar.

    Custo: Θ(n log n) no caso médio, independentemente de ``k``. Repare que o
    custo **não cai** quando ``k = 1``: essa insensibilidade a ``k`` é o próprio
    defeito.
    """
    _validate_k(k, len(arr))
    counters = counters if counters is not None else Counters()
    if k == 0:
        return []
    return quicksort(arr, 16, counters=counters)[:k]


def k_smallest_quickselect(
    arr: Sequence[T],
    k: int,
    counters: Counters | None = None,
    *,
    seed: int = RANDOM_SEED,
) -> list[T]:
    """Versão B: quickselect particiona, e só os ``k`` menores são ordenados.

    Duas etapas, e a conta de cada uma:

    1. ``quickselect`` leva o ``k``-ésimo menor à posição ``k−1`` e, de quebra,
       deixa os ``k`` menores no prefixo — Θ(n) no caso médio;
    2. ordenar esse prefixo de ``k`` elementos — O(k log k) (decisão 3).

    Total Θ(n + k log k). Para ``k`` pequeno isso é Θ(n): melhor **classe** que
    a versão A, não só melhor constante. Para ``k = n`` as duas coincidem, e é
    assim que tem de ser.

    Custo: Θ(n + k log k) no caso médio; Θ(n²) no pior caso de partição.
    """
    _validate_k(k, len(arr))
    counters = counters if counters is not None else Counters()
    if k == 0:
        return []
    _, particionado = quickselect_partitioned(
        arr, k - 1, seed=seed, counters=counters
    )
    return quicksort(particionado[:k], 16, seed=seed, counters=counters)


def k_smallest_bst(
    root: BSTNode[T, Any] | None, k: int, counters: Counters | None = None
) -> list[T]:
    """Versão C: travessia in-order **controlada**, a partir do nó raiz.

    Desce até o menor elemento e sobe visitando em ordem, parando assim que
    junta ``k`` chaves. Não percorre a árvore inteira nem ordena nada — a
    árvore já é a ordenação.

    Recebe o nó raiz (decisão de projeto 4) e é iterativa, para funcionar
    também na árvore degenerada de altura ``n`` que o exercício manda demonstrar.

    Multiplicidade conta: uma chave inserida três vezes ocupa três das ``k``
    posições, coerente com ``BinarySearchTree.in_order_keys``.

    Custo: Θ(h + k), onde ``h`` é a altura. Em árvore equilibrada,
    Θ(log n + k); na degenerada, Θ(n) — e o exercício pede para medir os dois.
    """
    if not isinstance(k, int) or isinstance(k, bool):
        raise InvalidKError(f"k deve ser int, recebido {type(k).__name__}")
    if k < 0:
        raise InvalidKError(f"k deve ser >= 0, recebido {k}")
    counters = counters if counters is not None else Counters()
    resultado: list[T] = []
    if k == 0 or root is None:
        return resultado

    pilha: list[BSTNode[T, Any]] = []
    atual: BSTNode[T, Any] | None = root
    while pilha or atual is not None:
        while atual is not None:
            pilha.append(atual)
            atual = atual.left
            if atual is not None:
                counters.count_hop()
        node = pilha.pop()
        for _ in range(node.multiplicity):
            resultado.append(node.key)
            if len(resultado) == k:
                return resultado
        atual = node.right
        if atual is not None:
            counters.count_hop()
    return resultado


def k_smallest_sort_counted(arr: Sequence[T], k: int) -> tuple[list[T], Counters]:
    """Versão A no protocolo ``(resultado, Counters)``."""
    counters = Counters()
    return k_smallest_sort(arr, k, counters), counters


def k_smallest_quickselect_counted(
    arr: Sequence[T], k: int
) -> tuple[list[T], Counters]:
    """Versão B no protocolo ``(resultado, Counters)``."""
    counters = Counters()
    return k_smallest_quickselect(arr, k, counters), counters


def k_smallest_bst_counted(
    root: BSTNode[T, Any] | None, k: int
) -> tuple[list[T], Counters]:
    """Versão C no protocolo ``(resultado, Counters)``."""
    counters = Counters()
    return k_smallest_bst(root, k, counters), counters


# ----------------------------------------------------------------------
# Experimentos do exercício 3
# ----------------------------------------------------------------------
def bench_deduplicate(
    sizes: Iterable[int] = (500, 1_000, 2_000, 4_000),
    distinct_ratios: Iterable[float] = (0.05, 0.5, 1.0),
    *,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Θ(n·u) contra Θ(n), variando o tamanho e a fração de valores distintos.

    ``distinct_ratios`` controla ``u/n``. A coluna importa porque o custo da
    versão lenta é ``Θ(n·u)``, não ``Θ(n²)`` cego: com 5% de distintos ela
    parece aceitável, com 100% ela é quadrática de verdade. Medir só o caso
    fácil esconderia o problema — que é exatamente o erro que o exercício 3
    manda corrigir.

    Previsão: ``lenta/n²`` converge para ``(u/n)/2`` e ``rapida/n`` fica
    aproximadamente constante.

    Custo: O(Σ n²) na versão lenta.
    """
    rows: list[dict[str, Any]] = []
    for n in sizes:
        if n <= 0:
            raise ValueError(f"tamanho deve ser positivo, recebido {n}")
        for fracao in distinct_ratios:
            if not 0.0 < fracao <= 1.0:
                raise ValueError(
                    f"fração de distintos deve estar em (0, 1], recebido {fracao}"
                )
            distintos = max(1, int(fracao * n))
            # `value % distintos` garante exatamente `distintos` valores
            # distintos. Sortear com repetição daria u ≈ 0,63n mesmo pedindo
            # 100%, e a coluna u_distintos mentiria.
            arr = [value % distintos for value in range(n)]
            random.Random(seed).shuffle(arr)

            lenta_counters = Counters()
            with stopwatch() as lenta_tempo:
                lenta = deduplicate_slow(arr, lenta_counters)

            rapida_counters = Counters()
            with stopwatch() as rapida_tempo:
                rapida = deduplicate_fast(arr, rapida_counters)

            if lenta != rapida:
                raise AssertionError(
                    f"as duas deduplicações discordaram (n={n}, fração={fracao})"
                )

            u = len(lenta)
            rows.append(
                {
                    "exercicio": "ex3",
                    "algoritmo": "deduplicate: lenta × rápida",
                    "n": n,
                    "padrao": f"{fracao:.0%} de valores distintos",
                    "fracao_distintos": fracao,
                    "u_distintos": u,
                    "lenta_comparacoes": lenta_counters.comparisons,
                    "lenta/n2": ratio(lenta_counters.comparisons, n, "n2"),
                    "lenta/(n*u)": lenta_counters.comparisons / (n * u),
                    "rapida_comparacoes": rapida_counters.comparisons,
                    "rapida/n": ratio(rapida_counters.comparisons, n, "n"),
                    "razao_lenta_sobre_rapida": (
                        lenta_counters.comparisons / rapida_counters.comparisons
                        if rapida_counters.comparisons
                        else math.inf
                    ),
                    "tempo_lenta_s": lenta_tempo[0],
                    "tempo_rapida_s": rapida_tempo[0],
                    "razao_tempo": (
                        lenta_tempo[0] / rapida_tempo[0] if rapida_tempo[0] else math.inf
                    ),
                    "memoria_extra_lenta": "Θ(1)",
                    "memoria_extra_rapida": f"Θ(u) — tabela com {u} chaves",
                }
            )
    return rows


def bench_k_smallest(
    sizes: Iterable[int] = (1_000, 4_000, 16_000, 64_000),
    ks: Iterable[int] = (1, 10, 100, 1_000),
    *,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Versão A (ordenar tudo) × versão B (quickselect), variando ``n`` e ``k``.

    Previsão: a versão A **não muda** com ``k`` — é sempre Θ(n log n), e essa
    insensibilidade é o defeito. A versão B custa Θ(n + k log k), então para
    ``k`` pequeno a razão ``comparações/n`` fica presa perto de uma constante.

    Custo: O(Σ n log n) por combinação, dominado pela versão A.
    """
    rows: list[dict[str, Any]] = []
    for n in sizes:
        if n <= 0:
            raise ValueError(f"tamanho deve ser positivo, recebido {n}")
        arr = generate_array(n, "aleatorio", seed=seed)
        for k in ks:
            if k > n:
                continue
            versao_a = Counters()
            with stopwatch() as tempo_a:
                resultado_a = k_smallest_sort(arr, k, versao_a)

            versao_b = Counters()
            with stopwatch() as tempo_b:
                resultado_b = k_smallest_quickselect(arr, k, versao_b, seed=seed)

            if resultado_a != resultado_b:
                raise AssertionError(
                    f"versões A e B discordaram (n={n}, k={k})"
                )
            rows.append(
                {
                    "exercicio": "ex3",
                    "algoritmo": "k_smallest: A (ordenar) × B (quickselect)",
                    "n": n,
                    "padrao": "aleatorio",
                    "k": k,
                    "k_relativo": k / n,
                    "A_comparacoes": versao_a.comparisons,
                    "A/n_log2n": ratio(versao_a.comparisons, n, "n_log2n"),
                    "A/n": ratio(versao_a.comparisons, n, "n"),
                    "B_comparacoes": versao_b.comparisons,
                    "B/n": ratio(versao_b.comparisons, n, "n"),
                    "B/n_log2n": ratio(versao_b.comparisons, n, "n_log2n"),
                    "razao_A_sobre_B": (
                        versao_a.comparisons / versao_b.comparisons
                        if versao_b.comparisons
                        else math.inf
                    ),
                    "tempo_A_s": tempo_a[0],
                    "tempo_B_s": tempo_b[0],
                }
            )
        del arr
    return rows


def bench_k_smallest_bst(
    sizes: Iterable[int] = (250, 500, 1_000, 2_000),
    ks: Iterable[int] = (1, 10, 100),
    *,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Versão C em três formas de árvore — e o pior caso que o enunciado pede.

    As três formas vêm da mesma lista de chaves, só muda a ordem de inserção:

    * ``aleatória`` — altura ``≈ 3·log₂n``; ``k_smallest_bst`` custa Θ(log n + k);
    * ``crescente`` — a árvore é uma espinha **à direita**. O menor elemento é a
      própria raiz, então descer até ele é O(1) e a travessia continua barata.
      Degenerar nem sempre dói na mesma operação;
    * ``decrescente`` — espinha **à esquerda**. Agora o menor elemento está no
      fundo: só chegar nele custa ``n−1`` saltos, e ``k_smallest_bst(1)``, que
      deveria ser a operação mais barata possível, vira Θ(n).

    É esse par crescente/decrescente que torna a demonstração honesta: "a BST
    degenerou" não é conclusão suficiente, é preciso dizer **qual** operação
    degrada e por quê.

    Custo: O(Σ n²) para construir as árvores degeneradas.
    """
    rows: list[dict[str, Any]] = []
    for n in sizes:
        if n <= 0:
            raise ValueError(f"tamanho deve ser positivo, recebido {n}")
        embaralhado = list(range(n))
        random.Random(seed).shuffle(embaralhado)
        formas: list[tuple[str, list[int]]] = [
            ("aleatória", embaralhado),
            ("degenerada crescente", list(range(n))),
            ("degenerada decrescente", list(range(n - 1, -1, -1))),
        ]
        for forma, keys in formas:
            tree: BinarySearchTree[int, None] = BinarySearchTree(keys)
            altura = tree.height()
            for k in ks:
                if k > n:
                    continue
                counters = Counters()
                with stopwatch() as elapsed:
                    resultado = k_smallest_bst(tree.root, k, counters)
                if resultado != list(range(k)):
                    raise AssertionError(
                        f"k_smallest_bst errou em {forma!r}, n={n}, k={k}"
                    )
                rows.append(
                    {
                        "exercicio": "ex3",
                        "algoritmo": "k_smallest_bst",
                        "n": n,
                        "padrao": forma,
                        "forma": forma,
                        "k": k,
                        "altura": altura,
                        "altura/log2n": (
                            altura / math.log2(n) if n > 1 else float("nan")
                        ),
                        "saltos": counters.hops,
                        "saltos_previstos_h_mais_k": altura + k,
                        "saltos/n": counters.hops / n,
                        "saltos/(log2n+k)": (
                            counters.hops / (math.log2(n) + k) if n > 1 else float("nan")
                        ),
                        "degenerada": altura == n and n > 1,
                        "tempo_s": elapsed[0],
                    }
                )
            del tree
    return rows

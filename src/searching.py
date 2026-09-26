"""Exercício 1 — Busca linear, busca binária e o impacto da organização.

Decisões de projeto
-------------------
1. **Uma sondagem = uma comparação de chaves, nas três buscas.**
   A busca binária poderia contar duas comparações por iteração (``==`` e
   depois ``<``). Aqui ela conta **uma** comparação de três vias por elemento
   inspecionado, exatamente como a busca linear conta uma por elemento visitado.
   Sem esse critério comum, "a binária faz 20 comparações e a linear 500.000"
   compararia unidades diferentes, e o exercício inteiro é sobre comparar as
   duas. O mesmo critério vale para a busca sobre lista encadeada.

2. **A verificação de pré-condição da busca binária é amostrada por padrão.**
   Verificar por completo que o vetor está ordenado custa ``n-1`` comparações —
   ou seja, O(n). Uma busca binária O(log n) precedida de uma verificação O(n)
   é O(n) no total, e nesse ponto a busca linear resolve o mesmo problema sem
   pré-condição nenhuma. Verificar por completo **destrói a razão de existir da
   busca binária**.
   Por isso o padrão é ``check="sampled"``: ``k`` pares adjacentes sorteados,
   O(k). Com ``k`` constante o total continua O(log n). Os três modos existem e
   são medidos em :func:`bench_fast_fail`:

   ==============  ==========  ==================================================
   ``check``       Custo       Garantia
   ==============  ==========  ==================================================
   ``"off"``       O(1)        nenhuma; contrato é do chamador
   ``"sampled"``   O(k)        detecta desordem grosseira com alta probabilidade
   ``"full"``      O(n)        certeza, ao preço de tornar a busca O(n)
   ==============  ==========  ==================================================

3. **Falha rápida falha alto.** Quando a verificação detecta desordem, a busca
   levanta :class:`UnsortedInputError` com o índice do par violado, em vez de
   devolver ``-1``. ``-1`` significa "não encontrei", e "o vetor está errado" é
   outra coisa: confundir os dois esconde bug do chamador.

4. **A busca em lista encadeada não reimplementa a lista.** Ela percorre a
   :class:`~src.linked_list.SinglyLinkedList` do exercício 10 pelo iterador
   público e conta por conta própria. Um teste prova que suas contagens são
   idênticas, comparação a comparação e salto a salto, às de
   ``SinglyLinkedList.search`` — é o que autoriza dizer que a diferença medida
   contra o array é só de custo prático, não de algoritmo.

5. **Cópias sempre aparecem na tabela, mesmo valendo zero.** Nenhuma das buscas
   move dados: as três fazem 0 cópias. A coluna fica no DataFrame assim mesmo,
   porque coluna ausente parece instrumentação faltando, e o item 1.2 da rubrica
   cobra "comparações **e** cópias".

Custo em Big O
--------------
=================================  ==========  ==========  ==================
Busca                              Melhor      Médio       Pior
=================================  ==========  ==========  ==================
``linear_search`` (array)          O(1)        O(n)        O(n)
``linear_search_linked`` (lista)   O(1)        O(n)        O(n)
``binary_search``                  O(1)        O(log n)    O(log n)
=================================  ==========  ==========  ==================

Array e lista encadeada fazem **o mesmo número de comparações** — o algoritmo é
o mesmo. O que muda é o custo de chegar ao próximo elemento: no array é
aritmética de ponteiro sobre memória contígua; na lista é seguir um ponteiro
para um nó alocado em outro lugar, o que custa um salto e estraga a localidade
de cache. :func:`bench_array_vs_linked` mede as duas coisas lado a lado.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence, TypeVar

from src.counters import RANDOM_SEED, Counters, ratio, stopwatch
from src.linked_list import SinglyLinkedList

T = TypeVar("T")

#: Padrões de entrada aceitos por :func:`generate_array`. Os três primeiros são
#: os que o exercício 1 pede; ``quase_ordenado`` existe para o exercício 2, que
#: reaproveita este gerador em vez de escrever o seu.
PATTERNS: tuple[str, ...] = ("ordenado", "reverso", "aleatorio", "quase_ordenado")

#: Modos de verificação da pré-condição da busca binária.
CHECK_MODES: tuple[str, ...] = ("off", "sampled", "full")

#: Número padrão de pares adjacentes amostrados na verificação rápida.
DEFAULT_SAMPLE_SIZE: int = 16


# ----------------------------------------------------------------------
# Exceções próprias (convenção 9: nunca `raise Exception(...)`)
# ----------------------------------------------------------------------
class SearchError(Exception):
    """Erro base deste módulo."""


class UnsortedInputError(SearchError):
    """A pré-condição de ordenação da busca binária falhou (decisão 3)."""


class InvalidPatternError(SearchError):
    """Padrão de entrada desconhecido em :func:`generate_array`."""


class InvalidCheckModeError(SearchError):
    """Modo de verificação desconhecido em :func:`binary_search`."""


# ----------------------------------------------------------------------
# Gerador de vetores
# ----------------------------------------------------------------------
def generate_array(
    n: int,
    pattern: str,
    *,
    seed: int = RANDOM_SEED,
    disorder: float = 0.05,
) -> list[int]:
    """Vetor de ``n`` inteiros distintos no padrão pedido.

    Os valores são sempre ``0..n-1``, só a ordem muda. Serem **distintos** não é
    detalhe: com repetições, "o índice encontrado" deixaria de ser único e os
    testes de corretude precisariam de ressalvas a cada asserção.

    * ``ordenado`` — crescente;
    * ``reverso`` — decrescente (pior caso da busca binária: a pré-condição
      falha logo no primeiro par);
    * ``aleatorio`` — embaralhado com ``random.Random(seed)``;
    * ``quase_ordenado`` — crescente com ``disorder·n`` trocas de pares
      adjacentes sorteados (usado pelo exercício 2).

    Custo: O(n).
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise InvalidPatternError(f"n deve ser int, recebido {type(n).__name__}")
    if n < 0:
        raise InvalidPatternError(f"n deve ser >= 0, recebido {n}")
    if pattern not in PATTERNS:
        raise InvalidPatternError(
            f"padrão desconhecido: {pattern!r}; disponíveis: {list(PATTERNS)}"
        )

    values = list(range(n))
    if pattern == "ordenado":
        return values
    if pattern == "reverso":
        values.reverse()
        return values

    rng = random.Random(seed)
    if pattern == "aleatorio":
        rng.shuffle(values)
        return values

    # quase_ordenado
    if not 0.0 <= disorder <= 1.0:
        raise InvalidPatternError(
            f"disorder deve estar em [0, 1], recebido {disorder}"
        )
    swaps = int(disorder * n)
    for _ in range(swaps):
        if n < 2:
            break
        index = rng.randrange(n - 1)
        values[index], values[index + 1] = values[index + 1], values[index]
    return values


# ----------------------------------------------------------------------
# Busca linear em array
# ----------------------------------------------------------------------
def linear_search(
    arr: Sequence[T], target: T, counters: Counters | None = None
) -> int:
    """Índice da primeira ocorrência de ``target``, ou ``-1``.

    Contagens: ``i+1`` comparações quando o alvo está em ``i``; ``n`` quando
    está ausente. Nenhum salto de ponteiro (o array é contíguo) e nenhuma cópia.

    Custo: O(1) no melhor caso, O(n) no médio e no pior.
    """
    counters = counters if counters is not None else Counters()
    for index, value in enumerate(arr):
        if counters.eq(value, target):
            return index
    return -1


def linear_search_counted(arr: Sequence[T], target: T) -> tuple[int, Counters]:
    """:func:`linear_search` no protocolo ``(resultado, Counters)``."""
    counters = Counters()
    return linear_search(arr, target, counters), counters


# ----------------------------------------------------------------------
# Busca linear em lista encadeada
# ----------------------------------------------------------------------
def linear_search_linked(
    items: SinglyLinkedList[T], target: T, counters: Counters | None = None
) -> int:
    """A mesma busca linear, agora percorrendo uma lista encadeada.

    Mesmo algoritmo, mesmas comparações — e um salto de ponteiro por avanço, que
    o array não paga. É essa a diferença que o exercício pede para explicar.

    Contagens: ``i+1`` comparações e ``i`` saltos quando o alvo está em ``i``;
    ``n`` comparações e ``n`` saltos quando está ausente. Idênticas, por
    construção, às de ``SinglyLinkedList.search`` — e há teste provando.

    Custo: O(1) no melhor caso, O(n) no médio e no pior.
    """
    counters = counters if counters is not None else Counters()
    index = 0
    for value in items:
        if counters.eq(value, target):
            return index
        counters.count_hop()  # avançar para o próximo nó custa um ponteiro
        index += 1
    return -1


def linear_search_linked_counted(
    items: SinglyLinkedList[T], target: T
) -> tuple[int, Counters]:
    """:func:`linear_search_linked` no protocolo ``(resultado, Counters)``."""
    counters = Counters()
    return linear_search_linked(items, target, counters), counters


# ----------------------------------------------------------------------
# Verificação de pré-condição
# ----------------------------------------------------------------------
def find_first_descent(
    arr: Sequence[T], counters: Counters | None = None
) -> int:
    """Menor ``i`` com ``arr[i] > arr[i+1]``, ou ``-1`` se o vetor é crescente.

    Verificação **completa**: ``n-1`` comparações no pior caso.

    Custo: O(n).
    """
    counters = counters if counters is not None else Counters()
    for index in range(len(arr) - 1):
        if counters.gt(arr[index], arr[index + 1]):
            return index
    return -1


def is_sorted(arr: Sequence[T], counters: Counters | None = None) -> bool:
    """``True`` se ``arr`` é não decrescente. Custo: O(n)."""
    return find_first_descent(arr, counters) == -1


@dataclass(slots=True)
class SortSample:
    """Resultado da verificação amostrada de ordenação.

    ``looks_sorted`` é uma afirmação **probabilística**: significa "nenhum dos
    pares olhados estava fora de ordem", não "o vetor está ordenado". É por isso
    que o campo não se chama ``is_sorted``.
    """

    looks_sorted: bool
    sampled_pairs: int
    comparisons: int
    first_violation: int | None

    def miss_probability(self, n: int, descents: int) -> float:
        """Probabilidade de esta amostragem NÃO ver nenhuma das ``descents``
        violações existentes, em um vetor de ``n`` elementos.

        ``(1 - d/(n-1))^k`` — é a conta que quantifica o risco de falso
        positivo. Custo: O(1).
        """
        pairs = n - 1
        if pairs <= 0 or descents <= 0:
            return 0.0
        return (1.0 - descents / pairs) ** self.sampled_pairs


def sample_sorted_check(
    arr: Sequence[T],
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    *,
    seed: int = RANDOM_SEED,
    counters: Counters | None = None,
) -> SortSample:
    """Falha rápida: olha ``sample_size`` pares adjacentes sorteados.

    Estratégia de custo **O(k)** contra O(n) da verificação completa e
    O(n log n) de simplesmente ordenar o vetor para garantir a pré-condição.
    Com ``k`` constante, uma busca binária precedida desta verificação continua
    O(log n).

    Risco de falso positivo (concluir "ordenado" para um vetor que não está):
    com ``d`` pares fora de ordem entre os ``n-1`` adjacentes, a chance de a
    amostra não pegar nenhum é ``(1 - d/(n-1))^k``. Em números, com ``k = 16``:

    * vetor **reverso** (``d = n-1``): risco 0 — a primeira amostra já detecta;
    * vetor **aleatório** (``d ≈ (n-1)/2``): risco ``0,5^16 ≈ 1,5·10⁻⁵``;
    * vetor ordenado com **uma única** inversão (``d = 1``, ``n = 10⁶``): risco
      ``≈ 0,99998`` — ou seja, a amostragem praticamente **não** detecta.

    A conclusão honesta é essa: amostrar serve para pegar desordem grosseira,
    que é o erro comum (passar o vetor errado, esquecer de ordenar, passar o
    reverso). Não serve para garantir integridade contra corrupção pontual. Quem
    precisa dessa garantia paga O(n) com ``check="full"`` — e aí deveria estar
    usando busca linear.

    Uma observação de segurança: com ``seed`` fixa a amostragem é determinística,
    o que é ótimo para reprodutibilidade e ruim contra adversário, porque quem
    conhece a semente sabe quais pares não serão olhados.

    Custo: O(k).
    """
    if sample_size < 1:
        raise ValueError(f"sample_size deve ser >= 1, recebido {sample_size}")
    counters = counters if counters is not None else Counters()
    pairs = len(arr) - 1
    if pairs <= 0:
        return SortSample(True, 0, 0, None)

    rng = random.Random(seed)
    effective = min(sample_size, pairs)
    indices = rng.sample(range(pairs), effective)
    before = counters.comparisons
    for index in indices:
        if counters.gt(arr[index], arr[index + 1]):
            return SortSample(
                looks_sorted=False,
                sampled_pairs=effective,
                comparisons=counters.comparisons - before,
                first_violation=index,
            )
    return SortSample(
        looks_sorted=True,
        sampled_pairs=effective,
        comparisons=counters.comparisons - before,
        first_violation=None,
    )


# ----------------------------------------------------------------------
# Busca binária
# ----------------------------------------------------------------------
def binary_search(
    sorted_arr: Sequence[T],
    target: T,
    counters: Counters | None = None,
    *,
    check: str = "sampled",
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> int:
    """Índice de ``target`` em vetor ordenado, ou ``-1``.

    Cada iteração inspeciona um elemento e conta **uma** comparação de três vias
    (decisão de projeto 1).

    ``check`` define o que fazer com a pré-condição (decisão de projeto 2):
    ``"off"`` confia no chamador, ``"sampled"`` (padrão) olha ``sample_size``
    pares em O(k), ``"full"`` verifica tudo em O(n). Nos dois últimos, desordem
    detectada levanta :class:`UnsortedInputError` com o índice do par violado —
    nunca ``-1``.

    As comparações da verificação entram no MESMO contador da busca. É de
    propósito: o custo da pré-condição é custo da chamada, e escondê-lo num
    contador separado faria a busca binária parecer mais barata do que é.

    Custo: O(log n) com ``check`` em ``{"off", "sampled"}``; O(n) com
    ``"full"``.
    """
    if check not in CHECK_MODES:
        raise InvalidCheckModeError(
            f"modo de verificação desconhecido: {check!r}; "
            f"disponíveis: {list(CHECK_MODES)}"
        )
    counters = counters if counters is not None else Counters()

    if check == "sampled":
        sample = sample_sorted_check(
            sorted_arr, sample_size, seed=seed, counters=counters
        )
        if not sample.looks_sorted:
            index = sample.first_violation
            raise UnsortedInputError(
                f"pré-condição de ordenação falhou: arr[{index}]="
                f"{sorted_arr[index]!r} > arr[{index + 1}]="  # type: ignore[index]
                f"{sorted_arr[index + 1]!r}. "  # type: ignore[index,operator]
                f"Detectado por amostragem de {sample.sampled_pairs} par(es) "
                f"em O(k); a amostragem detecta desordem grosseira, não garante "
                f"ordenação (use check='full' para certeza em O(n))."
            )
    elif check == "full":
        index = find_first_descent(sorted_arr, counters)
        if index != -1:
            raise UnsortedInputError(
                f"pré-condição de ordenação falhou: arr[{index}]="
                f"{sorted_arr[index]!r} > arr[{index + 1}]="
                f"{sorted_arr[index + 1]!r}. "
                f"Detectado por verificação completa em O(n)."
            )

    low, high = 0, len(sorted_arr) - 1
    while low <= high:
        mid = (low + high) // 2
        counters.count_comparison()  # uma sondagem de três vias
        probe = sorted_arr[mid]
        if probe == target:
            return mid
        if probe < target:  # type: ignore[operator]
            low = mid + 1
        else:
            high = mid - 1
    return -1


def binary_search_counted(
    sorted_arr: Sequence[T], target: T, *, check: str = "sampled"
) -> tuple[int, Counters]:
    """:func:`binary_search` no protocolo ``(resultado, Counters)``."""
    counters = Counters()
    return binary_search(sorted_arr, target, counters, check=check), counters


# ----------------------------------------------------------------------
# Auxiliares dos experimentos
# ----------------------------------------------------------------------
def _stratified_positions(n: int, samples: int) -> list[int]:
    """Posições igualmente espaçadas em ``0..n-1``.

    O caso médio é estimado por amostragem **estratificada** em vez de
    aleatória. A média teórica é a mesma — ``(n+1)/2`` comparações para a busca
    linear — mas a variância é praticamente zero, o que permite medir
    ``n = 10⁶`` com meia dúzia de buscas em vez de centenas. Com posições
    ``⌊n(2j+1)/(2s)⌋``, a média das posições é exatamente ``(n-1)/2``.

    Custo: O(samples).
    """
    samples = max(1, min(samples, n))
    return [(n * (2 * j + 1)) // (2 * samples) for j in range(samples)]


def _average_samples_for(n: int, requested: int, budget: int) -> int:
    """Quantas buscas de caso médio cabem no orçamento de comparações.

    Uma busca linear de caso médio custa ``~n/2`` comparações; com ``n = 10⁶`` e
    30 amostras seriam 15 milhões, o que tornaria o notebook lento sem melhorar
    a estimativa (ver :func:`_stratified_positions`). Custo: O(1).
    """
    if n <= 0:
        return 1
    affordable = max(1, budget // max(1, n))
    return max(2, min(requested, affordable))


# ----------------------------------------------------------------------
# Experimentos do exercício 1
# ----------------------------------------------------------------------
def bench_searches(
    sizes: Iterable[int] = (100, 1_000, 10_000, 100_000, 1_000_000),
    patterns: Iterable[str] = ("ordenado", "reverso", "aleatorio"),
    *,
    seed: int = RANDOM_SEED,
    average_samples: int = 16,
    comparison_budget: int = 4_000_000,
) -> list[dict[str, Any]]:
    """Tabela principal: busca linear nos 3 cenários e busca binária, por caso.

    Para cada ``n`` e cada cenário, mede a busca linear nos três casos:

    * **melhor** — alvo na posição 0: 1 comparação, independente de ``n``;
    * **médio** — alvos em posições estratificadas: ``≈ (n+1)/2`` comparações,
      isto é ``comparações/n ≈ 0,5``;
    * **pior** — alvo ausente: exatamente ``n`` comparações, ``comparações/n = 1``.

    A busca binária é medida só sobre o vetor ordenado — nos outros dois
    cenários a pré-condição falha, e isso é o assunto de
    :func:`demonstrate_precondition`. Os três casos dela:

    * **melhor** — alvo no primeiro ponto médio: 1 comparação;
    * **médio** e **pior** — ``≈ log₂ n`` comparações, com
      ``comparações/log₂n ≈ 1``.

    A coluna ``copies`` fica em 0 nas duas buscas e continua na tabela de
    propósito (decisão de projeto 5).

    Custo: O(Σ n · amostras).
    """
    rows: list[dict[str, Any]] = []
    for n in sizes:
        if n <= 0:
            raise ValueError(f"tamanho deve ser positivo, recebido {n}")
        samples = _average_samples_for(n, average_samples, comparison_budget)
        positions = _stratified_positions(n, samples)

        for pattern in patterns:
            if pattern not in PATTERNS:
                raise InvalidPatternError(f"padrão desconhecido: {pattern!r}")
            arr = generate_array(n, pattern, seed=seed)

            casos: list[tuple[str, list[T], int]] = [  # type: ignore[valid-type]
                ("melhor (1ª posição)", [arr[0]], 1),
                ("médio (posições estratificadas)", [arr[p] for p in positions], samples),
                ("pior (ausente)", [n + 1], 1),
            ]
            for caso, alvos, buscas in casos:
                counters = Counters()
                encontrados = 0
                with stopwatch() as elapsed:
                    for alvo in alvos:
                        if linear_search(arr, alvo, counters) != -1:
                            encontrados += 1
                por_busca = counters.comparisons / buscas
                rows.append(
                    {
                        "exercicio": "ex1",
                        "algoritmo": "linear_search (array)",
                        "estrutura": "array",
                        "n": n,
                        "padrao": pattern,
                        "caso": caso,
                        **counters.as_dict(),
                        "buscas": buscas,
                        "encontradas": encontrados,
                        "comparacoes_por_busca": por_busca,
                        "comparacoes_por_busca/n": ratio(por_busca, n, "n"),
                        "comparacoes_por_busca/log2n": ratio(por_busca, n, "log2n"),
                        "tempo_s": elapsed[0],
                    }
                )
            del arr

        # busca binária: só faz sentido sobre o vetor ordenado
        ordenado = generate_array(n, "ordenado", seed=seed)
        meio = (n - 1) // 2
        casos_binaria: list[tuple[str, list[int], int]] = [
            ("melhor (1º ponto médio)", [ordenado[meio]], 1),
            (
                "médio (posições estratificadas)",
                [ordenado[p] for p in positions],
                samples,
            ),
            ("pior (ausente)", [n + 1], 1),
        ]
        for caso, alvos, buscas in casos_binaria:
            counters = Counters()
            encontrados = 0
            with stopwatch() as elapsed:
                for alvo in alvos:
                    # check="off": a pré-condição é conhecida aqui, e incluir a
                    # verificação misturaria o custo dela com o da busca. O custo
                    # de cada modo é medido em bench_fast_fail.
                    if binary_search(ordenado, alvo, counters, check="off") != -1:
                        encontrados += 1
            por_busca = counters.comparisons / buscas
            rows.append(
                {
                    "exercicio": "ex1",
                    "algoritmo": "binary_search (array ordenado)",
                    "estrutura": "array",
                    "n": n,
                    "padrao": "ordenado",
                    "caso": caso,
                    **counters.as_dict(),
                    "buscas": buscas,
                    "encontradas": encontrados,
                    "comparacoes_por_busca": por_busca,
                    "comparacoes_por_busca/n": ratio(por_busca, n, "n"),
                    "comparacoes_por_busca/log2n": ratio(por_busca, n, "log2n"),
                    "tempo_s": elapsed[0],
                }
            )
        del ordenado
    return rows


def demonstrate_precondition(
    n: int = 1_000,
    *,
    seed: int = RANDOM_SEED,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> list[dict[str, Any]]:
    """O que a busca binária faz quando o vetor NÃO está ordenado, por modo.

    Uma linha por (cenário, modo de verificação), registrando se houve exceção,
    quantas comparações o modo custou e — no modo ``"off"`` — que resposta
    errada a busca devolveu em silêncio. Essa última linha é a justificativa de
    a verificação existir.

    Custo: O(n) por linha no modo ``"full"``, O(k) no ``"sampled"``.
    """
    if n < 2:
        raise ValueError(f"n deve ser >= 2, recebido {n}")
    rows: list[dict[str, Any]] = []
    for pattern in ("ordenado", "reverso", "aleatorio"):
        arr = generate_array(n, pattern, seed=seed)
        alvo = arr[n // 3]  # existe no vetor, em posição não trivial
        verdadeiro = arr.index(alvo)
        descents = sum(1 for i in range(n - 1) if arr[i] > arr[i + 1])

        for check in CHECK_MODES:
            counters = Counters()
            detectou = False
            resposta: int | None = None
            mensagem = ""
            try:
                resposta = binary_search(
                    arr, alvo, counters, check=check,
                    sample_size=sample_size, seed=seed,
                )
            except UnsortedInputError as error:
                detectou = True
                mensagem = str(error).split(".")[0]

            rows.append(
                {
                    "exercicio": "ex1",
                    "algoritmo": "binary_search — pré-condição",
                    "n": n,
                    "padrao": pattern,
                    "modo": check,
                    "pares_fora_de_ordem": descents,
                    "detectou": detectou,
                    "indice_correto": verdadeiro,
                    "indice_devolvido": resposta,
                    "resposta_errada_em_silencio": (
                        not detectou and resposta != verdadeiro
                    ),
                    **counters.as_dict(),
                    "custo_teorico": {
                        "off": "O(1)", "sampled": "O(k)", "full": "O(n)"
                    }[check],
                    "mensagem": mensagem,
                }
            )
    return rows


def bench_fast_fail(
    n: int = 100_000,
    sample_sizes: Iterable[int] = (1, 2, 4, 8, 16, 32),
    *,
    trials: int = 200,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Taxa de detecção medida da falha rápida × previsão ``(1-d/(n-1))^k``.

    Três vetores com graus de desordem muito diferentes:

    * ``reverso`` — ``d = n-1``, toda a estrutura violada;
    * ``aleatorio`` — ``d ≈ (n-1)/2``;
    * ``uma_inversao`` — ordenado com **um único** par trocado, ``d = 1``.

    Cada combinação (vetor, ``k``) roda ``trials`` amostragens com sementes
    diferentes — variar a semente é obrigatório aqui: com semente fixa a
    amostragem sempre olha os mesmos pares, e a "taxa de detecção" mediria uma
    única amostra repetida.

    A previsão teórica ao lado da taxa medida é a evidência quantitativa do
    risco de falso positivo que o enunciado pede discutir.

    Custo: O(trials · k) por linha, contra O(n) da verificação completa e
    O(n log n) de ordenar.
    """
    if trials < 1:
        raise ValueError(f"trials deve ser >= 1, recebido {trials}")
    if n < 4:
        raise ValueError(f"n deve ser >= 4, recebido {n}")

    uma_inversao = generate_array(n, "ordenado")
    meio = n // 2
    uma_inversao[meio], uma_inversao[meio + 1] = (
        uma_inversao[meio + 1],
        uma_inversao[meio],
    )

    vetores: list[tuple[str, list[int]]] = [
        ("reverso", generate_array(n, "reverso")),
        ("aleatorio", generate_array(n, "aleatorio", seed=seed)),
        ("uma_inversao", uma_inversao),
    ]

    rows: list[dict[str, Any]] = []
    for nome, arr in vetores:
        descents = sum(1 for index in range(n - 1) if arr[index] > arr[index + 1])
        for k in sample_sizes:
            if k < 1:
                raise ValueError(f"sample_size deve ser >= 1, recebido {k}")
            detectados = 0
            counters = Counters()
            with stopwatch() as elapsed:
                for trial in range(trials):
                    sample = sample_sorted_check(
                        arr, k, seed=seed + trial, counters=counters
                    )
                    if not sample.looks_sorted:
                        detectados += 1
            taxa = detectados / trials
            previsto = 1.0 - (1.0 - descents / (n - 1)) ** k
            rows.append(
                {
                    "exercicio": "ex1",
                    "algoritmo": "falha rápida — amostragem de pares",
                    "n": n,
                    "padrao": nome,
                    "k": k,
                    "pares_fora_de_ordem": descents,
                    "fracao_desordenada": descents / (n - 1),
                    "taxa_deteccao_medida": taxa,
                    "taxa_deteccao_prevista": previsto,
                    "risco_falso_positivo_previsto": 1.0 - previsto,
                    "comparacoes_por_verificacao": counters.comparisons / trials,
                    "comparacoes_verificacao_completa": n - 1,
                    "custo_relativo_vs_completa": (
                        (counters.comparisons / trials) / (n - 1)
                    ),
                    "custo_teorico_ordenar": n * math.log2(n),
                    "tempo_s": elapsed[0],
                }
            )
    return rows


def bench_array_vs_linked(
    sizes: Iterable[int] = (1_000, 2_000, 4_000, 8_000, 16_000),
    *,
    seed: int = RANDOM_SEED,
    repeats: int = 3,
) -> list[dict[str, Any]]:
    """A mesma busca linear em array e em lista encadeada, lado a lado.

    Previsão: **exatamente as mesmas comparações** (é o mesmo algoritmo) e
    ``hops = 0`` no array contra ``hops = n`` na lista. A diferença observável
    fica no tempo por comparação: o array percorre memória contígua e a lista
    salta de nó em nó.

    Os tamanhos param em 16.000 e não em 10⁶ de propósito: um nó de
    :class:`~src.linked_list.SinglyLinkedList` custa dezenas de bytes, então uma
    lista de 10⁶ elementos consumiria dezenas de MB para medir uma contagem que
    já se sabe idêntica por construção. O que precisa de 10⁶ é a curva de
    comparações, e essa está em :func:`bench_searches`.

    Custo: O(repeats · Σ n).
    """
    if repeats < 1:
        raise ValueError(f"repeats deve ser >= 1, recebido {repeats}")
    rows: list[dict[str, Any]] = []
    for n in sizes:
        if n <= 0:
            raise ValueError(f"tamanho deve ser positivo, recebido {n}")
        arr = generate_array(n, "aleatorio", seed=seed)
        linked: SinglyLinkedList[int] = SinglyLinkedList(arr)
        alvo = n + 1  # ausente: força a travessia completa nos dois

        for estrutura, alvo_busca in (("array", arr), ("lista encadeada", linked)):
            counters = Counters()
            melhor = float("inf")
            for attempt in range(repeats):
                run_counters = counters if attempt == 0 else Counters()
                with stopwatch() as elapsed:
                    if estrutura == "array":
                        resultado = linear_search(alvo_busca, alvo, run_counters)
                    else:
                        resultado = linear_search_linked(
                            alvo_busca, alvo, run_counters  # type: ignore[arg-type]
                        )
                melhor = min(melhor, elapsed[0])
                assert resultado == -1
            rows.append(
                {
                    "exercicio": "ex1",
                    "algoritmo": f"linear_search ({estrutura})",
                    "estrutura": estrutura,
                    "n": n,
                    "padrao": "aleatorio, alvo ausente (pior caso)",
                    "caso": "pior (ausente)",
                    **counters.as_dict(),
                    "comparacoes_teoricas": n,
                    "saltos_teoricos": 0 if estrutura == "array" else n,
                    "tempo_s": melhor,
                    "ns_por_comparacao": melhor * 1e9 / n,
                    "repeticoes": repeats,
                }
            )
        del arr, linked
    return rows


def bench_search_break_even(
    sizes: Iterable[int] = (1_000, 10_000, 100_000),
    query_counts: Iterable[int] = (1, 4, 16, 64, 256),
    *,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Quando vale a pena ordenar antes de buscar.

    Ordenar é O(n log n) e só se paga se houver consultas suficientes para
    diluir esse custo. Para ``m`` consultas em um vetor de ``n`` elementos:

    * buscar linearmente ``m`` vezes: ``m · n/2`` comparações;
    * ordenar e buscar binariamente: ``n·log₂n + m·log₂n`` comparações.

    O ponto de equilíbrio é ``m* = n·log₂n / (n/2 − log₂n) ≈ 2·log₂n`` — ou
    seja, ordenar compensa a partir de **poucas dezenas de consultas**, e o
    limiar cresce só logaritmicamente com ``n``.

    As colunas de busca são **medidas** (busca linear e binária instrumentadas,
    caso médio estratificado); a coluna de ordenação é o custo teórico
    ``n·log₂n``, porque as ordenações instrumentadas são o exercício 2 e ainda
    não existem aqui — está rotulada como estimativa.

    Custo: O(Σ n) por tamanho.
    """
    rows: list[dict[str, Any]] = []
    for n in sizes:
        if n <= 0:
            raise ValueError(f"tamanho deve ser positivo, recebido {n}")
        arr = generate_array(n, "aleatorio", seed=seed)
        ordenado = generate_array(n, "ordenado", seed=seed)
        positions = _stratified_positions(n, 8)

        linear_counters = Counters()
        for position in positions:
            linear_search(arr, arr[position], linear_counters)
        linear_por_busca = linear_counters.comparisons / len(positions)

        binaria_counters = Counters()
        for position in positions:
            binary_search(
                ordenado, ordenado[position], binaria_counters, check="off"
            )
        binaria_por_busca = binaria_counters.comparisons / len(positions)

        custo_ordenar = n * math.log2(n)
        for m in query_counts:
            if m < 1:
                raise ValueError(f"query_counts deve ser >= 1, recebido {m}")
            total_linear = m * linear_por_busca
            total_ordenado = custo_ordenar + m * binaria_por_busca
            rows.append(
                {
                    "exercicio": "ex1",
                    "algoritmo": "linear × (ordenar + binária)",
                    "n": n,
                    "padrao": "aleatorio",
                    "consultas": m,
                    "comparacoes_linear_por_busca": linear_por_busca,
                    "comparacoes_binaria_por_busca": binaria_por_busca,
                    "total_linear": total_linear,
                    "custo_ordenar_estimado": custo_ordenar,
                    "total_ordenar_mais_binaria": total_ordenado,
                    "vale_ordenar": total_ordenado < total_linear,
                    "equilibrio_teorico_m": (
                        custo_ordenar / (linear_por_busca - binaria_por_busca)
                        if linear_por_busca > binaria_por_busca
                        else float("nan")
                    ),
                }
            )
        del arr, ordenado
    return rows

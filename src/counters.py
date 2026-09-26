"""Instrumentação compartilhada por todos os exercícios do TP DR1_AT.

Decisão de projeto
------------------
A nota deste TP depende da coerência entre implementação, medição e análise em
Big O. Para que os números de exercícios diferentes sejam comparáveis, existe
UM único contador, definido aqui e reutilizado por todos os módulos de ``src/``.

Protocolo uniforme: toda função instrumentada devolve ``(resultado, Counters)``.
Nenhuma função escreve em um contador global — o contador é sempre criado pela
chamada de nível mais alto e propagado para baixo, de modo que dois experimentos
nunca contaminam as contagens um do outro.

Contadores
----------
``comparisons``
    Comparações de chaves (``<``, ``<=``, ``==`` entre elementos da entrada).
    Comparação de ponteiro contra ``None`` NÃO conta aqui: não é comparação de
    chave e inflaria artificialmente as ordenações.
``copies``
    Movimentações de dados. Critério fixo do enunciado: **uma troca conta como
    3 cópias** (``tmp = a; a = b; b = tmp``). Ver :data:`COPIES_PER_SWAP`.
``calls``
    Chamadas recursivas, usado para medir o ganho da memoization (ex 8) e a
    profundidade de recursão (ex 5, 7, 9).
``hops``
    Saltos de ponteiro ``no = no.next`` em estruturas encadeadas.

    Extensão aos três contadores obrigatórios, e a razão é o exercício 10: com
    ponteiro ``tail``, ``insert_last`` faz 0 comparações de chave e 0 cópias —
    exatamente como a versão sem ``tail``. Os dois contadores obrigatórios são
    cegos para a diferença entre O(1) e O(n) aqui, e sem ``hops`` a "evidência
    quantitativa" que a rubrica cobra não existiria. Mesmo motivo no ex 1
    (busca linear em array vs. em lista encadeada: mesmas comparações, custo de
    travessia diferente) e no ex 7 (caminho em lista encadeada vs. lista Python).
    ``hops`` é aditivo: quem não usa estrutura encadeada simplesmente o deixa 0.

Custo em Big O
--------------
Toda a instrumentação é O(1) por evento contado, portanto não altera a classe de
complexidade do algoritmo medido — só a constante multiplicativa.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator, MutableSequence, Sequence, TypeVar

T = TypeVar("T")

#: Critério fixo do enunciado: uma troca é contabilizada como 3 cópias.
COPIES_PER_SWAP: int = 3

#: Semente única de todos os experimentos, para que notebook, PDF e vídeo
#: mostrem exatamente os mesmos números.
RANDOM_SEED: int = 42


@dataclass
class Counters:
    """Contadores de um experimento instrumentado.

    Custo: O(1) por incremento; a própria instância é O(1) em memória.
    """

    comparisons: int = 0
    copies: int = 0
    calls: int = 0
    hops: int = 0

    # ------------------------------------------------------------------
    # Incrementos explícitos
    # ------------------------------------------------------------------
    def count_comparison(self, times: int = 1) -> None:
        """Registra ``times`` comparações de chave."""
        self.comparisons += times

    def count_copy(self, times: int = 1) -> None:
        """Registra ``times`` movimentações de dados."""
        self.copies += times

    def count_swap(self, times: int = 1) -> None:
        """Registra ``times`` trocas, cada uma valendo 3 cópias."""
        self.copies += COPIES_PER_SWAP * times

    def count_call(self, times: int = 1) -> None:
        """Registra ``times`` chamadas (recursivas ou de subproblema)."""
        self.calls += times

    def count_hop(self, times: int = 1) -> None:
        """Registra ``times`` saltos de ponteiro em estrutura encadeada."""
        self.hops += times

    # ------------------------------------------------------------------
    # Comparações que se contam sozinhas
    # ------------------------------------------------------------------
    # Decisão de projeto: as ordenações e buscas usam estes helpers em vez de
    # `a < b` cru. Assim é impossível esquecer de contar uma comparação, e a
    # contagem de bubble/selection/insertion fica comparável por construção.
    def lt(self, left: Any, right: Any) -> bool:
        """``left < right``, contando 1 comparação."""
        self.comparisons += 1
        return left < right

    def le(self, left: Any, right: Any) -> bool:
        """``left <= right``, contando 1 comparação."""
        self.comparisons += 1
        return left <= right

    def gt(self, left: Any, right: Any) -> bool:
        """``left > right``, contando 1 comparação."""
        self.comparisons += 1
        return left > right

    def ge(self, left: Any, right: Any) -> bool:
        """``left >= right``, contando 1 comparação."""
        self.comparisons += 1
        return left >= right

    def eq(self, left: Any, right: Any) -> bool:
        """``left == right``, contando 1 comparação."""
        self.comparisons += 1
        return left == right

    # ------------------------------------------------------------------
    # Movimentações que se contam sozinhas
    # ------------------------------------------------------------------
    def swap(self, seq: MutableSequence[T], i: int, j: int) -> None:
        """Troca ``seq[i]`` com ``seq[j]``, contando 3 cópias."""
        seq[i], seq[j] = seq[j], seq[i]
        self.copies += COPIES_PER_SWAP

    def write(self, seq: MutableSequence[T], index: int, value: T) -> None:
        """Escreve ``value`` em ``seq[index]``, contando 1 cópia."""
        seq[index] = value
        self.copies += 1

    def read(self, value: T) -> T:
        """Registra 1 cópia de leitura (ex.: guardar a chave em ``key = arr[i]``)."""
        self.copies += 1
        return value

    # ------------------------------------------------------------------
    # Composição
    # ------------------------------------------------------------------
    def __add__(self, other: "Counters") -> "Counters":
        return Counters(
            comparisons=self.comparisons + other.comparisons,
            copies=self.copies + other.copies,
            calls=self.calls + other.calls,
            hops=self.hops + other.hops,
        )

    def __iadd__(self, other: "Counters") -> "Counters":
        self.comparisons += other.comparisons
        self.copies += other.copies
        self.calls += other.calls
        self.hops += other.hops
        return self

    def absorb(self, other: "Counters") -> "Counters":
        """Soma ``other`` neste contador e devolve ``self`` (para encadear)."""
        self += other
        return self

    # ------------------------------------------------------------------
    # Leitura
    # ------------------------------------------------------------------
    @property
    def total(self) -> int:
        """Soma de todos os eventos contados."""
        return self.comparisons + self.copies + self.calls + self.hops

    def as_dict(self) -> dict[str, int]:
        """Contadores como dicionário, pronto para virar linha de DataFrame."""
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"comparisons={self.comparisons} copies={self.copies} "
            f"calls={self.calls} hops={self.hops}"
        )


# ----------------------------------------------------------------------
# Curvas teóricas e razões
# ----------------------------------------------------------------------
#: Nome de cada curva teórica -> função que a avalia em n.
CURVES: dict[str, Any] = {
    "n": lambda n: float(n),
    "n_log2n": lambda n: float(n) * math.log2(n) if n > 1 else float("nan"),
    "n2": lambda n: float(n) * float(n),
    "log2n": lambda n: math.log2(n) if n > 0 else float("nan"),
}


def ratio(measured: int | float, n: int, curve: str) -> float:
    """Razão entre o valor medido e a curva teórica ``curve`` avaliada em ``n``.

    Razão aproximadamente constante quando ``n`` cresce é a evidência de que o
    algoritmo pertence àquela classe. Devolve ``nan`` quando a curva não é
    definida para o ``n`` dado (ex.: ``n·log2 n`` para ``n <= 1``).

    Custo: O(1).
    """
    if curve not in CURVES:
        raise ValueError(
            f"curva desconhecida: {curve!r}; disponíveis: {sorted(CURVES)}"
        )
    denominator = CURVES[curve](n)
    if denominator is None or denominator == 0 or math.isnan(denominator):
        return float("nan")
    return measured / denominator


def make_row(
    *,
    exercise: str,
    algorithm: str,
    n: int,
    pattern: str = "-",
    counters: Counters | None = None,
    metric: str = "comparisons",
    curves: Sequence[str] = ("n", "n_log2n", "n2"),
    elapsed_s: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Monta uma linha de medição no formato único usado no notebook inteiro.

    Colunas fixas: exercício, algoritmo, n, padrão de entrada, os quatro
    contadores e a razão da métrica escolhida contra cada curva teórica pedida.

    Custo: O(len(curves)).
    """
    counters = counters if counters is not None else Counters()
    if metric not in counters.as_dict():
        raise ValueError(
            f"métrica desconhecida: {metric!r}; "
            f"disponíveis: {sorted(counters.as_dict())}"
        )
    measured = counters.as_dict()[metric]

    row: dict[str, Any] = {
        "exercicio": exercise,
        "algoritmo": algorithm,
        "n": n,
        "padrao": pattern,
        **counters.as_dict(),
        "metrica": metric,
    }
    for curve in curves:
        row[f"{metric}/{curve}"] = ratio(measured, n, curve)
    if elapsed_s is not None:
        row["tempo_s"] = elapsed_s
        row["tempo_us_por_op"] = (elapsed_s * 1e6 / n) if n else float("nan")
    if extra:
        row.update(extra)
    return row


@contextmanager
def stopwatch() -> Iterator[list[float]]:
    """Cronômetro de parede, para acompanhar (não substituir) as contagens.

    Uso::

        with stopwatch() as elapsed:
            ...
        print(elapsed[0])

    O tempo é ruído de máquina; a prova da classe de complexidade são as
    contagens. O tempo entra apenas para mostrar que a constante escondida pelo
    Big O também se comporta como o esperado.
    """
    marks: list[float] = []
    start = time.perf_counter()
    try:
        yield marks
    finally:
        marks.append(time.perf_counter() - start)

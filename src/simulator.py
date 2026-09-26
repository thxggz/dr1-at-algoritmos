"""Exercício 6 — Simulador de eventos com quatro filas e índice em BST.

Reutiliza a :class:`~src.stack_queue.Queue` de capacidade fixa do exercício 5 e
a :class:`~src.bst.BinarySearchTree` do exercício 2.

Decisões de projeto
-------------------
1. **A chave do índice é o identificador completo — como objeto, não como
   texto.** O enunciado pede a chave ``"B17"``. Se ela fosse a *string*
   ``"B17"``, a travessia in-order devolveria ``B1, B10, B11, …, B2, B20``,
   porque a ordem de texto compara caractere a caractere e ``'1' < '2'``. Um
   índice cuja listagem "em ordem" sai fora de ordem não serve para auditoria —
   que é exatamente a razão de o enunciado pedir o índice.
   A saída é :class:`ClientId`: ``str(ClientId("B", 17)) == "B17"``, e a ordem
   total é por ``(fila, sequência)``. A chave continua sendo o identificador
   completo; só não é uma string. Há teste que constrói a versão com string e
   mostra a listagem errada, para que a decisão não pareça preciosismo.

2. **O contador de sequência só avança em chegada bem-sucedida.** Se um
   overflow consumisse o número, o identificador ``B17`` passaria a significar
   "o 17º que *tentou* entrar" e a listagem do índice teria buracos sem
   explicação. Quem quer contar as tentativas frustradas tem a coluna própria
   no relatório de erros.

3. **Erro registra contexto e a simulação continua** — exigência do enunciado.
   Cada erro guarda o instante lógico, o evento, a fila, a ocupação no momento e
   a mensagem da exceção original (:class:`~src.stack_queue.QueueOverflowError`
   ou :class:`~src.stack_queue.QueueUnderflowError`). Sem contexto, "houve um
   overflow" não permite reconstruir nada.

4. **Caractere alfabético fora de A–D também é erro, não snapshot.**
   A regra do enunciado é "não alfabético dispara snapshot". Um ``z`` é
   alfabético, então não é snapshot; e não existe fila Z, então é entrada
   inválida. Tratá-lo como snapshot esconderia erro de digitação no roteiro de
   eventos.

5. **Espaço e quebra de linha são não alfabéticos e, portanto, disparam
   snapshot.** É a leitura literal do enunciado, e está documentada aqui porque
   surpreende: uma string de eventos "formatada" com espaços produz um snapshot
   por espaço.

Custo em Big O
--------------
Com ``e`` eventos, ``s`` clientes atendidos e ``h`` a altura do índice:

=================================  =================  ======================
Operação                           Custo              Observação
=================================  =================  ======================
chegada (``enqueue``)              O(1)               fila circular do ex 5
saída (``dequeue`` + indexar)      O(h)               o custo está no índice
snapshot                           O(q · c)           q filas, c ocupação
busca por identificador            O(h)               ``find``
listagem in-order                  O(s)               ``listar_atendidos``
simulação completa                 O(e + s·h)         —
=================================  =================  ======================

``h`` é o ponto de atenção, e :func:`bench_index_shape` mede isso: dentro de
cada fila os identificadores são **crescentes** (``A1, A2, A3, …``), que é
exatamente a ordem de inserção que degenera uma BST (exercício 3). Com 4 filas
o índice vira quatro espinhas penduradas, e ``h`` cresce como ``s/4`` em vez de
``log₂s``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

from src.bst import BinarySearchTree
from src.counters import RANDOM_SEED, Counters, stopwatch
from src.stack_queue import (
    Queue,
    QueueOverflowError,
    QueueUnderflowError,
)

#: Nomes das quatro filas do enunciado.
QUEUE_NAMES: tuple[str, ...] = ("A", "B", "C", "D")

#: Capacidade padrão de cada fila. Pequena de propósito: overflow é um caso de
#: borda que o enunciado manda exercitar, e com capacidade generosa ele nunca
#: aconteceria.
DEFAULT_CAPACITY: int = 8


# ----------------------------------------------------------------------
# Exceções próprias (convenção 9)
# ----------------------------------------------------------------------
class SimulatorError(Exception):
    """Erro base do simulador."""


class SimulatorConfigError(SimulatorError):
    """Configuração inválida (capacidade, nomes de fila)."""


class ClientIdError(SimulatorError):
    """Identificador de cliente malformado."""


class ClientNotFoundError(SimulatorError):
    """Identificador não está no índice de atendidos."""


# ----------------------------------------------------------------------
# Identificador
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ClientId:
    """Identificador completo de um cliente: fila e número sequencial.

    ``str(ClientId("B", 17)) == "B17"`` — é o identificador do enunciado. A
    diferença está na **ordem**: comparações usam ``(fila, sequência)``, então
    ``B2 < B17``, ao contrário do que a ordem de texto diria (decisão 1).
    """

    queue: str
    sequence: int

    def __str__(self) -> str:
        return f"{self.queue}{self.sequence}"

    def __repr__(self) -> str:
        return f"ClientId({str(self)!r})"

    def _chave(self) -> tuple[str, int]:
        return (self.queue, self.sequence)

    def __lt__(self, other: "ClientId") -> bool:
        return self._chave() < other._chave()

    def __le__(self, other: "ClientId") -> bool:
        return self._chave() <= other._chave()

    def __gt__(self, other: "ClientId") -> bool:
        return self._chave() > other._chave()

    def __ge__(self, other: "ClientId") -> bool:
        return self._chave() >= other._chave()

    @classmethod
    def parse(cls, text: str) -> "ClientId":
        """``"B17"`` → ``ClientId("B", 17)``. Custo: O(len(text)).

        Levanta :class:`ClientIdError` para texto malformado.
        """
        if not isinstance(text, str) or len(text) < 2:
            raise ClientIdError(
                f"identificador deve ser uma letra seguida de dígitos, "
                f"recebido {text!r}"
            )
        letra, numero = text[0], text[1:]
        if not letra.isalpha() or not numero.isdigit():
            raise ClientIdError(
                f"identificador malformado: {text!r} "
                "(esperado uma letra seguida de dígitos, como 'B17')"
            )
        return cls(letra.upper(), int(numero))


# ----------------------------------------------------------------------
# Registros
# ----------------------------------------------------------------------
@dataclass(slots=True)
class ServiceRecord:
    """Metadados do atendimento, guardados como valor no índice."""

    client: ClientId
    queue: str
    service_order: int
    arrival_tick: int
    service_tick: int

    @property
    def wait_ticks(self) -> int:
        """Instantes lógicos entre chegada e atendimento. Custo: O(1)."""
        return self.service_tick - self.arrival_tick


@dataclass(slots=True)
class LogEntry:
    """Uma linha do log da simulação."""

    tick: int
    event: str
    kind: str  # "chegada" | "saida" | "snapshot" | "erro"
    queue: str | None
    client: str | None
    message: str


@dataclass(slots=True)
class ErrorRecord:
    """Erro com contexto suficiente para reconstruir o que aconteceu."""

    tick: int
    event: str
    kind: str  # "overflow" | "underflow" | "evento_invalido"
    queue: str | None
    occupancy: int | None
    capacity: int | None
    message: str


@dataclass(slots=True)
class Snapshot:
    """Estado de todas as filas num instante lógico."""

    tick: int
    trigger: str
    sizes: dict[str, int]
    contents: dict[str, list[str]]

    def total(self) -> int:
        """Clientes em espera somando todas as filas. Custo: O(q)."""
        return sum(self.sizes.values())


@dataclass(slots=True)
class SimulationResult:
    """Tudo que a simulação produziu, pronto para virar tabela no notebook."""

    events: str
    log: list[LogEntry]
    snapshots: list[Snapshot]
    errors: list[ErrorRecord]
    index: BinarySearchTree[ClientId, ServiceRecord]
    arrivals: int
    departures: int
    remaining: dict[str, list[str]]
    counters: Counters
    elapsed_s: float

    def stats(self) -> dict[str, Any]:
        """Resumo numérico da simulação. Custo: O(n)."""
        relatorio = self.index.balance_report()
        overflow = sum(1 for erro in self.errors if erro.kind == "overflow")
        underflow = sum(1 for erro in self.errors if erro.kind == "underflow")
        invalidos = sum(
            1 for erro in self.errors if erro.kind == "evento_invalido"
        )
        esperas = [
            registro.wait_ticks for _, registro in self.index.in_order_items()
        ]
        return {
            "eventos": len(self.events),
            "chegadas": self.arrivals,
            "saidas": self.departures,
            "atendidos": len(self.index),
            "snapshots": len(self.snapshots),
            "erros": len(self.errors),
            "overflow": overflow,
            "underflow": underflow,
            "eventos_invalidos": invalidos,
            "em_espera_no_fim": sum(len(fila) for fila in self.remaining.values()),
            "indice_altura": relatorio["altura"],
            "indice_altura_ideal": relatorio["altura_ideal"],
            "indice_razao_altura": relatorio["razao_altura"],
            "espera_media": (sum(esperas) / len(esperas)) if esperas else 0.0,
            "espera_maxima": max(esperas) if esperas else 0,
            **self.counters.as_dict(),
            "tempo_s": self.elapsed_s,
        }


# ----------------------------------------------------------------------
# Simulador
# ----------------------------------------------------------------------
class CheckoutSimulator:
    """Simulador de checkout com quatro filas independentes.

    Regras de evento, na ordem em que são testadas:

    * **minúscula em A–D** → chegada na fila correspondente;
    * **maiúscula em A–D** → saída (atendimento) daquela fila;
    * **não alfabético** → snapshot do estado de todas as filas (decisão 5);
    * **alfabético fora de A–D** → evento inválido, registrado como erro
      (decisão 4).

    Overflow e underflow são registrados com contexto e a simulação continua
    (decisão 3).
    """

    __slots__ = ("_capacity", "_queue_names")

    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY,
        queue_names: Iterable[str] = QUEUE_NAMES,
    ) -> None:
        """Custo: O(q)."""
        nomes = tuple(nome.upper() for nome in queue_names)
        if not nomes:
            raise SimulatorConfigError("é preciso ao menos uma fila")
        if len(set(nomes)) != len(nomes):
            raise SimulatorConfigError(f"nomes de fila repetidos: {nomes}")
        for nome in nomes:
            if len(nome) != 1 or not nome.isalpha():
                raise SimulatorConfigError(
                    f"nome de fila deve ser uma única letra, recebido {nome!r}"
                )
        if not isinstance(capacity, int) or isinstance(capacity, bool):
            raise SimulatorConfigError(
                f"capacidade deve ser int, recebido {type(capacity).__name__}"
            )
        if capacity < 1:
            raise SimulatorConfigError(
                f"capacidade deve ser >= 1, recebido {capacity}"
            )
        self._capacity = capacity
        self._queue_names = nomes

    @property
    def capacity(self) -> int:
        """Capacidade de cada fila. Custo: O(1)."""
        return self._capacity

    @property
    def queue_names(self) -> tuple[str, ...]:
        """Nomes das filas. Custo: O(1)."""
        return self._queue_names

    def run(
        self, events: str, counters: Counters | None = None
    ) -> SimulationResult:
        """Executa a string de eventos e devolve tudo que aconteceu.

        Custo: O(e + s·h) — ``e`` eventos, ``s`` atendimentos, ``h`` a altura do
        índice. O termo ``s·h`` é o da indexação, e é o que domina quando o
        índice degenera (ver :func:`bench_index_shape`).
        """
        if not isinstance(events, str):
            raise SimulatorConfigError(
                f"events deve ser str, recebido {type(events).__name__}"
            )
        counters = counters if counters is not None else Counters()

        filas: dict[str, Queue[tuple[ClientId, int]]] = {
            nome: Queue(self._capacity) for nome in self._queue_names
        }
        sequencias: dict[str, int] = {nome: 0 for nome in self._queue_names}
        index: BinarySearchTree[ClientId, ServiceRecord] = BinarySearchTree()
        log: list[LogEntry] = []
        snapshots: list[Snapshot] = []
        errors: list[ErrorRecord] = []
        arrivals = departures = 0

        with stopwatch() as elapsed:
            for tick, event in enumerate(events, start=1):
                if not event.isalpha():
                    snapshot = self._snapshot(tick, event, filas)
                    snapshots.append(snapshot)
                    log.append(
                        LogEntry(
                            tick, event, "snapshot", None, None,
                            f"snapshot: {snapshot.total()} cliente(s) em espera",
                        )
                    )
                    continue

                nome = event.upper()
                if nome not in filas:
                    erro = ErrorRecord(
                        tick, event, "evento_invalido", None, None, None,
                        f"evento {event!r} é alfabético mas não corresponde a "
                        f"nenhuma fila ({', '.join(self._queue_names)}); "
                        "alfabético não vira snapshot (decisão de projeto 4)",
                    )
                    errors.append(erro)
                    log.append(
                        LogEntry(tick, event, "erro", None, None, erro.message)
                    )
                    continue

                fila = filas[nome]
                if event.islower():
                    proximo = ClientId(nome, sequencias[nome] + 1)
                    try:
                        fila.enqueue((proximo, tick), counters)
                    except QueueOverflowError as falha:
                        erro = ErrorRecord(
                            tick, event, "overflow", nome, len(fila),
                            self._capacity,
                            f"chegada recusada na fila {nome}: {falha}. "
                            f"O número {proximo.sequence} NÃO foi consumido "
                            "(decisão de projeto 2)",
                        )
                        errors.append(erro)
                        log.append(
                            LogEntry(tick, event, "erro", nome, None, erro.message)
                        )
                        continue
                    sequencias[nome] += 1
                    arrivals += 1
                    log.append(
                        LogEntry(
                            tick, event, "chegada", nome, str(proximo),
                            f"{proximo} entrou na fila {nome} "
                            f"({len(fila)}/{self._capacity})",
                        )
                    )
                else:
                    try:
                        cliente, chegada = fila.dequeue(counters)
                    except QueueUnderflowError as falha:
                        erro = ErrorRecord(
                            tick, event, "underflow", nome, 0, self._capacity,
                            f"atendimento sem cliente na fila {nome}: {falha}",
                        )
                        errors.append(erro)
                        log.append(
                            LogEntry(tick, event, "erro", nome, None, erro.message)
                        )
                        continue
                    departures += 1
                    registro = ServiceRecord(
                        client=cliente,
                        queue=nome,
                        service_order=departures,
                        arrival_tick=chegada,
                        service_tick=tick,
                    )
                    index.insert(cliente, registro, counters)
                    log.append(
                        LogEntry(
                            tick, event, "saida", nome, str(cliente),
                            f"{cliente} atendido (ordem {departures}, "
                            f"esperou {registro.wait_ticks} instante(s))",
                        )
                    )

        index.check_invariants()
        for fila in filas.values():
            fila.check_invariants()

        return SimulationResult(
            events=events,
            log=log,
            snapshots=snapshots,
            errors=errors,
            index=index,
            arrivals=arrivals,
            departures=departures,
            remaining={
                nome: [str(cliente) for cliente, _ in fila]
                for nome, fila in filas.items()
            },
            counters=counters,
            elapsed_s=elapsed[0],
        )

    def _snapshot(
        self,
        tick: int,
        trigger: str,
        filas: dict[str, Queue[tuple[ClientId, int]]],
    ) -> Snapshot:
        """Fotografa o estado das filas. Custo: O(q · ocupação)."""
        return Snapshot(
            tick=tick,
            trigger=trigger,
            sizes={nome: len(fila) for nome, fila in filas.items()},
            contents={
                nome: [str(cliente) for cliente, _ in fila]
                for nome, fila in filas.items()
            },
        )


# ----------------------------------------------------------------------
# Consultas ao índice
# ----------------------------------------------------------------------
def find_client(
    result: SimulationResult, identifier: str, counters: Counters | None = None
) -> ServiceRecord:
    """Busca um atendimento pelo identificador completo (``"B17"``).

    Levanta :class:`ClientNotFoundError` se o cliente não foi atendido — o que
    é diferente de "não existe": ele pode estar na fila até agora.

    Custo: O(h) — O(log s) em índice equilibrado, O(s) no degenerado.
    """
    chave = ClientId.parse(identifier)
    registro = result.index.get_or(chave, None, counters)
    if registro is None:
        raise ClientNotFoundError(
            f"cliente {identifier!r} não está no índice de atendidos "
            f"({len(result.index)} atendimento(s)). Ele pode não ter sido "
            "atendido ainda, ou nunca ter existido."
        )
    return registro


def list_served(
    result: SimulationResult, counters: Counters | None = None
) -> list[ServiceRecord]:
    """Todos os atendimentos em ordem crescente de identificador.

    A ordem é ``(fila, sequência)``: todos os ``A`` numericamente ordenados,
    depois os ``B``, e assim por diante (decisão de projeto 1).

    Custo: O(s).
    """
    return [registro for _, registro in result.index.in_order_items(counters)]


def list_served_by_queue(
    result: SimulationResult, queue: str
) -> list[ServiceRecord]:
    """Atendimentos de uma fila, em ordem numérica. Custo: O(s).

    Percorre o índice inteiro e filtra. Uma BST por fila deixaria isto em
    O(atendidos_da_fila), ao preço de manter quatro índices — troca que não
    compensa para os volumes deste exercício, e que fica registrada aqui como a
    otimização óbvia caso compensasse.
    """
    alvo = queue.upper()
    return [
        registro
        for _, registro in result.index.in_order_items()
        if registro.queue == alvo
    ]


#: Custo de cada operação do simulador, para a tabela do notebook.
COST_TABLE: list[dict[str, str]] = [
    {"operacao": "chegada (minúscula)", "custo": "O(1)",
     "por_que": "enqueue na fila circular de capacidade fixa do ex 5"},
    {"operacao": "saída (maiúscula)", "custo": "O(h)",
     "por_que": "dequeue O(1) mais a inserção no índice, que custa a altura"},
    {"operacao": "snapshot (não alfabético)", "custo": "O(q · c)",
     "por_que": "copia o conteúdo das q filas, c = ocupação"},
    {"operacao": "evento inválido", "custo": "O(1)",
     "por_que": "só registra o erro e segue"},
    {"operacao": "find_client", "custo": "O(h)",
     "por_que": "uma descida na BST"},
    {"operacao": "list_served (in-order)", "custo": "O(s)",
     "por_que": "visita cada atendimento uma vez"},
    {"operacao": "list_served_by_queue", "custo": "O(s)",
     "por_que": "percorre tudo e filtra; um índice por fila daria O(s_fila)"},
    {"operacao": "simulação completa", "custo": "O(e + s·h)",
     "por_que": "e eventos mais s inserções no índice"},
]


# ----------------------------------------------------------------------
# Geradores e experimentos
# ----------------------------------------------------------------------
def generate_events(
    length: int,
    *,
    seed: int = RANDOM_SEED,
    arrival_weight: float = 0.5,
    snapshot_weight: float = 0.05,
    invalid_weight: float = 0.02,
    queue_names: Iterable[str] = QUEUE_NAMES,
) -> str:
    """Gera uma string de eventos reprodutível.

    Os pesos controlam a mistura: chegadas, saídas, snapshots e eventos
    inválidos. Com ``arrival_weight`` pouco acima de 0,5 as filas enchem e o
    overflow aparece; abaixo, o underflow domina. Os dois casos de borda que o
    enunciado manda tratar precisam acontecer de verdade na simulação.

    Custo: O(length).
    """
    if length < 0:
        raise ValueError(f"length deve ser >= 0, recebido {length}")
    for nome, peso in (
        ("arrival_weight", arrival_weight),
        ("snapshot_weight", snapshot_weight),
        ("invalid_weight", invalid_weight),
    ):
        if not 0.0 <= peso <= 1.0:
            raise ValueError(f"{nome} deve estar em [0, 1], recebido {peso}")
    if snapshot_weight + invalid_weight > 1.0:
        raise ValueError("snapshot_weight + invalid_weight não pode passar de 1")

    nomes = tuple(nome.upper() for nome in queue_names)
    rng = random.Random(seed)
    eventos: list[str] = []
    for _ in range(length):
        sorteio = rng.random()
        if sorteio < snapshot_weight:
            eventos.append(rng.choice("0123456789.!*#"))
        elif sorteio < snapshot_weight + invalid_weight:
            eventos.append(rng.choice("xyzw"))
        else:
            fila = rng.choice(nomes)
            eventos.append(
                fila.lower() if rng.random() < arrival_weight else fila.upper()
            )
    return "".join(eventos)


def bench_simulation(
    lengths: Iterable[int] = (200, 1_000, 5_000, 20_000),
    *,
    capacity: int = DEFAULT_CAPACITY,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Simulações de tamanhos crescentes, com o custo do índice separado.

    A coluna que interessa é ``comparacoes_por_atendimento``: ela é o ``h``
    médio do índice, e cresce com o número de atendidos porque os
    identificadores chegam **crescentes dentro de cada fila** — o cenário de
    degeneração do exercício 3, reaparecendo num contexto aplicado.

    Custo: O(Σ e + Σ s·h).
    """
    simulador = CheckoutSimulator(capacity)
    rows: list[dict[str, Any]] = []
    for length in lengths:
        if length <= 0:
            raise ValueError(f"tamanho deve ser positivo, recebido {length}")
        eventos = generate_events(length, seed=seed)
        resultado = simulador.run(eventos)
        stats = resultado.stats()
        atendidos = stats["atendidos"]
        rows.append(
            {
                "exercicio": "ex6",
                "algoritmo": "simulador de checkout",
                "n": length,
                "padrao": f"capacidade {capacity}, seed {seed}",
                **stats,
                "comparacoes_por_atendimento": (
                    stats["comparisons"] / atendidos if atendidos else 0.0
                ),
                "altura/log2s": (
                    stats["indice_altura"] / math.log2(atendidos)
                    if atendidos > 1
                    else float("nan")
                ),
            }
        )
    return rows


def bench_index_shape(
    lengths: Iterable[int] = (400, 1_600, 6_400, 25_600),
    *,
    capacity: int = DEFAULT_CAPACITY,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Por que o índice degenera, e quanto custaria não deixar.

    Compara duas ordens de inserção dos MESMOS atendimentos:

    * ``cronológica`` — a ordem real da simulação. Dentro de cada fila os
      identificadores são crescentes, então o índice vira uma espinha por fila e
      a altura cresce como ``s/q``;
    * ``embaralhada`` — os mesmos registros inseridos em ordem aleatória, que é
      o que uma BST equilibrada (ou um embaralhamento prévio) entregaria.
      Altura ``≈ 3·log₂s``.

    A diferença entre as duas colunas é o preço de indexar chave que chega
    ordenada — e a justificativa quantitativa para trocar a BST simples por uma
    auto-balanceada caso o volume cresça.

    Custo: O(Σ e + Σ s²) no lado cronológico.
    """
    simulador = CheckoutSimulator(capacity)
    rows: list[dict[str, Any]] = []
    for length in lengths:
        if length <= 0:
            raise ValueError(f"tamanho deve ser positivo, recebido {length}")
        resultado = simulador.run(generate_events(length, seed=seed))
        registros = [
            registro
            for _, registro in sorted(
                (
                    (registro.service_order, registro)
                    for _, registro in resultado.index.in_order_items()
                ),
                key=lambda par: par[0],
            )
        ]
        atendidos = len(registros)
        if atendidos < 2:
            continue

        embaralhados = list(registros)
        random.Random(seed).shuffle(embaralhados)

        for ordem, sequencia in (
            ("cronológica", registros),
            ("embaralhada", embaralhados),
        ):
            counters = Counters()
            arvore: BinarySearchTree[ClientId, ServiceRecord] = BinarySearchTree()
            with stopwatch() as elapsed:
                for registro in sequencia:
                    arvore.insert(registro.client, registro, counters)
            arvore.check_invariants()
            relatorio = arvore.balance_report()

            busca = Counters()
            arvore.search(registros[-1].client, busca)

            rows.append(
                {
                    "exercicio": "ex6",
                    "algoritmo": "índice de atendidos — forma da BST",
                    "n": length,
                    "padrao": ordem,
                    "ordem_de_insercao": ordem,
                    "atendidos": atendidos,
                    "altura": relatorio["altura"],
                    "altura_ideal": relatorio["altura_ideal"],
                    "razao_altura": relatorio["razao_altura"],
                    "altura/log2s": relatorio["altura"] / math.log2(atendidos),
                    "altura/atendidos": relatorio["altura"] / atendidos,
                    "comparacoes_construcao": counters.comparisons,
                    "comparacoes_por_insercao": counters.comparisons / atendidos,
                    "comparacoes_busca": busca.comparisons,
                    "tempo_s": elapsed[0],
                }
            )
    return rows

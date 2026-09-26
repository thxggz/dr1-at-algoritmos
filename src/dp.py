"""Exercício 8 — Programação dinâmica e memoization com hashtable própria.

Problema escolhido: ``min_coins(amount, coins)`` — o menor número de moedas que
soma exatamente ``amount``, com moedas de valores ``coins`` em quantidade
ilimitada.

Decisões de projeto
-------------------
1. **``min_coins`` em vez de knapsack.** Os dois têm sobreposição de
   subproblemas, mas o estado do knapsack é o par ``(índice, capacidade)`` e o
   do ``min_coins`` é **um único inteiro**. Com estado unidimensional a árvore
   de subproblemas cabe na tela, a sobreposição é visível a olho nu
   (``f(7)`` aparece em vários ramos) e a justificativa da chave do memo fica
   verificável em vez de declarativa.

2. **O estado mínimo suficiente é ``amount``, e só.**
   A justificativa: o conjunto de moedas é fixo durante toda a execução, cada
   moeda tem quantidade ilimitada, e o custo ótimo de formar ``a`` não depende
   de quais moedas foram usadas antes nem da ordem em que foram escolhidas —
   depende apenas de ``a``. Formalmente, ``f(a) = 1 + min_c f(a − c)``: o lado
   direito só menciona ``a`` e o conjunto fixo ``coins``. Incluir o caminho
   percorrido no estado seria correto e inútil: criaria estados distintos com a
   mesma resposta e destruiria a sobreposição, que é justamente o que a
   memoization explora. :func:`bench_key_granularity` mede esse erro.

3. **O memo é a ``HashTableChained`` do exercício 4.** ``dict`` e
   ``functools.lru_cache`` estão proibidos pela convenção 9 — e há ganho
   colateral: as comparações feitas dentro dos buckets entram no contador, então
   o custo do memo aparece na tabela em vez de ser tratado como gratuito.

4. **A versão ingênua tem orçamento de chamadas.** Ela é exponencial: com
   ``coins=[1,3,4]`` o número de chamadas cresce como ``≈1,8^amount``, e
   ``amount=40`` já não termina. :class:`CallBudgetExceeded` corta a execução e
   informa onde parou — melhor do que travar o notebook, e a própria exceção
   vira evidência do custo.

5. **A árvore de subproblemas registra o motivo de cada nó parar.**
   ``base``, ``memo`` (estado já calculado), ``inviavel`` ou ``expandido``. Sem
   o motivo, a árvore memoizada pareceria só "menor"; com ele dá para contar
   exatamente quantas expansões a memoization evitou.

Custo em Big O
--------------
Com ``a`` = ``amount``, ``k`` = número de moedas e ``m`` = menor moeda:

==========================  =========================  ===================
Versão                      Tempo                      Memória
==========================  =========================  ===================
recursiva pura              Θ(k^(a/m)) — exponencial   Θ(a/m) de pilha
memoizada (topo-baixo)      Θ(a·k)                     Θ(a) + Θ(a/m) pilha
iterativa (baixo-cima)      Θ(a·k)                     Θ(a), sem pilha
==========================  =========================  ===================

A memoization muda a **classe**: de exponencial para pseudopolinomial Θ(a·k).
"Pseudopolinomial" porque ``a`` é o *valor* da entrada, não o tamanho dela — um
``amount`` com 40 bits custa Θ(2⁴⁰·k). Vale registrar: chamar Θ(a·k) de
"polinomial" seria erro de análise.

A versão iterativa tem a mesma classe de tempo da memoizada e troca a pilha de
chamadas por um laço, o que remove o teto de ``sys.getrecursionlimit()`` —
a mesma lição dos exercícios 5, 7 e 9.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from src.counters import Counters, ratio, stopwatch
from src.hashtable import HashTableChained
from src.stack_queue import Queue

#: Devolvido quando não existe combinação de moedas que some ``amount``.
INFEASIBLE: int = -1

#: Orçamento padrão de chamadas da versão ingênua (decisão de projeto 4).
DEFAULT_CALL_BUDGET: int = 2_000_000


# ----------------------------------------------------------------------
# Exceções próprias (convenção 9)
# ----------------------------------------------------------------------
class DPError(Exception):
    """Erro base deste módulo."""


class InvalidCoinsError(DPError):
    """Conjunto de moedas inválido (vazio, com valor <= 0 ou repetido)."""


class InvalidAmountError(DPError):
    """``amount`` negativo ou não inteiro."""


class CallBudgetExceeded(DPError):
    """A versão ingênua passou do orçamento de chamadas (decisão 4)."""


class StateNotFoundError(DPError):
    """Estado procurado não aparece na árvore de subproblemas."""


# ----------------------------------------------------------------------
# Validação
# ----------------------------------------------------------------------
def validate_problem(amount: int, coins: Sequence[int]) -> tuple[int, ...]:
    """Valida a entrada e devolve as moedas normalizadas (ordenadas, únicas).

    Ordenar não muda a resposta, mas torna a ordem de expansão determinística —
    e portanto a árvore de subproblemas reprodutível, que é requisito da
    seção 5 do CLAUDE.md.

    Custo: O(k log k).
    """
    if not isinstance(amount, int) or isinstance(amount, bool):
        raise InvalidAmountError(
            f"amount deve ser int, recebido {type(amount).__name__}"
        )
    if amount < 0:
        raise InvalidAmountError(f"amount deve ser >= 0, recebido {amount}")
    if not coins:
        raise InvalidCoinsError("é preciso ao menos uma moeda")
    for coin in coins:
        if not isinstance(coin, int) or isinstance(coin, bool):
            raise InvalidCoinsError(
                f"moeda deve ser int, recebida {type(coin).__name__}"
            )
        if coin <= 0:
            raise InvalidCoinsError(f"moeda deve ser > 0, recebida {coin}")
    unicas = tuple(sorted(set(coins)))
    if len(unicas) != len(coins):
        raise InvalidCoinsError(
            f"moedas repetidas em {list(coins)}: repetição não muda a resposta "
            "e multiplicaria as transições da árvore de subproblemas"
        )
    return unicas


# ----------------------------------------------------------------------
# Versão recursiva pura
# ----------------------------------------------------------------------
@dataclass(slots=True)
class DPResult:
    """Resultado de uma resolução, com as medidas que o exercício pede."""

    value: int
    counters: Counters
    strategy: str
    distinct_states: int
    max_depth: int
    memo: HashTableChained[int, int] | None = None

    def as_tuple(self) -> tuple[int, Counters]:
        """O par ``(resultado, Counters)`` do protocolo. Custo: O(1)."""
        return self.value, self.counters


def min_coins_naive(
    amount: int,
    coins: Sequence[int],
    counters: Counters | None = None,
    *,
    call_budget: int = DEFAULT_CALL_BUDGET,
) -> DPResult:
    """Recursão pura, sem memo: ``f(a) = 1 + min_c f(a − c)``.

    Conta ``calls`` (todas as invocações) e ``distinct_states`` (quantos valores
    de ``a`` diferentes chegaram a ser visitados). A distância entre os dois é a
    própria sobreposição de subproblemas — e é o número que a memoization
    transforma em ganho.

    Levanta :class:`CallBudgetExceeded` ao passar de ``call_budget`` chamadas.

    Custo: Θ(k^(a/m)) em tempo — exponencial —, Θ(a/m) de pilha de chamadas.
    """
    moedas = validate_problem(amount, coins)
    counters = counters if counters is not None else Counters()
    vistos: set[int] = set()
    mais_fundo = [0]

    def resolver(restante: int, profundidade: int) -> float:
        counters.count_call()
        if counters.calls > call_budget:
            raise CallBudgetExceeded(
                f"a versão ingênua passou de {call_budget:,} chamadas com "
                f"amount={amount} e coins={list(moedas)}. O crescimento é "
                "exponencial: é exatamente o que a memoization resolve."
            )
        if profundidade > mais_fundo[0]:
            mais_fundo[0] = profundidade
        vistos.add(restante)

        if restante == 0:
            return 0
        melhor: float = math.inf
        for moeda in moedas:
            if counters.gt(moeda, restante):
                continue
            candidato = resolver(restante - moeda, profundidade + 1)
            if counters.lt(candidato + 1, melhor):
                melhor = candidato + 1
        return melhor

    resultado = resolver(amount, 1)
    return DPResult(
        value=INFEASIBLE if math.isinf(resultado) else int(resultado),
        counters=counters,
        strategy="recursiva pura",
        distinct_states=len(vistos),
        max_depth=mais_fundo[0],
    )


def min_coins_memo(
    amount: int, coins: Sequence[int], counters: Counters | None = None
) -> DPResult:
    """A mesma recursão, com memo na :class:`HashTableChained` do exercício 4.

    A chave é o estado mínimo suficiente — o valor restante (decisão 2). Cada
    estado é resolvido **uma vez**; as visitas seguintes custam uma consulta
    O(1) na tabela.

    O memo fica em ``DPResult.memo``, para que o notebook possa inspecionar a
    tabela de estados resolvidos sem recalcular.

    Custo: Θ(a·k) em tempo, Θ(a) de memo mais Θ(a/m) de pilha de chamadas.
    """
    moedas = validate_problem(amount, coins)
    counters = counters if counters is not None else Counters()
    memo: HashTableChained[int, int] = HashTableChained()
    mais_fundo = [0]

    def resolver(restante: int, profundidade: int) -> float:
        counters.count_call()
        if profundidade > mais_fundo[0]:
            mais_fundo[0] = profundidade
        if restante == 0:
            return 0
        # get_or faz UMA travessia do bucket (ver decisão 4 do ex 4)
        guardado = memo.get_or(restante, None, counters)
        if guardado is not None:
            return math.inf if guardado == INFEASIBLE else guardado

        melhor: float = math.inf
        for moeda in moedas:
            if counters.gt(moeda, restante):
                continue
            candidato = resolver(restante - moeda, profundidade + 1)
            if counters.lt(candidato + 1, melhor):
                melhor = candidato + 1
        memo.put(
            restante, INFEASIBLE if math.isinf(melhor) else int(melhor), counters
        )
        return melhor

    resultado = resolver(amount, 1)
    return DPResult(
        value=INFEASIBLE if math.isinf(resultado) else int(resultado),
        counters=counters,
        strategy="memoizada (hashtable própria)",
        distinct_states=len(memo),
        max_depth=mais_fundo[0],
        memo=memo,
    )


def min_coins_bottom_up(
    amount: int, coins: Sequence[int], counters: Counters | None = None
) -> DPResult:
    """A mesma DP de baixo para cima, sem recursão nenhuma.

    Resolve ``f(0), f(1), …, f(a)`` em ordem. Mesma classe de tempo da
    memoizada, Θ(a·k), e **nenhuma** pilha de chamadas — some o teto de
    ``sys.getrecursionlimit()``, que nas versões recursivas limita ``amount`` a
    cerca de ``1000·m``.

    O preço: calcula **todos** os estados de 0 a ``a``, inclusive os que a
    versão topo-baixo nunca visitaria. Com moedas grandes e ``amount`` que não
    as alcança, a memoizada visita menos estados — e é isso que
    :func:`bench_memoization` mostra na coluna de estados distintos.

    Custo: Θ(a·k) em tempo, Θ(a) de memória, O(1) de pilha.
    """
    moedas = validate_problem(amount, coins)
    counters = counters if counters is not None else Counters()
    melhor: list[float] = [math.inf] * (amount + 1)
    melhor[0] = 0
    for alvo in range(1, amount + 1):
        counters.count_call()
        for moeda in moedas:
            if counters.gt(moeda, alvo):
                continue
            candidato = melhor[alvo - moeda] + 1
            if counters.lt(candidato, melhor[alvo]):
                counters.write(melhor, alvo, candidato)  # type: ignore[arg-type]
    resultado = melhor[amount]
    return DPResult(
        value=INFEASIBLE if math.isinf(resultado) else int(resultado),
        counters=counters,
        strategy="iterativa (baixo-cima)",
        distinct_states=amount + 1,
        max_depth=0,
    )


def min_coins_naive_counted(
    amount: int, coins: Sequence[int]
) -> tuple[int, Counters]:
    """:func:`min_coins_naive` no protocolo ``(resultado, Counters)``."""
    return min_coins_naive(amount, coins).as_tuple()


def min_coins_memo_counted(
    amount: int, coins: Sequence[int]
) -> tuple[int, Counters]:
    """:func:`min_coins_memo` no protocolo ``(resultado, Counters)``."""
    return min_coins_memo(amount, coins).as_tuple()


# ----------------------------------------------------------------------
# Árvore de subproblemas
# ----------------------------------------------------------------------
@dataclass(slots=True)
class SubproblemNode:
    """Nó da árvore de subproblemas: um estado e as transições que saíram dele.

    ``status`` diz **por que** o nó parou (decisão de projeto 5):

    * ``"expandido"`` — gerou filhos, um por moeda aplicável;
    * ``"base"`` — estado 0, o caso base;
    * ``"memo"`` — estado já resolvido antes; a memoization cortou aqui;
    * ``"inviavel"`` — nenhuma moeda cabe no restante.
    """

    state: int
    status: str
    coin_used: int | None = None
    value: int | None = None
    children: list["SubproblemNode"] = field(default_factory=list)

    def is_leaf(self) -> bool:
        """Custo: O(1)."""
        return not self.children

    def __repr__(self) -> str:
        moeda = f" via {self.coin_used}" if self.coin_used is not None else ""
        return f"SubproblemNode(f({self.state}){moeda}, {self.status})"


def build_subproblem_tree(
    amount: int,
    coins: Sequence[int],
    *,
    memoize: bool = True,
    call_budget: int = DEFAULT_CALL_BUDGET,
    counters: Counters | None = None,
) -> tuple[SubproblemNode, DPResult]:
    """Árvore dos subproblemas visitados em **uma** execução.

    Cada nó é um estado ``f(a)``; cada filho é a transição por uma moeda, isto é
    ``f(a − c)``. Com ``memoize=False`` sai a árvore de recursão completa
    (exponencial); com ``memoize=True`` sai a árvore podada, em que estados já
    resolvidos viram folhas ``"memo"``.

    Comparar as duas é a forma visual de mostrar a sobreposição: a quantidade de
    folhas ``"memo"`` é exatamente o número de reexpansões que a memoization
    evitou.

    Custo: o mesmo da resolução correspondente, mais Θ(nós) de memória para a
    árvore.
    """
    moedas = validate_problem(amount, coins)
    counters = counters if counters is not None else Counters()
    memo: HashTableChained[int, int] = HashTableChained()
    vistos: set[int] = set()
    mais_fundo = [0]

    def construir(
        restante: int, moeda_usada: int | None, profundidade: int
    ) -> tuple[SubproblemNode, float]:
        counters.count_call()
        if counters.calls > call_budget:
            raise CallBudgetExceeded(
                f"a expansão passou de {call_budget:,} nós com amount={amount}, "
                f"coins={list(moedas)} e memoize={memoize}."
            )
        if profundidade > mais_fundo[0]:
            mais_fundo[0] = profundidade
        vistos.add(restante)

        if restante == 0:
            return SubproblemNode(0, "base", moeda_usada, 0), 0

        if memoize:
            guardado = memo.get_or(restante, None, counters)
            if guardado is not None:
                return (
                    SubproblemNode(restante, "memo", moeda_usada, guardado),
                    math.inf if guardado == INFEASIBLE else guardado,
                )

        filhos: list[SubproblemNode] = []
        melhor: float = math.inf
        for moeda in moedas:
            if counters.gt(moeda, restante):
                continue
            filho, custo = construir(restante - moeda, moeda, profundidade + 1)
            filhos.append(filho)
            if counters.lt(custo + 1, melhor):
                melhor = custo + 1

        valor = INFEASIBLE if math.isinf(melhor) else int(melhor)
        if memoize:
            memo.put(restante, valor, counters)
        status = "inviavel" if not filhos else "expandido"
        return SubproblemNode(restante, status, moeda_usada, valor, filhos), melhor

    raiz, resultado = construir(amount, None, 1)
    return raiz, DPResult(
        value=INFEASIBLE if math.isinf(resultado) else int(resultado),
        counters=counters,
        strategy=(
            "árvore memoizada" if memoize else "árvore de recursão completa"
        ),
        distinct_states=len(memo) if memoize else len(vistos),
        max_depth=mais_fundo[0],
        memo=memo if memoize else None,
    )


def count_tree(root: SubproblemNode) -> dict[str, int]:
    """Nós da árvore por ``status``, mais o total. Custo: Θ(nós), iterativa."""
    contagem = {"total": 0, "expandido": 0, "base": 0, "memo": 0, "inviavel": 0}
    pendentes: list[SubproblemNode] = [root]
    while pendentes:
        node = pendentes.pop()
        contagem["total"] += 1
        contagem[node.status] += 1
        pendentes.extend(node.children)
    return contagem


def search_subproblem_tree(
    root: SubproblemNode, state: int, counters: Counters | None = None
) -> list[SubproblemNode]:
    """Todos os nós da árvore que representam ``state``, em largura.

    Devolve uma **lista** porque o mesmo estado costuma aparecer em vários
    ramos — é essa multiplicidade que a memoization elimina, e escondê-la atrás
    de "o primeiro que achar" apagaria a evidência.

    Percurso em largura com a :class:`~src.stack_queue.Queue` do exercício 5, o
    que dá os nós em ordem de profundidade crescente: o primeiro da lista é a
    ocorrência mais rasa.

    **Custo: Θ(V)**, ``V`` = nós da árvore — e é esse o ponto da discussão. A
    árvore não é indexada por estado, então buscar nela é varredura. O memo,
    que é a mesma informação numa hashtable, responde em O(1). Guardar a árvore
    serve para *explicar* a execução, não para consultar durante ela; usá-la
    como índice desfaria o ganho da DP.

    Levanta :class:`StateNotFoundError` se o estado não aparece.
    """
    counters = counters if counters is not None else Counters()
    total = count_tree(root)["total"]
    fila: Queue[SubproblemNode] = Queue(max(1, total))
    fila.enqueue(root, counters)
    encontrados: list[SubproblemNode] = []
    while not fila.is_empty():
        node = fila.dequeue(counters)
        if counters.eq(node.state, state):
            encontrados.append(node)
        for child in node.children:
            fila.enqueue(child, counters)
    if not encontrados:
        raise StateNotFoundError(
            f"o estado f({state}) não aparece nesta árvore de {total} nó(s)"
        )
    return encontrados


def tree_to_string(root: SubproblemNode, *, max_lines: int = 60) -> str:
    """Árvore em texto indentado, para o notebook. Custo: Θ(nós)."""
    linhas: list[str] = []
    pendentes: list[tuple[SubproblemNode, int]] = [(root, 0)]
    while pendentes and len(linhas) < max_lines:
        node, nivel = pendentes.pop()
        # Sem setas unicode: o console do Windows usa cp1252 e quebraria na
        # exibição, e o PDF passa por esse mesmo caminho.
        marca = {
            "expandido": "", "base": "  <- base",
            "memo": "  <- memo (ja resolvido)", "inviavel": "  <- inviavel",
        }[node.status]
        via = f" [+{node.coin_used}]" if node.coin_used is not None else ""
        linhas.append("  " * nivel + f"f({node.state}){via}{marca}")
        for child in reversed(node.children):
            pendentes.append((child, nivel + 1))
    if pendentes:
        linhas.append(f"... ({len(pendentes)} subárvore(s) omitida(s))")
    return "\n".join(linhas)


# ----------------------------------------------------------------------
# Experimentos do exercício 8
# ----------------------------------------------------------------------
def bench_memoization(
    amounts: Iterable[int] = (8, 12, 16, 20, 24),
    coins: Sequence[int] = (1, 3, 4),
    *,
    call_budget: int = DEFAULT_CALL_BUDGET,
) -> list[dict[str, Any]]:
    """Ingênua × memoizada × iterativa, lado a lado.

    Colunas que importam:

    * ``chamadas_ingenua`` cresce como ``≈1,8^amount`` com ``coins=[1,3,4]``;
    * ``chamadas_memo`` cresce **linearmente**, porque cada estado é resolvido
      uma vez e cada resolução dispara ``k`` chamadas;
    * ``estados_distintos`` é ``amount`` — a prova de que o espaço de estados é
      pequeno e de que a explosão da versão ingênua é toda de repetição;
    * ``reducao_de_chamadas`` é o ganho quantitativo que o item 3.5 da rubrica
      pede.

    Custo: exponencial no lado ingênuo — por isso ``amounts`` para em 24.
    """
    rows: list[dict[str, Any]] = []
    for amount in amounts:
        memoizada = min_coins_memo(amount, coins)
        iterativa = min_coins_bottom_up(amount, coins)

        estourou = False
        try:
            ingenua = min_coins_naive(amount, coins, call_budget=call_budget)
            chamadas_ingenua: int | None = ingenua.counters.calls
            valor_ingenuo: int | None = ingenua.value
            estados_ingenua: int | None = ingenua.distinct_states
        except CallBudgetExceeded:
            estourou = True
            chamadas_ingenua = None
            valor_ingenuo = None
            estados_ingenua = None

        if not estourou and valor_ingenuo != memoizada.value:
            raise AssertionError(
                f"ingênua e memoizada discordaram em amount={amount}"
            )
        if iterativa.value != memoizada.value:
            raise AssertionError(
                f"iterativa e memoizada discordaram em amount={amount}"
            )

        rows.append(
            {
                "exercicio": "ex8",
                "algoritmo": "min_coins: ingênua × memoizada × iterativa",
                "n": amount,
                "padrao": f"coins={list(coins)}",
                "amount": amount,
                "moedas": len(set(coins)),
                "resposta": memoizada.value,
                "estados_distintos": memoizada.distinct_states,
                "chamadas_ingenua": chamadas_ingenua,
                "chamadas_memo": memoizada.counters.calls,
                "chamadas_iterativa": iterativa.counters.calls,
                "orcamento_estourado": estourou,
                "reducao_de_chamadas": (
                    chamadas_ingenua / memoizada.counters.calls
                    if chamadas_ingenua
                    else math.inf
                ),
                "ingenua/1.8^a": (
                    chamadas_ingenua / (1.8**amount) if chamadas_ingenua else None
                ),
                "memo/(a*k)": (
                    memoizada.counters.calls / (amount * len(set(coins)))
                    if amount
                    else float("nan")
                ),
                "comparacoes_ingenua": (
                    ingenua.counters.comparisons if not estourou else None
                ),
                "comparacoes_memo": memoizada.counters.comparisons,
                "profundidade_memo": memoizada.max_depth,
                "estados_visitados_ingenua": estados_ingenua,
            }
        )
    return rows


def bench_memo_scaling(
    amounts: Iterable[int] = (100, 200, 400, 800, 1_600),
    coins: Sequence[int] = (1, 3, 4, 7),
) -> list[dict[str, Any]]:
    """A memoizada em escalas que a ingênua não alcança.

    Previsão: ``chamadas/(a·k)`` fica aproximadamente constante — é a evidência
    de Θ(a·k). A versão iterativa serve de controle: mesma classe, sem pilha.

    Os quatro primeiros ``amounts`` cabem na pilha; o último (1.600) **não**,
    e é de propósito: a memoizada continua recursiva e sua profundidade é
    ``a/m``, então com ``m = 1`` ela quebra pouco acima de
    ``sys.getrecursionlimit()``. A linha com ``memo_quebrou=True`` é evidência,
    não falha — mostra o teto na mesma tabela em que se mede a classe.

    Custo: Θ(Σ a·k).
    """
    rows: list[dict[str, Any]] = []
    moedas = validate_problem(1, coins)
    for amount in amounts:
        if amount <= 0:
            raise ValueError(f"amount deve ser positivo, recebido {amount}")

        iterativa = min_coins_bottom_up(amount, coins)
        with stopwatch() as tempo_iter:
            min_coins_bottom_up(amount, coins)

        quebrou = False
        try:
            with stopwatch() as tempo_memo:
                memoizada = min_coins_memo(amount, coins)
        except RecursionError:
            quebrou = True

        rows.append(
            {
                "exercicio": "ex8",
                "algoritmo": "min_coins memoizada × iterativa",
                "n": amount,
                "padrao": f"coins={list(moedas)}",
                "amount": amount,
                "moedas": len(moedas),
                "resposta": iterativa.value,
                "memo_quebrou": quebrou,
                "chamadas_memo": None if quebrou else memoizada.counters.calls,
                "memo/(a*k)": (
                    None
                    if quebrou
                    else memoizada.counters.calls / (amount * len(moedas))
                ),
                "comparacoes_memo": None if quebrou else memoizada.counters.comparisons,
                "profundidade_memo": None if quebrou else memoizada.max_depth,
                "chamadas_iterativa": iterativa.counters.calls,
                "comparacoes_iterativa": iterativa.counters.comparisons,
                "iterativa/(a*k)": iterativa.counters.comparisons
                / (amount * len(moedas)),
                "tempo_memo_s": None if quebrou else tempo_memo[0],
                "tempo_iterativa_s": tempo_iter[0],
            }
        )
    return rows


def bench_subproblem_tree(
    amounts: Iterable[int] = (8, 12, 16, 20),
    coins: Sequence[int] = (1, 3, 4),
    *,
    call_budget: int = DEFAULT_CALL_BUDGET,
) -> list[dict[str, Any]]:
    """Tamanho da árvore de subproblemas com e sem memo.

    A coluna ``folhas_memo`` conta os nós em que a memoization cortou a
    expansão. Cada uma dessas folhas é uma subárvore inteira que não foi
    construída — é a sobreposição, contada.

    Também mede o custo de :func:`search_subproblem_tree`: Θ(V) contra O(1) da
    consulta ao memo, que é a discussão que o enunciado pede.

    Custo: exponencial no lado sem memo.
    """
    rows: list[dict[str, Any]] = []
    for amount in amounts:
        raiz_memo, resultado_memo = build_subproblem_tree(
            amount, coins, memoize=True, call_budget=call_budget
        )
        contagem_memo = count_tree(raiz_memo)

        estourou = False
        try:
            raiz_cheia, _ = build_subproblem_tree(
                amount, coins, memoize=False, call_budget=call_budget
            )
            contagem_cheia: dict[str, int] | None = count_tree(raiz_cheia)
        except CallBudgetExceeded:
            estourou = True
            contagem_cheia = None

        alvo = max(1, amount // 2)
        busca = Counters()
        with stopwatch() as tempo_busca:
            ocorrencias = search_subproblem_tree(raiz_memo, alvo, busca)

        consulta_memo = Counters()
        assert resultado_memo.memo is not None
        resultado_memo.memo.get_or(alvo, None, consulta_memo)

        rows.append(
            {
                "exercicio": "ex8",
                "algoritmo": "árvore de subproblemas",
                "n": amount,
                "padrao": f"coins={list(coins)}",
                "amount": amount,
                "nos_com_memo": contagem_memo["total"],
                "nos_sem_memo": None if estourou else contagem_cheia["total"],  # type: ignore[index]
                "orcamento_estourado": estourou,
                "reducao_de_nos": (
                    contagem_cheia["total"] / contagem_memo["total"]  # type: ignore[index]
                    if not estourou
                    else math.inf
                ),
                "folhas_memo": contagem_memo["memo"],
                "folhas_base": contagem_memo["base"],
                "nos_expandidos": contagem_memo["expandido"],
                "estado_buscado": alvo,
                "ocorrencias_na_arvore": len(ocorrencias),
                "nos_visitados_na_busca": contagem_memo["total"],
                "comparacoes_busca_arvore": busca.comparisons,
                "comparacoes_consulta_memo": consulta_memo.comparisons,
                "vantagem_do_memo": (
                    busca.comparisons / max(1, consulta_memo.comparisons)
                ),
                "tempo_busca_s": tempo_busca[0],
            }
        )
    return rows


def bench_key_granularity(
    amount: int = 18,
    coins: Sequence[int] = (1, 3, 4),
    *,
    call_budget: int = DEFAULT_CALL_BUDGET,
) -> list[dict[str, Any]]:
    """Por que a chave do memo tem de ser o estado **mínimo** suficiente.

    Compara três granularidades de chave sobre o mesmo problema:

    * ``restante`` — o estado mínimo suficiente (decisão de projeto 2);
    * ``(restante, profundidade)`` — correta, porém fina demais: estados iguais
      em profundidades diferentes viram chaves diferentes, então quase nenhuma
      consulta acerta e o memo vira peso morto;
    * ``(restante, última moeda)`` — o erro sutil: parece razoável guardar por
      qual moeda se chegou, mas isso multiplica o espaço de estados por ``k``
      sem mudar nenhuma resposta.

    As três dão **o mesmo valor** — a diferença está só no número de chamadas e
    de chaves guardadas. É isso que transforma "a chave deve ser o estado
    mínimo" de afirmação em medição.

    Custo: pode ser exponencial na granularidade mais fina.
    """
    moedas = validate_problem(amount, coins)
    rows: list[dict[str, Any]] = []

    for rotulo, chave in (
        ("restante (mínimo suficiente)", lambda r, d, c: r),
        ("(restante, profundidade)", lambda r, d, c: (r, d)),
        ("(restante, última moeda)", lambda r, d, c: (r, c if c else 0)),
    ):
        counters = Counters()
        memo: HashTableChained[Any, int] = HashTableChained()
        estourou = False

        def resolver(restante: int, profundidade: int, ultima: int | None) -> float:
            counters.count_call()
            if counters.calls > call_budget:
                raise CallBudgetExceeded(
                    f"granularidade {rotulo!r} passou de {call_budget:,} chamadas"
                )
            if restante == 0:
                return 0
            k = chave(restante, profundidade, ultima)
            guardado = memo.get_or(k, None, counters)
            if guardado is not None:
                return math.inf if guardado == INFEASIBLE else guardado
            melhor: float = math.inf
            for moeda in moedas:
                if counters.gt(moeda, restante):
                    continue
                candidato = resolver(restante - moeda, profundidade + 1, moeda)
                if counters.lt(candidato + 1, melhor):
                    melhor = candidato + 1
            memo.put(k, INFEASIBLE if math.isinf(melhor) else int(melhor), counters)
            return melhor

        try:
            valor = resolver(amount, 1, None)
        except CallBudgetExceeded:
            estourou = True
            valor = math.inf

        rows.append(
            {
                "exercicio": "ex8",
                "algoritmo": "granularidade da chave do memo",
                "n": amount,
                "padrao": rotulo,
                "chave": rotulo,
                "amount": amount,
                "resposta": (
                    None
                    if estourou
                    else (INFEASIBLE if math.isinf(valor) else int(valor))
                ),
                "orcamento_estourado": estourou,
                "chaves_guardadas": len(memo),
                "chamadas": counters.calls,
                "comparacoes": counters.comparisons,
                "chaves_por_amount": len(memo) / amount if amount else float("nan"),
            }
        )
    return rows

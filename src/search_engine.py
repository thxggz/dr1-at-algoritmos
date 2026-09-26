"""Exercício 12 — Motor de indexação e consulta (integrador).

Reúne, obrigatoriamente, oito estruturas dos exercícios anteriores:

===========================  ==============================================
Estrutura                    Onde é usada aqui
===========================  ==============================================
``HashTableChained`` (ex 4)  índice invertido termo → ocorrências; memo da DP
``SinglyLinkedList`` (ex 10) lista de ocorrências de cada termo
``BinarySearchTree`` (ex 2)  árvore de termos, para listagem e busca
``Stack`` (ex 5)             shunting-yard e avaliação da consulta
``Queue`` (ex 5)             percurso em largura da BST (relatório)
``quicksort`` (ex 9)         ranqueamento dos documentos por score
recursão (ex 7)              distância de edição
DP com memo (ex 8)           distância de edição memoizada
===========================  ==============================================

Decisões de projeto
-------------------
1. **As listas de ocorrências nascem ordenadas por ``doc_id``, de graça.**
   Os documentos são indexados em ordem crescente de ``doc_id`` e cada
   ocorrência é anexada ao **fim** da lista com ``insert_last``, que é O(1)
   graças ao ponteiro ``tail`` do exercício 10. Manter a ordem não custa nada e
   é o que habilita a otimização 1: interseção e união por varredura simultânea,
   Θ(m+n), em vez de comparação par a par, Θ(m·n).

2. **Dois índices sobre os mesmos termos, com papéis diferentes.**
   A hashtable responde "onde aparece este termo?" em O(1) médio; a BST responde
   "quais termos existem, em ordem?" e "existe algum termo entre X e Y?" em
   O(h). Nenhuma das duas faz bem o trabalho da outra: hashtable não tem ordem,
   BST não tem O(1). O custo é Θ(V) de memória a mais, com ``V`` = vocabulário.

3. **``NOT`` é unário e tem a maior precedência**, seguido de ``AND`` e depois
   ``OR`` — a convenção usual de busca booleana. ``NOT x`` custa Θ(N) porque
   precisa do universo de documentos; por isso o avaliador reordena ``a AND
   NOT b`` para uma diferença direta quando pode, em vez de materializar o
   complemento.

4. **O score usa frequência e posição**, como o enunciado pede:
   ``score(d) = Σ_t tf(t,d) · (1 + 1/(1 + primeira_posição(t,d)))``.
   O termo posicional vale no máximo o dobro e no mínimo quase 1, então ele
   desempata sem dominar a frequência. É uma escolha, não uma verdade — e está
   isolada em :func:`score_document` para ser trocada sem tocar no resto.

5. **A distância de edição memoiza na ``HashTableChained``**, com chave
   ``(i, j)`` — o estado mínimo suficiente, pela mesma razão do exercício 8: o
   custo de alinhar os prefixos ``a[:i]`` e ``b[:j]`` não depende de como se
   chegou até eles.

Custo em Big O
--------------
Com ``D`` documentos, ``T`` tokens no total, ``V`` termos distintos, ``h`` a
altura da BST, ``q`` termos na consulta e ``P`` o tamanho das listas envolvidas:

============================  =======================  =======================
Etapa                         Caso médio               Pior caso
============================  =======================  =======================
tokenizar + indexar           Θ(T + V·h)               Θ(T + V²)
``postings(termo)``           O(1 + α)                 O(V)
listar termos em ordem        Θ(V)                     Θ(V)
parsing (shunting-yard)       Θ(q)                     Θ(q)
avaliar a consulta            Θ(P)                     Θ(P + D) com ``NOT``
ranquear                      Θ(R + R·log R)           Θ(R²)
sugerir termo                 Θ(V·L²) sem poda         —
relatório BFS                 Θ(V)                     Θ(V)
============================  =======================  =======================

``Θ(T + V·h)`` na indexação: cada token custa O(1) na hashtable, e cada termo
**novo** custa uma inserção O(h) na BST. Como os termos chegam em ordem de
texto (quase aleatória), ``h ≈ 3·log₂V`` e o termo ``V·h`` é pequeno perto de
``T``. Se os documentos chegassem com vocabulário já ordenado, a BST degeneraria
e a indexação viraria Θ(T + V²) — o pior caso da tabela, que é o mesmo do
exercício 3 aparecendo pela terceira vez.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from src.bst import BinarySearchTree
from src.counters import Counters, ratio, stopwatch
from src.hashtable import HashTableChained
from src.linked_list import SinglyLinkedList
from src.sorting import quicksort
from src.stack_queue import (
    Queue,
    Stack,
    StackUnderflowError,
)

#: Operadores booleanos e sua precedência. Maior liga mais forte (decisão 3).
PRECEDENCE: dict[str, int] = {"OR": 1, "AND": 2, "NOT": 3}

#: Operadores unários.
UNARY: frozenset[str] = frozenset({"NOT"})

#: Corte padrão de distância na sugestão de termos.
DEFAULT_MAX_DISTANCE: int = 2


# ----------------------------------------------------------------------
# Exceções próprias (convenção 9)
# ----------------------------------------------------------------------
class SearchEngineError(Exception):
    """Erro base do motor de busca."""


class EmptyQueryError(SearchEngineError):
    """Consulta vazia."""


class UnbalancedParenthesesError(SearchEngineError):
    """Parênteses desbalanceados na consulta."""


class MalformedQueryError(SearchEngineError):
    """Consulta sintaticamente inválida (operador sem operandos, sobra…)."""


class TermNotFoundError(SearchEngineError):
    """Termo ausente do vocabulário."""


class DuplicateDocumentError(SearchEngineError):
    """``doc_id`` já indexado."""


# ----------------------------------------------------------------------
# Tokenização
# ----------------------------------------------------------------------
def normalize(text: str) -> str:
    """Minúsculas e sem acento. Custo: O(len(text)).

    Remover acento faz ``"análise"`` e ``"analise"`` virarem o mesmo termo — o
    que é quase sempre o que o usuário quer numa busca em português, e o que
    evita que o índice guarde dois termos para a mesma palavra. O preço é perder
    a distinção de pares como ``"e"``/``"é"``; a troca é consciente.
    """
    sem_acento = unicodedata.normalize("NFKD", text)
    return "".join(
        caractere for caractere in sem_acento if not unicodedata.combining(caractere)
    ).lower()


def tokenize(text: str, counters: Counters | None = None) -> list[str]:
    """Texto → lista de termos normalizados, na ordem em que aparecem.

    Quebra em qualquer caractere que não seja letra ou dígito, então pontuação
    e hífen separam termos. A posição de cada termo na lista é a posição que vai
    para o índice — é ela que sustenta o componente posicional do score.

    Custo: Θ(len(text)).
    """
    counters = counters if counters is not None else Counters()
    termos: list[str] = []
    atual: list[str] = []
    for caractere in normalize(text):
        if caractere.isalnum():
            atual.append(caractere)
        elif atual:
            termos.append("".join(atual))
            counters.count_copy()
            atual = []
    if atual:
        termos.append("".join(atual))
        counters.count_copy()
    return termos


# ----------------------------------------------------------------------
# Ocorrências
# ----------------------------------------------------------------------
@dataclass(slots=True)
class Occurrence:
    """Ocorrências de um termo em um documento: o id e todas as posições."""

    doc_id: int
    positions: list[int] = field(default_factory=list)

    @property
    def frequency(self) -> int:
        """Quantas vezes o termo aparece no documento. Custo: O(1)."""
        return len(self.positions)

    @property
    def first_position(self) -> int:
        """Primeira posição do termo no documento. Custo: O(1)."""
        return self.positions[0]

    def __repr__(self) -> str:
        return f"Occurrence(doc={self.doc_id}, tf={self.frequency})"


# ----------------------------------------------------------------------
# Índice invertido
# ----------------------------------------------------------------------
class InvertedIndex:
    """Índice invertido: hashtable de termos + BST de termos.

    Invariantes (ver :meth:`check_invariants`):

    * toda lista de ocorrências está **ordenada por ``doc_id``** e sem repetição;
    * a hashtable e a BST contêm exatamente o mesmo conjunto de termos;
    * o valor guardado na BST é a frequência de documentos do termo, igual ao
      comprimento da lista de ocorrências correspondente.
    """

    __slots__ = ("_postings", "_terms", "_documents", "_token_count")

    def __init__(self) -> None:
        """Custo: O(1)."""
        self._postings: HashTableChained[
            str, SinglyLinkedList[Occurrence]
        ] = HashTableChained()
        self._terms: BinarySearchTree[str, int] = BinarySearchTree()
        self._documents: list[tuple[int, str]] = []
        self._token_count = 0

    # -- leitura -------------------------------------------------------
    @property
    def document_count(self) -> int:
        """Número de documentos indexados. Custo: O(1)."""
        return len(self._documents)

    @property
    def vocabulary_size(self) -> int:
        """Número de termos distintos. Custo: O(1)."""
        return len(self._postings)

    @property
    def token_count(self) -> int:
        """Total de tokens indexados. Custo: O(1)."""
        return self._token_count

    @property
    def terms_tree(self) -> BinarySearchTree[str, int]:
        """A BST de termos, para o relatório de diagnóstico. Custo: O(1)."""
        return self._terms

    def document_ids(self) -> list[int]:
        """Todos os ``doc_id``, em ordem crescente. Custo: Θ(D)."""
        return [doc_id for doc_id, _ in self._documents]

    def document_text(self, doc_id: int) -> str:
        """Texto original do documento. Custo: Θ(D)."""
        for indexado, texto in self._documents:
            if indexado == doc_id:
                return texto
        raise TermNotFoundError(f"documento {doc_id} não está indexado")

    def __repr__(self) -> str:
        return (
            f"InvertedIndex(docs={self.document_count}, "
            f"termos={self.vocabulary_size}, tokens={self._token_count})"
        )

    # -- indexação -----------------------------------------------------
    def add_document(
        self, doc_id: int, text: str, counters: Counters | None = None
    ) -> int:
        """Indexa um documento; devolve quantos tokens ele tinha.

        Os ``doc_id`` devem chegar em ordem **crescente** — é o que mantém as
        listas de ocorrências ordenadas sem custo nenhum (decisão de projeto 1).
        Ordem decrescente ou repetida levanta erro em vez de silenciosamente
        quebrar a invariante de que a otimização 1 depende.

        Custo: Θ(t + novos·h) — ``t`` tokens do documento, ``novos`` termos
        inéditos, ``h`` a altura da BST.
        """
        if not isinstance(doc_id, int) or isinstance(doc_id, bool):
            raise SearchEngineError(
                f"doc_id deve ser int, recebido {type(doc_id).__name__}"
            )
        if self._documents and doc_id <= self._documents[-1][0]:
            raise DuplicateDocumentError(
                f"doc_id {doc_id} não é maior que o último indexado "
                f"({self._documents[-1][0]}). Os documentos precisam chegar em "
                "ordem crescente para que as listas de ocorrências nasçam "
                "ordenadas (decisão de projeto 1)."
            )
        counters = counters if counters is not None else Counters()
        termos = tokenize(text, counters)
        self._documents.append((doc_id, text))
        self._token_count += len(termos)

        # Agrupa as posições deste documento antes de tocar no índice: assim
        # cada termo do documento gera UMA ocorrência, e a lista continua com
        # no máximo um nó por documento.
        posicoes: HashTableChained[str, list[int]] = HashTableChained()
        for posicao, termo in enumerate(termos):
            lista = posicoes.get_or(termo, None, counters)
            if lista is None:
                posicoes.put(termo, [posicao], counters)
            else:
                lista.append(posicao)

        for termo, lista_posicoes in posicoes.items():
            ocorrencias = self._postings.get_or(termo, None, counters)
            if ocorrencias is None:
                ocorrencias = SinglyLinkedList()
                self._postings.put(termo, ocorrencias, counters)
                # Só termo NOVO entra na BST. Reinserir a cada documento
                # incrementaria a multiplicidade do nó, e a listagem in-order
                # passaria a repetir o termo uma vez por documento — além de
                # transformar o custo de indexação de Θ(V·h) em Θ(P·h).
                self._terms.insert(termo, None, counters)
            ocorrencias.insert_last(Occurrence(doc_id, lista_posicoes), counters)
        return len(termos)

    def add_documents(
        self, documents: Iterable[str], counters: Counters | None = None
    ) -> int:
        """Indexa uma lista de textos, numerando de 0 em diante. Custo: Θ(T)."""
        counters = counters if counters is not None else Counters()
        total = 0
        for doc_id, texto in enumerate(documents):
            total += self.add_document(doc_id, texto, counters)
        return total

    # -- consulta ao índice --------------------------------------------
    def postings(
        self, term: str, counters: Counters | None = None
    ) -> SinglyLinkedList[Occurrence]:
        """Lista de ocorrências do termo; vazia se ele não existe.

        Custo: O(1 + α) no caso médio.
        """
        counters = counters if counters is not None else Counters()
        encontrado = self._postings.get_or(normalize(term), None, counters)
        return encontrado if encontrado is not None else SinglyLinkedList()

    def doc_ids_for(
        self, term: str, counters: Counters | None = None
    ) -> list[int]:
        """``doc_id`` em que o termo aparece, **em ordem crescente**.

        Custo: O(1 + α) para achar a lista, Θ(df) para percorrê-la.
        """
        counters = counters if counters is not None else Counters()
        resultado: list[int] = []
        primeiro = True
        for ocorrencia in self.postings(term, counters):
            if not primeiro:
                counters.count_hop()
            primeiro = False
            resultado.append(ocorrencia.doc_id)
        return resultado

    def document_frequency(
        self, term: str, counters: Counters | None = None
    ) -> int:
        """Em quantos documentos o termo aparece. Custo: O(1 + α)."""
        return len(self.postings(term, counters))

    def contains_term(self, term: str, counters: Counters | None = None) -> bool:
        """Custo: O(1 + α)."""
        return self._postings.contains(normalize(term), counters)

    def search_term_in_tree(
        self, term: str, counters: Counters | None = None
    ) -> int:
        """Confirma o termo **na BST** e devolve sua frequência de documentos.

        Existe ao lado de :meth:`contains_term` de propósito: as duas respondem
        a mesma pergunta por caminhos diferentes, e comparar as contagens é o
        que torna visível a diferença entre O(1 + α) e O(h) (decisão 2).

        A BST guarda **só o termo**, não a contagem: a frequência vem da lista de
        ocorrências, que é onde o dado mora. Guardá-la também na árvore criaria
        duas fontes de verdade para o mesmo número, e manter as duas em dia
        custaria um O(h) por par (termo, documento) — Θ(P·h) em vez de Θ(V·h).

        Custo: O(h) para confirmar, O(1 + α) para contar.
        """
        counters = counters if counters is not None else Counters()
        normalizado = normalize(term)
        if not self._terms.contains(normalizado, counters):
            raise TermNotFoundError(f"termo {term!r} não está no vocabulário")
        return len(self.postings(normalizado, counters))

    def terms_in_order(self, counters: Counters | None = None) -> list[str]:
        """Todos os termos em ordem alfabética. Custo: Θ(V).

        É o que a hashtable **não** faz: iterar a tabela dá ordem de bucket.
        """
        counters = counters if counters is not None else Counters()
        # in_order_items dá um par por NÓ; in_order_keys repetiria pela
        # multiplicidade, que aqui é sempre 1 mas cuja semântica é outra.
        return [termo for termo, _ in self._terms.in_order_items(counters)]

    def terms_with_prefix(
        self, prefix: str, counters: Counters | None = None
    ) -> list[str]:
        """Termos que começam com ``prefix``, em ordem. Custo: Θ(V).

        Percorre a BST inteira. Uma busca por faixa aproveitaria a ordem e
        custaria O(h + resultado) — otimização registrada e não feita, porque o
        ex 12 já tem duas otimizações medidas e esta não é uma delas.
        """
        counters = counters if counters is not None else Counters()
        alvo = normalize(prefix)
        return [
            termo
            for termo, _ in self._terms.in_order_items(counters)
            if termo.startswith(alvo)
        ]

    # -- relatório de diagnóstico (BFS com Queue) ----------------------
    def bfs_report(self, counters: Counters | None = None) -> dict[str, Any]:
        """Relatório da BST de termos por percurso em **largura**.

        Usa a :class:`~src.stack_queue.Queue` do exercício 5. Reporta níveis,
        termos por nível, balanceamento aproximado e os termos da raiz e do
        nível mais largo.

        A medida de balanceamento é ``altura / altura_ideal``, com
        ``altura_ideal = ⌈log₂(V+1)⌉``: 1,0 é uma árvore completa, ``≈3`` é o
        esperado para inserção aleatória e ``V/log₂V`` é a degeneração.

        Custo: Θ(V) em tempo, Θ(largura máxima) de memória na fila.
        """
        counters = counters if counters is not None else Counters()
        relatorio = self._terms.balance_report()
        por_nivel: list[list[str]] = []
        for nivel, node in self._terms.level_order_nodes(counters):
            while len(por_nivel) <= nivel:
                por_nivel.append([])
            por_nivel[nivel].append(node.key)

        contagem = [len(termos) for termos in por_nivel]
        nivel_mais_largo = (
            contagem.index(max(contagem)) if contagem else -1
        )
        return {
            "termos": relatorio["nos"],
            "niveis": len(por_nivel),
            "altura": relatorio["altura"],
            "altura_ideal": relatorio["altura_ideal"],
            "balanceamento": relatorio["razao_altura"],
            "degenerada": relatorio["degenerada"],
            "profundidade_minima": relatorio["profundidade_minima"],
            "folhas": relatorio["folhas"],
            "termos_por_nivel": contagem,
            "nivel_mais_largo": nivel_mais_largo,
            "largura_maxima": max(contagem) if contagem else 0,
            "raiz": por_nivel[0][0] if por_nivel else None,
            "amostra_nivel_mais_largo": (
                por_nivel[nivel_mais_largo][:8] if contagem else []
            ),
            "saltos_bfs": counters.hops,
        }

    # -- verificação ---------------------------------------------------
    def check_invariants(self) -> None:
        """Valida as invariantes; levanta :class:`SearchEngineError`. Custo: Θ(V + P)."""
        self._postings.check_invariants()
        self._terms.check_invariants()

        termos_hash = set(self._postings.keys())
        termos_bst = set(self._terms)
        if termos_hash != termos_bst:
            faltando = termos_hash ^ termos_bst
            raise SearchEngineError(
                f"hashtable e BST discordam em {len(faltando)} termo(s): "
                f"{sorted(faltando)[:5]}"
            )

        for termo, ocorrencias in self._postings.items():
            ocorrencias.check_invariants()
            anterior = -1
            for ocorrencia in ocorrencias:
                if ocorrencia.doc_id <= anterior:
                    raise SearchEngineError(
                        f"lista de {termo!r} fora de ordem: {ocorrencia.doc_id} "
                        f"depois de {anterior}"
                    )
                if not ocorrencia.positions:
                    raise SearchEngineError(
                        f"ocorrência de {termo!r} no doc {ocorrencia.doc_id} "
                        "sem posição nenhuma"
                    )
                anterior = ocorrencia.doc_id
        for node in self._terms.in_order_nodes():
            if node.multiplicity != 1:
                raise SearchEngineError(
                    f"o termo {node.key!r} está na BST com multiplicidade "
                    f"{node.multiplicity}; cada termo deve ocupar um nó só, "
                    "senão a listagem in-order o repete"
                )


# ----------------------------------------------------------------------
# Operações sobre listas ordenadas de doc_ids (otimização 1)
# ----------------------------------------------------------------------
def intersect_sorted(
    left: Sequence[int], right: Sequence[int], counters: Counters | None = None
) -> list[int]:
    """Interseção por varredura simultânea de duas listas **ordenadas**.

    **Otimização 1 do exercício.** A versão ingênua compara cada elemento de uma
    lista com todos os da outra: Θ(m·n). Esta aproveita a ordem e anda um
    ponteiro de cada vez: Θ(m+n). Como as listas de ocorrências nascem ordenadas
    (decisão de projeto 1), a otimização não custa preparação nenhuma.

    Custo: Θ(m + n) em tempo, Θ(min(m,n)) de saída.
    """
    counters = counters if counters is not None else Counters()
    resultado: list[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if counters.eq(left[i], right[j]):
            resultado.append(left[i])
            i += 1
            j += 1
        elif counters.lt(left[i], right[j]):
            i += 1
        else:
            j += 1
    return resultado


def intersect_naive(
    left: Sequence[int], right: Sequence[int], counters: Counters | None = None
) -> list[int]:
    """Interseção par a par — a versão que a otimização 1 substitui.

    Custo: Θ(m · n). Existe só para medir o ganho; nada no motor a usa.
    """
    counters = counters if counters is not None else Counters()
    resultado: list[int] = []
    for esquerda in left:
        for direita in right:
            if counters.eq(esquerda, direita):
                resultado.append(esquerda)
                break
    return resultado


def union_sorted(
    left: Sequence[int], right: Sequence[int], counters: Counters | None = None
) -> list[int]:
    """União por varredura simultânea. Custo: Θ(m + n)."""
    counters = counters if counters is not None else Counters()
    resultado: list[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if counters.eq(left[i], right[j]):
            resultado.append(left[i])
            i += 1
            j += 1
        elif counters.lt(left[i], right[j]):
            resultado.append(left[i])
            i += 1
        else:
            resultado.append(right[j])
            j += 1
    resultado.extend(left[i:])
    resultado.extend(right[j:])
    return resultado


def difference_sorted(
    left: Sequence[int], right: Sequence[int], counters: Counters | None = None
) -> list[int]:
    """``left − right`` por varredura simultânea. Custo: Θ(m + n)."""
    counters = counters if counters is not None else Counters()
    resultado: list[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if counters.eq(left[i], right[j]):
            i += 1
            j += 1
        elif counters.lt(left[i], right[j]):
            resultado.append(left[i])
            i += 1
        else:
            j += 1
    resultado.extend(left[i:])
    return resultado


# ----------------------------------------------------------------------
# Consulta booleana: parsing e avaliação
# ----------------------------------------------------------------------
def tokenize_query(query: str) -> list[str]:
    """Consulta → tokens, com ``AND``/``OR``/``NOT`` e parênteses separados.

    Operadores são reconhecidos sem distinção de caixa; qualquer outra palavra
    vira termo normalizado.

    Custo: Θ(len(query)).
    """
    if not isinstance(query, str):
        raise EmptyQueryError(
            f"consulta deve ser str, recebido {type(query).__name__}"
        )
    tokens: list[str] = []
    atual: list[str] = []

    def fechar() -> None:
        if atual:
            palavra = "".join(atual)
            tokens.append(
                palavra.upper() if palavra.upper() in PRECEDENCE else normalize(palavra)
            )
            atual.clear()

    for caractere in query:
        if caractere in "()":
            fechar()
            tokens.append(caractere)
        elif caractere.isspace():
            fechar()
        else:
            atual.append(caractere)
    fechar()
    if not tokens:
        raise EmptyQueryError("consulta vazia")
    return tokens


def to_postfix(tokens: Sequence[str], counters: Counters | None = None) -> list[str]:
    """Infixa → pós-fixa pelo **shunting-yard**, com a ``Stack`` do exercício 5.

    Precedência: ``NOT`` > ``AND`` > ``OR`` (decisão de projeto 3). ``NOT`` é
    unário e associativo à direita, então ele **não** desempilha outro ``NOT``
    de mesma precedência — sem essa ressalva, ``NOT NOT a`` sairia errado.

    Validações: parênteses desbalanceados levantam
    :class:`UnbalancedParenthesesError` dizendo qual lado sobrou.

    Custo: Θ(q) em tempo e memória.
    """
    counters = counters if counters is not None else Counters()
    saida: list[str] = []
    operadores: Stack[str] = Stack(max(1, len(tokens)))

    for token in tokens:
        if token == "(":
            operadores.push(token, counters)
        elif token == ")":
            achou = False
            while not operadores.is_empty():
                topo = operadores.pop(counters)
                if topo == "(":
                    achou = True
                    break
                saida.append(topo)
            if not achou:
                raise UnbalancedParenthesesError(
                    "parêntese ')' sem '(' correspondente"
                )
        elif token in PRECEDENCE:
            while not operadores.is_empty():
                topo = operadores.peek()
                if topo == "(":
                    break
                # unário é associativo à direita: não desempilha o de igual
                # precedência, senão NOT NOT a vira NOT a
                if PRECEDENCE[topo] > PRECEDENCE[token] or (
                    PRECEDENCE[topo] == PRECEDENCE[token] and token not in UNARY
                ):
                    saida.append(operadores.pop(counters))
                else:
                    break
            operadores.push(token, counters)
        else:
            saida.append(token)

    while not operadores.is_empty():
        topo = operadores.pop(counters)
        if topo == "(":
            raise UnbalancedParenthesesError(
                "parêntese '(' sem ')' correspondente"
            )
        saida.append(topo)
    return saida


def evaluate_postfix(
    postfix: Sequence[str],
    index: InvertedIndex,
    counters: Counters | None = None,
) -> list[int]:
    """Avalia a consulta pós-fixa e devolve os ``doc_id`` **ordenados**.

    Usa uma :class:`~src.stack_queue.Stack` de listas ordenadas. Todas as
    combinações são feitas pelas rotinas Θ(m+n) da otimização 1; o universo de
    documentos só é materializado quando aparece um ``NOT``, que é a única
    operação Θ(D).

    Custo: Θ(P) no total das listas envolvidas, mais Θ(D) por ``NOT``.
    """
    counters = counters if counters is not None else Counters()
    if not postfix:
        raise EmptyQueryError("expressão de consulta vazia")
    pilha: Stack[list[int]] = Stack(max(1, len(postfix)))

    for token in postfix:
        if token == "NOT":
            try:
                operando = pilha.pop(counters)
            except StackUnderflowError as falha:
                raise MalformedQueryError(
                    f"'NOT' sem operando: {falha}"
                ) from falha
            pilha.push(
                difference_sorted(index.document_ids(), operando, counters), counters
            )
        elif token in PRECEDENCE:
            try:
                direita = pilha.pop(counters)
                esquerda = pilha.pop(counters)
            except StackUnderflowError as falha:
                raise MalformedQueryError(
                    f"operador {token!r} precisa de 2 operandos: {falha}"
                ) from falha
            combinar = intersect_sorted if token == "AND" else union_sorted
            pilha.push(combinar(esquerda, direita, counters), counters)
        else:
            pilha.push(index.doc_ids_for(token, counters), counters)

    if len(pilha) != 1:
        raise MalformedQueryError(
            f"a consulta terminou com {len(pilha)} resultados parciais, "
            "esperado 1 — provavelmente faltou um operador entre dois termos"
        )
    return pilha.pop(counters)


def run_query(
    query: str, index: InvertedIndex, counters: Counters | None = None
) -> list[int]:
    """Tokeniza, converte para pós-fixa e avalia. Custo: Θ(q + P)."""
    counters = counters if counters is not None else Counters()
    return evaluate_postfix(
        to_postfix(tokenize_query(query), counters), index, counters
    )


# ----------------------------------------------------------------------
# Ranqueamento
# ----------------------------------------------------------------------
@dataclass(slots=True)
class ScoredDocument:
    """Documento com pontuação, para ordenação pelo quicksort do exercício 9."""

    doc_id: int
    score: float
    matched_terms: int
    total_frequency: int

    def _chave(self) -> tuple[float, int]:
        # score decrescente; empate desfeito pelo doc_id crescente, para que a
        # saída seja determinística (requisito da seção 5 do CLAUDE.md)
        return (-self.score, self.doc_id)

    def __lt__(self, other: "ScoredDocument") -> bool:
        return self._chave() < other._chave()

    def __le__(self, other: "ScoredDocument") -> bool:
        return self._chave() <= other._chave()

    def __gt__(self, other: "ScoredDocument") -> bool:
        return self._chave() > other._chave()

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ScoredDocument) and self._chave() == other._chave()
        )

    def __repr__(self) -> str:
        return f"ScoredDocument(doc={self.doc_id}, score={self.score:.3f})"


def score_document(
    doc_id: int,
    terms: Sequence[str],
    index: InvertedIndex,
    counters: Counters | None = None,
) -> ScoredDocument:
    """Pontua um documento por frequência e posição (decisão de projeto 4).

    ``score = Σ_t tf(t,d) · (1 + 1/(1 + primeira_posição(t,d)))``

    O fator posicional vale 2 quando o termo abre o documento e tende a 1
    quando ele aparece tarde — desempata sem dominar a frequência.

    Custo: Θ(Σ_t df(t)) — percorre a lista de ocorrências de cada termo.
    """
    counters = counters if counters is not None else Counters()
    total = 0.0
    encontrados = 0
    frequencia = 0
    for termo in terms:
        primeiro = True
        for ocorrencia in index.postings(termo, counters):
            if not primeiro:
                counters.count_hop()
            primeiro = False
            if ocorrencia.doc_id != doc_id:
                continue
            peso_posicional = 1.0 + 1.0 / (1.0 + ocorrencia.first_position)
            total += ocorrencia.frequency * peso_posicional
            encontrados += 1
            frequencia += ocorrencia.frequency
            break
    return ScoredDocument(doc_id, total, encontrados, frequencia)


def rank_documents(
    doc_ids: Sequence[int],
    terms: Sequence[str],
    index: InvertedIndex,
    counters: Counters | None = None,
    *,
    short: int = 16,
) -> list[ScoredDocument]:
    """Pontua e ordena com o **quicksort instrumentado do exercício 9**.

    Ordena por score decrescente, com o ``doc_id`` crescente desempatando.
    O quicksort em array não é estável, então o desempate explícito é o que
    garante saída determinística — sem ele, dois documentos de mesmo score
    poderiam trocar de lugar entre execuções.

    Custo: Θ(R·Σdf) para pontuar e Θ(R log R) para ordenar, ``R`` = resultados.
    """
    counters = counters if counters is not None else Counters()
    pontuados = [
        score_document(doc_id, terms, index, counters) for doc_id in doc_ids
    ]
    return quicksort(pontuados, short, counters=counters)


def query_terms(tokens: Sequence[str]) -> list[str]:
    """Só os termos de uma consulta, sem operadores nem parênteses.

    São eles que entram no score: ``NOT`` e parênteses mudam *quais* documentos
    voltam, não *o quanto* cada um casa com a busca.

    Custo: Θ(q).
    """
    return [
        token
        for token in tokens
        if token not in PRECEDENCE and token not in ("(", ")")
    ]


# ----------------------------------------------------------------------
# Sugestão de termo: distância de edição com memoization (ex 8)
# ----------------------------------------------------------------------
def edit_distance(
    left: str, right: str, counters: Counters | None = None
) -> int:
    """Distância de Levenshtein, recursiva com memo na ``HashTableChained``.

    Chave do memo: ``(i, j)`` — o estado mínimo suficiente (decisão 5). Alinhar
    ``left[:i]`` com ``right[:j]`` não depende de como se chegou até esses
    prefixos, então o par de índices basta.

    Custo: Θ(|a|·|b|) em tempo e memória, contra Θ(3^max(|a|,|b|)) da recursão
    sem memo — a mesma mudança de classe do exercício 8.
    """
    counters = counters if counters is not None else Counters()
    memo: HashTableChained[tuple[int, int], int] = HashTableChained()

    def resolver(i: int, j: int) -> int:
        counters.count_call()
        if i == 0:
            return j
        if j == 0:
            return i
        chave = (i, j)
        guardado = memo.get_or(chave, None, counters)
        if guardado is not None:
            return guardado

        if counters.eq(left[i - 1], right[j - 1]):
            resultado = resolver(i - 1, j - 1)
        else:
            resultado = 1 + min(
                resolver(i - 1, j),  # remoção
                resolver(i, j - 1),  # inserção
                resolver(i - 1, j - 1),  # substituição
            )
        memo.put(chave, resultado, counters)
        return resultado

    return resolver(len(left), len(right))


def edit_distance_naive(
    left: str, right: str, counters: Counters | None = None
) -> int:
    """A mesma recursão **sem** memo — só para medir o ganho.

    Custo: Θ(3^max(|a|,|b|)). Não use com palavras de mais de ~9 letras.
    """
    counters = counters if counters is not None else Counters()

    def resolver(i: int, j: int) -> int:
        counters.count_call()
        if i == 0:
            return j
        if j == 0:
            return i
        if counters.eq(left[i - 1], right[j - 1]):
            return resolver(i - 1, j - 1)
        return 1 + min(
            resolver(i - 1, j), resolver(i, j - 1), resolver(i - 1, j - 1)
        )

    return resolver(len(left), len(right))


@dataclass(slots=True)
class Suggestion:
    """Sugestão de termo com a distância e o custo de tê-la encontrado."""

    term: str
    distance: int
    document_frequency: int

    def __repr__(self) -> str:
        return f"Suggestion({self.term!r}, d={self.distance})"


def suggest_terms(
    term: str,
    index: InvertedIndex,
    counters: Counters | None = None,
    *,
    max_distance: int = DEFAULT_MAX_DISTANCE,
    limit: int = 5,
    prune_by_length: bool = True,
) -> list[Suggestion]:
    """Termos do vocabulário mais próximos de ``term``.

    **Otimização 2 do exercício:** ``prune_by_length``. A distância de edição
    entre duas palavras é **pelo menos** a diferença de comprimento delas —
    cada operação muda o tamanho em no máximo 1. Então um candidato cujo
    comprimento difere de ``term`` em mais de ``max_distance`` pode ser
    descartado **sem calcular nada**, trocando um Θ(L²) por uma comparação
    O(1). O ganho é medido em :func:`bench_suggestion`.

    Sem a poda, o custo é Θ(V·L²); com ela, Θ(V + C·L²), onde ``C`` é o número
    de candidatos que sobrevivem ao filtro.

    Empates são desfeitos por frequência de documentos (maior primeiro) e depois
    por ordem alfabética, para que a saída seja determinística.

    Custo: Θ(V + C·L²) com poda, Θ(V·L²) sem.
    """
    if max_distance < 0:
        raise ValueError(f"max_distance deve ser >= 0, recebido {max_distance}")
    if limit < 1:
        raise ValueError(f"limit deve ser >= 1, recebido {limit}")
    counters = counters if counters is not None else Counters()
    alvo = normalize(term)
    candidatos: list[Suggestion] = []

    for candidato in index.terms_in_order(counters):
        if prune_by_length and abs(len(candidato) - len(alvo)) > max_distance:
            counters.count_comparison()  # o filtro O(1) que substitui o Θ(L²)
            continue
        distancia = edit_distance(alvo, candidato, counters)
        if distancia <= max_distance and distancia > 0:
            candidatos.append(
                Suggestion(
                    candidato, distancia, index.document_frequency(candidato, counters)
                )
            )

    candidatos.sort(key=lambda s: (s.distance, -s.document_frequency, s.term))
    return candidatos[:limit]


# ----------------------------------------------------------------------
# Fachada
# ----------------------------------------------------------------------
@dataclass(slots=True)
class QueryResult:
    """Resultado completo de uma consulta, pronto para o notebook."""

    query: str
    tokens: list[str]
    postfix: list[str]
    doc_ids: list[int]
    ranked: list[ScoredDocument]
    suggestions: dict[str, list[Suggestion]]
    counters: Counters
    elapsed_s: float

    def top(self, k: int = 3) -> list[ScoredDocument]:
        """Os ``k`` melhores. Custo: O(k)."""
        return self.ranked[:k]


class SearchEngine:
    """Fachada: indexar, consultar, ranquear e sugerir."""

    __slots__ = ("index",)

    def __init__(self, documents: Iterable[str] | None = None) -> None:
        """Custo: Θ(T)."""
        self.index = InvertedIndex()
        if documents is not None:
            self.index.add_documents(documents)

    def search(
        self,
        query: str,
        counters: Counters | None = None,
        *,
        max_distance: int = DEFAULT_MAX_DISTANCE,
    ) -> QueryResult:
        """Executa a cadeia completa e devolve tudo que aconteceu.

        Para cada termo da consulta que **não** está no vocabulário, chama o
        sugeridor. Termos presentes não geram sugestão: gastar Θ(V·L²) para
        sugerir alternativa a uma palavra que existe seria trabalho jogado fora.

        Custo: Θ(q + P + R log R + A·V), ``A`` = termos ausentes.
        """
        counters = counters if counters is not None else Counters()
        with stopwatch() as elapsed:
            tokens = tokenize_query(query)
            postfix = to_postfix(tokens, counters)
            doc_ids = evaluate_postfix(postfix, self.index, counters)
            termos = query_terms(tokens)
            ranked = rank_documents(doc_ids, termos, self.index, counters)
            sugestoes: dict[str, list[Suggestion]] = {}
            for termo in termos:
                if not self.index.contains_term(termo, counters):
                    sugestoes[termo] = suggest_terms(
                        termo, self.index, counters, max_distance=max_distance
                    )
        return QueryResult(
            query=query,
            tokens=tokens,
            postfix=postfix,
            doc_ids=doc_ids,
            ranked=ranked,
            suggestions=sugestoes,
            counters=counters,
            elapsed_s=elapsed[0],
        )


# ----------------------------------------------------------------------
# Corpus reprodutível para os experimentos
# ----------------------------------------------------------------------
#: Vocabulário base dos corpora sintéticos. Palavras do próprio domínio do
#: trabalho, para que as consultas de exemplo façam sentido.
BASE_VOCABULARY: tuple[str, ...] = (
    "algoritmo", "ordenacao", "busca", "arvore", "binaria", "lista",
    "encadeada", "pilha", "fila", "tabela", "hash", "colisao", "recursao",
    "memoization", "programacao", "dinamica", "complexidade", "assintotica",
    "comparacao", "copia", "particao", "pivo", "quicksort", "insertion",
    "selection", "bubble", "invariante", "ponteiro", "no", "indice",
    "documento", "termo", "consulta", "ranqueamento", "score", "frequencia",
    "posicao", "vocabulario", "degenerado", "balanceado",
)


#: Sílabas usadas para compor vocabulário sintético de tamanho arbitrário.
_SYLLABLES: tuple[str, ...] = (
    "ra", "ti", "lo", "men", "de", "ca", "so", "pre", "tra", "vi", "nu",
    "es", "gor", "al", "in", "con", "per", "mo", "ta", "bri",
)


def build_vocabulary(
    size: int, *, seed: int = 42, base: Sequence[str] = BASE_VOCABULARY
) -> list[str]:
    """Vocabulário reprodutível de ``size`` termos distintos.

    Começa pelos termos do domínio do trabalho (para que as consultas de exemplo
    façam sentido) e completa compondo sílabas. As palavras compostas têm de 4 a
    18 letras, e essa **variedade de comprimento** não é enfeite: é ela que dá
    sentido à otimização 2, cuja poda descarta candidatos pelo tamanho. Um
    vocabulário de palavras todas do mesmo tamanho mediria a poda em zero.

    Custo: Θ(size) esperado.
    """
    import random as _random

    if size < 1:
        raise ValueError(f"size deve ser >= 1, recebido {size}")
    rng = _random.Random(seed)
    vocabulario = list(dict.fromkeys(base))[:size]
    vistos = set(vocabulario)
    while len(vocabulario) < size:
        palavra = "".join(
            rng.choice(_SYLLABLES) for _ in range(rng.randint(2, 6))
        )
        if palavra not in vistos:
            vistos.add(palavra)
            vocabulario.append(palavra)
    return vocabulario


def generate_corpus(
    documents: int,
    *,
    seed: int = 42,
    words_per_document: int = 40,
    vocabulary_size: int = 40,
    vocabulary: Sequence[str] | None = None,
) -> list[str]:
    """Corpus sintético reprodutível.

    As palavras são sorteadas com peso ``1/rank`` (lei de Zipf aproximada), que
    é como o vocabulário se distribui em texto real: poucos termos muito
    frequentes e uma cauda longa de termos raros. Sortear uniformemente daria
    listas de ocorrências todas do mesmo tamanho e esconderia o efeito que a
    otimização 1 explora — interseção entre uma lista curta e uma longa.

    ``vocabulary_size`` controla ``V`` independentemente de ``documents``, o que
    é necessário porque as duas variáveis aparecem separadas na análise: a
    indexação é Θ(T + V·h), e medir só corpora com ``V`` saturado não distingue
    os dois termos.

    Custo: Θ(documents · words_per_document + vocabulary_size).
    """
    import random as _random

    if documents < 1:
        raise ValueError(f"documents deve ser >= 1, recebido {documents}")
    if words_per_document < 1:
        raise ValueError(
            f"words_per_document deve ser >= 1, recebido {words_per_document}"
        )
    termos = (
        list(vocabulary)
        if vocabulary is not None
        else build_vocabulary(vocabulary_size, seed=seed)
    )
    rng = _random.Random(seed)
    pesos = [1.0 / (posicao + 1) for posicao in range(len(termos))]
    corpus: list[str] = []
    for _ in range(documents):
        palavras = rng.choices(termos, weights=pesos, k=words_per_document)
        corpus.append(" ".join(palavras) + ".")
    return corpus


# ----------------------------------------------------------------------
# Relatório técnico de desempenho (item 4.5 da rubrica)
# ----------------------------------------------------------------------
def performance_report(
    engine: "SearchEngine",
    queries: Sequence[str],
    *,
    missing_term: str = "algoritmoo",
) -> dict[str, Any]:
    """Relatório técnico consolidado: custo de cada etapa da cadeia.

    O item 4.5 da rubrica pede "relatório técnico de desempenho", e não apenas
    análise assintótica solta. Este é ele: mede indexação, consulta,
    ranqueamento e sugestão **no mesmo índice**, com as contagens separadas por
    etapa, para que dê para dizer qual delas domina.

    Custo: o das etapas medidas.
    """
    index = engine.index
    diagnostico = index.bfs_report()

    por_consulta: list[dict[str, Any]] = []
    for consulta in queries:
        tokens = tokenize_query(consulta)
        termos = query_terms(tokens)

        parsing = Counters()
        with stopwatch() as tempo_parsing:
            postfix = to_postfix(tokens, parsing)

        avaliacao = Counters()
        with stopwatch() as tempo_avaliacao:
            doc_ids = evaluate_postfix(postfix, index, avaliacao)

        ranqueamento = Counters()
        with stopwatch() as tempo_ranqueamento:
            ranked = rank_documents(doc_ids, termos, index, ranqueamento)

        por_consulta.append(
            {
                "consulta": consulta,
                "termos": len(termos),
                "resultados": len(doc_ids),
                "parsing_comparacoes": parsing.comparisons,
                "parsing_copias": parsing.copies,
                "avaliacao_comparacoes": avaliacao.comparisons,
                "avaliacao_saltos": avaliacao.hops,
                "ranqueamento_comparacoes": ranqueamento.comparisons,
                "ranqueamento_copias": ranqueamento.copies,
                "tempo_parsing_s": tempo_parsing[0],
                "tempo_avaliacao_s": tempo_avaliacao[0],
                "tempo_ranqueamento_s": tempo_ranqueamento[0],
                "top": [(d.doc_id, round(d.score, 3)) for d in ranked[:3]],
            }
        )

    com_poda = Counters()
    with stopwatch() as tempo_com_poda:
        sugestoes = suggest_terms(
            missing_term, index, com_poda, prune_by_length=True
        )
    sem_poda = Counters()
    with stopwatch() as tempo_sem_poda:
        suggest_terms(missing_term, index, sem_poda, prune_by_length=False)

    return {
        "indice": {
            "documentos": index.document_count,
            "tokens": index.token_count,
            "vocabulario": index.vocabulary_size,
            "tokens_por_documento": (
                index.token_count / index.document_count
                if index.document_count
                else 0.0
            ),
        },
        "bst_de_termos": diagnostico,
        "consultas": por_consulta,
        "sugestao": {
            "termo_ausente": missing_term,
            "sugestoes": [(s.term, s.distance) for s in sugestoes],
            "comparacoes_com_poda": com_poda.comparisons,
            "comparacoes_sem_poda": sem_poda.comparisons,
            "chamadas_com_poda": com_poda.calls,
            "chamadas_sem_poda": sem_poda.calls,
            "reducao_de_chamadas": (
                sem_poda.calls / com_poda.calls if com_poda.calls else math.inf
            ),
            "tempo_com_poda_s": tempo_com_poda[0],
            "tempo_sem_poda_s": tempo_sem_poda[0],
        },
    }


# ----------------------------------------------------------------------
# Experimentos do exercício 12
# ----------------------------------------------------------------------
def bench_indexing(
    sizes: Iterable[int] = (50, 200, 800, 3_200),
    *,
    seed: int = 42,
    words_per_document: int = 40,
    vocabulary_size: int = 400,
) -> list[dict[str, Any]]:
    """Custo de indexar, por tamanho de corpus.

    Previsão: ``comparações/T`` fica aproximadamente constante — a indexação é
    Θ(T + V·h), e como o vocabulário satura (lei de Zipf: corpus maior não traz
    termos novos na mesma proporção), o termo ``T`` domina e a curva vira linear.

    A coluna ``altura_bst`` mostra que a árvore de termos **não** degenera:
    os termos chegam em ordem de texto, que é quase aleatória.

    Custo: Θ(Σ T).
    """
    rows: list[dict[str, Any]] = []
    for documents in sizes:
        if documents <= 0:
            raise ValueError(f"tamanho deve ser positivo, recebido {documents}")
        corpus = generate_corpus(
            documents,
            seed=seed,
            words_per_document=words_per_document,
            vocabulary_size=vocabulary_size,
        )
        index = InvertedIndex()
        counters = Counters()
        with stopwatch() as elapsed:
            tokens = index.add_documents(corpus, counters)
        index.check_invariants()
        relatorio = index.terms_tree.balance_report()

        rows.append(
            {
                "exercicio": "ex12",
                "algoritmo": "indexação",
                "n": tokens,
                "padrao": f"{documents} documentos, {words_per_document} palavras",
                "documentos": documents,
                "tokens": tokens,
                "vocabulario": index.vocabulary_size,
                "ocorrencias": sum(
                    len(lista) for _, lista in index._postings.items()
                ),
                **counters.as_dict(),
                "comparacoes/tokens": counters.comparisons / tokens,
                "altura_bst": relatorio["altura"],
                "altura_ideal": relatorio["altura_ideal"],
                "balanceamento": relatorio["razao_altura"],
                "tempo_s": elapsed[0],
                "tempo_us_por_token": elapsed[0] * 1e6 / tokens,
            }
        )
    return rows


def bench_intersection(
    sizes: Iterable[int] = (100, 400, 1_600, 6_400),
    *,
    overlap: float = 0.1,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """**Otimização 1** medida: varredura ordenada Θ(m+n) × par a par Θ(m·n).

    Duas listas ordenadas de ``n`` doc_ids com ``overlap`` de interseção. As
    duas rotinas devolvem exatamente o mesmo resultado; o que muda é a contagem.

    Previsão: ``ordenada/n`` fica constante (Θ(m+n) com m=n) e ``ingenua/n²``
    fica constante (Θ(m·n)). A razão entre as duas cresce linearmente com ``n``.

    Custo: Θ(Σ n²) no lado ingênuo.
    """
    import random as _random

    rows: list[dict[str, Any]] = []
    for n in sizes:
        if n <= 0:
            raise ValueError(f"tamanho deve ser positivo, recebido {n}")
        if not 0.0 <= overlap <= 1.0:
            raise ValueError(f"overlap deve estar em [0, 1], recebido {overlap}")
        rng = _random.Random(seed)
        universo = 4 * n
        comuns = sorted(rng.sample(range(universo), int(overlap * n)))
        restantes = [valor for valor in range(universo) if valor not in set(comuns)]
        rng.shuffle(restantes)
        faltam = n - len(comuns)
        esquerda = sorted(comuns + restantes[:faltam])
        direita = sorted(comuns + restantes[faltam : 2 * faltam])

        ordenada = Counters()
        with stopwatch() as tempo_ordenada:
            resultado_ordenado = intersect_sorted(esquerda, direita, ordenada)

        ingenua = Counters()
        with stopwatch() as tempo_ingenuo:
            resultado_ingenuo = intersect_naive(esquerda, direita, ingenua)

        if resultado_ordenado != resultado_ingenuo:
            raise AssertionError("as duas interseções discordaram")

        rows.append(
            {
                "exercicio": "ex12",
                "algoritmo": "interseção: ordenada × par a par",
                "n": n,
                "padrao": f"listas de {n}, sobreposição {overlap:.0%}",
                "resultado": len(resultado_ordenado),
                "ordenada_comparacoes": ordenada.comparisons,
                "ordenada/n": ratio(ordenada.comparisons, n, "n"),
                "ingenua_comparacoes": ingenua.comparisons,
                "ingenua/n2": ratio(ingenua.comparisons, n, "n2"),
                "razao": ingenua.comparisons / max(1, ordenada.comparisons),
                "tempo_ordenada_s": tempo_ordenada[0],
                "tempo_ingenua_s": tempo_ingenuo[0],
            }
        )
    return rows


def bench_suggestion(
    vocabulary_sizes: Iterable[int] = (200, 800, 3_200),
    *,
    missing_term: str = "algoritmoo",
    max_distance: int = DEFAULT_MAX_DISTANCE,
    seed: int = 42,
    documents: int = 200,
) -> list[dict[str, Any]]:
    """**Otimização 2** medida: poda por diferença de comprimento.

    A distância de edição entre duas palavras é pelo menos a diferença de
    comprimento delas, porque cada operação muda o tamanho em no máximo 1.
    Então um candidato fora da faixa ``|len(a) − len(b)| <= max_distance`` pode
    ser descartado por uma comparação O(1), em vez de um cálculo Θ(L²).

    As duas versões devolvem **a mesma lista** — a poda não descarta nenhum
    candidato válido, e isso é testado. O que muda é o custo.

    Custo: Θ(V·L²) na versão sem poda.
    """
    rows: list[dict[str, Any]] = []
    for tamanho_vocabulario in vocabulary_sizes:
        if tamanho_vocabulario <= 0:
            raise ValueError(
                f"tamanho deve ser positivo, recebido {tamanho_vocabulario}"
            )
        index = InvertedIndex()
        index.add_documents(
            generate_corpus(
                documents,
                seed=seed,
                words_per_document=80,
                vocabulary_size=tamanho_vocabulario,
            )
        )

        com_poda = Counters()
        with stopwatch() as tempo_com:
            resultado_com = suggest_terms(
                missing_term, index, com_poda,
                max_distance=max_distance, prune_by_length=True,
            )
        sem_poda = Counters()
        with stopwatch() as tempo_sem:
            resultado_sem = suggest_terms(
                missing_term, index, sem_poda,
                max_distance=max_distance, prune_by_length=False,
            )
        if resultado_com != resultado_sem:
            raise AssertionError(
                "a poda mudou o resultado — ela deveria ser exata"
            )

        vocabulario = index.vocabulary_size
        rows.append(
            {
                "exercicio": "ex12",
                "algoritmo": "sugestão: com poda × sem poda",
                "n": vocabulario,
                "padrao": f"vocabulário pedido: {tamanho_vocabulario}",
                "documentos": documents,
                "vocabulario": vocabulario,
                "termo_ausente": missing_term,
                "sugestoes": [(s.term, s.distance) for s in resultado_com],
                "chamadas_com_poda": com_poda.calls,
                "chamadas_sem_poda": sem_poda.calls,
                "comparacoes_com_poda": com_poda.comparisons,
                "comparacoes_sem_poda": sem_poda.comparisons,
                "reducao_de_chamadas": (
                    sem_poda.calls / com_poda.calls if com_poda.calls else math.inf
                ),
                "chamadas_por_termo_sem_poda": sem_poda.calls / vocabulario,
                "tempo_com_poda_s": tempo_com[0],
                "tempo_sem_poda_s": tempo_sem[0],
                "razao_tempo": (
                    tempo_sem[0] / tempo_com[0] if tempo_com[0] else math.inf
                ),
            }
        )
    return rows


def bench_edit_distance(
    pairs: Sequence[tuple[str, str]] = (
        ("arvore", "arvre"),
        ("algoritmo", "algoritimo"),
        ("ordenacao", "ordencao"),
        ("complexidade", "complexidad"),
    ),
) -> list[dict[str, Any]]:
    """A DP dentro do ex 12: distância de edição com e sem memo.

    Mesma mudança de classe do exercício 8, agora num módulo aplicado:
    Θ(3^L) → Θ(|a|·|b|). A coluna ``estados`` é ``|a|·|b|``, o tamanho do espaço
    de estados — e as chamadas da versão memoizada ficam na mesma ordem dele.

    Custo: exponencial no lado sem memo; por isso os pares têm no máximo ~12
    letras.
    """
    rows: list[dict[str, Any]] = []
    for esquerda, direita in pairs:
        com_memo = Counters()
        with stopwatch() as tempo_memo:
            distancia = edit_distance(esquerda, direita, com_memo)
        sem_memo = Counters()
        with stopwatch() as tempo_sem:
            distancia_sem = edit_distance_naive(esquerda, direita, sem_memo)
        if distancia != distancia_sem:
            raise AssertionError(
                f"as duas versões discordaram em {esquerda!r}/{direita!r}"
            )
        estados = len(esquerda) * len(direita)
        rows.append(
            {
                "exercicio": "ex12",
                "algoritmo": "distância de edição: memo × ingênua",
                "n": estados,
                "padrao": f"{esquerda!r} vs {direita!r}",
                "esquerda": esquerda,
                "direita": direita,
                "distancia": distancia,
                "estados_possiveis": estados,
                "chamadas_com_memo": com_memo.calls,
                "chamadas_sem_memo": sem_memo.calls,
                "memo/estados": com_memo.calls / estados,
                "reducao_de_chamadas": sem_memo.calls / com_memo.calls,
                "tempo_memo_s": tempo_memo[0],
                "tempo_sem_memo_s": tempo_sem[0],
            }
        )
    return rows


def bench_index_lookup(
    vocabulary_sizes: Iterable[int] = (200, 800, 3_200),
    *,
    seed: int = 42,
    documents: int = 200,
) -> list[dict[str, Any]]:
    """Hashtable × BST para a **mesma** pergunta (decisão de projeto 2).

    Procura todos os termos do vocabulário pelos dois caminhos e compara as
    comparações por consulta. Previsão: a hashtable fica em torno de 1 + α/2,
    constante; a BST cresce como a altura, ``≈ 3·log₂V``.

    O experimento não existe para eleger um vencedor — existe para mostrar que
    as duas estruturas pagam preços diferentes pelo mesmo resultado, e que a BST
    é mantida pelo que a hashtable **não** faz: listar em ordem.

    Custo: Θ(Σ V·h).
    """
    rows: list[dict[str, Any]] = []
    for tamanho_vocabulario in vocabulary_sizes:
        if tamanho_vocabulario <= 0:
            raise ValueError(
                f"tamanho deve ser positivo, recebido {tamanho_vocabulario}"
            )
        index = InvertedIndex()
        index.add_documents(
            generate_corpus(
                documents,
                seed=seed,
                words_per_document=80,
                vocabulary_size=tamanho_vocabulario,
            )
        )
        termos = index.terms_in_order()

        na_hash = Counters()
        with stopwatch() as tempo_hash:
            for termo in termos:
                index.contains_term(termo, na_hash)

        na_bst = Counters()
        with stopwatch() as tempo_bst:
            for termo in termos:
                index.terms_tree.contains(termo, na_bst)

        ordenado = Counters()
        with stopwatch() as tempo_ordenado:
            listagem = index.terms_in_order(ordenado)

        relatorio = index.terms_tree.balance_report()
        rows.append(
            {
                "exercicio": "ex12",
                "algoritmo": "busca de termo: hashtable × BST",
                "n": len(termos),
                "padrao": f"vocabulário pedido: {tamanho_vocabulario}",
                "vocabulario": len(termos),
                "altura_bst": relatorio["altura"],
                "hash_comparacoes_por_busca": na_hash.comparisons / len(termos),
                "bst_comparacoes_por_busca": na_bst.comparisons / len(termos),
                "bst/hash": (
                    (na_bst.comparisons / na_hash.comparisons)
                    if na_hash.comparisons
                    else math.inf
                ),
                "bst_por_busca/log2V": (
                    (na_bst.comparisons / len(termos)) / math.log2(len(termos))
                    if len(termos) > 1
                    else float("nan")
                ),
                "listagem_ordenada_ok": listagem == sorted(listagem),
                "tempo_hash_s": tempo_hash[0],
                "tempo_bst_s": tempo_bst[0],
                "tempo_listagem_s": tempo_ordenado[0],
            }
        )
    return rows

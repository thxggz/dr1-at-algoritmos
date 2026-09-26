"""Exercício 4 — Hashtable com colisões resolvidas por encadeamento.

Reutilizada pelo exercício 8 (memo da programação dinâmica) e pelo exercício 12
(índice invertido). É a estrutura mais reaproveitada do trabalho depois do
:class:`~src.counters.Counters`.

Decisões de projeto
-------------------
1. **Função de hash própria e determinística (FNV-1a de 64 bits).**
   ``hash()`` de ``str`` em CPython é *salgado* por processo (``PYTHONHASHSEED``):
   a mesma chave cai em buckets diferentes a cada execução. Com ``hash()``
   embutido, as contagens de comparação mudariam a cada run e o notebook, o PDF
   e o vídeo mostrariam números diferentes — exatamente o que a seção 5 do
   CLAUDE.md proíbe. FNV-1a é determinístico, tem boa dispersão e cabe em 15
   linhas, então a reprodutibilidade sai de graça.
2. **O bucket é uma lista encadeada de verdade, e é a do exercício 10.**
   :class:`HashBucket` herda de :class:`~src.linked_list.SinglyLinkedList` e
   acrescenta as operações por *chave* (a classe base busca por *valor*). Herdar
   em vez de reescrever preserva o nó, ``insert_last`` O(1), ``_unlink`` com
   manutenção do ``tail``, ``__len__`` O(1) e ``check_invariants`` — e cada
   bucket é validável com o mesmo código já testado no ex 10.
   Cada operação faz **uma única travessia**: buscar e desligar acontecem no
   mesmo laço, então a contagem medida é a contagem real, não o dobro.
3. **Tipos de chave restritos, de propósito.** São aceitos ``str``, ``bytes``,
   ``int``/``bool``, ``None`` e ``tuple`` desses. Qualquer outro tipo levanta
   :class:`UnsupportedKeyError`. O motivo é a decisão 1: hashear um objeto
   arbitrário exigiria cair no ``hash()`` embutido e perder a reprodutibilidade.
   ``bool`` é normalizado para ``int`` porque ``True == 1`` em Python — se os
   dois caíssem em buckets diferentes mas comparassem iguais dentro do bucket, a
   estrutura teria duas chaves "iguais" em lugares diferentes. ``float`` fica de
   fora pelo mesmo motivo (``1 == 1.0``).
4. **``get`` de chave ausente levanta**, não devolve ``None``. ``None`` é um
   valor legítimo; devolvê-lo tornaria "não existe" indistinguível de "existe e
   vale ``None``". Quem precisa do caminho sem exceção usa
   :meth:`HashTableChained.get_or` (uma travessia) ou ``in`` (idem) — é o que o
   memo do ex 8 usa.

Custo em Big O
--------------
Com ``m`` buckets e ``n`` pares, o fator de carga é ``α = n/m``.

==================  ==========================  ===================
Operação            Caso médio (hash uniforme)  Pior caso
==================  ==========================  ===================
``put`` (nova)      O(1 + α), amortizado O(1)   O(n)
``put`` (update)     O(1 + α)                    O(n)
``get``             O(1 + α)                    O(n)
``delete``          O(1 + α)                    O(n)
``__len__``         O(1)                        O(1)
``rehash``          O(n + m)                    O(n + m)
==================  ==========================  ===================

Busca **bem-sucedida** percorre em média metade da cadeia: ``1 + α/2``
comparações. Busca **mal-sucedida** percorre a cadeia inteira: ``α``
comparações. As duas previsões são conferidas em
:func:`bench_alpha_vs_comparisons`.

Como o redimensionamento mantém ``α ≤ 0,75``, o termo ``1 + α`` é constante e o
custo médio é O(1) — mas isso é uma **promessa condicional**, válida só enquanto
o hash espalha. :func:`bench_adversarial` quebra a condição de propósito, com
chaves reais escolhidas para caírem todas no mesmo bucket, e mostra a
degradação para Θ(n).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Generic, Iterable, Iterator, TypeVar

from src.counters import RANDOM_SEED, Counters, make_row, stopwatch
from src.linked_list import SinglyLinkedList

K = TypeVar("K")
V = TypeVar("V")

#: Capacidade inicial. Potência de dois para que ``% capacidade`` seja um
#: mascaramento de bits e para que o argumento do cenário adversarial
#: (colisão que sobrevive a todos os redimensionamentos) seja exato.
DEFAULT_CAPACITY: int = 8

#: Limiar de fator de carga que dispara o redimensionamento.
DEFAULT_LOAD_FACTOR_THRESHOLD: float = 0.75

#: Fator de crescimento no rehash. Dobrar é o que torna o custo total dos
#: rehashes uma série geométrica (Σ n/2^k < 2n), isto é, O(1) amortizado por
#: inserção — verificado em :func:`bench_rehash_amortizado`.
GROWTH_FACTOR: int = 2


# ----------------------------------------------------------------------
# Exceções próprias (convenção 9: nunca `raise Exception(...)`)
# ----------------------------------------------------------------------
class HashTableError(Exception):
    """Erro base de :class:`HashTableChained`."""


class KeyNotFoundError(HashTableError):
    """A chave pedida não está na tabela."""


class UnsupportedKeyError(HashTableError):
    """Tipo de chave sem codificação determinística em bytes (ver decisão 3)."""


class HashTableConfigError(HashTableError):
    """Parâmetro de construção inválido (capacidade, limiar de carga)."""


class HashTableInvariantError(HashTableError):
    """Invariante estrutural violada (bug de implementação, não de uso)."""


# ----------------------------------------------------------------------
# Função de hash determinística
# ----------------------------------------------------------------------
_FNV_OFFSET_BASIS_64: int = 0xCBF29CE484222325
_FNV_PRIME_64: int = 0x100000001B3
_MASK_64: int = 0xFFFFFFFFFFFFFFFF


def fnv1a_64(data: bytes) -> int:
    """Hash FNV-1a de 64 bits. Custo: O(len(data)).

    Escolhido por ser determinístico entre execuções (ao contrário de ``hash()``
    para ``str``), de dispersão conhecida e simples o bastante para ser defendido
    linha a linha no vídeo.
    """
    digest = _FNV_OFFSET_BASIS_64
    for byte in data:
        digest ^= byte
        digest = (digest * _FNV_PRIME_64) & _MASK_64
    return digest


def key_to_bytes(key: Any) -> bytes:
    """Codificação determinística e injetora da chave em bytes.

    O prefixo de tipo (``s:``, ``i:``, ...) evita que chaves de tipos diferentes
    com a mesma grafia colidam por construção — sem ele, ``"1"`` e ``1``
    gerariam os mesmos bytes.

    Custo: O(tamanho da chave).

    Levanta :class:`UnsupportedKeyError` para tipos sem codificação definida.
    """
    if key is None:
        return b"n:"
    if isinstance(key, bytes):
        return b"b:" + key
    if isinstance(key, str):
        return b"s:" + key.encode("utf-8")
    if isinstance(key, bool):
        # bool antes de int: True == 1 em Python, então os dois precisam gerar
        # os MESMOS bytes, senão cairiam em buckets diferentes comparando iguais.
        return b"i:" + str(int(key)).encode("ascii")
    if isinstance(key, int):
        return b"i:" + str(key).encode("ascii")
    if isinstance(key, tuple):
        parts = b",".join(key_to_bytes(item) for item in key)
        return b"t:(" + parts + b")"
    raise UnsupportedKeyError(
        f"tipo de chave não suportado: {type(key).__name__}. "
        "Aceitos: str, bytes, int, bool, None e tuple desses. "
        "O motivo é reprodutibilidade: hashear outro tipo exigiria o hash() "
        "embutido, que é aleatorizado por processo para str."
    )


def default_hash(key: Any) -> int:
    """Hash determinístico da chave. Custo: O(tamanho da chave)."""
    return fnv1a_64(key_to_bytes(key))


# ----------------------------------------------------------------------
# Par armazenado e bucket
# ----------------------------------------------------------------------
@dataclass(slots=True)
class Entry(Generic[K, V]):
    """Par (chave, valor) guardado em um nó do bucket.

    ``value`` é mutável de propósito: atualizar uma chave existente é uma
    escrita no lugar, sem inserir nó novo — é o que o enunciado pede em
    "atualização de valor para chave existente sem inserir duplicata".
    """

    key: K
    value: V


class HashBucket(SinglyLinkedList["Entry[K, V]"], Generic[K, V]):
    """Lista encadeada de pares (chave, valor) — um bucket da tabela.

    Herda de :class:`~src.linked_list.SinglyLinkedList` (decisão de projeto 2) e
    acrescenta as operações por chave. A classe base compara *valores* inteiros
    com ``==``; aqui a comparação é só da chave, e o valor é atualizado no lugar.

    Todo método faz uma única travessia: ``remove`` desliga o nó no mesmo laço em
    que o encontra, usando o ``_unlink`` herdado (que mantém ``_tail`` e
    ``_size`` corretos).
    """

    __slots__ = ()

    def find(self, key: K, counters: Counters) -> Entry[K, V] | None:
        """Par com a chave ``key``, ou ``None``. Custo: O(comprimento da cadeia).

        Conta 1 comparação por nó visitado e 1 salto por ponteiro seguido.
        """
        current = self._head
        while current is not None:
            if counters.eq(current.value.key, key):
                return current.value
            current = current.next
            counters.count_hop()
        return None

    def put(self, key: K, value: V, counters: Counters) -> bool:
        """Insere ou atualiza. Devolve ``True`` se a chave é nova.

        Custo: O(comprimento da cadeia).
        """
        entry = self.find(key, counters)
        if entry is not None:
            entry.value = value
            counters.count_copy()
            return False
        self.insert_last(Entry(key, value), counters)
        return True

    def remove(self, key: K, counters: Counters) -> Entry[K, V] | None:
        """Remove e devolve o par, ou ``None`` se a chave não está no bucket.

        Custo: O(comprimento da cadeia), em uma única travessia.
        """
        previous = None
        current = self._head
        while current is not None:
            if counters.eq(current.value.key, key):
                self._unlink(previous, current)
                return current.value
            previous = current
            current = current.next
            counters.count_hop()
        return None

    def keys(self) -> Iterator[K]:
        """Chaves do bucket, na ordem de inserção. Custo: O(comprimento)."""
        for entry in self:
            yield entry.key


# ----------------------------------------------------------------------
# Tabela
# ----------------------------------------------------------------------
class HashTableChained(Generic[K, V]):
    """Tabela hash com resolução de colisão por encadeamento separado.

    Invariantes mantidas por todos os métodos (ver :meth:`check_invariants`):

    * ``len(self) == Σ len(bucket)``;
    * toda entrada está no bucket que seu hash indica;
    * nenhuma chave aparece duas vezes;
    * ``load_factor() <= threshold`` depois de qualquer ``put``;
    * cada bucket satisfaz as invariantes de :class:`SinglyLinkedList`.
    """

    __slots__ = (
        "_buckets",
        "_size",
        "_threshold",
        "_hash_function",
        "_resizes",
        "_entries_rehashed",
    )

    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY,
        *,
        load_factor_threshold: float = DEFAULT_LOAD_FACTOR_THRESHOLD,
        hash_function: Any = default_hash,
    ) -> None:
        """Cria a tabela vazia.

        ``load_factor_threshold`` pode ser ``float("inf")`` para **desligar** o
        redimensionamento — é assim que :func:`bench_alpha_vs_comparisons` faz α
        crescer de propósito para validar a fórmula ``1 + α/2``.

        ``hash_function`` é injetável para permitir cenários controlados nos
        experimentos, sem gambiarra dentro da classe.

        Custo: O(capacity).
        """
        if not isinstance(capacity, int) or isinstance(capacity, bool):
            raise HashTableConfigError(
                f"capacidade deve ser int, recebido {type(capacity).__name__}"
            )
        if capacity < 1:
            raise HashTableConfigError(
                f"capacidade deve ser >= 1, recebido {capacity}"
            )
        if capacity & (capacity - 1) != 0:
            raise HashTableConfigError(
                f"capacidade deve ser potência de dois, recebido {capacity}. "
                "Potência de dois mantém o redimensionamento como duplicação "
                "exata e torna o cenário adversarial demonstrável."
            )
        if load_factor_threshold <= 0 or math.isnan(load_factor_threshold):
            raise HashTableConfigError(
                f"limiar de carga deve ser > 0, recebido {load_factor_threshold}"
            )

        self._buckets: list[HashBucket[K, V]] = [
            HashBucket() for _ in range(capacity)
        ]
        self._size = 0
        self._threshold = load_factor_threshold
        self._hash_function = hash_function
        self._resizes = 0
        self._entries_rehashed = 0

    # -- leitura -------------------------------------------------------
    def __len__(self) -> int:
        """Número de pares. Custo: O(1)."""
        return self._size

    def is_empty(self) -> bool:
        """``True`` se a tabela não tem pares. Custo: O(1)."""
        return self._size == 0

    @property
    def capacity(self) -> int:
        """Número de buckets. Custo: O(1)."""
        return len(self._buckets)

    @property
    def resizes(self) -> int:
        """Quantos rehashes já aconteceram. Custo: O(1)."""
        return self._resizes

    @property
    def entries_rehashed(self) -> int:
        """Total de pares movidos somando todos os rehashes. Custo: O(1)."""
        return self._entries_rehashed

    def load_factor(self) -> float:
        """``α = n/m``. Custo: O(1)."""
        return self._size / len(self._buckets)

    def bucket_index(self, key: K) -> int:
        """Índice do bucket da chave. Custo: O(tamanho da chave)."""
        return self._hash_function(key) % len(self._buckets)

    def bucket_lengths(self) -> list[int]:
        """Comprimento de cada bucket, para análise de dispersão. Custo: O(m)."""
        return [len(bucket) for bucket in self._buckets]

    def items(self) -> Iterator[tuple[K, V]]:
        """Pares (chave, valor), em ordem de bucket. Custo: O(n + m)."""
        for bucket in self._buckets:
            for entry in bucket:
                yield entry.key, entry.value

    def keys(self) -> Iterator[K]:
        """Chaves, em ordem de bucket. Custo: O(n + m)."""
        for key, _ in self.items():
            yield key

    def values(self) -> Iterator[V]:
        """Valores, em ordem de bucket. Custo: O(n + m)."""
        for _, value in self.items():
            yield value

    def __iter__(self) -> Iterator[K]:
        """Itera pelas chaves. Custo: O(n + m)."""
        return self.keys()

    def __str__(self) -> str:
        """Custo: O(n + m)."""
        pairs = ", ".join(f"{key!r}: {value!r}" for key, value in self.items())
        return "{" + pairs + "}"

    def __repr__(self) -> str:
        return (
            f"HashTableChained(size={self._size}, capacity={self.capacity}, "
            f"alpha={self.load_factor():.3f})"
        )

    # -- operações principais ------------------------------------------
    def put(self, key: K, value: V, counters: Counters | None = None) -> None:
        """Insere o par ou atualiza o valor de uma chave já existente.

        Nunca cria chave duplicada: se a chave já está no bucket, o valor é
        escrito no lugar (1 cópia) e ``__len__`` não muda.

        Custo: O(1 + α) no caso médio; O(n) no pior. Amortizado O(1) contando o
        redimensionamento, porque dobrar a capacidade torna o custo total dos
        rehashes uma série geométrica.
        """
        counters = counters if counters is not None else Counters()
        bucket = self._buckets[self.bucket_index(key)]
        if bucket.put(key, value, counters):
            self._size += 1
            self._maybe_resize(counters)

    def put_counted(self, key: K, value: V) -> tuple[None, Counters]:
        """:meth:`put` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        self.put(key, value, counters)
        return None, counters

    def put_if_absent(
        self, key: K, value: V, counters: Counters | None = None
    ) -> bool:
        """Insere só se a chave ainda não existir; devolve ``True`` se inseriu.

        Faz **uma** travessia do bucket. A alternativa óbvia — ``contains()``
        seguido de ``put()`` — percorre a cadeia duas vezes e dobra as
        comparações medidas, o que num cenário de colisão transforma ``n(n−1)/2``
        em ``n(n−1)``. É o caminho que a deduplicação do exercício 3 usa.

        Custo: O(1 + α) no caso médio; O(n) no pior.
        """
        counters = counters if counters is not None else Counters()
        bucket = self._buckets[self.bucket_index(key)]
        if bucket.find(key, counters) is not None:
            return False
        bucket.insert_last(Entry(key, value), counters)
        self._size += 1
        self._maybe_resize(counters)
        return True

    def get(self, key: K, counters: Counters | None = None) -> V:
        """Valor da chave. Levanta :class:`KeyNotFoundError` se ausente.

        Custo: O(1 + α) no caso médio; O(n) no pior.
        """
        counters = counters if counters is not None else Counters()
        bucket = self._buckets[self.bucket_index(key)]
        entry = bucket.find(key, counters)
        if entry is None:
            raise KeyNotFoundError(
                f"chave {key!r} não está na tabela ({self._size} par(es), "
                f"bucket {self.bucket_index(key)} de {self.capacity})"
            )
        return entry.value

    def get_counted(self, key: K) -> tuple[V, Counters]:
        """:meth:`get` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        return self.get(key, counters), counters

    def get_or(
        self, key: K, default: Any = None, counters: Counters | None = None
    ) -> Any:
        """Valor da chave, ou ``default`` se ausente — **uma só travessia**.

        É o caminho que o memo do ex 8 usa: ``contains`` seguido de ``get``
        percorreria o bucket duas vezes e dobraria as comparações medidas.

        Custo: O(1 + α) no caso médio.
        """
        counters = counters if counters is not None else Counters()
        bucket = self._buckets[self.bucket_index(key)]
        entry = bucket.find(key, counters)
        return default if entry is None else entry.value

    def delete(self, key: K, counters: Counters | None = None) -> V:
        """Remove a chave e devolve o valor que estava guardado.

        Levanta :class:`KeyNotFoundError` se a chave não existe — remover algo
        que não está lá é erro do chamador, não silêncio.

        A tabela **não encolhe**: liberar buckets exigiria um segundo limiar e,
        com ele, o risco de oscilação (inserir/remover em torno da fronteira
        dispararia rehash a cada operação). Decisão registrada aqui porque o
        custo é memória que não volta.

        Custo: O(1 + α) no caso médio; O(n) no pior.
        """
        counters = counters if counters is not None else Counters()
        bucket = self._buckets[self.bucket_index(key)]
        entry = bucket.remove(key, counters)
        if entry is None:
            raise KeyNotFoundError(
                f"delete({key!r}): chave não está na tabela "
                f"({self._size} par(es))"
            )
        self._size -= 1
        return entry.value

    def delete_counted(self, key: K) -> tuple[V, Counters]:
        """:meth:`delete` no protocolo ``(resultado, Counters)``."""
        counters = Counters()
        return self.delete(key, counters), counters

    def contains(self, key: K, counters: Counters | None = None) -> bool:
        """``True`` se a chave está na tabela. Custo: O(1 + α) no caso médio."""
        counters = counters if counters is not None else Counters()
        bucket = self._buckets[self.bucket_index(key)]
        return bucket.find(key, counters) is not None

    def __contains__(self, key: object) -> bool:
        return self.contains(key)  # type: ignore[arg-type]

    # -- redimensionamento ---------------------------------------------
    def _maybe_resize(self, counters: Counters) -> None:
        """Dispara o rehash se o fator de carga passou do limiar. Custo: O(1)
        para testar, O(n + m) quando redimensiona."""
        if self.load_factor() > self._threshold:
            self._resize(len(self._buckets) * GROWTH_FACTOR, counters)

    def _resize(self, new_capacity: int, counters: Counters) -> None:
        """Reconstrói a tabela com ``new_capacity`` buckets.

        Nenhuma comparação de chave é contada aqui, e o motivo importa: as
        chaves já são distintas por invariante, então o rehash só recalcula
        índices e move nós. Contar comparações no rehash poluiria a fórmula
        ``1 + α/2``, que é sobre busca.

        Custo: O(n + m).
        """
        old_buckets = self._buckets
        self._buckets = [HashBucket() for _ in range(new_capacity)]
        moved = 0
        for bucket in old_buckets:
            for entry in bucket:
                index = self.bucket_index(entry.key)
                self._buckets[index].insert_last(entry, counters)
                moved += 1
        self._resizes += 1
        self._entries_rehashed += moved

    # -- estatísticas e verificação ------------------------------------
    def stats(self) -> dict[str, Any]:
        """Retrato da dispersão, para a análise do notebook. Custo: O(n + m)."""
        lengths = self.bucket_lengths()
        occupied = [length for length in lengths if length > 0]
        mean = (sum(lengths) / len(lengths)) if lengths else 0.0
        variance = (
            sum((length - mean) ** 2 for length in lengths) / len(lengths)
            if lengths
            else 0.0
        )
        return {
            "n": self._size,
            "capacidade": self.capacity,
            "alpha": self.load_factor(),
            "buckets_ocupados": len(occupied),
            "buckets_vazios": len(lengths) - len(occupied),
            "cadeia_maxima": max(lengths) if lengths else 0,
            "cadeia_media": mean,
            "cadeia_media_ocupados": (
                sum(occupied) / len(occupied) if occupied else 0.0
            ),
            "desvio_padrao_cadeia": math.sqrt(variance),
            "redimensionamentos": self._resizes,
            "pares_reposicionados": self._entries_rehashed,
        }

    def check_invariants(self) -> None:
        """Valida as invariantes; levanta :class:`HashTableInvariantError`.

        Custo: O(n + m).
        """
        capacity = len(self._buckets)
        if capacity < 1 or capacity & (capacity - 1) != 0:
            raise HashTableInvariantError(
                f"capacidade deve ser potência de dois, está {capacity}"
            )

        total = 0
        seen: set[bytes] = set()
        for index, bucket in enumerate(self._buckets):
            bucket.check_invariants()  # invariantes da lista encadeada do ex 10
            total += len(bucket)
            for entry in bucket:
                expected = self.bucket_index(entry.key)
                if expected != index:
                    raise HashTableInvariantError(
                        f"chave {entry.key!r} está no bucket {index} "
                        f"mas seu hash aponta para {expected}"
                    )
                fingerprint = key_to_bytes(entry.key)
                if fingerprint in seen:
                    raise HashTableInvariantError(
                        f"chave {entry.key!r} aparece mais de uma vez na tabela"
                    )
                seen.add(fingerprint)

        if total != self._size:
            raise HashTableInvariantError(
                f"_size={self._size} mas os buckets somam {total} par(es)"
            )
        if math.isfinite(self._threshold) and self.load_factor() > self._threshold:
            raise HashTableInvariantError(
                f"fator de carga {self.load_factor():.3f} passou do limiar "
                f"{self._threshold} sem redimensionar"
            )


# ----------------------------------------------------------------------
# Cenário adversarial
# ----------------------------------------------------------------------
#: Cache das buscas por chaves colidentes. A procura é por força bruta e o
#: notebook chama a função várias vezes com o mesmo (prefixo, módulo, resíduo);
#: sem cache, cada chamada repetiria o mesmo varrimento. Guarda a lista já
#: encontrada e o próximo candidato a testar.
_COLLISION_CACHE: dict[tuple[str, int, int], tuple[list[str], int]] = {}


def find_colliding_keys(
    count: int,
    *,
    modulus: int = 1024,
    prefix: str = "adv",
    residue: int = 0,
    max_candidates: int = 5_000_000,
) -> list[str]:
    """Chaves ``str`` reais que caem todas no mesmo bucket.

    Procura por força bruta chaves cujo :func:`default_hash` seja congruente a
    ``residue`` módulo ``modulus``. Com ``modulus`` potência de dois, isso é
    mais forte do que colidir em uma capacidade específica: se
    ``h₁ ≡ h₂ (mod 2^k)``, então ``h₁ ≡ h₂ (mod 2^j)`` para todo ``j ≤ k``.
    Ou seja, as chaves continuam no mesmo bucket **em todas as capacidades até
    ``modulus``**, e o redimensionamento automático não as separa.

    É esse o ponto do cenário adversarial: não basta a tabela crescer. Quem
    conhece a função de hash escolhe chaves imunes ao crescimento, e a promessa
    O(1) vira Θ(n).

    Custo esperado: O(count · modulus) tentativas de hash.
    """
    if count < 1:
        raise ValueError(f"count deve ser >= 1, recebido {count}")
    if modulus < 1 or modulus & (modulus - 1) != 0:
        raise ValueError(f"modulus deve ser potência de dois, recebido {modulus}")

    cache_key = (prefix, modulus, residue)
    found, candidate = _COLLISION_CACHE.get(cache_key, ([], 0))
    found = list(found)
    while len(found) < count:
        if candidate >= max_candidates:
            raise ValueError(
                f"não encontrei {count} chaves colidentes em {max_candidates} "
                f"candidatos (modulus={modulus}); reduza count ou modulus"
            )
        key = f"{prefix}{candidate}"
        if default_hash(key) % modulus == residue:
            found.append(key)
        candidate += 1
    _COLLISION_CACHE[cache_key] = (found, candidate)
    return found[:count]


# ----------------------------------------------------------------------
# Experimentos do exercício 4
# ----------------------------------------------------------------------
def bench_alpha_vs_comparisons(
    capacity: int = 64,
    sizes: Iterable[int] = (16, 32, 64, 128, 256, 512),
    *,
    seed: int = RANDOM_SEED,
    probes: int = 400,
) -> list[dict[str, Any]]:
    """Valida ``1 + α/2`` (busca com sucesso) e ``α`` (busca sem sucesso).

    O redimensionamento é **desligado** (``load_factor_threshold = inf``) e a
    capacidade fica fixa, de modo que α cresce junto com n. É o único jeito de
    ver a dependência em α: com redimensionamento ligado, α nunca passa de 0,75
    e a curva vira uma reta horizontal (que é justamente o que
    :func:`bench_resizing_keeps_cost_constant` mostra).

    Duas linhas por tamanho: uma para busca bem-sucedida, outra para
    mal-sucedida, cada uma com a previsão teórica ao lado da medição.

    Custo: O(Σ (n + probes · α)).
    """
    rows: list[dict[str, Any]] = []
    for n in sizes:
        if n <= 0:
            raise ValueError(f"tamanho deve ser positivo, recebido {n}")
        rng = random.Random(seed)
        table: HashTableChained[str, int] = HashTableChained(
            capacity, load_factor_threshold=float("inf")
        )
        keys = [f"chave-{index}" for index in range(n)]
        for value, key in enumerate(keys):
            table.put(key, value)
        table.check_invariants()
        alpha = table.load_factor()

        for label, sample, theoretical in (
            (
                "busca com sucesso",
                [rng.choice(keys) for _ in range(probes)],
                1 + alpha / 2,
            ),
            (
                "busca sem sucesso",
                [f"ausente-{index}" for index in range(probes)],
                alpha,
            ),
        ):
            counters = Counters()
            with stopwatch() as elapsed:
                for key in sample:
                    table.get_or(key, None, counters)
            measured = counters.comparisons / probes
            rows.append(
                {
                    "exercicio": "ex4",
                    "algoritmo": f"HashTableChained.get — {label}",
                    "n": n,
                    "padrao": label,
                    "capacidade": table.capacity,
                    "alpha": alpha,
                    **counters.as_dict(),
                    "sondagens": probes,
                    "comparacoes_por_busca": measured,
                    "previsao_teorica": theoretical,
                    "medido/teorico": (
                        measured / theoretical if theoretical else float("nan")
                    ),
                    "cadeia_maxima": max(table.bucket_lengths()),
                    "tempo_s": elapsed[0],
                }
            )
    return rows


def bench_resizing_keeps_cost_constant(
    sizes: Iterable[int] = (500, 2_000, 8_000, 32_000),
    *,
    seed: int = RANDOM_SEED,
    probes: int = 500,
) -> list[dict[str, Any]]:
    """Com redimensionamento ligado, o custo de busca não cresce com n.

    É a evidência do O(1) **médio**: n cresce 64×, α fica presa abaixo de 0,75
    pelo redimensionamento, e as comparações por busca ficam praticamente
    constantes. A razão ``comparações_por_busca / n`` despenca — o que descarta
    Θ(n) — enquanto o valor absoluto não se move.

    Custo: O(Σ n).
    """
    rows: list[dict[str, Any]] = []
    for n in sizes:
        if n <= 0:
            raise ValueError(f"tamanho deve ser positivo, recebido {n}")
        rng = random.Random(seed)
        table: HashTableChained[str, int] = HashTableChained()
        keys = [f"chave-{index}" for index in range(n)]
        build_counters = Counters()
        with stopwatch() as build_elapsed:
            for value, key in enumerate(keys):
                table.put(key, value, build_counters)
        table.check_invariants()

        sample = [rng.choice(keys) for _ in range(probes)]
        probe_counters = Counters()
        with stopwatch() as probe_elapsed:
            for key in sample:
                table.get(key, probe_counters)

        stats = table.stats()
        rows.append(
            {
                "exercicio": "ex4",
                "algoritmo": "HashTableChained — carga e busca",
                "n": n,
                "padrao": "chaves distintas, redimensionamento ligado",
                "capacidade": table.capacity,
                "alpha": table.load_factor(),
                "comparacoes_construcao": build_counters.comparisons,
                "comparacoes_por_insercao": build_counters.comparisons / n,
                "comparacoes_por_busca": probe_counters.comparisons / probes,
                "comparacoes_por_busca/n": probe_counters.comparisons / probes / n,
                "previsao_teorica": 1 + table.load_factor() / 2,
                "cadeia_maxima": stats["cadeia_maxima"],
                "buckets_vazios": stats["buckets_vazios"],
                "redimensionamentos": stats["redimensionamentos"],
                "pares_reposicionados": stats["pares_reposicionados"],
                "reposicionados/n": stats["pares_reposicionados"] / n,
                "tempo_construcao_s": build_elapsed[0],
                "tempo_busca_s": probe_elapsed[0],
            }
        )
    return rows


def bench_rehash_amortizado(
    sizes: Iterable[int] = (100, 1_000, 10_000, 100_000),
) -> list[dict[str, Any]]:
    """Mostra que o custo total dos rehashes é O(n), isto é O(1) amortizado.

    Inserindo n chaves a partir da capacidade ``c₀``, os rehashes movem
    ``c₀ + 2c₀ + 4c₀ + ... < 2n/0,75`` pares. A coluna ``reposicionados/n`` é a
    evidência: tem de ficar limitada por uma constante enquanto n cresce 1000×,
    e não crescer com n.

    Custo: O(Σ n).
    """
    rows: list[dict[str, Any]] = []
    for n in sizes:
        if n <= 0:
            raise ValueError(f"tamanho deve ser positivo, recebido {n}")
        table: HashTableChained[int, int] = HashTableChained()
        counters = Counters()
        with stopwatch() as elapsed:
            for value in range(n):
                table.put(value, value, counters)
        table.check_invariants()
        rows.append(
            {
                "exercicio": "ex4",
                "algoritmo": "HashTableChained — custo amortizado do rehash",
                "n": n,
                "padrao": "n inserções a partir da capacidade inicial 8",
                "capacidade": table.capacity,
                "alpha": table.load_factor(),
                "redimensionamentos": table.resizes,
                "pares_reposicionados": table.entries_rehashed,
                "reposicionados/n": table.entries_rehashed / n,
                "copias_totais": counters.copies,
                "copias/n": counters.copies / n,
                "comparacoes/n": counters.comparisons / n,
                "tempo_s": elapsed[0],
                "tempo_us_por_insercao": elapsed[0] * 1e6 / n,
            }
        )
    return rows


def bench_adversarial(
    sizes: Iterable[int] = (50, 100, 200, 400),
    *,
    modulus: int = 1024,
    probes: int = 100,
) -> list[dict[str, Any]]:
    """Chaves escolhidas para colidir: a promessa O(1) cai para Θ(n).

    Para cada n, compara duas tabelas **idênticas em configuração** (mesma
    função de hash, mesmo limiar, redimensionamento ligado):

    * **benigno** — chaves ``chave-0..n-1``, espalhadas pelo hash;
    * **adversarial** — chaves de :func:`find_colliding_keys`, congruentes
      módulo ``modulus`` e portanto no mesmo bucket em qualquer capacidade até
      ``modulus`` (por isso ``modulus`` tem de ser ≥ a maior capacidade que a
      tabela vai atingir, senão o redimensionamento separa as chaves).

    **Modelo de ataque.** O adversário controla as chaves inseridas *e* as
    consultadas — é assim que um hash flooding funciona: o atacante enche a
    estrutura e depois consulta. Por isso as sondagens do cenário adversarial
    também são chaves colidentes (ausentes da tabela): uma chave qualquer cairia
    num bucket vazio e mediria 0 comparações, o que descreveria um ataque que
    ninguém faria. No cenário benigno as sondagens são chaves ausentes comuns,
    que é o controle correto.

    Resultado esperado: no lado adversarial a busca sem sucesso percorre a
    cadeia inteira, ``comparações_por_busca = n`` e
    ``comparações_por_busca / n = 1`` — assinatura de Θ(n). No lado benigno a
    mesma razão tende a zero.

    Custo: O(Σ n²) no lado adversarial.
    """
    rows: list[dict[str, Any]] = []
    for n in sizes:
        if n <= 0:
            raise ValueError(f"tamanho deve ser positivo, recebido {n}")

        adversarial_pool = find_colliding_keys(n + probes, modulus=modulus)
        cenarios: list[tuple[str, list[str], list[str]]] = [
            (
                "benigno",
                [f"chave-{index}" for index in range(n)],
                [f"chave-{index}" for index in range(n, n + probes)],
            ),
            ("adversarial", adversarial_pool[:n], adversarial_pool[n:]),
        ]

        for label, keys, probe_keys in cenarios:
            table: HashTableChained[str, int] = HashTableChained()
            build_counters = Counters()
            for value, key in enumerate(keys):
                table.put(key, value, build_counters)
            table.check_invariants()

            stats = table.stats()
            if label == "adversarial" and stats["buckets_ocupados"] != 1:
                raise ValueError(
                    f"modulus={modulus} é pequeno demais para n={n}: a tabela "
                    f"chegou à capacidade {table.capacity} e as chaves se "
                    f"espalharam por {stats['buckets_ocupados']} buckets. "
                    "Use um modulus >= a capacidade final."
                )

            miss_counters = Counters()
            with stopwatch() as elapsed:
                for key in probe_keys:
                    table.get_or(key, None, miss_counters)

            rows.append(
                make_row(
                    exercise="ex4",
                    algorithm=f"HashTableChained.get — cenário {label}",
                    n=n,
                    pattern=label,
                    counters=miss_counters,
                    metric="comparisons",
                    curves=("n", "n2"),
                    elapsed_s=elapsed[0],
                    extra={
                        "cenario": label,
                        "capacidade": table.capacity,
                        "alpha": table.load_factor(),
                        "cadeia_maxima": stats["cadeia_maxima"],
                        "buckets_ocupados": stats["buckets_ocupados"],
                        "sondagens": len(probe_keys),
                        "comparacoes_por_busca": (
                            miss_counters.comparisons / len(probe_keys)
                        ),
                        "comparacoes_por_busca/n": (
                            miss_counters.comparisons / len(probe_keys) / n
                        ),
                        "comparacoes_construcao": build_counters.comparisons,
                    },
                )
            )
    return rows

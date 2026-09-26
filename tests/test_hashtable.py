"""Testes do exercício 4 — HashTableChained com encadeamento por lista.

Três exigências do enunciado viram teste explícito aqui:

* "atualização de valor para chave existente **sem inserir duplicata**";
* "demonstre por teste que **todas as chaves permanecem acessíveis após o
  rehash**";
* "cenário adversarial com chaves que colidem".

Além disso, a reprodutibilidade — que sustenta todos os números do notebook — é
testada contra vetores literais do FNV-1a: se alguém trocar a função de hash
própria pelo ``hash()`` embutido, estes testes quebram na hora.
"""

from __future__ import annotations

import random

import pytest

from src.counters import Counters
from src.hashtable import (
    DEFAULT_CAPACITY,
    DEFAULT_LOAD_FACTOR_THRESHOLD,
    Entry,
    HashBucket,
    HashTableChained,
    HashTableConfigError,
    HashTableInvariantError,
    KeyNotFoundError,
    UnsupportedKeyError,
    bench_adversarial,
    bench_alpha_vs_comparisons,
    bench_rehash_amortizado,
    bench_resizing_keeps_cost_constant,
    default_hash,
    find_colliding_keys,
    fnv1a_64,
    key_to_bytes,
)
from src.linked_list import SinglyLinkedList


# ----------------------------------------------------------------------
# Função de hash — determinismo é requisito, não detalhe
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"", 14695981039346656037),  # offset basis do FNV-1a de 64 bits
        (b"a", 12638187200555641996),  # vetor de teste conhecido do FNV-1a
        (b"hello", 11831194018420276491),
    ],
)
def test_fnv1a_bate_com_os_vetores_de_referencia(data: bytes, expected: int) -> None:
    assert fnv1a_64(data) == expected


def test_hash_e_estavel_entre_execucoes() -> None:
    """Valor literal: se alguém trocar por hash() embutido, isto quebra.

    hash() de str em CPython é salgado por PYTHONHASHSEED; com ele, os números
    do notebook mudariam a cada execução.
    """
    assert default_hash("termo") == 6982619395706773855
    assert default_hash(42) == 8630990101202016552
    assert default_hash(("a", 1)) == 2244445208850429220


def test_codificacao_da_chave_separa_tipos() -> None:
    assert key_to_bytes("1") != key_to_bytes(1)
    assert key_to_bytes(None) == b"n:"
    assert key_to_bytes(b"x") == b"b:x"
    assert key_to_bytes(("a", 1, None)) == b"t:(s:a,i:1,n:)"


def test_bool_e_int_equivalentes_geram_a_mesma_chave() -> None:
    """True == 1 em Python: se hasheassem diferente, a tabela teria duas
    entradas que comparam iguais em buckets distintos."""
    assert key_to_bytes(True) == key_to_bytes(1)
    assert key_to_bytes(False) == key_to_bytes(0)

    table: HashTableChained[object, str] = HashTableChained()
    table.put(1, "um")
    table.put(True, "verdadeiro")
    assert len(table) == 1
    assert table.get(1) == "verdadeiro"
    table.check_invariants()


@pytest.mark.parametrize("bad_key", [1.5, [1, 2], {1: 2}, {1, 2}, object()])
def test_tipo_de_chave_nao_suportado_levanta_com_explicacao(bad_key: object) -> None:
    with pytest.raises(UnsupportedKeyError, match="reprodutibilidade"):
        key_to_bytes(bad_key)


def test_tupla_com_elemento_invalido_tambem_levanta() -> None:
    with pytest.raises(UnsupportedKeyError):
        key_to_bytes(("ok", 1.5))


# ----------------------------------------------------------------------
# Bucket — é uma lista encadeada de verdade, a do exercício 10
# ----------------------------------------------------------------------
def test_bucket_e_uma_lista_encadeada_do_exercicio_10() -> None:
    bucket: HashBucket[str, int] = HashBucket()
    assert isinstance(bucket, SinglyLinkedList)
    assert len(bucket) == 0
    bucket.check_invariants()  # método herdado


def test_bucket_insere_atualiza_e_remove_por_chave() -> None:
    bucket: HashBucket[str, int] = HashBucket()
    counters = Counters()
    assert bucket.put("a", 1, counters) is True
    assert bucket.put("b", 2, counters) is True
    assert bucket.put("a", 10, counters) is False  # atualização, não inserção
    assert len(bucket) == 2
    assert list(bucket.keys()) == ["a", "b"]

    entry = bucket.find("a", counters)
    assert entry is not None and entry.value == 10

    removed = bucket.remove("a", counters)
    assert removed is not None and removed.key == "a"
    assert len(bucket) == 1
    assert bucket.remove("ausente", counters) is None
    bucket.check_invariants()


def test_bucket_conta_uma_comparacao_por_no_visitado() -> None:
    bucket: HashBucket[str, int] = HashBucket()
    for index in range(5):
        bucket.put(f"k{index}", index, Counters())

    for position in range(5):
        counters = Counters()
        bucket.find(f"k{position}", counters)
        assert counters.comparisons == position + 1
        assert counters.hops == position

    ausente = Counters()
    assert bucket.find("k99", ausente) is None
    assert ausente.comparisons == 5  # percorreu a cadeia inteira
    assert ausente.hops == 5


def test_bucket_remove_faz_uma_unica_travessia() -> None:
    """Buscar e desligar no mesmo laço: a contagem medida é a real, não o dobro."""
    bucket: HashBucket[str, int] = HashBucket()
    for index in range(5):
        bucket.put(f"k{index}", index, Counters())
    counters = Counters()
    bucket.remove("k4", counters)
    assert counters.comparisons == 5  # e não 10
    assert counters.hops == 4


def test_bucket_mantem_tail_apos_remover_o_ultimo() -> None:
    bucket: HashBucket[str, int] = HashBucket()
    for index in range(3):
        bucket.put(f"k{index}", index, Counters())
    bucket.remove("k2", Counters())
    bucket.put("novo", 99, Counters())
    assert list(bucket.keys()) == ["k0", "k1", "novo"]
    bucket.check_invariants()


# ----------------------------------------------------------------------
# Tabela — operações básicas
# ----------------------------------------------------------------------
def test_tabela_nova_esta_vazia() -> None:
    table: HashTableChained[str, int] = HashTableChained()
    assert len(table) == 0
    assert table.is_empty() is True
    assert table.capacity == DEFAULT_CAPACITY
    assert table.load_factor() == 0.0
    assert list(table.items()) == []
    table.check_invariants()


def test_put_get_delete_e_len() -> None:
    table: HashTableChained[str, int] = HashTableChained()
    table.put("a", 1)
    table.put("b", 2)
    table.put("c", 3)
    assert len(table) == 3
    assert (table.get("a"), table.get("b"), table.get("c")) == (1, 2, 3)
    assert table.delete("b") == 2
    assert len(table) == 2
    assert "b" not in table
    assert "a" in table
    table.check_invariants()


def test_atualizar_chave_existente_nao_cria_duplicata() -> None:
    """Exigência literal do enunciado."""
    table: HashTableChained[str, int] = HashTableChained()
    for repetition in range(50):
        table.put("mesma", repetition)
    assert len(table) == 1
    assert table.get("mesma") == 49
    assert list(table.keys()) == ["mesma"]
    assert sum(table.bucket_lengths()) == 1
    table.check_invariants()


def test_get_de_chave_ausente_levanta_com_contexto() -> None:
    table: HashTableChained[str, int] = HashTableChained()
    table.put("a", 1)
    with pytest.raises(KeyNotFoundError, match="não está na tabela"):
        table.get("z")


def test_delete_de_chave_ausente_levanta() -> None:
    table: HashTableChained[str, int] = HashTableChained()
    with pytest.raises(KeyNotFoundError, match="delete"):
        table.delete("z")


def test_valor_none_e_distinguivel_de_chave_ausente() -> None:
    """Decisão de projeto 4: por isso get levanta em vez de devolver None."""
    table: HashTableChained[str, object] = HashTableChained()
    table.put("presente", None)
    assert table.get("presente") is None
    assert "presente" in table
    with pytest.raises(KeyNotFoundError):
        table.get("ausente")
    assert table.get_or("ausente", "padrão") == "padrão"


def test_get_or_faz_uma_unica_travessia() -> None:
    table: HashTableChained[str, int] = HashTableChained(1)
    for index in range(6):
        table.put(f"k{index}", index, Counters())

    contains_then_get = Counters()
    table.contains("k5", contains_then_get)
    table.get("k5", contains_then_get)

    single = Counters()
    table.get_or("k5", None, single)

    assert single.comparisons * 2 == contains_then_get.comparisons


def test_itens_chaves_e_valores_sao_coerentes() -> None:
    table: HashTableChained[str, int] = HashTableChained()
    esperado = {f"k{index}": index for index in range(30)}
    for key, value in esperado.items():
        table.put(key, value)
    assert dict(table.items()) == esperado
    assert sorted(table.keys()) == sorted(esperado)
    assert sorted(table.values()) == sorted(esperado.values())
    assert sorted(iter(table)) == sorted(esperado)


def test_str_e_repr() -> None:
    table: HashTableChained[str, int] = HashTableChained()
    table.put("a", 1)
    assert str(table) == "{'a': 1}"
    assert "size=1" in repr(table)
    assert "capacity=8" in repr(table)


# ----------------------------------------------------------------------
# Redimensionamento e rehash
# ----------------------------------------------------------------------
def test_redimensiona_quando_passa_do_limiar() -> None:
    table: HashTableChained[int, int] = HashTableChained(8)
    assert table.capacity == 8
    for value in range(6):  # 6/8 = 0,75 -> ainda não passa
        table.put(value, value)
    assert table.capacity == 8
    assert table.resizes == 0
    table.put(6, 6)  # 7/8 = 0,875 > 0,75 -> dobra
    assert table.capacity == 16
    assert table.resizes == 1
    table.check_invariants()


def test_todas_as_chaves_continuam_acessiveis_apos_o_rehash() -> None:
    """Exigência literal do enunciado."""
    table: HashTableChained[str, int] = HashTableChained(8)
    esperado = {f"chave-{index}": index * 7 for index in range(500)}
    for key, value in esperado.items():
        table.put(key, value)

    assert table.resizes > 0  # o rehash realmente aconteceu
    assert table.capacity > 8
    assert len(table) == len(esperado)
    for key, value in esperado.items():
        assert table.get(key) == value
    assert dict(table.items()) == esperado
    table.check_invariants()


def test_rehash_mantem_o_fator_de_carga_abaixo_do_limiar() -> None:
    table: HashTableChained[int, int] = HashTableChained()
    for value in range(2_000):
        table.put(value, value)
        assert table.load_factor() <= DEFAULT_LOAD_FACTOR_THRESHOLD
    table.check_invariants()


def test_rehash_nao_conta_comparacoes_de_chave() -> None:
    """As chaves já são distintas por invariante; contar comparações no rehash
    poluiria a fórmula 1 + α/2, que é sobre busca."""
    table: HashTableChained[int, int] = HashTableChained(8)
    for value in range(6):
        table.put(value, value)
    antes = Counters()
    for value in range(6):
        table.contains(value, antes)

    rehash = Counters()
    table.put(6, 6, rehash)  # esta inserção dispara o redimensionamento
    assert table.resizes == 1
    # a única comparação é a da busca no bucket de destino, que estava vazio
    assert rehash.comparisons == 0
    assert rehash.copies == 1 + 7  # 1 inserção + 7 pares reposicionados


def test_custo_total_do_rehash_e_linear() -> None:
    """Dobrar a capacidade torna Σ reposicionamentos uma série geométrica."""
    table: HashTableChained[int, int] = HashTableChained()
    n = 10_000
    for value in range(n):
        table.put(value, value)
    assert table.entries_rehashed < 3 * n  # limitado por constante, não por n
    assert table.entries_rehashed / n < 3


def test_redimensionamento_pode_ser_desligado() -> None:
    table: HashTableChained[int, int] = HashTableChained(
        8, load_factor_threshold=float("inf")
    )
    for value in range(100):
        table.put(value, value)
    assert table.capacity == 8
    assert table.resizes == 0
    assert table.load_factor() == 12.5
    assert len(table) == 100
    table.check_invariants()


def test_tabela_nao_encolhe_ao_remover() -> None:
    """Decisão documentada: encolher exigiria um segundo limiar e traria
    oscilação de rehash em torno da fronteira."""
    table: HashTableChained[int, int] = HashTableChained()
    for value in range(100):
        table.put(value, value)
    capacidade_cheia = table.capacity
    for value in range(100):
        table.delete(value)
    assert table.is_empty() is True
    assert table.capacity == capacidade_cheia
    table.check_invariants()


# ----------------------------------------------------------------------
# Configuração inválida
# ----------------------------------------------------------------------
@pytest.mark.parametrize("bad_capacity", [0, -1, 3, 7, 100])
def test_capacidade_invalida_levanta(bad_capacity: int) -> None:
    with pytest.raises(HashTableConfigError):
        HashTableChained(bad_capacity)


@pytest.mark.parametrize("bad_capacity", ["8", 8.0, None, True])
def test_capacidade_nao_inteira_levanta(bad_capacity: object) -> None:
    with pytest.raises(HashTableConfigError):
        HashTableChained(bad_capacity)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_threshold", [0, -0.5, float("nan")])
def test_limiar_de_carga_invalido_levanta(bad_threshold: float) -> None:
    with pytest.raises(HashTableConfigError, match="limiar"):
        HashTableChained(8, load_factor_threshold=bad_threshold)


# ----------------------------------------------------------------------
# check_invariants tem de FALHAR em estrutura corrompida
# ----------------------------------------------------------------------
def test_check_invariants_detecta_size_errado() -> None:
    table: HashTableChained[str, int] = HashTableChained()
    table.put("a", 1)
    table._size = 5
    with pytest.raises(HashTableInvariantError, match="somam"):
        table.check_invariants()


def test_check_invariants_detecta_entrada_no_bucket_errado() -> None:
    table: HashTableChained[str, int] = HashTableChained()
    table.put("a", 1)
    correto = table.bucket_index("a")
    errado = (correto + 1) % table.capacity
    entry = table._buckets[correto].remove("a", Counters())
    assert entry is not None
    table._buckets[errado].insert_last(entry)
    with pytest.raises(HashTableInvariantError, match="bucket"):
        table.check_invariants()


def test_check_invariants_detecta_chave_duplicada() -> None:
    table: HashTableChained[str, int] = HashTableChained()
    table.put("a", 1)
    indice = table.bucket_index("a")
    table._buckets[indice].insert_last(Entry("a", 2))
    table._size += 1
    with pytest.raises(HashTableInvariantError, match="mais de uma vez"):
        table.check_invariants()


def test_check_invariants_detecta_carga_acima_do_limiar() -> None:
    table: HashTableChained[int, int] = HashTableChained(8)
    for value in range(6):
        table.put(value, value)
    # injeta pares sem passar pelo put, pulando o redimensionamento
    for value in range(100, 110):
        table._buckets[table.bucket_index(value)].insert_last(Entry(value, value))
        table._size += 1
    with pytest.raises(HashTableInvariantError, match="limiar"):
        table.check_invariants()


# ----------------------------------------------------------------------
# Cenário adversarial
# ----------------------------------------------------------------------
def test_find_colliding_keys_devolve_chaves_realmente_colidentes() -> None:
    keys = find_colliding_keys(20, modulus=512)
    assert len(keys) == len(set(keys)) == 20
    assert all(default_hash(key) % 512 == 0 for key in keys)
    # e, por serem congruentes módulo 2^9, colidem em toda capacidade menor
    for capacity in (8, 16, 32, 64, 128, 256, 512):
        assert len({default_hash(key) % capacity for key in keys}) == 1


def test_find_colliding_keys_valida_parametros() -> None:
    with pytest.raises(ValueError, match="count"):
        find_colliding_keys(0)
    with pytest.raises(ValueError, match="potência de dois"):
        find_colliding_keys(5, modulus=1000)
    with pytest.raises(ValueError, match="não encontrei"):
        # prefixo virgem: o cache de outras chamadas não pode satisfazer este
        find_colliding_keys(5, modulus=1024, prefix="zzz", max_candidates=10)


def test_chaves_adversariais_degradam_a_busca_para_linear() -> None:
    n = 200
    pool = find_colliding_keys(n + 1, modulus=1024)
    keys, sonda_ausente = pool[:n], pool[n]
    table: HashTableChained[str, int] = HashTableChained()
    for value, key in enumerate(keys):
        table.put(key, value)
    table.check_invariants()

    # o redimensionamento aconteceu e ainda assim tudo está em um só bucket
    assert table.resizes > 0
    assert table.stats()["buckets_ocupados"] == 1
    assert table.stats()["cadeia_maxima"] == n

    # a sondagem precisa ser uma chave COLIDENTE ausente: o modelo de ataque é
    # o adversário controlando também a consulta. Uma chave qualquer cairia num
    # bucket vazio e mediria 0 comparações, descrevendo um ataque que ninguém faz.
    ausente = Counters()
    table.get_or(sonda_ausente, None, ausente)
    assert ausente.comparisons == n  # Θ(n), não O(1)

    ultimo = Counters()
    table.get(keys[-1], ultimo)
    assert ultimo.comparisons == n

    # controle: chave ausente que NÃO colide não mede nada
    fora = Counters()
    table.get_or("chave-comum-ausente", None, fora)
    assert fora.comparisons == 0


def test_cenario_benigno_espalha_de_verdade() -> None:
    """Controle do teste anterior: com as mesmas n chaves, mas sem adversário."""
    n = 200
    table: HashTableChained[str, int] = HashTableChained()
    for value in range(n):
        table.put(f"chave-{value}", value)
    stats = table.stats()
    assert stats["buckets_ocupados"] > n / 2
    assert stats["cadeia_maxima"] <= 5
    ausente = Counters()
    table.get_or("nao-existe-mesmo", None, ausente)
    assert ausente.comparisons <= 5


# ----------------------------------------------------------------------
# Experimentos
# ----------------------------------------------------------------------
def test_bench_alpha_confirma_a_formula_1_mais_alpha_sobre_2() -> None:
    rows = bench_alpha_vs_comparisons(
        capacity=32, sizes=(32, 64, 128, 256), probes=1_000
    )
    com_sucesso = [row for row in rows if row["padrao"] == "busca com sucesso"]
    sem_sucesso = [row for row in rows if row["padrao"] == "busca sem sucesso"]
    assert len(com_sucesso) == len(sem_sucesso) == 4

    # A tolerância é de 30%, e não de 5%, por um motivo que vale registrar: a
    # previsão 1 + α/2 é uma média sobre funções de hash aleatórias. Uma
    # instância concreta com m = 32 buckets tem dispersão realizada que pode ser
    # melhor ou pior que a Poisson idealizada. O que a evidência mostra é que a
    # razão medido/teórico fica presa perto de 1 enquanto α cresce 8×.
    for row in com_sucesso:
        assert row["alpha"] == row["n"] / 32
        assert row["previsao_teorica"] == 1 + row["alpha"] / 2
        assert row["medido/teorico"] == pytest.approx(1.0, abs=0.30)

    for row in sem_sucesso:
        assert row["previsao_teorica"] == row["alpha"]
        # busca sem sucesso percorre a cadeia inteira: média == α
        assert row["medido/teorico"] == pytest.approx(1.0, abs=0.30)

    # e o custo cresce com α, o que é o ponto
    medidos = [row["comparacoes_por_busca"] for row in sem_sucesso]
    assert medidos == sorted(medidos)
    assert medidos[-1] > 5 * medidos[0]


def test_bench_resizing_mantem_o_custo_de_busca_constante() -> None:
    rows = bench_resizing_keeps_cost_constant((500, 2_000, 8_000), probes=200)
    assert len(rows) == 3
    for row in rows:
        assert row["alpha"] <= DEFAULT_LOAD_FACTOR_THRESHOLD
        assert row["comparacoes_por_busca"] < 2.0

    por_busca = [row["comparacoes_por_busca"] for row in rows]
    assert max(por_busca) / min(por_busca) < 1.5  # praticamente constante
    razao_contra_n = [row["comparacoes_por_busca/n"] for row in rows]
    assert razao_contra_n == sorted(razao_contra_n, reverse=True)  # despenca


def test_bench_rehash_mostra_custo_amortizado_constante() -> None:
    rows = bench_rehash_amortizado((100, 1_000, 10_000))
    assert len(rows) == 3
    for row in rows:
        assert row["reposicionados/n"] < 3
        assert row["redimensionamentos"] > 0
    # n cresce 100x e reposicionados/n não cresce
    razoes = [row["reposicionados/n"] for row in rows]
    assert max(razoes) / min(razoes) < 2


def test_bench_adversarial_separa_os_dois_cenarios() -> None:
    rows = bench_adversarial((50, 100), probes=50)
    benignos = [row for row in rows if row["cenario"] == "benigno"]
    adversariais = [row for row in rows if row["cenario"] == "adversarial"]
    assert len(benignos) == len(adversariais) == 2

    for row in adversariais:
        assert row["buckets_ocupados"] == 1
        assert row["cadeia_maxima"] == row["n"]
        # comparações por busca == n exatamente: assinatura de Θ(n)
        assert row["comparacoes_por_busca"] == row["n"]
        assert row["comparacoes_por_busca/n"] == pytest.approx(1.0)

    for row in benignos:
        assert row["comparacoes_por_busca"] < 3
        assert row["comparacoes_por_busca/n"] < 0.1


@pytest.mark.parametrize(
    ("nome", "chamada"),
    [
        ("alpha", lambda: bench_alpha_vs_comparisons(32, [0])),
        ("resizing", lambda: bench_resizing_keeps_cost_constant([0])),
        ("rehash", lambda: bench_rehash_amortizado([0])),
        ("adversarial", lambda: bench_adversarial([0])),
    ],
)
def test_benchs_validam_tamanho(nome: str, chamada: object) -> None:
    with pytest.raises(ValueError, match="tamanho deve ser positivo"):
        chamada()  # type: ignore[operator]


# ----------------------------------------------------------------------
# Modelo de referência: a tabela se comporta como um dict
# ----------------------------------------------------------------------
def test_sequencia_longa_aleatoria_bate_com_o_dict_de_referencia() -> None:
    """dict como referência de corretude é permitido pela convenção 9."""
    rng = random.Random(42)
    table: HashTableChained[str, int] = HashTableChained()
    model: dict[str, int] = {}

    for step in range(5_000):
        key = f"k{rng.randrange(300)}"
        acao = rng.choice(["put", "put", "delete", "get"])
        if acao == "put":
            value = rng.randrange(1_000)
            table.put(key, value)
            model[key] = value
        elif acao == "delete":
            if key in model:
                assert table.delete(key) == model.pop(key)
            else:
                with pytest.raises(KeyNotFoundError):
                    table.delete(key)
        else:
            if key in model:
                assert table.get(key) == model[key]
            else:
                assert table.get_or(key, "faltou") == "faltou"

        assert len(table) == len(model)
        if step % 250 == 0:
            table.check_invariants()

    table.check_invariants()
    assert dict(table.items()) == model

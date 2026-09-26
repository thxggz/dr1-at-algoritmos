"""Testes do exercício 6 — simulador de eventos com quatro filas e índice em BST.

O teste mais importante do arquivo é
``test_chave_como_string_produziria_listagem_errada``: ele constrói de propósito
a versão que o enunciado parece pedir ao pé da letra — chave ``"B17"`` como
string — e mostra que a listagem in-order sai fora de ordem. Sem ele, a decisão
de projeto 1 seria só uma opinião no docstring.
"""

from __future__ import annotations

import math

import pytest

from src.bst import BinarySearchTree
from src.counters import Counters
from src.simulator import (
    COST_TABLE,
    DEFAULT_CAPACITY,
    QUEUE_NAMES,
    CheckoutSimulator,
    ClientId,
    ClientIdError,
    ClientNotFoundError,
    SimulatorConfigError,
    bench_index_shape,
    bench_simulation,
    find_client,
    generate_events,
    list_served,
    list_served_by_queue,
)


# ----------------------------------------------------------------------
# ClientId
# ----------------------------------------------------------------------
def test_identificador_se_escreve_como_o_enunciado_pede() -> None:
    assert str(ClientId("B", 17)) == "B17"
    assert repr(ClientId("B", 17)) == "ClientId('B17')"


def test_identificador_ordena_por_fila_e_numero_nao_por_texto() -> None:
    """Decisão de projeto 1: B2 < B17, ao contrário da ordem de texto."""
    assert ClientId("B", 2) < ClientId("B", 17)
    assert "B17" < "B2"  # a ordem de texto diria o contrário
    assert ClientId("A", 99) < ClientId("B", 1)
    assert sorted([ClientId("B", 10), ClientId("B", 2), ClientId("A", 30)]) == [
        ClientId("A", 30), ClientId("B", 2), ClientId("B", 10)
    ]


def test_identificador_parse() -> None:
    assert ClientId.parse("B17") == ClientId("B", 17)
    assert ClientId.parse("a1") == ClientId("A", 1)
    for ruim in ("", "B", "17", "BB1", "B1x", 17):
        with pytest.raises(ClientIdError):
            ClientId.parse(ruim)  # type: ignore[arg-type]


def test_chave_como_string_produziria_listagem_errada() -> None:
    """A demonstração do problema que a decisão de projeto 1 evita."""
    ids = [f"B{numero}" for numero in range(1, 21)]

    como_texto: BinarySearchTree[str, None] = BinarySearchTree(ids)
    listagem_texto = list(como_texto)
    assert listagem_texto[:4] == ["B1", "B10", "B11", "B12"]  # errado
    assert listagem_texto != ids

    como_objeto: BinarySearchTree[ClientId, None] = BinarySearchTree(
        [ClientId.parse(texto) for texto in ids]
    )
    assert [str(chave) for chave in como_objeto] == ids  # certo


# ----------------------------------------------------------------------
# Regras de evento
# ----------------------------------------------------------------------
def test_minuscula_e_chegada_e_maiuscula_e_saida() -> None:
    resultado = CheckoutSimulator().run("aA")
    assert resultado.arrivals == 1
    assert resultado.departures == 1
    assert [entrada.kind for entrada in resultado.log] == ["chegada", "saida"]
    assert len(resultado.index) == 1


def test_identificador_e_sequencial_por_fila() -> None:
    resultado = CheckoutSimulator().run("aabAABb")
    atendidos = [str(registro.client) for registro in list_served(resultado)]
    assert atendidos == ["A1", "A2", "B1"]


def test_qualquer_nao_alfabetico_dispara_snapshot() -> None:
    """Decisão de projeto 5: espaço e quebra de linha também."""
    resultado = CheckoutSimulator().run("a1b c\td!e")
    gatilhos = [snapshot.trigger for snapshot in resultado.snapshots]
    assert gatilhos == ["1", " ", "\t", "!"]
    assert len(resultado.snapshots) == 4


def test_snapshot_fotografa_todas_as_filas() -> None:
    resultado = CheckoutSimulator().run("aabc*")
    snapshot = resultado.snapshots[0]
    assert snapshot.sizes == {"A": 2, "B": 1, "C": 1, "D": 0}
    assert snapshot.contents["A"] == ["A1", "A2"]
    assert snapshot.contents["D"] == []
    assert snapshot.total() == 4


def test_alfabetico_fora_das_filas_e_erro_e_nao_snapshot() -> None:
    """Decisão de projeto 4: um 'z' é erro de digitação, não snapshot."""
    resultado = CheckoutSimulator().run("azZ")
    assert len(resultado.snapshots) == 0
    assert len(resultado.errors) == 2
    assert all(erro.kind == "evento_invalido" for erro in resultado.errors)
    assert "não corresponde a nenhuma fila" in resultado.errors[0].message


def test_string_de_eventos_vazia() -> None:
    resultado = CheckoutSimulator().run("")
    assert resultado.log == []
    assert resultado.arrivals == 0
    assert len(resultado.index) == 0


# ----------------------------------------------------------------------
# Overflow e underflow — com contexto, e a simulação continua
# ----------------------------------------------------------------------
def test_overflow_registra_contexto_e_a_simulacao_continua() -> None:
    resultado = CheckoutSimulator(capacity=2).run("aaaaA")
    overflow = [erro for erro in resultado.errors if erro.kind == "overflow"]
    assert len(overflow) == 2
    erro = overflow[0]
    assert erro.queue == "A"
    assert erro.occupancy == 2
    assert erro.capacity == 2
    assert "capacidade 2 esgotada" in erro.message
    # a simulação seguiu: o 'A' final foi processado
    assert resultado.departures == 1


def test_underflow_registra_contexto_e_a_simulacao_continua() -> None:
    resultado = CheckoutSimulator().run("AAaA")
    underflow = [erro for erro in resultado.errors if erro.kind == "underflow"]
    assert len(underflow) == 2
    assert underflow[0].queue == "A"
    assert underflow[0].occupancy == 0
    assert "fila vazia" in underflow[0].message
    assert resultado.arrivals == 1
    assert resultado.departures == 1


def test_overflow_nao_consome_o_numero_sequencial() -> None:
    """Decisão de projeto 2: B17 é o 17º que ENTROU, não o 17º que tentou."""
    resultado = CheckoutSimulator(capacity=2).run("aaaaaAAA")
    atendidos = [str(registro.client) for registro in list_served(resultado)]
    assert atendidos == ["A1", "A2"]  # sem buracos na numeração
    # e a próxima chegada continua de onde parou
    seguinte = CheckoutSimulator(capacity=2).run("aaaaaAAa")
    assert [entrada.client for entrada in seguinte.log if entrada.kind == "chegada"] == [
        "A1", "A2", "A3"
    ]


def test_todos_os_erros_tem_instante_e_evento() -> None:
    resultado = CheckoutSimulator(capacity=1).run("aaAAz")
    assert len(resultado.errors) == 3
    for erro in resultado.errors:
        assert erro.tick >= 1
        assert erro.event
        assert erro.message


# ----------------------------------------------------------------------
# Filas independentes e FIFO
# ----------------------------------------------------------------------
def test_as_quatro_filas_sao_independentes() -> None:
    resultado = CheckoutSimulator().run("abcdABCD")
    atendidos = [str(registro.client) for registro in list_served(resultado)]
    assert atendidos == ["A1", "B1", "C1", "D1"]
    assert all(len(fila) == 0 for fila in resultado.remaining.values())


def test_atendimento_respeita_a_ordem_de_chegada_dentro_da_fila() -> None:
    resultado = CheckoutSimulator().run("aaaAAA")
    ordens = [
        (registro.service_order, str(registro.client))
        for registro in list_served(resultado)
    ]
    assert ordens == [(1, "A1"), (2, "A2"), (3, "A3")]


def test_clientes_nao_atendidos_ficam_na_fila() -> None:
    resultado = CheckoutSimulator().run("aaabbA")
    assert resultado.remaining["A"] == ["A2", "A3"]
    assert resultado.remaining["B"] == ["B1", "B2"]
    assert resultado.remaining["C"] == []


# ----------------------------------------------------------------------
# Metadados e índice
# ----------------------------------------------------------------------
def test_metadados_do_atendimento() -> None:
    resultado = CheckoutSimulator().run("a..A")
    registro = find_client(resultado, "A1")
    assert registro.queue == "A"
    assert registro.service_order == 1
    assert registro.arrival_tick == 1
    assert registro.service_tick == 4
    assert registro.wait_ticks == 3


def test_busca_por_identificador() -> None:
    resultado = CheckoutSimulator().run("abAB")
    assert find_client(resultado, "A1").queue == "A"
    assert find_client(resultado, "b1").queue == "B"  # parse normaliza a caixa
    with pytest.raises(ClientNotFoundError, match="não está no índice"):
        find_client(resultado, "C9")


def test_busca_distingue_nao_atendido_de_inexistente_na_mensagem() -> None:
    resultado = CheckoutSimulator().run("aa")  # A1 e A2 chegaram, ninguém saiu
    with pytest.raises(ClientNotFoundError, match="pode não ter sido atendido"):
        find_client(resultado, "A1")


def test_listagem_in_order_sai_em_ordem_numerica_por_fila() -> None:
    eventos = "a" * 12 + "A" * 12 + "b" * 3 + "B" * 3
    resultado = CheckoutSimulator(capacity=20).run(eventos)
    atendidos = [str(registro.client) for registro in list_served(resultado)]
    assert atendidos == [f"A{numero}" for numero in range(1, 13)] + [
        "B1", "B2", "B3"
    ]
    # o ponto: A2 vem antes de A10, que a ordem de texto inverteria
    assert atendidos.index("A2") < atendidos.index("A10")


def test_listagem_por_fila() -> None:
    resultado = CheckoutSimulator().run("ababABAB")
    assert [str(r.client) for r in list_served_by_queue(resultado, "A")] == [
        "A1", "A2"
    ]
    assert [str(r.client) for r in list_served_by_queue(resultado, "b")] == [
        "B1", "B2"
    ]
    assert list_served_by_queue(resultado, "D") == []


def test_indice_mantem_invariantes_apos_simulacao_longa() -> None:
    resultado = CheckoutSimulator().run(generate_events(5_000))
    resultado.index.check_invariants()
    assert len(resultado.index) == resultado.departures


# ----------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------
def test_simulador_valida_configuracao() -> None:
    with pytest.raises(SimulatorConfigError, match="capacidade deve ser >= 1"):
        CheckoutSimulator(0)
    with pytest.raises(SimulatorConfigError, match="capacidade deve ser int"):
        CheckoutSimulator("8")  # type: ignore[arg-type]
    with pytest.raises(SimulatorConfigError, match="ao menos uma fila"):
        CheckoutSimulator(4, [])
    with pytest.raises(SimulatorConfigError, match="repetidos"):
        CheckoutSimulator(4, ["A", "a"])
    with pytest.raises(SimulatorConfigError, match="única letra"):
        CheckoutSimulator(4, ["AB"])
    with pytest.raises(SimulatorConfigError, match="events deve ser str"):
        CheckoutSimulator().run(123)  # type: ignore[arg-type]


def test_simulador_aceita_outro_conjunto_de_filas() -> None:
    resultado = CheckoutSimulator(4, ["X", "Y"]).run("xyXY")
    assert resultado.departures == 2
    assert [str(r.client) for r in list_served(resultado)] == ["X1", "Y1"]
    # e 'a' deixa de ser fila válida
    assert CheckoutSimulator(4, ["X"]).run("a").errors[0].kind == "evento_invalido"


def test_capacidade_e_nomes_padrao() -> None:
    simulador = CheckoutSimulator()
    assert simulador.capacity == DEFAULT_CAPACITY
    assert simulador.queue_names == QUEUE_NAMES


# ----------------------------------------------------------------------
# Gerador de eventos
# ----------------------------------------------------------------------
def test_gerador_e_reprodutivel() -> None:
    assert generate_events(500, seed=42) == generate_events(500, seed=42)
    assert generate_events(500, seed=1) != generate_events(500, seed=2)
    assert len(generate_events(500)) == 500
    assert generate_events(0) == ""


def test_gerador_produz_os_dois_casos_de_borda() -> None:
    """Os casos que o enunciado manda tratar precisam acontecer de verdade."""
    resultado = CheckoutSimulator(capacity=4).run(
        generate_events(5_000, arrival_weight=0.55)
    )
    tipos = {erro.kind for erro in resultado.errors}
    assert "overflow" in tipos
    assert "underflow" in tipos
    assert "evento_invalido" in tipos
    assert len(resultado.snapshots) > 0


def test_gerador_valida_parametros() -> None:
    with pytest.raises(ValueError, match="length deve ser >= 0"):
        generate_events(-1)
    with pytest.raises(ValueError, match="arrival_weight"):
        generate_events(10, arrival_weight=2.0)
    with pytest.raises(ValueError, match="não pode passar de 1"):
        generate_events(10, snapshot_weight=0.8, invalid_weight=0.5)


# ----------------------------------------------------------------------
# Estatísticas e experimentos
# ----------------------------------------------------------------------
def test_stats_fecha_as_contas() -> None:
    resultado = CheckoutSimulator(capacity=3).run(generate_events(2_000))
    stats = resultado.stats()
    assert stats["eventos"] == 2_000
    assert stats["atendidos"] == stats["saidas"] == resultado.departures
    assert stats["erros"] == (
        stats["overflow"] + stats["underflow"] + stats["eventos_invalidos"]
    )
    # todo evento é chegada, saída, snapshot ou erro
    assert (
        stats["chegadas"] + stats["saidas"] + stats["snapshots"] + stats["erros"]
        == stats["eventos"]
    )
    # todo cliente que chegou ou foi atendido ou ficou na fila
    assert stats["chegadas"] == stats["atendidos"] + stats["em_espera_no_fim"]


def test_bench_simulation_roda_e_mede() -> None:
    rows = bench_simulation([200, 1_000])
    assert len(rows) == 2
    for row in rows:
        assert row["atendidos"] > 0
        assert row["comparacoes_por_atendimento"] > 0
        assert row["indice_altura"] >= row["indice_altura_ideal"]


def test_bench_simulation_valida_tamanho() -> None:
    with pytest.raises(ValueError, match="tamanho deve ser positivo"):
        bench_simulation([0])


def test_bench_index_shape_expoe_a_degeneracao() -> None:
    """Identificadores crescentes por fila degeneram o índice — ex 3 de novo."""
    rows = bench_index_shape([1_600, 6_400])
    cronologicas = [row for row in rows if row["ordem_de_insercao"] == "cronológica"]
    embaralhadas = [row for row in rows if row["ordem_de_insercao"] == "embaralhada"]
    assert len(cronologicas) == len(embaralhadas) == 2

    for crono, baralho in zip(cronologicas, embaralhadas):
        assert crono["atendidos"] == baralho["atendidos"]
        # a ordem cronológica produz árvore muito mais alta
        assert crono["altura"] > 5 * baralho["altura"]
        # e por isso custa muito mais para construir e para buscar
        assert crono["comparacoes_por_insercao"] > baralho["comparacoes_por_insercao"]
        # a embaralhada fica na faixa esperada de ~3·log2 s
        assert baralho["altura/log2s"] < 4.0
        # a cronológica cresce como uma fração de s, não como log s
        assert crono["altura/atendidos"] > 0.1


def test_bench_index_shape_valida_tamanho() -> None:
    with pytest.raises(ValueError, match="tamanho deve ser positivo"):
        bench_index_shape([0])


def test_tabela_de_custo_cobre_as_operacoes() -> None:
    operacoes = {row["operacao"] for row in COST_TABLE}
    for esperada in (
        "chegada (minúscula)",
        "saída (maiúscula)",
        "snapshot (não alfabético)",
        "find_client",
        "list_served (in-order)",
    ):
        assert esperada in operacoes
    for row in COST_TABLE:
        assert set(row) == {"operacao", "custo", "por_que"}
        assert row["por_que"]

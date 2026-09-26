"""Testes do exercício 12 — motor de indexação e consulta.

Além da corretude, a suíte verifica as duas coisas que o enunciado cobra do
integrador: que as **oito estruturas** dos exercícios anteriores estão mesmo
sendo usadas (e não reimplementadas ali dentro), e que as **duas otimizações**
são exatas — produzem o mesmo resultado da versão que substituem, e só mudam o
custo.
"""

from __future__ import annotations

import math

import pytest

from src.bst import BinarySearchTree
from src.counters import Counters
from src.hashtable import HashTableChained
from src.linked_list import SinglyLinkedList
from src.search_engine import (
    BASE_VOCABULARY,
    DEFAULT_MAX_DISTANCE,
    PRECEDENCE,
    DuplicateDocumentError,
    EmptyQueryError,
    InvertedIndex,
    MalformedQueryError,
    Occurrence,
    ScoredDocument,
    SearchEngine,
    SearchEngineError,
    TermNotFoundError,
    UnbalancedParenthesesError,
    bench_edit_distance,
    bench_index_lookup,
    bench_indexing,
    bench_intersection,
    bench_suggestion,
    build_vocabulary,
    difference_sorted,
    edit_distance,
    edit_distance_naive,
    evaluate_postfix,
    generate_corpus,
    intersect_naive,
    intersect_sorted,
    normalize,
    performance_report,
    query_terms,
    rank_documents,
    run_query,
    score_document,
    suggest_terms,
    to_postfix,
    tokenize,
    tokenize_query,
    union_sorted,
)

DOCS = [
    "Algoritmos de ordenação: bubble sort, insertion sort e selection sort.",
    "Estruturas de dados: lista encadeada, pilha, fila e árvore binária.",
    "Análise de algoritmos com notação Big O e contagem de comparações.",
    "A árvore binária de busca degenera quando a entrada já está ordenada.",
    "Tabela hash com encadeamento resolve colisões usando lista encadeada.",
]


@pytest.fixture()
def motor() -> SearchEngine:
    return SearchEngine(DOCS)


# ----------------------------------------------------------------------
# Tokenização
# ----------------------------------------------------------------------
def test_normalize_tira_acento_e_caixa() -> None:
    assert normalize("Análise") == "analise"
    assert normalize("ÁRVORE") == "arvore"
    assert normalize("coração") == "coracao"


def test_tokenize_quebra_em_nao_alfanumerico() -> None:
    assert tokenize("Bubble sort, insertion-sort!") == [
        "bubble", "sort", "insertion", "sort"
    ]
    assert tokenize("") == []
    assert tokenize("   ...   ") == []
    assert tokenize("a1 b2") == ["a1", "b2"]


def test_tokenize_preserva_a_ordem_que_vira_posicao() -> None:
    termos = tokenize("um dois tres dois")
    assert termos == ["um", "dois", "tres", "dois"]
    assert termos.index("um") == 0


# ----------------------------------------------------------------------
# Índice invertido — estrutura e invariantes
# ----------------------------------------------------------------------
def test_indice_usa_as_estruturas_dos_exercicios_anteriores(
    motor: SearchEngine,
) -> None:
    """A exigência de integração do enunciado, verificada e não presumida."""
    index = motor.index
    assert isinstance(index._postings, HashTableChained)  # ex 4
    assert isinstance(index.terms_tree, BinarySearchTree)  # ex 2
    assert isinstance(index.postings("arvore"), SinglyLinkedList)  # ex 10


def test_indice_conta_documentos_termos_e_tokens(motor: SearchEngine) -> None:
    assert motor.index.document_count == 5
    assert motor.index.vocabulary_size > 25
    assert motor.index.token_count > 40
    motor.index.check_invariants()


def test_ocorrencia_guarda_doc_id_e_posicoes() -> None:
    index = InvertedIndex()
    index.add_document(0, "sort sort busca sort")
    ocorrencias = index.postings("sort").to_list()
    assert len(ocorrencias) == 1
    assert ocorrencias[0].doc_id == 0
    assert ocorrencias[0].positions == [0, 1, 3]
    assert ocorrencias[0].frequency == 3
    assert ocorrencias[0].first_position == 0


def test_um_no_por_documento_na_lista_de_ocorrencias() -> None:
    """Termo repetido no mesmo documento vira UMA ocorrência com n posições."""
    index = InvertedIndex()
    index.add_document(0, "a a a a")
    index.add_document(1, "a")
    assert len(index.postings("a")) == 2
    assert index.document_frequency("a") == 2


def test_listas_de_ocorrencias_nascem_ordenadas(motor: SearchEngine) -> None:
    """Decisão de projeto 1 — é o que habilita a otimização 1."""
    for termo in motor.index.terms_in_order():
        ids = motor.index.doc_ids_for(termo)
        assert ids == sorted(ids)
        assert len(ids) == len(set(ids))


def test_documentos_fora_de_ordem_sao_rejeitados() -> None:
    index = InvertedIndex()
    index.add_document(5, "a")
    with pytest.raises(DuplicateDocumentError, match="ordem crescente"):
        index.add_document(3, "b")
    with pytest.raises(DuplicateDocumentError):
        index.add_document(5, "c")
    with pytest.raises(SearchEngineError, match="doc_id deve ser int"):
        index.add_document("6", "d")  # type: ignore[arg-type]


def test_listagem_em_ordem_nao_repete_termos(motor: SearchEngine) -> None:
    """A BST guarda um nó por termo; repetir inseriria multiplicidade."""
    termos = motor.index.terms_in_order()
    assert termos == sorted(termos)
    assert len(termos) == len(set(termos)) == motor.index.vocabulary_size


def test_termos_com_prefixo(motor: SearchEngine) -> None:
    assert motor.index.terms_with_prefix("enc") == ["encadeada", "encadeamento"]
    assert motor.index.terms_with_prefix("zzz") == []


def test_busca_de_termo_pelos_dois_caminhos(motor: SearchEngine) -> None:
    assert motor.index.contains_term("árvore") is True  # acento é normalizado
    assert motor.index.search_term_in_tree("arvore") == 2
    with pytest.raises(TermNotFoundError, match="não está no vocabulário"):
        motor.index.search_term_in_tree("inexistente")


def test_termo_ausente_devolve_lista_vazia(motor: SearchEngine) -> None:
    assert len(motor.index.postings("inexistente")) == 0
    assert motor.index.doc_ids_for("inexistente") == []
    assert motor.index.document_frequency("inexistente") == 0


def test_check_invariants_detecta_lista_fora_de_ordem(
    motor: SearchEngine,
) -> None:
    lista = motor.index.postings("arvore")
    lista.insert_last(Occurrence(0, [0]))  # doc 0 depois do doc 3
    with pytest.raises(SearchEngineError, match="fora de ordem"):
        motor.index.check_invariants()


def test_check_invariants_detecta_divergencia_entre_hash_e_bst(
    motor: SearchEngine,
) -> None:
    motor.index.terms_tree.insert("fantasma", None)
    with pytest.raises(SearchEngineError, match="discordam"):
        motor.index.check_invariants()


def test_check_invariants_detecta_termo_com_multiplicidade(
    motor: SearchEngine,
) -> None:
    motor.index.terms_tree.insert("arvore", None)  # incrementa multiplicidade
    with pytest.raises(SearchEngineError, match="multiplicidade"):
        motor.index.check_invariants()


# ----------------------------------------------------------------------
# Operações sobre listas ordenadas (otimização 1)
# ----------------------------------------------------------------------
def test_intersecao_uniao_e_diferenca() -> None:
    a = [1, 3, 5, 7, 9]
    b = [3, 4, 5, 6]
    assert intersect_sorted(a, b) == [3, 5]
    assert union_sorted(a, b) == [1, 3, 4, 5, 6, 7, 9]
    assert difference_sorted(a, b) == [1, 7, 9]
    assert difference_sorted(b, a) == [4, 6]


def test_operacoes_em_listas_vazias() -> None:
    assert intersect_sorted([], [1, 2]) == []
    assert union_sorted([], [1, 2]) == [1, 2]
    assert difference_sorted([1, 2], []) == [1, 2]
    assert difference_sorted([], [1, 2]) == []


def test_as_duas_intersecoes_dao_o_mesmo_resultado() -> None:
    """A otimização 1 é exata: só o custo muda."""
    import random

    rng = random.Random(42)
    for _ in range(100):
        a = sorted(rng.sample(range(200), rng.randrange(0, 40)))
        b = sorted(rng.sample(range(200), rng.randrange(0, 40)))
        assert intersect_sorted(a, b) == intersect_naive(a, b)


def test_intersecao_ordenada_e_linear_e_a_ingenua_quadratica() -> None:
    n = 400
    a = list(range(0, 2 * n, 2))
    b = list(range(0, 2 * n, 3))
    rapida = Counters()
    intersect_sorted(a, b, rapida)
    lenta = Counters()
    intersect_naive(a, b, lenta)
    assert rapida.comparisons < 3 * len(a) + len(b)  # Θ(m+n)
    assert lenta.comparisons > 10 * rapida.comparisons


# ----------------------------------------------------------------------
# Parsing da consulta (shunting-yard com Stack)
# ----------------------------------------------------------------------
def test_tokenize_query_separa_operadores_e_parenteses() -> None:
    assert tokenize_query("arvore AND binaria") == ["arvore", "AND", "binaria"]
    assert tokenize_query("(a OR b) and NOT c") == [
        "(", "a", "OR", "b", ")", "AND", "NOT", "c"
    ]
    assert tokenize_query("Árvore") == ["arvore"]


def test_tokenize_query_rejeita_vazio() -> None:
    for vazio in ("", "   ", "\n"):
        with pytest.raises(EmptyQueryError):
            tokenize_query(vazio)
    with pytest.raises(EmptyQueryError, match="deve ser str"):
        tokenize_query(42)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("consulta", "esperado"),
    [
        ("a AND b", ["a", "b", "AND"]),
        ("a OR b AND c", ["a", "b", "c", "AND", "OR"]),  # AND liga mais forte
        ("(a OR b) AND c", ["a", "b", "OR", "c", "AND"]),
        ("NOT a AND b", ["a", "NOT", "b", "AND"]),  # NOT liga mais forte
        ("a AND NOT b", ["a", "b", "NOT", "AND"]),
        ("NOT NOT a", ["a", "NOT", "NOT"]),  # unário é associativo à direita
        ("((a))", ["a"]),
    ],
)
def test_shunting_yard_respeita_a_precedencia(
    consulta: str, esperado: list[str]
) -> None:
    assert to_postfix(tokenize_query(consulta)) == esperado


def test_parenteses_desbalanceados_levantam() -> None:
    with pytest.raises(UnbalancedParenthesesError, match=r"'\(' sem"):
        to_postfix(tokenize_query("(a AND b"))
    with pytest.raises(UnbalancedParenthesesError, match=r"'\)' sem"):
        to_postfix(tokenize_query("a AND b)"))


def test_precedencia_declarada_bate_com_o_esperado() -> None:
    assert PRECEDENCE["NOT"] > PRECEDENCE["AND"] > PRECEDENCE["OR"]


# ----------------------------------------------------------------------
# Avaliação da consulta
# ----------------------------------------------------------------------
def test_consulta_de_termo_unico(motor: SearchEngine) -> None:
    assert run_query("arvore", motor.index) == [1, 3]
    assert run_query("lista", motor.index) == [1, 4]


def test_consulta_com_and_or_e_not(motor: SearchEngine) -> None:
    assert run_query("arvore AND binaria", motor.index) == [1, 3]
    assert run_query("pilha OR tabela", motor.index) == [1, 4]
    assert run_query("algoritmos AND NOT ordenacao", motor.index) == [2]
    assert run_query("(lista OR tabela) AND encadeada", motor.index) == [1, 4]


def test_consulta_devolve_doc_ids_ordenados(motor: SearchEngine) -> None:
    for consulta in ("de", "lista OR arvore", "NOT pilha"):
        resultado = run_query(consulta, motor.index)
        assert resultado == sorted(resultado)


def test_not_usa_o_universo_de_documentos(motor: SearchEngine) -> None:
    todos = motor.index.document_ids()
    com_pilha = run_query("pilha", motor.index)
    assert run_query("NOT pilha", motor.index) == [
        doc for doc in todos if doc not in com_pilha
    ]
    assert run_query("NOT NOT pilha", motor.index) == com_pilha


def test_termo_ausente_na_consulta_nao_quebra(motor: SearchEngine) -> None:
    assert run_query("inexistente", motor.index) == []
    assert run_query("inexistente AND arvore", motor.index) == []
    assert run_query("inexistente OR arvore", motor.index) == [1, 3]


def test_consulta_malformada_levanta(motor: SearchEngine) -> None:
    with pytest.raises(MalformedQueryError, match="2 operandos"):
        run_query("AND arvore", motor.index)
    with pytest.raises(MalformedQueryError, match="sem operando"):
        run_query("NOT", motor.index)
    with pytest.raises(MalformedQueryError, match="faltou um operador"):
        run_query("arvore binaria", motor.index)
    with pytest.raises(EmptyQueryError):
        evaluate_postfix([], motor.index)


# ----------------------------------------------------------------------
# Ranqueamento
# ----------------------------------------------------------------------
def test_score_usa_frequencia_e_posicao() -> None:
    index = InvertedIndex()
    index.add_document(0, "alvo lixo lixo lixo")  # 1x, posição 0
    index.add_document(1, "lixo lixo lixo alvo")  # 1x, posição 3
    index.add_document(2, "lixo alvo alvo lixo")  # 2x, posição 1
    cedo = score_document(0, ["alvo"], index)
    tarde = score_document(1, ["alvo"], index)
    frequente = score_document(2, ["alvo"], index)

    assert cedo.score == pytest.approx(1 * 2.0)  # 1 + 1/(1+0)
    assert tarde.score == pytest.approx(1 * 1.25)  # 1 + 1/(1+3)
    assert frequente.score == pytest.approx(2 * 1.5)  # 1 + 1/(1+1)
    assert cedo.score > tarde.score  # posição desempata
    assert frequente.score > cedo.score  # frequência domina


def test_ranqueamento_ordena_por_score_decrescente(motor: SearchEngine) -> None:
    doc_ids = run_query("arvore OR lista", motor.index)
    ranked = rank_documents(doc_ids, ["arvore", "lista"], motor.index)
    scores = [documento.score for documento in ranked]
    assert scores == sorted(scores, reverse=True)
    assert {documento.doc_id for documento in ranked} == set(doc_ids)


def test_ranqueamento_desempata_por_doc_id_e_e_deterministico() -> None:
    index = InvertedIndex()
    for doc_id in range(4):
        index.add_document(doc_id, "igual")
    ranked = rank_documents([0, 1, 2, 3], ["igual"], index)
    assert [documento.doc_id for documento in ranked] == [0, 1, 2, 3]
    outra = rank_documents([3, 2, 1, 0], ["igual"], index)
    assert [d.doc_id for d in outra] == [0, 1, 2, 3]


def test_ranqueamento_usa_o_quicksort_instrumentado(
    motor: SearchEngine,
) -> None:
    counters = Counters()
    doc_ids = run_query("de", motor.index)
    rank_documents(doc_ids, ["de"], motor.index, counters)
    assert counters.calls > 0  # o quicksort registrou chamadas recursivas
    assert counters.comparisons > 0


def test_query_terms_descarta_operadores() -> None:
    tokens = tokenize_query("(a OR b) AND NOT c")
    assert query_terms(tokens) == ["a", "b", "c"]


def test_ranqueamento_de_resultado_vazio(motor: SearchEngine) -> None:
    assert rank_documents([], ["arvore"], motor.index) == []


# ----------------------------------------------------------------------
# Distância de edição e sugestão
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("a", "b", "esperado"),
    [
        ("", "", 0),
        ("abc", "", 3),
        ("", "abc", 3),
        ("abc", "abc", 0),
        ("arvore", "arvre", 1),
        ("kitten", "sitting", 3),
        ("saturday", "sunday", 3),
    ],
)
def test_distancia_de_edicao(a: str, b: str, esperado: int) -> None:
    assert edit_distance(a, b) == esperado
    assert edit_distance_naive(a, b) == esperado


def test_distancia_de_edicao_e_simetrica() -> None:
    import random

    rng = random.Random(42)
    letras = "abcde"
    for _ in range(60):
        a = "".join(rng.choice(letras) for _ in range(rng.randrange(0, 7)))
        b = "".join(rng.choice(letras) for _ in range(rng.randrange(0, 7)))
        assert edit_distance(a, b) == edit_distance(b, a)


def test_memo_reduz_drasticamente_as_chamadas() -> None:
    com_memo = Counters()
    edit_distance("complexidade", "complexidad", com_memo)
    sem_memo = Counters()
    edit_distance_naive("complexidade", "complexidad", sem_memo)
    assert sem_memo.calls > 1_000 * com_memo.calls


def test_sugestao_encontra_o_termo_mais_proximo(motor: SearchEngine) -> None:
    sugestoes = suggest_terms("arvre", motor.index)
    assert sugestoes[0].term == "arvore"
    assert sugestoes[0].distance == 1
    assert all(s.distance <= DEFAULT_MAX_DISTANCE for s in sugestoes)


def test_sugestao_nao_sugere_o_proprio_termo(motor: SearchEngine) -> None:
    for sugestao in suggest_terms("arvore", motor.index):
        assert sugestao.term != "arvore"
        assert sugestao.distance > 0


def test_poda_por_comprimento_e_exata(motor: SearchEngine) -> None:
    """A otimização 2 não pode descartar candidato válido."""
    for termo in ("arvre", "list", "tabla", "algoritmoo", "xyz"):
        com = suggest_terms(termo, motor.index, prune_by_length=True)
        sem = suggest_terms(termo, motor.index, prune_by_length=False)
        assert com == sem


def test_poda_reduz_o_custo(motor: SearchEngine) -> None:
    com = Counters()
    suggest_terms("algoritmoo", motor.index, com, prune_by_length=True)
    sem = Counters()
    suggest_terms("algoritmoo", motor.index, sem, prune_by_length=False)
    assert com.calls < sem.calls


def test_sugestao_valida_parametros(motor: SearchEngine) -> None:
    with pytest.raises(ValueError, match="max_distance"):
        suggest_terms("a", motor.index, max_distance=-1)
    with pytest.raises(ValueError, match="limit"):
        suggest_terms("a", motor.index, limit=0)


# ----------------------------------------------------------------------
# Relatório BFS (Queue do ex 5)
# ----------------------------------------------------------------------
def test_relatorio_bfs_descreve_a_arvore_de_termos(motor: SearchEngine) -> None:
    relatorio = motor.index.bfs_report()
    assert relatorio["termos"] == motor.index.vocabulary_size
    assert relatorio["niveis"] == relatorio["altura"]
    assert sum(relatorio["termos_por_nivel"]) == relatorio["termos"]
    assert relatorio["termos_por_nivel"][0] == 1  # a raiz
    assert relatorio["balanceamento"] >= 1.0
    assert relatorio["raiz"] in motor.index.terms_in_order()
    assert relatorio["largura_maxima"] == max(relatorio["termos_por_nivel"])


def test_relatorio_bfs_de_indice_vazio() -> None:
    relatorio = InvertedIndex().bfs_report()
    assert relatorio["termos"] == 0
    assert relatorio["niveis"] == 0
    assert relatorio["raiz"] is None


# ----------------------------------------------------------------------
# Fachada
# ----------------------------------------------------------------------
def test_search_devolve_a_cadeia_completa(motor: SearchEngine) -> None:
    resultado = motor.search("arvore AND binaria")
    assert resultado.query == "arvore AND binaria"
    assert resultado.postfix == ["arvore", "binaria", "AND"]
    assert resultado.doc_ids == [1, 3]
    assert len(resultado.ranked) == 2
    assert resultado.suggestions == {}
    assert len(resultado.top(1)) == 1
    assert resultado.elapsed_s >= 0


def test_search_sugere_so_para_termos_ausentes(motor: SearchEngine) -> None:
    resultado = motor.search("arvre OR arvore")
    assert "arvre" in resultado.suggestions
    assert "arvore" not in resultado.suggestions  # existe: não gasta Θ(V·L²)
    assert resultado.suggestions["arvre"][0].term == "arvore"


def test_motor_vazio_responde_sem_quebrar() -> None:
    vazio = SearchEngine()
    resultado = vazio.search("qualquer")
    assert resultado.doc_ids == []
    assert resultado.ranked == []
    assert resultado.suggestions == {"qualquer": []}


# ----------------------------------------------------------------------
# Corpus sintético
# ----------------------------------------------------------------------
def test_vocabulario_sintetico_e_reprodutivel_e_do_tamanho_pedido() -> None:
    a = build_vocabulary(300)
    b = build_vocabulary(300)
    assert a == b
    assert len(a) == len(set(a)) == 300
    assert a[: len(BASE_VOCABULARY)] == list(BASE_VOCABULARY)
    comprimentos = {len(palavra) for palavra in a}
    assert len(comprimentos) > 5  # variedade de comprimento: a poda depende dela
    with pytest.raises(ValueError, match="size deve ser >= 1"):
        build_vocabulary(0)


def test_corpus_e_reprodutivel_e_valida_parametros() -> None:
    assert generate_corpus(10, seed=1) == generate_corpus(10, seed=1)
    assert generate_corpus(10, seed=1) != generate_corpus(10, seed=2)
    assert len(generate_corpus(7)) == 7
    with pytest.raises(ValueError, match="documents deve ser >= 1"):
        generate_corpus(0)
    with pytest.raises(ValueError, match="words_per_document"):
        generate_corpus(5, words_per_document=0)


# ----------------------------------------------------------------------
# Relatório técnico e experimentos
# ----------------------------------------------------------------------
def test_relatorio_tecnico_separa_as_etapas(motor: SearchEngine) -> None:
    relatorio = performance_report(
        motor, ["arvore AND binaria", "lista OR pilha"], missing_term="arvre"
    )
    assert relatorio["indice"]["documentos"] == 5
    assert relatorio["bst_de_termos"]["termos"] == motor.index.vocabulary_size
    assert len(relatorio["consultas"]) == 2
    for consulta in relatorio["consultas"]:
        assert consulta["parsing_comparacoes"] >= 0
        assert consulta["avaliacao_comparacoes"] >= 0
        assert consulta["resultados"] >= 0
    sugestao = relatorio["sugestao"]
    assert sugestao["reducao_de_chamadas"] >= 1.0
    assert sugestao["sugestoes"][0][0] == "arvore"


def test_bench_indexing_mostra_custo_linear_nos_tokens() -> None:
    rows = bench_indexing([50, 200, 800])
    assert len(rows) == 3
    razoes = [row["comparacoes/tokens"] for row in rows]
    assert all(1.0 < razao < 6.0 for razao in razoes)
    # com o vocabulário saturando, a razão converge
    assert razoes[-1] < razoes[0]
    for row in rows:
        assert row["altura_bst"] >= row["altura_ideal"]
        assert row["balanceamento"] < 4.0  # a BST de termos não degenera


def test_bench_intersection_mede_a_otimizacao_1() -> None:
    rows = bench_intersection([100, 400, 1_600])
    assert len(rows) == 3
    ordenadas = [row["ordenada/n"] for row in rows]
    ingenuas = [row["ingenua/n2"] for row in rows]
    # ordenada: razão contra n estável => Θ(m+n)
    assert max(ordenadas) / min(ordenadas) < 1.1
    # ingênua: razão contra n² estável => Θ(m·n)
    assert max(ingenuas) / min(ingenuas) < 1.2
    # e a vantagem cresce linearmente com n
    razoes = [row["razao"] for row in rows]
    assert razoes == sorted(razoes)
    assert razoes[-1] > 10 * razoes[0]


def test_bench_suggestion_mede_a_otimizacao_2() -> None:
    rows = bench_suggestion([200, 800])
    assert len(rows) == 2
    for row in rows:
        assert row["reducao_de_chamadas"] > 1.5
        assert row["chamadas_com_poda"] < row["chamadas_sem_poda"]
        assert row["sugestoes"]  # achou alguma sugestão


def test_bench_edit_distance_mostra_a_mudanca_de_classe() -> None:
    rows = bench_edit_distance()
    for row in rows:
        assert row["chamadas_com_memo"] < row["chamadas_sem_memo"]
        assert row["memo/estados"] < 2.0  # Θ(|a|·|b|)
    assert max(row["reducao_de_chamadas"] for row in rows) > 1_000


def test_bench_index_lookup_separa_hash_de_bst() -> None:
    rows = bench_index_lookup([200, 800])
    assert len(rows) == 2
    for row in rows:
        assert row["hash_comparacoes_por_busca"] < 2.0  # O(1 + α)
        assert row["bst_por_busca/log2V"] < 2.5  # O(log V)
        assert row["bst/hash"] > 3
        assert row["listagem_ordenada_ok"] is True
    # a hashtable não cresce; a BST cresce com log2 V
    assert rows[-1]["bst_comparacoes_por_busca"] > rows[0]["bst_comparacoes_por_busca"]


@pytest.mark.parametrize(
    "chamada",
    [
        lambda: bench_indexing([0]),
        lambda: bench_intersection([0]),
        lambda: bench_suggestion([0]),
        lambda: bench_index_lookup([0]),
    ],
)
def test_benchs_validam_tamanho(chamada: object) -> None:
    with pytest.raises(ValueError, match="tamanho deve ser positivo"):
        chamada()  # type: ignore[operator]

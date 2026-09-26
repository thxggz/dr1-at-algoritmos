"""Testes da árvore binária de busca (definida no ex 2, usada em 2, 3, 6, 8, 12).

Dois pontos merecem destaque:

* **`delete` nos três casos** — folha, um filho e dois filhos por sucessor
  in-order. Nenhum enunciado pede remoção; o item 4.4 da rubrica pede. Cada
  caso é verificado estruturalmente (quem aponta para quem), não só pelo
  resultado da travessia.
* **`check_invariants` com limites herdados** — uma árvore pode ter todo nó
  coerente com seus filhos diretos e ainda violar a ordenação global. Esse é o
  bug clássico da estrutura, e há um teste que o constrói de propósito.
"""

from __future__ import annotations

import math
import random

import pytest

from src.counters import Counters
from src.bst import (
    BSTInvariantError,
    BSTKeyNotFoundError,
    BSTNode,
    BinarySearchTree,
    EmptyTreeError,
    bench_degenerate,
    bench_insert_shapes,
    bench_k_smallest,
    bst_sort,
)


def arvore_exemplo() -> BinarySearchTree[int, None]:
    """Árvore perfeita de 7 nós, altura 3:

    ::

            50
          /    \\
        30      70
       /  \\    /  \\
     20   40  60   80
    """
    return BinarySearchTree([50, 30, 70, 20, 40, 60, 80])


# ----------------------------------------------------------------------
# Nó
# ----------------------------------------------------------------------
def test_no_guarda_chave_valor_e_multiplicidade() -> None:
    node: BSTNode[int, str] = BSTNode(5, "cinco")
    assert (node.key, node.value, node.multiplicity) == (5, "cinco", 1)
    assert node.is_leaf() is True
    assert node.child_count() == 0
    node.left = BSTNode(3)
    assert node.is_leaf() is False
    assert node.child_count() == 1


def test_no_usa_slots_e_repr_curto() -> None:
    node: BSTNode[int, None] = BSTNode(5)
    with pytest.raises(AttributeError):
        node.cor = "vermelho"  # type: ignore[attr-defined]
    assert repr(node) == "BSTNode(5)"
    node.multiplicity = 3
    assert repr(node) == "BSTNode(5x3)"


# ----------------------------------------------------------------------
# Estado inicial
# ----------------------------------------------------------------------
def test_arvore_vazia() -> None:
    tree: BinarySearchTree[int, None] = BinarySearchTree()
    assert len(tree) == 0
    assert tree.node_count == 0
    assert tree.is_empty() is True
    assert tree.root is None
    assert list(tree) == []
    assert tree.height() == 0
    assert tree.min_depth() == 0
    assert tree.nodes_per_level() == []
    assert str(tree) == "[]"
    tree.check_invariants()


def test_operacoes_que_exigem_no_levantam_em_arvore_vazia() -> None:
    tree: BinarySearchTree[int, None] = BinarySearchTree()
    with pytest.raises(EmptyTreeError, match="min_key"):
        tree.min_key()
    with pytest.raises(EmptyTreeError, match="max_key"):
        tree.max_key()
    with pytest.raises(BSTKeyNotFoundError):
        tree.search(1)
    with pytest.raises(BSTKeyNotFoundError):
        tree.delete(1)


# ----------------------------------------------------------------------
# Inserção
# ----------------------------------------------------------------------
def test_insercao_monta_a_arvore_esperada() -> None:
    tree = arvore_exemplo()
    assert tree.node_count == 7
    assert len(tree) == 7
    assert tree.height() == 3
    assert list(tree) == [20, 30, 40, 50, 60, 70, 80]
    assert tree.root is not None
    assert tree.root.key == 50
    assert tree.root.left is not None and tree.root.left.key == 30
    assert tree.root.right is not None and tree.root.right.key == 70
    tree.check_invariants()


def test_insercao_devolve_se_criou_no() -> None:
    tree: BinarySearchTree[int, str] = BinarySearchTree()
    assert tree.insert(5, "a") is True
    assert tree.insert(3, "b") is True
    assert tree.insert(5, "c") is False  # chave repetida


def test_chave_repetida_vira_multiplicidade_e_nao_no_novo() -> None:
    """Decisão de projeto 2: a mesma estrutura serve de mapa e multiconjunto."""
    tree: BinarySearchTree[int, str] = BinarySearchTree()
    for _ in range(4):
        tree.insert(7, "sete")
    assert tree.node_count == 1
    assert len(tree) == 4
    assert list(tree.in_order_keys()) == [7, 7, 7, 7]
    assert list(tree) == [7]  # __iter__ dá as chaves distintas
    tree.check_invariants()


def test_chave_repetida_atualiza_o_valor() -> None:
    tree: BinarySearchTree[str, int] = BinarySearchTree()
    tree.insert("a", 1)
    tree.insert("a", 2)
    assert tree.search("a") == 2
    assert tree.node_count == 1


def test_ordem_de_insercao_muda_a_forma_e_nao_o_conteudo() -> None:
    aleatoria = BinarySearchTree([50, 30, 70, 20, 40, 60, 80])
    ordenada = BinarySearchTree(range(7))
    assert list(aleatoria) == [20, 30, 40, 50, 60, 70, 80]
    assert list(ordenada) == list(range(7))
    assert aleatoria.height() == 3  # equilibrada
    assert ordenada.height() == 7  # espinha: virou lista encadeada
    ordenada.check_invariants()


def test_insercao_conta_uma_comparacao_por_no_visitado() -> None:
    tree = arvore_exemplo()
    counters = Counters()
    tree.insert(45, None, counters)  # desce 50 -> 30 -> 40 -> insere
    assert counters.comparisons == 3
    assert counters.hops == 2
    assert counters.copies == 1


def test_construcao_ordenada_custa_exatamente_n_vezes_n_menos_1_sobre_2() -> None:
    for n in (10, 50, 200):
        counters = Counters()
        tree: BinarySearchTree[int, None] = BinarySearchTree()
        for key in range(n):
            tree.insert(key, None, counters)
        assert counters.comparisons == n * (n - 1) // 2
        assert tree.height() == n


def test_insert_protocolo_resultado_counters() -> None:
    tree: BinarySearchTree[int, None] = BinarySearchTree()
    criou, counters = tree.insert_counted(1)
    assert criou is True and counters.copies == 1


# ----------------------------------------------------------------------
# Busca
# ----------------------------------------------------------------------
def test_busca_encontra_todas_as_chaves() -> None:
    tree: BinarySearchTree[int, str] = BinarySearchTree()
    for key in (50, 30, 70, 20, 40):
        tree.insert(key, f"v{key}")
    for key in (50, 30, 70, 20, 40):
        assert tree.search(key) == f"v{key}"
        assert key in tree
        assert tree.contains(key) is True


def test_busca_de_chave_ausente_levanta_com_contexto() -> None:
    tree = arvore_exemplo()
    with pytest.raises(BSTKeyNotFoundError, match="não está na árvore"):
        tree.search(99)
    assert 99 not in tree
    assert tree.get_or(99, "padrão") == "padrão"


def test_valor_none_e_distinguivel_de_chave_ausente() -> None:
    tree: BinarySearchTree[int, None] = BinarySearchTree()
    tree.insert(1, None)
    assert tree.search(1) is None
    assert 1 in tree
    with pytest.raises(BSTKeyNotFoundError):
        tree.search(2)


def test_min_e_max() -> None:
    tree = arvore_exemplo()
    assert tree.min_key() == 20
    assert tree.max_key() == 80


def test_busca_na_espinha_custa_n_comparacoes() -> None:
    """Numa árvore degenerada a busca é uma busca em lista encadeada."""
    n = 300
    tree: BinarySearchTree[int, None] = BinarySearchTree(range(n))
    counters = Counters()
    tree.search(n - 1, counters)
    assert counters.comparisons == n
    assert counters.hops == n - 1


def test_busca_em_arvore_equilibrada_custa_log() -> None:
    tree = arvore_exemplo()
    counters = Counters()
    tree.search(80, counters)
    assert counters.comparisons == 3 == tree.height()


# ----------------------------------------------------------------------
# Remoção — os três casos (item 4.4 da rubrica)
# ----------------------------------------------------------------------
def test_delete_caso_folha() -> None:
    tree = arvore_exemplo()
    assert tree.delete(20) is None
    assert tree.node_count == 6
    assert list(tree) == [30, 40, 50, 60, 70, 80]
    esquerda = tree.root.left  # type: ignore[union-attr]
    assert esquerda is not None and esquerda.key == 30
    assert esquerda.left is None  # o pai passou a apontar para None
    tree.check_invariants()


def test_delete_caso_um_filho() -> None:
    tree = arvore_exemplo()
    tree.delete(20)  # deixa 30 com um único filho (40)
    tree.delete(30)
    assert tree.node_count == 5
    assert list(tree) == [40, 50, 60, 70, 80]
    esquerda = tree.root.left  # type: ignore[union-attr]
    # o avô passou a apontar direto para o neto
    assert esquerda is not None and esquerda.key == 40
    tree.check_invariants()


def test_delete_caso_dois_filhos_usa_o_sucessor_in_order() -> None:
    tree = arvore_exemplo()
    tree.delete(30)  # 30 tem dois filhos: 20 e 40; sucessor in-order é 40
    assert tree.node_count == 6
    assert list(tree) == [20, 40, 50, 60, 70, 80]
    esquerda = tree.root.left  # type: ignore[union-attr]
    assert esquerda is not None and esquerda.key == 40
    assert esquerda.left is not None and esquerda.left.key == 20
    assert esquerda.right is None  # o sucessor saiu do lugar antigo
    tree.check_invariants()


def test_delete_da_raiz_com_dois_filhos() -> None:
    tree = arvore_exemplo()
    tree.delete(50)  # sucessor in-order de 50 é 60
    assert tree.root is not None and tree.root.key == 60
    assert list(tree) == [20, 30, 40, 60, 70, 80]
    direita = tree.root.right
    assert direita is not None and direita.key == 70
    assert direita.left is None  # 60 saiu de onde estava
    tree.check_invariants()


def test_delete_da_raiz_sozinha() -> None:
    tree: BinarySearchTree[int, None] = BinarySearchTree([1])
    tree.delete(1)
    assert tree.is_empty() is True
    assert tree.root is None
    tree.check_invariants()


def test_delete_decrementa_multiplicidade_antes_de_remover_o_no() -> None:
    tree: BinarySearchTree[int, None] = BinarySearchTree([5, 5, 5])
    assert tree.node_count == 1 and len(tree) == 3
    tree.delete(5)
    assert tree.node_count == 1 and len(tree) == 2
    tree.delete(5)
    tree.delete(5)
    assert tree.is_empty() is True
    tree.check_invariants()


def test_delete_com_all_copies_remove_o_no_inteiro() -> None:
    tree: BinarySearchTree[int, None] = BinarySearchTree([5, 5, 5, 3, 9])
    tree.delete(5, all_copies=True)
    assert tree.node_count == 2
    assert len(tree) == 2
    assert list(tree.in_order_keys()) == [3, 9]
    tree.check_invariants()


def test_delete_de_chave_ausente_levanta() -> None:
    tree = arvore_exemplo()
    with pytest.raises(BSTKeyNotFoundError, match="delete"):
        tree.delete(99)
    assert tree.node_count == 7  # nada foi alterado


def test_delete_ate_esvaziar_em_ordem_aleatoria() -> None:
    rng = random.Random(42)
    keys = list(range(200))
    rng.shuffle(keys)
    tree: BinarySearchTree[int, None] = BinarySearchTree(keys)
    rng.shuffle(keys)
    for index, key in enumerate(keys):
        tree.delete(key)
        if index % 25 == 0:
            tree.check_invariants()
            assert list(tree) == sorted(keys[index + 1 :])
    assert tree.is_empty() is True
    tree.check_invariants()


def test_delete_protocolo_resultado_counters() -> None:
    tree: BinarySearchTree[int, str] = BinarySearchTree()
    tree.insert(1, "um")
    value, counters = tree.delete_counted(1)
    assert value == "um"
    assert counters.comparisons == 1


# ----------------------------------------------------------------------
# Travessias
# ----------------------------------------------------------------------
def test_in_order_pre_order_e_level_order() -> None:
    tree = arvore_exemplo()
    assert [node.key for node in tree.in_order_nodes()] == [
        20, 30, 40, 50, 60, 70, 80
    ]
    assert [node.key for node in tree.pre_order_nodes()] == [
        50, 30, 20, 40, 70, 60, 80
    ]
    assert [(level, node.key) for level, node in tree.level_order_nodes()] == [
        (0, 50), (1, 30), (1, 70), (2, 20), (2, 40), (2, 60), (2, 80)
    ]


def test_in_order_items_traz_os_valores() -> None:
    tree: BinarySearchTree[int, str] = BinarySearchTree()
    for key in (2, 1, 3):
        tree.insert(key, f"v{key}")
    assert list(tree.in_order_items()) == [(1, "v1"), (2, "v2"), (3, "v3")]


def test_travessias_funcionam_em_arvore_degenerada() -> None:
    """Decisão de projeto 1: iterativas, senão o experimento do ex 3 não roda."""
    n = 3_000  # bem acima do limite de recursão do Python
    tree: BinarySearchTree[int, None] = BinarySearchTree(range(n))
    assert tree.height() == n
    assert list(tree) == list(range(n))
    assert len(list(tree.pre_order_nodes())) == n
    assert [level for level, _ in tree.level_order_nodes()] == list(range(n))
    assert tree.min_depth() == n
    tree.check_invariants()


def test_level_order_de_arvore_vazia_nao_quebra() -> None:
    tree: BinarySearchTree[int, None] = BinarySearchTree()
    assert list(tree.level_order_nodes()) == []


# ----------------------------------------------------------------------
# k_smallest — travessia controlada (ex 3)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("k", [0, 1, 3, 7, 20])
def test_k_smallest_devolve_as_k_menores(k: int) -> None:
    tree = arvore_exemplo()
    esperado = [20, 30, 40, 50, 60, 70, 80][:k]
    assert tree.k_smallest(k) == esperado


def test_k_smallest_conta_multiplicidade() -> None:
    tree: BinarySearchTree[int, None] = BinarySearchTree([5, 1, 5, 9, 5])
    assert tree.k_smallest(4) == [1, 5, 5, 5]


def test_k_smallest_para_cedo_e_nao_percorre_tudo() -> None:
    """A diferença é assintótica: O(h + k) contra O(n)."""
    n = 5_000
    keys = list(range(n))
    random.Random(42).shuffle(keys)
    tree: BinarySearchTree[int, None] = BinarySearchTree(keys)

    curto = Counters()
    tree.k_smallest(5, curto)
    longo = Counters()
    list(tree.in_order_keys(longo))

    assert curto.hops < longo.hops / 100
    assert curto.hops <= tree.height() + 5


def test_k_smallest_valida_k() -> None:
    tree = arvore_exemplo()
    with pytest.raises(ValueError, match="k deve ser >= 0"):
        tree.k_smallest(-1)
    with pytest.raises(ValueError, match="k deve ser int"):
        tree.k_smallest(1.5)  # type: ignore[arg-type]


def test_k_smallest_protocolo_resultado_counters() -> None:
    resultado, counters = arvore_exemplo().k_smallest_counted(2)
    assert resultado == [20, 30]
    assert counters.hops >= 0


# ----------------------------------------------------------------------
# Forma da árvore
# ----------------------------------------------------------------------
def test_relatorio_de_balanceamento_de_arvore_perfeita() -> None:
    tree = arvore_exemplo()
    relatorio = tree.balance_report()
    assert relatorio["nos"] == 7
    assert relatorio["altura"] == 3
    assert relatorio["altura_ideal"] == 3
    assert relatorio["razao_altura"] == 1.0
    assert relatorio["profundidade_minima"] == 3
    assert relatorio["folhas"] == 4
    assert relatorio["nos_por_nivel"] == [1, 2, 4]
    assert relatorio["degenerada"] is False


def test_relatorio_de_balanceamento_de_arvore_degenerada() -> None:
    n = 100
    tree: BinarySearchTree[int, None] = BinarySearchTree(range(n))
    relatorio = tree.balance_report()
    assert relatorio["altura"] == n
    assert relatorio["altura_ideal"] == math.ceil(math.log2(n + 1))
    assert relatorio["razao_altura"] > 14  # 100 / 7
    assert relatorio["nos_por_nivel"] == [1] * n
    assert relatorio["folhas"] == 1
    assert relatorio["degenerada"] is True


# ----------------------------------------------------------------------
# check_invariants tem de FALHAR em estrutura corrompida
# ----------------------------------------------------------------------
def test_check_invariants_detecta_violacao_global_com_pais_locais_corretos() -> None:
    """O bug clássico: cada nó é coerente com os filhos diretos, e mesmo assim
    a ordenação global está quebrada. Comparação local não pega isso."""
    tree: BinarySearchTree[int, None] = BinarySearchTree()
    tree._root = BSTNode(50, None, 1, BSTNode(30, None, 1, None, BSTNode(60)), None)
    tree._nodes = 3
    tree._size = 3
    # coerência local: 30 < 50 e 60 > 30 — tudo certo olhando pai e filho
    assert tree._root.left.key < tree._root.key  # type: ignore[union-attr]
    assert tree._root.left.right.key > tree._root.left.key  # type: ignore[union-attr]
    # mas 60 está na subárvore ESQUERDA de 50:
    with pytest.raises(BSTInvariantError, match="limite herdado"):
        tree.check_invariants()


def test_check_invariants_detecta_contagem_de_nos_errada() -> None:
    tree = arvore_exemplo()
    tree._nodes = 99
    with pytest.raises(BSTInvariantError, match="nó\\(s\\) alcançáveis"):
        tree.check_invariants()


def test_check_invariants_detecta_soma_de_multiplicidades_errada() -> None:
    tree = arvore_exemplo()
    tree._size = 99
    with pytest.raises(BSTInvariantError, match="multiplicidades somam"):
        tree.check_invariants()


def test_check_invariants_detecta_multiplicidade_invalida() -> None:
    tree = arvore_exemplo()
    tree._root.multiplicity = 0  # type: ignore[union-attr]
    tree._size -= 1
    with pytest.raises(BSTInvariantError, match="multiplicidade"):
        tree.check_invariants()


def test_check_invariants_detecta_no_compartilhado() -> None:
    tree: BinarySearchTree[int, None] = BinarySearchTree()
    compartilhado: BSTNode[int, None] = BSTNode(30)
    tree._root = BSTNode(50, None, 1, compartilhado, compartilhado)
    tree._nodes = 3
    tree._size = 3
    with pytest.raises(BSTInvariantError):
        tree.check_invariants()


def test_check_invariants_detecta_raiz_ausente_com_contagem() -> None:
    tree: BinarySearchTree[int, None] = BinarySearchTree()
    tree._nodes = 5
    with pytest.raises(BSTInvariantError, match="sem raiz"):
        tree.check_invariants()


# ----------------------------------------------------------------------
# Ordenação por BST (ex 2)
# ----------------------------------------------------------------------
def test_bst_sort_ordena_e_devolve_a_arvore() -> None:
    valores = [5, 3, 9, 1, 7]
    ordenado, tree = bst_sort(valores)
    assert ordenado == sorted(valores)
    assert tree.node_count == 5


def test_bst_sort_preserva_repeticoes() -> None:
    valores = [5, 3, 9, 3, 5, 5]
    ordenado, tree = bst_sort(valores)
    assert ordenado == sorted(valores)  # multiconjunto, não conjunto
    assert tree.node_count == 3
    assert len(tree) == 6


def test_bst_sort_bate_com_sorted_em_entradas_aleatorias() -> None:
    rng = random.Random(42)
    for _ in range(20):
        valores = [rng.randrange(50) for _ in range(rng.randrange(1, 80))]
        assert bst_sort(valores)[0] == sorted(valores)


def test_bst_sort_conta_o_custo_da_construcao() -> None:
    counters = Counters()
    bst_sort(range(100), counters)  # entrada já ordenada: pior caso
    assert counters.comparisons == 100 * 99 // 2


# ----------------------------------------------------------------------
# Experimentos
# ----------------------------------------------------------------------
def test_bench_insert_shapes_separa_n_log_n_de_n2() -> None:
    rows = bench_insert_shapes([125, 250, 500])
    aleatorios = [row for row in rows if row["padrao"] == "aleatorio"]
    ordenados = [row for row in rows if row["padrao"] == "ordenado"]
    reversos = [row for row in rows if row["padrao"] == "reverso"]
    assert len(aleatorios) == len(ordenados) == len(reversos) == 3

    for row in aleatorios:
        # ≈ 1,386·n·log2n − 1,85n comparações. O termo linear domina nestes
        # tamanhos, então a razão medida fica em 1,0–1,2 e não em 1,386.
        n = row["n"]
        previsto = 1.386 * n * math.log2(n) - 1.85 * n
        assert row["comparisons"] == pytest.approx(previsto, rel=0.20)
        assert 0.9 < row["comparisons/n_log2n"] < 1.4
        assert row["degenerada"] is False
        # altura esperada ≈ 4,31·ln n ≈ 2,99·log2 n, aproximada por baixo
        assert row["altura/log2n"] < 3.5

    for row in ordenados + reversos:
        n = row["n"]
        assert row["comparisons"] == n * (n - 1) // 2 == row["comparacoes_teoricas"]
        assert row["comparisons/n2"] == pytest.approx(0.5, abs=0.01)
        assert row["altura"] == n
        assert row["degenerada"] is True

    # a razão contra n·log2n é estável no aleatório e explode no ordenado
    razoes_aleatorio = [row["comparisons/n_log2n"] for row in aleatorios]
    razoes_ordenado = [row["comparisons/n_log2n"] for row in ordenados]
    assert max(razoes_aleatorio) / min(razoes_aleatorio) < 1.3
    assert razoes_ordenado[-1] / razoes_ordenado[0] > 2.5


def test_bench_degenerate_mostra_a_busca_virando_linear() -> None:
    rows = bench_degenerate([125, 250, 500])
    aleatorias = [row for row in rows if row["forma"] == "aleatória"]
    espinhas = [row for row in rows if row["forma"].startswith("degenerada")]
    assert len(aleatorias) == len(espinhas) == 3

    for row in espinhas:
        n = row["n"]
        assert row["altura"] == n
        assert row["altura/n"] == 1.0
        # a busca pela maior chave custa n: é uma busca em lista encadeada
        assert row["comparacoes_busca_maior"] == n
        assert row["busca/n"] == 1.0
        assert row["construcao/n2"] == pytest.approx(0.5, abs=0.01)

    for row in aleatorias:
        # busca em O(log n): razão contra log2 n presa perto de uma constante
        assert row["busca/log2n"] < 3
        assert row["altura/log2n"] < 6

    # e a travessia in-order custa O(n) nos DOIS casos: nem tudo degrada
    for aleatoria, espinha in zip(aleatorias, espinhas):
        assert aleatoria["saltos_travessia"] == espinha["saltos_travessia"]


def test_bench_k_smallest_mostra_a_economia_da_parada_antecipada() -> None:
    rows = bench_k_smallest(n=2_000, ks=(1, 10, 100, 2_000))
    assert len(rows) == 4
    por_k = {row["k"]: row for row in rows}

    assert por_k[1]["economia"] > 0.99
    assert por_k[10]["economia"] > 0.98
    assert por_k[100]["economia"] > 0.8
    assert por_k[2_000]["economia"] == pytest.approx(0.0, abs=0.01)

    # os saltos crescem com k; a travessia completa não muda
    saltos = [por_k[k]["saltos_controlada"] for k in (1, 10, 100, 2_000)]
    assert saltos == sorted(saltos)
    completos = {por_k[k]["saltos_completa"] for k in (1, 10, 100, 2_000)}
    assert len(completos) == 1


@pytest.mark.parametrize(
    "chamada",
    [
        lambda: bench_insert_shapes([0]),
        lambda: bench_degenerate([0]),
        lambda: bench_k_smallest(n=0),
    ],
)
def test_benchs_validam_tamanho(chamada: object) -> None:
    with pytest.raises(ValueError, match="tamanho deve ser positivo"):
        chamada()  # type: ignore[operator]


def test_bench_insert_shapes_rejeita_padrao_desconhecido() -> None:
    with pytest.raises(ValueError, match="padrão desconhecido"):
        bench_insert_shapes([10], patterns=["torto"])


def test_bench_k_smallest_valida_k() -> None:
    with pytest.raises(ValueError, match="k deve ser >= 0"):
        bench_k_smallest(n=10, ks=(-1,))

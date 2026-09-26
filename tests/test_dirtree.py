"""Testes do exercício 7 — recursão sobre árvore de diretórios em memória.

Dois pontos com peso de rubrica:

* a **política de remoção** (item de qualidade do enunciado): nó interno com
  filhos exige ``recursive=True``, e o erro diz quantos descendentes seriam
  apagados;
* a **travessia recursiva** com caso base explícito, verificada contra uma
  versão iterativa que produz exatamente a mesma lista — e que continua
  funcionando na hierarquia funda em que a recursiva quebra.
"""

from __future__ import annotations

import sys

import pytest

from src.counters import Counters
from src.dirtree import (
    DIRECTORY,
    FILE,
    PATH_STACKS,
    SEPARATOR,
    DirectoryNotEmptyError,
    DirectoryTree,
    DirectoryTreeError,
    DirectoryTreeInvariantError,
    DirNode,
    InvalidNodeTypeError,
    InvalidPathError,
    LinkedPathStack,
    ListPathStack,
    NotADirectoryTreeError,
    ParentNotFoundError,
    PathExistsError,
    PathNotFoundError,
    bench_path_stack,
    bench_search_by_width,
    bench_walk,
    build_balanced_tree,
    build_deep_tree,
    join_path,
    split_path,
    walk,
    walk_iterative,
)


def arvore_exemplo() -> DirectoryTree:
    """::

        /
          projeto/
            src/
              main.py
              util.py
            docs/
              leia.md
            README.md
    """
    tree = DirectoryTree()
    tree.insert("/projeto", DIRECTORY)
    tree.insert("/projeto/src", DIRECTORY)
    tree.insert("/projeto/src/main.py", FILE)
    tree.insert("/projeto/src/util.py", FILE)
    tree.insert("/projeto/docs", DIRECTORY)
    tree.insert("/projeto/docs/leia.md", FILE)
    tree.insert("/projeto/README.md", FILE)
    return tree


# ----------------------------------------------------------------------
# Caminhos
# ----------------------------------------------------------------------
def test_split_e_join_de_caminho() -> None:
    assert split_path("/") == []
    assert split_path("/a") == ["a"]
    assert split_path("/a/b/c") == ["a", "b", "c"]
    assert split_path("/a/b/") == ["a", "b"]  # barra final é tolerada
    assert join_path([]) == "/"
    assert join_path(["a", "b"]) == "/a/b"


@pytest.mark.parametrize(
    "ruim", ["", "a/b", "/a//b", "/a/./b", "/a/../b", 17]
)
def test_caminho_invalido_levanta(ruim: object) -> None:
    with pytest.raises(InvalidPathError):
        split_path(ruim)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Nó
# ----------------------------------------------------------------------
def test_no_guarda_nome_tipo_e_filhos() -> None:
    node = DirNode("src", DIRECTORY)
    assert node.name == "src"
    assert node.is_dir() and not node.is_file()
    assert node.is_leaf() and node.child_count() == 0
    node.children.insert_last(DirNode("main.py", FILE))
    assert not node.is_leaf() and node.child_count() == 1


def test_no_com_tipo_invalido_levanta() -> None:
    with pytest.raises(InvalidNodeTypeError, match="tipo deve ser"):
        DirNode("x", "atalho")


def test_no_usa_slots() -> None:
    node = DirNode("x")
    with pytest.raises(AttributeError):
        node.dono = "kikea"  # type: ignore[attr-defined]


def test_find_child_conta_comparacoes_e_saltos() -> None:
    pai = DirNode("raiz")
    for indice in range(5):
        pai.children.insert_last(DirNode(f"e{indice}"))
    counters = Counters()
    assert pai.find_child("e4", counters) is not None
    assert counters.comparisons == 5  # busca linear O(k)
    assert counters.hops == 4
    ausente = Counters()
    assert pai.find_child("zzz", ausente) is None
    assert ausente.comparisons == 5


# ----------------------------------------------------------------------
# Pilhas de caminho
# ----------------------------------------------------------------------
@pytest.mark.parametrize("classe", list(PATH_STACKS.values()))
def test_pilha_de_caminho_push_pop_e_to_path(classe: object) -> None:
    pilha = classe()  # type: ignore[operator]
    assert pilha.is_empty() and len(pilha) == 0
    assert pilha.to_path() == "/"
    pilha.push("projeto")
    pilha.push("src")
    assert len(pilha) == 2
    assert pilha.segments() == ["projeto", "src"]
    assert pilha.to_path() == "/projeto/src"
    assert pilha.pop() == "src"
    assert pilha.to_path() == "/projeto"
    pilha.check_invariants()


@pytest.mark.parametrize("classe", list(PATH_STACKS.values()))
def test_pilha_de_caminho_aceita_segmentos_iniciais(classe: object) -> None:
    pilha = classe(["a", "b", "c"])  # type: ignore[operator]
    assert pilha.to_path() == "/a/b/c"
    assert len(pilha) == 3


def test_as_duas_pilhas_se_comportam_igual() -> None:
    encadeada = LinkedPathStack()
    do_python = ListPathStack()
    for segmento in ("a", "b", "c", "d"):
        encadeada.push(segmento)
        do_python.push(segmento)
        assert encadeada.to_path() == do_python.to_path()
        assert len(encadeada) == len(do_python)
    while len(encadeada):
        assert encadeada.pop() == do_python.pop()
        assert encadeada.to_path() == do_python.to_path()


def test_pilha_encadeada_conta_saltos_e_a_do_python_nao() -> None:
    """A diferença observável: montar o caminho segue ponteiros de um lado só."""
    segmentos = [f"n{indice}" for indice in range(50)]
    encadeada = LinkedPathStack(segmentos)
    do_python = ListPathStack(segmentos)

    c1 = Counters()
    encadeada.to_path(c1)
    c2 = Counters()
    do_python.to_path(c2)
    assert c1.hops == 49
    assert c2.hops == 0


# ----------------------------------------------------------------------
# walk recursivo
# ----------------------------------------------------------------------
def test_walk_devolve_caminhos_completos_em_pre_ordem() -> None:
    tree = arvore_exemplo()
    assert tree.walk() == [
        "/",
        "/projeto",
        "/projeto/src",
        "/projeto/src/main.py",
        "/projeto/src/util.py",
        "/projeto/docs",
        "/projeto/docs/leia.md",
        "/projeto/README.md",
    ]


def test_walk_da_arvore_vazia_devolve_so_a_raiz() -> None:
    assert DirectoryTree().walk() == ["/"]


def test_walk_nao_produz_barra_dupla() -> None:
    """A raiz tem nome vazio; empilhá-la geraria '//projeto'."""
    for caminho in arvore_exemplo().walk():
        assert "//" not in caminho
        assert caminho.startswith(SEPARATOR)


def test_walk_recursivo_e_iterativo_produzem_o_mesmo() -> None:
    for tree in (arvore_exemplo(), build_balanced_tree(4, 3)):
        assert tree.walk() == walk_iterative(tree.root)


def test_walk_funciona_com_as_duas_pilhas_de_caminho() -> None:
    tree = arvore_exemplo()
    assert tree.walk(path_stack_class=LinkedPathStack) == tree.walk(
        path_stack_class=ListPathStack
    )


def test_walk_conta_uma_chamada_por_no() -> None:
    tree = build_balanced_tree(3, 3)
    counters = Counters()
    caminhos = tree.walk(counters)
    assert counters.calls == len(caminhos) == len(tree) + 1


def test_walk_recursivo_quebra_onde_o_iterativo_nao_quebra() -> None:
    """Decisão de projeto 4: Θ(d) de pilha de chamadas tem teto."""
    profundidade = 3 * sys.getrecursionlimit()
    tree = build_deep_tree(profundidade)
    assert tree.depth() == profundidade

    with pytest.raises(RecursionError):
        tree.walk()

    caminhos = walk_iterative(tree.root)
    assert len(caminhos) == profundidade + 1
    assert caminhos[-1].endswith(f"n{profundidade}")


# ----------------------------------------------------------------------
# insert
# ----------------------------------------------------------------------
def test_insert_cria_diretorios_e_arquivos() -> None:
    tree = DirectoryTree()
    assert tree.is_empty()
    tree.insert("/a", DIRECTORY)
    tree.insert("/a/b.txt", FILE)
    assert len(tree) == 2
    assert tree.search("/a").is_dir()
    assert tree.search("/a/b.txt").is_file()
    tree.check_invariants()


def test_insert_exige_pai_existente() -> None:
    tree = DirectoryTree()
    with pytest.raises(ParentNotFoundError, match="create_parents=True"):
        tree.insert("/a/b/c")
    assert len(tree) == 0  # nada foi criado pela tentativa


def test_insert_com_create_parents() -> None:
    tree = DirectoryTree()
    tree.insert("/a/b/c", FILE, create_parents=True)
    assert len(tree) == 3
    assert tree.search("/a").is_dir()
    assert tree.search("/a/b").is_dir()
    assert tree.search("/a/b/c").is_file()
    tree.check_invariants()


def test_insert_rejeita_caminho_duplicado() -> None:
    tree = arvore_exemplo()
    with pytest.raises(PathExistsError, match="já existe"):
        tree.insert("/projeto/src")
    with pytest.raises(PathExistsError, match="raiz"):
        tree.insert("/")


def test_insert_rejeita_criar_abaixo_de_arquivo() -> None:
    tree = arvore_exemplo()
    with pytest.raises(NotADirectoryTreeError, match="é arquivo"):
        tree.insert("/projeto/README.md/x", FILE)
    with pytest.raises(NotADirectoryTreeError):
        tree.insert("/projeto/README.md/x/y", FILE, create_parents=True)


def test_insert_rejeita_tipo_invalido() -> None:
    with pytest.raises(InvalidNodeTypeError):
        DirectoryTree().insert("/a", "link")


def test_insert_protocolo_resultado_counters() -> None:
    tree = DirectoryTree()
    node, counters = tree.insert_counted("/a", DIRECTORY)
    assert node.name == "a"
    assert counters.copies >= 1


# ----------------------------------------------------------------------
# search
# ----------------------------------------------------------------------
def test_search_encontra_todos_os_caminhos_do_walk() -> None:
    tree = arvore_exemplo()
    for caminho in tree.walk():
        assert tree.search(caminho) is not None
        assert caminho in tree


def test_search_de_caminho_inexistente_levanta_dizendo_onde_parou() -> None:
    tree = arvore_exemplo()
    with pytest.raises(PathNotFoundError, match="'nao_existe'"):
        tree.search("/projeto/nao_existe")
    assert "/projeto/nao_existe" not in tree
    assert tree.exists("/projeto/src") is True
    assert tree.exists("/projeto/nada") is False


def test_search_abaixo_de_arquivo_levanta_erro_especifico() -> None:
    tree = arvore_exemplo()
    with pytest.raises(NotADirectoryTreeError, match="é arquivo"):
        tree.search("/projeto/README.md/x")


def test_search_custa_d_vezes_k() -> None:
    tree = build_balanced_tree(3, 8)
    counters = Counters()
    tree.search("/d0_7/d1_7/d2_7", counters)
    assert counters.comparisons == 3 * 8  # pior caso: último de cada nível


# ----------------------------------------------------------------------
# delete — a política do enunciado
# ----------------------------------------------------------------------
def test_delete_de_arquivo_e_de_diretorio_vazio_dispensa_a_bandeira() -> None:
    tree = arvore_exemplo()
    assert tree.delete("/projeto/README.md") == 1
    tree.insert("/projeto/vazio", DIRECTORY)
    assert tree.delete("/projeto/vazio") == 1
    tree.check_invariants()


def test_delete_de_diretorio_com_filhos_exige_recursive() -> None:
    """Decisão de projeto 3: o erro diz quantos nós seriam apagados."""
    tree = arvore_exemplo()
    with pytest.raises(DirectoryNotEmptyError) as falha:
        tree.delete("/projeto/src")
    mensagem = str(falha.value)
    assert "2 filho(s) diretos" in mensagem
    assert "2 descendente(s)" in mensagem
    assert "recursive=True" in mensagem
    # e nada foi removido pela tentativa
    assert len(tree) == 7
    tree.check_invariants()


def test_delete_recursivo_remove_a_subarvore_inteira() -> None:
    tree = arvore_exemplo()
    removidos = tree.delete("/projeto/src", recursive=True)
    assert removidos == 3  # o diretório mais os dois arquivos
    assert len(tree) == 4
    assert "/projeto/src" not in tree
    assert "/projeto/src/main.py" not in tree
    assert "/projeto/docs/leia.md" in tree  # o irmão ficou intacto
    tree.check_invariants()


def test_delete_da_raiz_e_proibido() -> None:
    with pytest.raises(DirectoryTreeError, match="raiz não pode ser removida"):
        arvore_exemplo().delete("/")


def test_delete_de_caminho_inexistente_levanta() -> None:
    tree = arvore_exemplo()
    with pytest.raises(PathNotFoundError):
        tree.delete("/projeto/nada")
    with pytest.raises(PathNotFoundError):
        tree.delete("/nada/nada")


def test_delete_abaixo_de_arquivo_levanta() -> None:
    tree = arvore_exemplo()
    with pytest.raises(NotADirectoryTreeError):
        tree.delete("/projeto/README.md/x")


def test_delete_ate_esvaziar_mantem_as_contas() -> None:
    tree = build_balanced_tree(3, 3)
    total = len(tree)
    removidos = tree.delete("/d0_0", recursive=True)
    assert len(tree) == total - removidos
    tree.check_invariants()
    for caminho in ("/d0_1", "/d0_2"):
        tree.delete(caminho, recursive=True)
    assert tree.is_empty()
    assert tree.walk() == ["/"]
    tree.check_invariants()


def test_delete_protocolo_resultado_counters() -> None:
    tree = arvore_exemplo()
    removidos, counters = tree.delete_counted("/projeto/src", recursive=True)
    assert removidos == 3
    assert counters.comparisons > 0


# ----------------------------------------------------------------------
# check_invariants tem de FALHAR em estrutura corrompida
# ----------------------------------------------------------------------
def test_check_invariants_detecta_arquivo_com_filho() -> None:
    tree = arvore_exemplo()
    arquivo = tree.search("/projeto/README.md")
    arquivo.children.insert_last(DirNode("clandestino", FILE))
    tree._size += 1
    with pytest.raises(DirectoryTreeInvariantError, match="não pode ter filhos"):
        tree.check_invariants()


def test_check_invariants_detecta_nomes_repetidos_no_mesmo_diretorio() -> None:
    tree = arvore_exemplo()
    projeto = tree.search("/projeto")
    projeto.children.insert_last(DirNode("src", DIRECTORY))
    tree._size += 1
    with pytest.raises(DirectoryTreeInvariantError, match="dois filhos chamados"):
        tree.check_invariants()


def test_check_invariants_detecta_contagem_errada() -> None:
    tree = arvore_exemplo()
    tree._size = 99
    with pytest.raises(DirectoryTreeInvariantError, match="nó\\(s\\) fora da raiz"):
        tree.check_invariants()


def test_check_invariants_detecta_nome_com_separador() -> None:
    tree = arvore_exemplo()
    tree.search("/projeto").children.insert_last(DirNode("a/b", FILE))
    tree._size += 1
    with pytest.raises(DirectoryTreeInvariantError, match="não pode conter"):
        tree.check_invariants()


def test_check_invariants_detecta_raiz_trocada() -> None:
    tree = DirectoryTree()
    tree._root = DirNode("raiz_errada", DIRECTORY)
    with pytest.raises(DirectoryTreeInvariantError, match="nome vazio"):
        tree.check_invariants()


# ----------------------------------------------------------------------
# Construtores
# ----------------------------------------------------------------------
def test_arvore_balanceada_tem_o_tamanho_previsto() -> None:
    for depth, branching in ((1, 2), (2, 3), (3, 2)):
        tree = build_balanced_tree(depth, branching)
        esperado = sum(branching**nivel for nivel in range(1, depth + 1))
        assert len(tree) == esperado
        assert tree.depth() == depth
        tree.check_invariants()


def test_arvore_balanceada_com_arquivos() -> None:
    tree = build_balanced_tree(2, 2, files_per_dir=3)
    arquivos = [c for c in tree.walk() if c.endswith(".txt")]
    assert len(arquivos) == 4 * 3  # 4 folhas × 3 arquivos
    tree.check_invariants()


def test_arvore_funda() -> None:
    tree = build_deep_tree(50)
    assert len(tree) == 50
    assert tree.depth() == 50
    assert tree.search("/n1/n2/n3").name == "n3"
    tree.check_invariants()


def test_construtores_validam_parametros() -> None:
    with pytest.raises(ValueError, match="depth deve ser >= 0"):
        build_balanced_tree(-1, 2)
    with pytest.raises(ValueError, match="branching deve ser >= 1"):
        build_balanced_tree(2, 0)
    with pytest.raises(ValueError, match="files_per_dir"):
        build_balanced_tree(2, 2, files_per_dir=-1)
    with pytest.raises(ValueError, match="depth deve ser >= 0"):
        build_deep_tree(-1)


# ----------------------------------------------------------------------
# Experimentos
# ----------------------------------------------------------------------
def test_bench_walk_mostra_que_o_custo_segue_os_caracteres() -> None:
    rows = bench_walk([2, 3, 4])
    assert len(rows) == 3
    for row in rows:
        assert row["calls_igual_a_nos"] is True
        assert row["caminhos"] == row["n"]
        assert row["caracteres_por_no"] > 0
    # caracteres por nó crescem com a profundidade: é o termo d de Θ(n·d)
    por_no = [row["caracteres_por_no"] for row in rows]
    assert por_no == sorted(por_no)


def test_bench_walk_valida_profundidade() -> None:
    with pytest.raises(ValueError, match="depth deve ser >= 0"):
        bench_walk([-1])


def test_bench_path_stack_compara_as_duas_implementacoes() -> None:
    rows = bench_path_stack([100, 400], repeats=1)
    assert len(rows) == 4
    encadeadas = [row for row in rows if row["implementacao"] == "lista encadeada"]
    do_python = [row for row in rows if row["implementacao"] == "list do Python"]
    assert len(encadeadas) == len(do_python) == 2

    for row in encadeadas:
        assert row["hops"] > 0  # montar o caminho percorre ponteiros
        assert row["hops/n"] == pytest.approx(1.0, abs=0.05)
    for row in do_python:
        assert row["hops"] == 0  # memória contígua: nada a percorrer

    # as duas são O(1) por operação: tempo por operação praticamente estável
    for grupo in (encadeadas, do_python):
        por_operacao = [row["tempo_us_por_operacao"] for row in grupo]
        assert max(por_operacao) / min(por_operacao) < 3.0


def test_bench_path_stack_valida_parametros() -> None:
    with pytest.raises(ValueError, match="repeats deve ser >= 1"):
        bench_path_stack([10], repeats=0)
    with pytest.raises(ValueError, match="profundidade deve ser positiva"):
        bench_path_stack([0])


def test_bench_search_por_largura_confirma_d_vezes_k() -> None:
    rows = bench_search_by_width([4, 16, 64], depth=3)
    assert len(rows) == 3
    for row in rows:
        assert row["comparacoes/(d*k)"] == pytest.approx(1.0, abs=0.01)
        assert row["comparacoes_se_fosse_hash"] == 3
        assert row["ganho_potencial"] == pytest.approx(row["largura"], rel=0.01)
    # o ganho potencial de trocar por hashtable cresce com a largura
    ganhos = [row["ganho_potencial"] for row in rows]
    assert ganhos == sorted(ganhos)


def test_bench_search_valida_largura() -> None:
    with pytest.raises(ValueError, match="largura deve ser >= 1"):
        bench_search_by_width([0])

"""Quebra o notebook único em 12 notebooks do Colab, um por exercício.

Uso::

    python gerar_colab.py

Por que existe
--------------
O professor abre o trabalho **exercício por exercício**, por link. Um notebook de
64 páginas obriga a rodar os doze exercícios para ver um só, e os experimentos
pesados do ex 2 e do ex 12 levam minutos. Doze notebooks resolvem isso: cada link
abre um exercício que roda em segundos.

O que cada notebook gerado contém, nesta ordem
----------------------------------------------
1. **Célula de preparação** — acha a raiz do projeto subindo os diretórios e, se
   não achar (é o caso no Colab), clona o repositório. Funciona igual na máquina
   local e no Colab, sem `if` de ambiente espalhado pelo notebook.
2. **Capa** — nome, qual exercício, e a lista de links para os outros onze.
3. **Seção 0** — a instrumentação compartilhada, idêntica à do notebook único.
4. **Imports compartilhados** — os três símbolos que, no notebook único, vinham
   da seção de outro exercício (ver :data:`IMPORTS_COMPARTILHADOS`). Sem eles o
   notebook isolado quebraria, e o script confere isso automaticamente.
5. **O exercício** — as células originais, sem uma linha alterada.

O código continua vivendo em ``src/``: os notebooks importam e demonstram, como
manda a seção 3 do CLAUDE.md. Nada de classe colada dentro do notebook.
"""

from __future__ import annotations

import ast
import builtins
import json
import sys
from pathlib import Path

import nbformat

RAIZ = Path(__file__).resolve().parent
NOTEBOOK = RAIZ / "notebook" / "DR1_AT.ipynb"
DESTINO = RAIZ / "colab"

#: Dono e nome do repositório público que o Colab clona.
USUARIO_GITHUB = "thxggz"
REPOSITORIO = "dr1-at-algoritmos"
BRANCH = "main"

#: Células do notebook original que formam a seção 0 (capa + instrumentação).
CELULAS_DE_SETUP = range(1, 4)

#: Símbolos que o notebook único definia na seção de um exercício e outro
#: exercício reaproveitava. Num notebook isolado eles precisam ser importados de
#: novo. A lista é conferida por :func:`conferir_dependencias`, então esquecer um
#: falha o build em vez de gerar notebook quebrado.
IMPORTS_COMPARTILHADOS: dict[str, str] = {
    "generate_array": "from src.searching import generate_array",
    "BinarySearchTree": "from src.bst import BinarySearchTree",
    "SinglyLinkedList": "from src.linked_list import SinglyLinkedList",
}

#: Apelido curto de cada exercício, usado no nome do arquivo e nos links.
APELIDOS: dict[int, str] = {
    1: "buscas",
    2: "ordenacoes_quadraticas",
    3: "otimizacao",
    4: "hashtable",
    5: "pilha_fila_expressao",
    6: "simulador_de_filas",
    7: "recursao_diretorios",
    8: "programacao_dinamica",
    9: "quicksort_quickselect",
    10: "lista_encadeada",
    11: "lista_dupla_deque",
    12: "motor_de_indexacao",
}

NOME_COMPLETO = "Henrique Goldstein Maluhy Mendes Amaral"


def url_colab(numero: int) -> str:
    """Link que abre o notebook do exercício ``numero`` direto no Colab."""
    arquivo = f"ex{numero:02d}_{APELIDOS[numero]}.ipynb"
    return (
        f"https://colab.research.google.com/github/{USUARIO_GITHUB}/"
        f"{REPOSITORIO}/blob/{BRANCH}/colab/{arquivo}"
    )


def celula_de_preparacao() -> str:
    """Código que garante que ``src/`` existe e que o cwd é a raiz do projeto.

    Sobe os diretórios procurando ``src/counters.py``. Se não achar, está no
    Colab com o disco vazio, então clona o repositório. Idempotente: rodar duas
    vezes não clona duas vezes.
    """
    return f'''# --- preparação do ambiente (rode esta célula primeiro) ---
# Acha a raiz do projeto; se não existir, estamos no Colab e o repositório é clonado.
import os
import pathlib
import subprocess

REPOSITORIO = "https://github.com/{USUARIO_GITHUB}/{REPOSITORIO}.git"
DESTINO_CLONE = pathlib.Path("/content/{REPOSITORIO}")


def raiz_do_projeto() -> pathlib.Path | None:
    """Primeiro diretório, subindo a partir do atual, que contenha src/counters.py."""
    for base in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:
        if (base / "src" / "counters.py").exists():
            return base
    return None


raiz = raiz_do_projeto()
if raiz is None:
    if not DESTINO_CLONE.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", REPOSITORIO, str(DESTINO_CLONE)],
            check=True,
        )
    raiz = DESTINO_CLONE
os.chdir(raiz)
print(f"raiz do projeto: {{raiz}}")
print("o código vive em src/; este notebook importa e demonstra")'''


def capa(numero: int, titulo: str) -> str:
    """Markdown de abertura, com o índice de links para os outros exercícios."""
    linhas = [
        f"# {titulo}",
        "",
        f"**{NOME_COMPLETO}** · Infnet · DR1_AT — Algoritmos e Estruturas de Dados",
        "",
        "Este notebook cobre **um** exercício. O código está em `src/`, testado por "
        "`pytest`; aqui ele é importado e demonstrado. A primeira célula prepara o "
        "ambiente — rode-a antes das outras, ou use **Ambiente de execução → "
        "Executar tudo**.",
        "",
        "**Os doze exercícios:**",
        "",
    ]
    for outro in sorted(APELIDOS):
        marca = "**→ você está aqui**" if outro == numero else ""
        rotulo = f"Exercício {outro}"
        if outro == numero:
            linhas.append(f"{outro}. {rotulo} {marca}")
        else:
            linhas.append(f"{outro}. [{rotulo}]({url_colab(outro)})")
    linhas += [
        "",
        "---",
        "",
    ]
    return "\n".join(linhas)


def _nomes(fonte: str) -> tuple[set[str], set[str]]:
    """Nomes ligados e nomes lidos em um trecho de código."""
    ligados: set[str] = set()
    lidos: set[str] = set()
    try:
        arvore = ast.parse(fonte)
    except SyntaxError:
        return ligados, lidos
    for no in ast.walk(arvore):
        if isinstance(no, ast.Name):
            (ligados if isinstance(no.ctx, ast.Store) else lidos).add(no.id)
        elif isinstance(no, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            ligados.add(no.name)
        elif isinstance(no, ast.arg):
            ligados.add(no.arg)
        elif isinstance(no, (ast.Import, ast.ImportFrom)):
            for alias in no.names:
                ligados.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(no, ast.ExceptHandler) and no.name:
            ligados.add(no.name)
    return ligados, lidos


def secoes(cells: list) -> dict[str, tuple[int, int]]:
    """Título de nível 1 -> faixa de células [início, fim) que ele cobre."""
    titulos = [
        i for i, c in enumerate(cells)
        if c["cell_type"] == "markdown"
        and any(l.startswith("# ") for l in "".join(c["source"]).splitlines())
    ]
    mapa: dict[str, tuple[int, int]] = {}
    for posicao, indice in enumerate(titulos):
        titulo = next(
            l[2:].strip()
            for l in "".join(cells[indice]["source"]).splitlines()
            if l.startswith("# ")
        )
        fim = titulos[posicao + 1] if posicao + 1 < len(titulos) else len(cells)
        mapa[titulo] = (indice, fim)
    return mapa


def conferir_dependencias(cells: list, faixa: tuple[int, int], base: set[str]) -> list[str]:
    """Nomes que o exercício usa e que ninguém definiu — o build falha se houver."""
    fonte = "\n".join(
        "".join(cells[i]["source"])
        for i in range(*faixa)
        if cells[i]["cell_type"] == "code"
    )
    ligados, lidos = _nomes(fonte)
    return sorted(lidos - ligados - base)


def main() -> int:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    cells = nb.cells
    mapa = secoes(cells)

    setup = [cells[i] for i in CELULAS_DE_SETUP]
    fonte_setup = "\n".join(c.source for c in setup if c.cell_type == "code")
    # A base NÃO inclui IMPORTS_COMPARTILHADOS: é justamente o que falta que
    # decide quais imports acrescentar, e o que falta fora dessa lista é erro.
    base = set(_nomes(fonte_setup)[0]) | set(dir(builtins))

    DESTINO.mkdir(exist_ok=True)
    gerados: list[tuple[int, str, Path]] = []

    for titulo, faixa in mapa.items():
        if not titulo.startswith("Exercício"):
            continue
        numero = int(titulo.split()[1])

        faltando = conferir_dependencias(cells, faixa, base)
        precisa = [nome for nome in faltando if nome in IMPORTS_COMPARTILHADOS]
        desconhecidos = [nome for nome in faltando if nome not in IMPORTS_COMPARTILHADOS]
        if desconhecidos:
            print(
                f"ERRO: exercício {numero} usa nomes que ninguém define: {desconhecidos}\n"
                f"       acrescente-os a IMPORTS_COMPARTILHADOS."
            )
            return 1

        novo = nbformat.v4.new_notebook(metadata=nb.metadata)
        novo.cells.append(nbformat.v4.new_code_cell(celula_de_preparacao()))
        novo.cells.append(nbformat.v4.new_markdown_cell(capa(numero, titulo)))
        novo.cells.extend(setup)
        if precisa:
            novo.cells.append(
                nbformat.v4.new_code_cell(
                    "# Estruturas definidas na seção de outro exercício do notebook\n"
                    "# único, reimportadas aqui para este notebook rodar sozinho.\n"
                    + "\n".join(IMPORTS_COMPARTILHADOS[n] for n in precisa)
                )
            )
        novo.cells.extend(cells[faixa[0]:faixa[1]])

        arquivo = DESTINO / f"ex{numero:02d}_{APELIDOS[numero]}.ipynb"
        nbformat.write(novo, arquivo)
        gerados.append((numero, titulo, arquivo))
        extra = f" + {len(precisa)} import(s)" if precisa else ""
        print(f"  ex {numero:>2}: {len(novo.cells):>3} células{extra} -> {arquivo.name}")

    if len(gerados) != 12:
        print(f"ERRO: gerei {len(gerados)} notebooks, esperava 12")
        return 1
    print(f"\n{len(gerados)} notebooks em {DESTINO.relative_to(RAIZ)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

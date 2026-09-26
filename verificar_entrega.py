"""Confere, num comando só, que a entrega está completa e íntegra.

Uso::

    python verificar_entrega.py

Por que existe
--------------
São **2 tentativas de envio** e um prazo. Conferir na mão, item por item, é
exatamente o tipo de tarefa em que se esquece de um — e o esquecimento que
motivou este script foi real: o PDF estava cortando o lado direito das tabelas
largas sem emitir aviso nenhum, e as colunas perdidas eram justamente as de razão
contra a curva teórica, que são a prova de cada afirmação de Big O do trabalho.

Cada conferência abaixo responde a uma pergunta que o professor pode fazer, e
falha ruidosamente em vez de passar em silêncio.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from build_pdf import (
    NOME_BASE,
    NOTEBOOK,
    SAIDA_PDF,
    _linhas_largas_do_notebook,
    _normalizar,
    _titulos_do_notebook,
)

RAIZ = Path(__file__).resolve().parent
FIGURAS = RAIZ / "figures"

#: Evidência que o RUBRICA.md cita para cada item, e como ela pode aparecer
#: escrita no PDF. Se um número citado lá não está no PDF entregue, o item está
#: apoiado em prova que o professor não consegue ver.
EVIDENCIAS: list[tuple[str, str, tuple[str, ...]]] = [
    ("1.1", "tratamento de não encontrado", ("UnsortedInputError",)),
    ("1.2", "mesmas comparações no reverso", ("7.998.000", "7998000")),
    ("1.2", "cópias separam trocar de deslocar", ("23.994.000", "23994000")),
    ("1.3", "melhor/médio/pior caso", ("pior caso", "melhor caso")),
    ("1.4", "otimização com equivalência provada", ("deduplicate_slow",)),
    ("1.5", "trade-off justificado", ("trade-off", "trade off")),
    ("2.1", "custo médio 1+α/2", ("1,23", "1.23")),
    ("2.2", "buffer circular com voltas", ("8.743", "8743")),
    ("2.3", "parsing com pilha", ("shunting", "Shunting")),
    ("2.4", "simulador com rastreabilidade", ("B17",)),
    ("2.5", "casos de borda", ("overflow", "underflow")),
    ("3.1", "recursão vs iteração", ("walk_iterative", "walk iterativo")),
    ("3.2", "sobreposição de subproblemas", ("142.129", "142129")),
    ("3.3", "quicksort por padrão de entrada", ("median3", "mediana de três")),
    ("3.4", "profundidade de recursão", ("25.601", "25601")),
    ("3.5", "ganho quantitativo da DP", ("2.090", "2090")),
    ("4.1", "lista encadeada completa", ("SinglyLinkedList",)),
    ("4.2", "lista comparada com array", ("1,8", "1.8")),
    ("4.3", "deque O(1) nas duas pontas", ("1,32", "1.32")),
    ("4.4", "BST com deleção mantendo invariantes", ("sucessor in-order",)),
    ("4.5", "relatório técnico de desempenho", ("performance_report",)),
]


@dataclass
class Resultado:
    """Resposta de uma conferência: passou, e o número que sustenta isso."""

    nome: str
    ok: bool
    detalhe: str

    def linha(self) -> str:
        marca = "OK  " if self.ok else "FALHA"
        return f"  [{marca}] {self.nome}: {self.detalhe}"


def conferir_testes() -> Resultado:
    """`pytest -q` verde é pré-condição de qualquer exercício estar pronto."""
    processo = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header"],
        cwd=RAIZ, capture_output=True, text=True, timeout=1800,
    )
    saida = (processo.stdout or "") + (processo.stderr or "")
    achado = re.search(r"(\d+) passed", saida)
    total = achado.group(1) if achado else "?"
    falhou = re.search(r"(\d+) (failed|error)", saida)
    return Resultado(
        "testes",
        processo.returncode == 0 and not falhou,
        f"{total} passando" if processo.returncode == 0 else f"saída: {saida[-200:]}",
    )


def conferir_notebook() -> Resultado:
    """Notebook executado ponta a ponta, sem célula com erro."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    codigo = [c for c in nb["cells"] if c["cell_type"] == "code"]
    com_saida = [c for c in codigo if c.get("outputs")]
    erros = [
        s.get("ename")
        for c in nb["cells"]
        for s in c.get("outputs", [])
        if s.get("output_type") == "error"
    ]
    ok = not erros and len(com_saida) >= len(codigo) * 0.9
    detalhe = f"{len(com_saida)}/{len(codigo)} células com saída, {len(erros)} erro(s)"
    return Resultado("notebook executado", ok, detalhe)


def conferir_nomes() -> Resultado:
    """O enunciado exige `nome_sobrenome_DR1_AT.pdf`; nome errado é nota perdida."""
    pdf = RAIZ / f"{NOME_BASE}.pdf"
    return Resultado(
        "nome do arquivo",
        pdf.exists(),
        f"{pdf.name} ({pdf.stat().st_size / 1_048_576:.1f} MB)" if pdf.exists()
        else "PDF não encontrado",
    )


def _texto_do_pdf() -> str:
    from pypdf import PdfReader

    return _normalizar(
        "".join((p.extract_text() or "") for p in PdfReader(SAIDA_PDF).pages)
    )


def conferir_exercicios_no_pdf(texto: str) -> Resultado:
    """Os 12 títulos precisam estar visíveis, na ordem 1 a 12."""
    achados = [
        i for i in range(1, 13) if re.search(rf"Exerc[ií]cio {i}(?!\d)", texto)
    ]
    faltando = [i for i in range(1, 13) if i not in achados]
    return Resultado(
        "exercícios no PDF",
        not faltando,
        f"{len(achados)}/12 títulos" + (f", faltando {faltando}" if faltando else ""),
    )


def conferir_marcadores() -> Resultado:
    """PDF navegável: um marcador por seção no painel do leitor."""
    from pypdf import PdfReader

    def achatar(itens) -> list:
        saida = []
        for item in itens:
            saida.extend(achatar(item) if isinstance(item, list) else [item])
        return saida

    leitor = PdfReader(SAIDA_PDF)
    marcadores = achatar(leitor.outline)
    esperados = len(_titulos_do_notebook())
    return Resultado(
        "marcadores do PDF",
        len(marcadores) >= esperados,
        f"{len(marcadores)} marcadores em {len(leitor.pages)} páginas",
    )


def conferir_nada_cortado(texto: str) -> Resultado:
    """Nenhuma linha larga pode ter perdido o final na impressão."""
    largas = sorted(_linhas_largas_do_notebook(minimo=100), key=len, reverse=True)
    perdidas = []
    conferidas = 0
    for linha in largas:
        cauda = " ".join(linha.split()[-4:])
        if len(cauda) < 8:
            continue
        conferidas += 1
        if cauda not in texto:
            perdidas.append(cauda)
    return Resultado(
        "nada cortado no PDF",
        not perdidas,
        f"{conferidas} linhas largas conferidas (a maior com {len(largas[0])} colunas)"
        if not perdidas else f"{len(perdidas)} cortadas, ex.: ...{perdidas[0][:60]}",
    )


def conferir_figuras() -> Resultado:
    """Os gráficos precisam existir como arquivo e estar embutidos no PDF."""
    from pypdf import PdfReader

    pngs = sorted(FIGURAS.glob("*.png"))
    com_imagem = sum(
        1 for p in PdfReader(SAIDA_PDF).pages
        if "/XObject" in (p.get("/Resources") or {})
    )
    return Resultado(
        "figuras", len(pngs) >= 12 and com_imagem > 0,
        f"{len(pngs)} PNGs em figures/, {com_imagem} páginas do PDF com imagem",
    )


def conferir_evidencias(texto: str) -> Resultado:
    """Cada item da rubrica tem, no PDF, a evidência que o RUBRICA.md promete."""
    ausentes = [
        f"{item} ({desc})"
        for item, desc, formas in EVIDENCIAS
        if not any(forma in texto for forma in formas)
    ]
    return Resultado(
        "evidências da rubrica",
        not ausentes,
        f"{len(EVIDENCIAS) - len(ausentes)}/{len(EVIDENCIAS)} presentes no PDF"
        + (f"; AUSENTES: {ausentes}" if ausentes else ""),
    )


def main() -> int:
    print("Conferência da entrega — TP DR1_AT\n")

    resultados = [conferir_testes(), conferir_notebook(), conferir_nomes()]
    if resultados[-1].ok:
        texto = _texto_do_pdf()
        resultados += [
            conferir_exercicios_no_pdf(texto),
            conferir_marcadores(),
            conferir_nada_cortado(texto),
            conferir_figuras(),
            conferir_evidencias(texto),
        ]

    for resultado in resultados:
        print(resultado.linha())

    falhas = [r for r in resultados if not r.ok]
    print()
    if falhas:
        print(f"{len(falhas)} conferência(s) FALHARAM — não envie ainda.")
        return 1
    print(f"As {len(resultados)} conferências passaram. A entrega está íntegra.")
    print("\nFalta só o que o script não faz por você:")
    print("  1. gravar o vídeo de 5 a 8 min seguindo ROTEIRO_VIDEO.md")
    print("  2. subir no Drive da Infnet com 'qualquer pessoa com o link'")
    print("  3. enviar PDF + link a partir de 22/09/2026 07h00 (2 tentativas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

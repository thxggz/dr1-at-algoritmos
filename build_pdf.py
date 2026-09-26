"""Converte o notebook executado em `Henrique_Goldstein_DR1_AT.pdf`.

Uso::

    python -m nbconvert --to notebook --execute --inplace notebook/DR1_AT.ipynb
    python build_pdf.py

Por que não `nbconvert --to pdf` direto
---------------------------------------
Esse caminho exige uma instalação de LaTeX (XeLaTeX + pacotes), que não existe na
máquina deste trabalho e cuja instalação é de gigabytes. O caminho usado aqui é
**notebook → HTML → PDF**, com três motores tentados em ordem de preferência:

0. **Chrome ou Edge em modo headless** (``--print-to-pdf``) — o caminho
   preferido, porque não exige instalar nada: o Windows já traz o Edge e a
   máquina deste trabalho também tem o Chrome.
1. **Playwright/Chromium** (``nbconvert --to webpdf``) — alternativa:
   respeita CSS, mantém os títulos como marcadores navegáveis e não corta tabela
   larga. Requer ``pip install "nbconvert[webpdf]"`` e ``playwright install chromium``.
2. **wkhtmltopdf** — se estiver no PATH.
3. **HTML apenas** — sempre funciona. O script gera o HTML, avisa e instrui a
   imprimir para PDF pelo navegador (Ctrl+P → "Salvar como PDF"), que é o
   procedimento manual de um minuto e produz PDF navegável.

Em todos os caminhos, o HTML intermediário fica salvo ao lado do PDF, para que
haja sempre um artefato utilizável mesmo se a conversão automática falhar.

Por que existe CSS de impressão próprio
---------------------------------------
O Chrome headless imprimia o HTML **cortando o lado direito das tabelas largas**,
sem emitir aviso nenhum. Duas causas somadas: o Jupyter marca as áreas de saída
com ``overflow: auto`` (que na impressão vira corte, não barra de rolagem) e a
folha A4 retrato não cabe uma linha de 197 colunas. O prejuízo não era estético:
as colunas de razão contra a curva teórica — que são a prova de cada afirmação de
Big O deste trabalho — ficam à direita nas tabelas, exatamente onde o corte caía.

:data:`CSS_IMPRESSAO` resolve com folha em paisagem, código em 7pt e
``overflow: visible``. A escolha do 7pt é medida, não chutada: a mediana das
linhas tem 63 colunas e o percentil 99,5 tem 197, então a folha é dimensionada
para :data:`LARGURA_ALVO_COLUNAS` colunas, o que deixa 17 linhas de 2.245 (0,8%)
dependendo da quebra automática, que existe como rede de segurança para que nada
seja perdido em silêncio nunca mais.

E :func:`conferir_nada_cortado` fecha o ciclo: depois de gerar o PDF, confere que
o final das linhas mais largas do notebook realmente está no texto do PDF. É o
mesmo princípio do resto do trabalho — a afirmação vem com a medição que a
sustenta.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
NOTEBOOK = RAIZ / "notebook" / "DR1_AT.ipynb"

#: Nome exigido pela regra de formatação do enunciado.
NOME_BASE = "Henrique_Goldstein_DR1_AT"
SAIDA_PDF = RAIZ / f"{NOME_BASE}.pdf"
SAIDA_HTML = RAIZ / f"{NOME_BASE}.html"


def _rodar(comando: list[str], descricao: str) -> bool:
    """Executa um comando e diz se deu certo, sem derrubar o script."""
    print(f"  → {descricao}")
    try:
        resultado = subprocess.run(
            comando, cwd=RAIZ, capture_output=True, text=True, timeout=1800
        )
    except (OSError, subprocess.TimeoutExpired) as erro:
        print(f"    falhou: {erro}")
        return False
    if resultado.returncode != 0:
        cauda = (resultado.stderr or resultado.stdout or "").strip().splitlines()
        for linha in cauda[-6:]:
            print(f"    {linha}")
        return False
    return True


def notebook_foi_executado() -> bool:
    """``True`` se o notebook tem saídas — ou seja, foi executado.

    Gerar PDF de notebook sem saída produziria um documento com o código e
    nenhuma evidência, que é exatamente o contrário do que o trabalho pede.
    """
    import nbformat

    nb = nbformat.read(NOTEBOOK, as_version=4)
    codigo = [celula for celula in nb.cells if celula.cell_type == "code"]
    com_saida = [celula for celula in codigo if celula.outputs]
    print(f"  {len(com_saida)} de {len(codigo)} células de código têm saída")
    erros = [
        (indice, saida.get("ename"))
        for indice, celula in enumerate(nb.cells)
        for saida in celula.get("outputs", [])
        if saida.get("output_type") == "error"
    ]
    if erros:
        print(f"  ATENÇÃO: {len(erros)} célula(s) com erro: {erros[:5]}")
        return False
    return len(com_saida) > len(codigo) * 0.9


def gerar_html() -> bool:
    """Notebook → HTML. É a base dos três caminhos."""
    return _rodar(
        [
            sys.executable, "-m", "nbconvert", "--to", "html",
            "--output", SAIDA_HTML.name, "--output-dir", str(RAIZ),
            str(NOTEBOOK),
        ],
        "nbconvert → HTML",
    )


#: Quantas colunas de texto monoespaçado a folha impressa precisa acomodar.
#: Medido no notebook: mediana 63, percentil 95 igual a 141, percentil 99,5 igual
#: a 197. Em 190 sobram 17 linhas de 2.245 (0,8%) para a quebra automática.
LARGURA_ALVO_COLUNAS = 190

#: CSS aplicado só na impressão. Ver a explicação no docstring do módulo.
CSS_IMPRESSAO = """
<style id="css-impressao-dr1at">
@page { size: A4 landscape; margin: 8mm 8mm 11mm 8mm; }

/* A causa raiz do corte: na tela `overflow: auto` dá barra de rolagem, mas na
   impressão simplesmente descarta o que passa da borda. */
.jp-OutputArea, .jp-OutputArea-output, .jp-OutputArea-child,
.jp-Cell-outputWrapper, .jp-RenderedText, .jp-InputArea, div.output_subarea {
  overflow: visible !important;
  max-width: none !important;
}

/* Código e tabelas de medição na largura cheia da folha em paisagem. */
pre, code, kbd, samp,
.jp-InputArea-editor, .highlight pre, .jp-RenderedText pre,
.jp-OutputArea-output pre, div.output_subarea pre {
  font-size: 7pt !important;
  line-height: 1.25 !important;
  white-space: pre-wrap !important;   /* rede de segurança: nunca perder texto */
  overflow-wrap: break-word !important;
  tab-size: 4;
}

/* Prosa em coluna estreita: linha de texto corrida com 281mm é ilegível, e a
   análise em Big O é justamente texto que o professor precisa ler. */
.jp-RenderedMarkdown { max-width: 185mm; }
.jp-RenderedMarkdown pre, .jp-RenderedMarkdown table { max-width: none; }

/* Não partir gráfico, tabela ou saída no meio de uma quebra de página. */
.jp-OutputArea-child, .jp-RenderedImage, table, tr, img { break-inside: avoid; }
h1, h2, h3 { break-after: avoid; page-break-after: avoid; }
h1 { break-before: page; page-break-before: page; }
h1:first-of-type { break-before: auto; page-break-before: auto; }

img { max-width: 100% !important; height: auto !important; }

/* Os gráficos têm fundo claro; sem isto o Chrome pode descartá-lo. */
body { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
</style>
"""


def injetar_css_de_impressao() -> bool:
    """Insere :data:`CSS_IMPRESSAO` no ``<head>`` do HTML já gerado.

    Injetar depois do nbconvert, em vez de usar um template próprio, mantém o
    HTML padrão do Jupyter intacto — o arquivo continua abrindo e sendo lido
    normalmente no navegador, e o CSS só muda o que a impressão faz.

    Idempotente: se o CSS já está lá, não duplica.
    """
    html = SAIDA_HTML.read_text(encoding="utf-8")
    if 'id="css-impressao-dr1at"' in html:
        print("  → CSS de impressão já presente")
        return True
    if "</head>" not in html:
        print("  → não achei </head> no HTML; seguindo sem o CSS")
        return False
    html = html.replace("</head>", CSS_IMPRESSAO + "</head>", 1)
    SAIDA_HTML.write_text(html, encoding="utf-8")
    print(f"  → CSS de impressão injetado (paisagem, 7pt, "
          f"alvo de {LARGURA_ALVO_COLUNAS} colunas)")
    return True


#: Navegadores que sabem imprimir PDF por linha de comando, em ordem de
#: preferência. Chrome e Edge já vêm instalados na máquina deste trabalho, então
#: este caminho não exige instalar nada — por isso é o primeiro tentado.
NAVEGADORES: tuple[str, ...] = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)


def encontrar_navegador() -> str | None:
    """Primeiro navegador disponível, do PATH ou dos caminhos padrão."""
    for nome in ("chrome", "msedge", "chromium"):
        caminho = shutil.which(nome)
        if caminho:
            return caminho
    for caminho in NAVEGADORES:
        if Path(caminho).exists():
            return caminho
    return None


def tentar_navegador_headless() -> bool:
    """Caminho preferido: Chrome/Edge em modo headless, zero instalação.

    ``--virtual-time-budget`` dá tempo de a página montar antes de imprimir: o
    HTML tem dezenas de imagens embutidas em base64, e sem essa folga o PDF sai
    com os gráficos em branco.
    """
    navegador = encontrar_navegador()
    if navegador is None:
        print("  → nenhum navegador Chromium encontrado")
        return False
    nome = Path(navegador).stem
    if SAIDA_PDF.exists():
        SAIDA_PDF.unlink()
    sucesso = _rodar(
        [
            navegador, "--headless=new", "--disable-gpu", "--no-sandbox",
            "--no-pdf-header-footer", "--virtual-time-budget=60000",
            f"--print-to-pdf={SAIDA_PDF}", SAIDA_HTML.as_uri(),
        ],
        f"{nome} headless → PDF",
    )
    if not sucesso:
        return False
    if not SAIDA_PDF.exists() or SAIDA_PDF.stat().st_size < 50_000:
        print("    o PDF saiu vazio ou pequeno demais; descartando")
        return False
    return True


def tentar_webpdf() -> bool:
    """Caminho preferido: Chromium via Playwright, PDF navegável."""
    try:
        import playwright  # noqa: F401
    except ImportError:
        print("  → webpdf indisponível (playwright não instalado)")
        return False
    return _rodar(
        [
            sys.executable, "-m", "nbconvert", "--to", "webpdf",
            "--allow-chromium-download",
            "--output", SAIDA_PDF.stem, "--output-dir", str(RAIZ),
            str(NOTEBOOK),
        ],
        "nbconvert → webpdf (Chromium)",
    )


def tentar_wkhtmltopdf() -> bool:
    """Segundo caminho: wkhtmltopdf, se estiver no PATH."""
    executavel = shutil.which("wkhtmltopdf")
    if executavel is None:
        print("  → wkhtmltopdf não encontrado no PATH")
        return False
    return _rodar(
        [
            executavel, "--enable-local-file-access",
            "--outline", "--outline-depth", "3",
            "--print-media-type", "--page-size", "A4",
            str(SAIDA_HTML), str(SAIDA_PDF),
        ],
        "wkhtmltopdf → PDF",
    )


def _titulos_do_notebook() -> list[str]:
    """Títulos de nível 1 do notebook, na ordem em que aparecem.

    A lista de marcadores vem do próprio notebook em vez de ser redigitada aqui,
    então renomear uma seção não deixa o índice do PDF desatualizado.
    """
    import nbformat

    nb = nbformat.read(NOTEBOOK, as_version=4)
    titulos: list[str] = []
    for celula in nb.cells:
        if celula.cell_type != "markdown":
            continue
        for linha in celula.source.splitlines():
            if linha.startswith("# "):
                titulos.append(" ".join(linha[2:].split()))
    return titulos


def _normalizar(texto: str) -> str:
    """Colapsa espaços, porque a extração de texto do PDF quebra linha sozinha."""
    return " ".join(texto.split())


def adicionar_marcadores() -> bool:
    """Adiciona ao PDF um marcador por seção, para o painel de navegação.

    O Chrome headless imprime sem marcador nenhum: em 82 páginas, isso obriga o
    professor a navegar rolando. O enunciado pede PDF navegável, então os
    marcadores entram aqui, depois da impressão.

    A busca é monotônica — cada título só é procurado a partir da página onde o
    anterior foi achado. Sem isso, o marcador do "Exercício 10" cairia na página
    da introdução, que menciona todos os exercícios antes de qualquer um começar.
    """
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        print("  → pypdf não instalado; PDF fica sem marcadores "
              "(pip install pypdf)")
        return False

    titulos = _titulos_do_notebook()
    if not titulos:
        print("  → nenhum título de nível 1 no notebook")
        return False

    leitor = PdfReader(SAIDA_PDF)
    texto_por_pagina = [
        _normalizar(pagina.extract_text() or "") for pagina in leitor.pages
    ]

    escritor = PdfWriter(clone_from=str(SAIDA_PDF))
    pagina_minima = 0
    encontrados = 0
    for titulo in titulos:
        alvo = _normalizar(titulo)
        # Tenta o título inteiro; se a extração mangueou o travessão, tenta um
        # prefixo cada vez mais curto, até 12 caracteres.
        candidatos = [alvo]
        for corte in (40, 25, 12):
            if len(alvo) > corte:
                candidatos.append(alvo[:corte])
        pagina_achada = None
        for candidato in candidatos:
            for indice in range(pagina_minima, len(texto_por_pagina)):
                if candidato in texto_por_pagina[indice]:
                    pagina_achada = indice
                    break
            if pagina_achada is not None:
                break
        if pagina_achada is None:
            print(f"    aviso: não localizei a página de {titulo!r}")
            continue
        escritor.add_outline_item(titulo, pagina_achada)
        pagina_minima = pagina_achada
        encontrados += 1

    escritor.write(str(SAIDA_PDF))
    print(f"  → {encontrados} de {len(titulos)} marcadores adicionados")
    return encontrados > 0


def _linhas_largas_do_notebook(minimo: int) -> list[str]:
    """Linhas de código e de saída do notebook com mais de ``minimo`` colunas."""
    import nbformat

    ansi = re.compile(r"\x1b\[[0-9;]*m")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    largas: list[str] = []
    for celula in nb.cells:
        blocos: list[str] = []
        if celula.cell_type == "code":
            blocos.append(celula.source)
        for saida in celula.get("outputs", []):
            dado = saida.get("text") or (saida.get("data") or {}).get("text/plain")
            if isinstance(dado, list):
                dado = "".join(dado)
            if isinstance(dado, str):
                blocos.append(dado)
        for bloco in blocos:
            for linha in bloco.splitlines():
                limpa = ansi.sub("", linha).rstrip()
                # O pandas quebra DataFrame largo sozinho e marca a continuação
                # com "\" no fim da linha. Esse caractere não sobrevive colado ao
                # texto na extração do PDF, e mantê-lo geraria alarme falso.
                limpa = limpa.rstrip("\\").rstrip()
                if len(limpa) > minimo:
                    largas.append(limpa)
    return largas


def conferir_nada_cortado(amostras: int = 25) -> bool:
    """Confere que o final das linhas mais largas sobreviveu no PDF.

    Este é o teste de regressão do bug que motivou o CSS de impressão. Pega as
    linhas mais largas do notebook, procura as últimas palavras de cada uma no
    texto extraído do PDF e reprova se alguma desapareceu.

    Custo: O(p + a·t), com ``p`` páginas, ``a`` amostras e ``t`` o texto do PDF.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        print("  → pypdf não instalado; não consigo conferir "
              "(pip install pypdf)")
        return True

    largas = _linhas_largas_do_notebook(minimo=110)
    if not largas:
        print("  → nenhuma linha larga a conferir")
        return True
    largas.sort(key=len, reverse=True)

    texto = _normalizar(
        "".join(
            (pagina.extract_text() or "") for pagina in PdfReader(SAIDA_PDF).pages
        )
    )

    perdidas: list[str] = []
    conferidas = 0
    for linha in largas[:amostras]:
        cauda = " ".join(linha.split()[-4:])
        if len(cauda) < 8:
            continue
        conferidas += 1
        if cauda not in texto:
            perdidas.append(cauda)

    if perdidas:
        print(f"  ATENÇÃO: {len(perdidas)} de {conferidas} linhas largas "
              f"perderam o final no PDF. Exemplos:")
        for cauda in perdidas[:4]:
            print(f"    ...{cauda}")
        return False
    print(f"  → {conferidas} linhas largas conferidas, nenhuma cortada "
          f"(a mais larga tem {len(largas[0])} colunas)")
    return True


def main() -> int:
    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument(
        "--pular-verificacao", action="store_true",
        help="gera o PDF mesmo que o notebook não pareça executado",
    )
    argumentos = analisador.parse_args()

    print(f"notebook: {NOTEBOOK}")
    if not NOTEBOOK.exists():
        print("ERRO: notebook não encontrado.")
        return 1

    print("\n[1/5] conferindo se o notebook foi executado")
    if not notebook_foi_executado() and not argumentos.pular_verificacao:
        print(
            "\nERRO: o notebook não está executado (ou tem células com erro).\n"
            "Rode primeiro:\n"
            "  python -m nbconvert --to notebook --execute --inplace "
            "notebook/DR1_AT.ipynb\n"
            "Ou repita com --pular-verificacao se souber o que está fazendo."
        )
        return 1

    print("\n[2/5] gerando HTML")
    if not gerar_html():
        print("ERRO: não consegui gerar o HTML.")
        return 1
    print(f"  HTML salvo em {SAIDA_HTML.name} "
          f"({SAIDA_HTML.stat().st_size / 1_048_576:.1f} MB)")

    print("\n[3/5] aplicando o CSS de impressão")
    injetar_css_de_impressao()

    print("\n[4/5] gerando PDF")
    if tentar_navegador_headless() or tentar_webpdf() or tentar_wkhtmltopdf():
        print(f"  PDF gerado ({SAIDA_PDF.stat().st_size / 1_048_576:.1f} MB)")

        print("\n[5/5] conferindo o PDF")
        adicionar_marcadores()
        intacto = conferir_nada_cortado()

        print(f"\nPRONTO: {SAIDA_PDF.name} "
              f"({SAIDA_PDF.stat().st_size / 1_048_576:.1f} MB)")
        if not intacto:
            print(
                "AVISO: há conteúdo cortado no PDF. Confira o CSS de impressão "
                "ou reduza a largura das tabelas mais largas antes de entregar."
            )
            return 1
        return 0

    print(
        f"\nNenhum conversor automático funcionou — mas o HTML está pronto.\n"
        f"\nPara fechar o PDF em um minuto:\n"
        f"  1. abra {SAIDA_HTML.name} no navegador\n"
        f"  2. Ctrl+P → Destino: 'Salvar como PDF'\n"
        f"  3. marque 'Gráficos de fundo', margens 'Padrão'\n"
        f"  4. salve como {SAIDA_PDF.name} na raiz do projeto\n"
        f"\nO PDF sai navegável, com os títulos de cada exercício visíveis.\n"
        f"\nPara automatizar nas próximas vezes:\n"
        f'  pip install "nbconvert[webpdf]" && playwright install chromium'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

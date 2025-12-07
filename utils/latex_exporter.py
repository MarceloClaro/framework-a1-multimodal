import subprocess
from pathlib import Path
from typing import Optional


OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


def build_latex_document(article_text: str) -> str:
    """Monta o conteúdo LaTeX a partir do texto do artigo."""
    safe_text = article_text.replace("_", "\\_")
    latex = rf"""
\documentclass[12pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage[brazil]{{babel}}
\usepackage{{geometry}}
\usepackage{{hyperref}}
\geometry{{margin=2.5cm}}

\title{{Framework Multimodal com Agentes Evolutivos}}
\author{{Marcelo Claro et al.}}
\date{{\today}}

\begin{{document}}

\maketitle

\section*{{Artigo Gerado Automaticamente}}
{safe_text}

\end{{document}}
"""
    return latex


def save_latex_file(article_text: str, filename: str = "artigo.tex") -> Path:
    """Salva o artigo em arquivo .tex."""
    tex_source = build_latex_document(article_text)
    tex_path = OUTPUT_DIR / filename
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_source)
    return tex_path


def compile_pdf(tex_path: Path) -> Optional[Path]:
    """Compila o arquivo .tex em PDF usando pdflatex. Retorna caminho do PDF ou None."""
    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path.name],
            cwd=tex_path.parent,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        pdf_path = tex_path.with_suffix(".pdf")
        return pdf_path if pdf_path.exists() else None
    except Exception:
        return None

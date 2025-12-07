import streamlit as st
from utils.article_builder import generate_article
from utils.exporter import save_json, save_csv
from utils.latex_exporter import save_latex_file, compile_pdf

st.title("📜 Gerador Automático de Artigo Científico (Qualis A1)")

# Verifica se relatórios existem
if "reports" not in st.session_state:
    st.warning("Primeiro execute os agentes para gerar os relatórios.")
    st.stop()

reports = st.session_state["reports"]

# Botão para gerar artigo
if st.button("Gerar artigo completo"):
    article = generate_article(reports)
    st.session_state["article"] = article
    st.success("Artigo gerado com sucesso!")

# Exibe o texto do artigo
article_text = st.session_state.get("article", "")
st.text_area("Artigo gerado:", value=article_text, height=700)

# Colunas para exportação
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Salvar artigo em JSON"):
        path = save_json({"artigo": article_text}, "artigo.json")
        st.success(f"Artigo salvo em {path}")

with col2:
    if st.button("Exportar LaTeX (.tex)"):
        tex_path = save_latex_file(article_text, "artigo.tex")
        st.success(f"LaTeX salvo em {tex_path}")
        with open(tex_path, "rb") as f:
            st.download_button(
                "Baixar artigo.tex", data=f.read(), file_name="artigo.tex", mime="application/x-tex"
            )

with col3:
    if st.button("Gerar PDF"):
        tex_path = save_latex_file(article_text, "artigo.tex")
        pdf_path = compile_pdf(tex_path)
        if pdf_path:
            st.success(f"PDF gerado em {pdf_path}")
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Baixar artigo.pdf", data=f.read(), file_name="artigo.pdf", mime="application/pdf"
                )
        else:
            st.error("Falha na compilação do PDF. Verifique se o pdflatex está instalado no sistema.")

import streamlit as st
from utils.metrics import compute_classification_metrics

st.title("📈 Estatística & Métricas")

results = st.session_state.get("results")
if not results:
    st.warning("Nenhum resultado disponível. Execute a classificação primeiro.")
    st.stop()

task_type = st.session_state.get("task_type", "binária")
classes = st.session_state.get("classes", [])

if task_type not in ["binária", "multiclasse"]:
    st.info("Para classificação multirrótulo conceitual, não há métricas quantitativas padrão.")
    st.stop()

st.subheader("Rótulos verdadeiros (um por linha, na mesma ordem dos arquivos carregados)")
labels_input = st.text_area("Rótulos verdadeiros:")
if st.button("Calcular métricas"):
    y_true = [l.strip() for l in labels_input.split("\n") if l.strip()]
    if len(y_true) != len(results):
        st.error("Número de rótulos verdadeiros diferente do número de previsões.")
    else:
        y_pred = [r.get("predicted_label") for r in results]
        metrics_dict = compute_classification_metrics(y_true, y_pred, task_type, classes)
        st.session_state["metrics_dict"] = metrics_dict
        st.json(metrics_dict)

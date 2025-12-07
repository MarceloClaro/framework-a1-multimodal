import streamlit as st
import pandas as pd
import plotly.express as px
from utils.metrics import compute_classification_metrics

st.title("📈 Dashboard de Resultados")

results = st.session_state.get("results")
if not results:
    st.warning("Nenhum resultado disponível. Execute a classificação primeiro.")
    st.stop()

# Constrói DataFrame com previsões
df = pd.DataFrame([
    {
        "file_name": r["file_name"],
        "hash": r["hash"],
        "predicted_label": r.get("predicted_label"),
    }
    for r in results
])

st.subheader("Distribuição das classes previstas")
count_df = df["predicted_label"].value_counts().reset_index()
count_df.columns = ["Classe", "Contagem"]
fig = px.bar(count_df, x="Classe", y="Contagem")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Matriz de confusão")
labels_input = st.text_area("Rótulos verdadeiros (um por linha na mesma ordem dos arquivos carregados):")
if st.button("Gerar matriz de confusão"):
    true_labels = [l.strip() for l in labels_input.split("\n") if l.strip()]
    if len(true_labels) != len(df):
        st.error("Número de rótulos verdadeiros diferente do número de previsões.")
    else:
        # Define conjunto completo de classes
        all_classes = sorted(list(set(true_labels + df["predicted_label"].tolist())))
        # Determina tipo de tarefa para métricas
        task_type = "multiclasse" if len(all_classes) > 2 else "binária"
        metrics_dict = compute_classification_metrics(true_labels, df["predicted_label"].tolist(), task_type, all_classes)
        cm = metrics_dict["confusion_matrix"]
        cm_df = pd.DataFrame(cm).T
        fig_cm = px.imshow(
            cm_df.values,
            x=cm_df.columns,
            y=cm_df.index,
            labels={"x": "Predito", "y": "Verdadeiro", "color": "Contagem"},
            text_auto=True,
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.json({"accuracy": metrics_dict["accuracy"], "macro_f1": metrics_dict.get("macro_f1")})

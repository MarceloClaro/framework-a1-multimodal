import streamlit as st
from agents.analyst import (
    scientific_analysis,
    critic_review,
    replication_protocol,
    stats_report,
    evolutionary_agent,
)
from agents.specialists import clinical_agent, industrial_agent, pedagogical_agent
from utils.metrics import compute_classification_metrics
from utils.db import log_run

st.title("🤠 Agentes Inteligentes")

# Verifica se há resultados disponíveis
results = st.session_state.get("results")
if not results:
    st.warning("Nenhum resultado de classificação disponível. Execute a classificação primeiro.")
    st.stop()

# Entrada da chave da API
api_key = st.text_input("Insira novamente a Gemini API Key:", type="password")

# Recupera parâmetros da sessão
domain = st.session_state.get("domain_choice", "geral")
task_type = st.session_state.get("task_type", "binária")
classes = st.session_state.get("classes", [])

# Possível cálculo de métricas
metrics_dict = None
if task_type in ["binária", "multiclasse"]:
    st.subheader("Rótulos verdadeiros (opcional, um por linha)")
    labels_input = st.text_area("Rótulos verdadeiros:", "")
    true_labels = [l.strip() for l in labels_input.split("\n") if l.strip()]
    if st.button("Calcular métricas"):
        if len(true_labels) != len(results):
            st.error("Número de rótulos verdadeiros diferente do número de previsões.")
        else:
            y_pred = [r.get("predicted_label") for r in results]
            metrics_dict = compute_classification_metrics(true_labels, y_pred, task_type, classes)
            st.session_state["metrics_dict"] = metrics_dict
            st.json(metrics_dict)

# Executa agentes quando acionado
if st.button("Executar agentes"):
    # Garante que métricas anteriores sejam usadas se disponíveis
    metrics_to_pass = metrics_dict or st.session_state.get("metrics_dict")

    # Executa cada agente
    report_scientific = scientific_analysis(results, task_type, classes, api_key, domain)
    report_critic = critic_review(results, task_type, classes, api_key, domain)
    report_repl = replication_protocol(results, task_type, classes, api_key, domain)
    report_stats = stats_report(results, task_type, classes, metrics_to_pass, api_key, domain)
    evo_report = evolutionary_agent(
        {
            "científico": report_scientific,
            "crítico": report_critic,
            "replicação": report_repl,
            "estatística": report_stats,
        },
        api_key=api_key,
        domain=domain,
    )
    specialist_report = ""
    if domain == "medicina":
        specialist_report = clinical_agent(results, api_key)
    elif domain == "indústria":
        specialist_report = industrial_agent(results, api_key)
    elif domain == "educação":
        specialist_report = pedagogical_agent(results, api_key)

    # Compila todos os relatórios
    reports = {
        "científico": report_scientific,
        "crítico": report_critic,
        "replicação": report_repl,
        "estatística": report_stats,
        "evolutivo": evo_report,
        "especializado": specialist_report,
    }
    st.session_state["reports"] = reports

    # Registra a execução no banco de dados
    try:
        run_id = log_run(
            domain=domain,
            task_type=task_type,
            model_name=st.session_state.get("model_choice", "gemini-1.5-pro"),
            results=results,
            metrics_dict=metrics_to_pass,
            reports=reports,
        )
        st.info(f"Execução registrada com run_id = {run_id}")
    except Exception as e:
        st.error(f"Erro ao registrar execução: {e}")

    st.success("Agentes executados com sucesso!")
    st.json(reports)

import streamlit as st
from agents.classifier import classify_objects, classify_objects_parallel
from utils.hashing import hash_file
from utils.parallel import init_ray
from utils.training import load_model, predict_images
import os

st.title("Classificação Multimodal")

# Inicializa resultados no session state
if "results" not in st.session_state:
    st.session_state["results"] = None

# Captura chave da API
api_key = st.text_input("Insira sua Gemini API Key:", type="password")

# Selecao do modelo LLM (somente modelo suportado)
model_choice = st.selectbox(
    "Seleção do Modelo:",
    ["gemini-1.5-pro"]
)

# Tipo de tarefa
task_type = st.selectbox(
    "Tipo de classificação:",
    ["binária", "multiclasse", "multirrotulo (conceitual)"]
)

# Dominio de aplicação
domain_choice = st.selectbox(
    "Domínio:",
    ["geral", "medicina", "indústria", "educação"]
)

# classes possíveis input
classes_str = st.text_input("Classes possíveis (separadas por vírgula):")
classes = [c.strip() for c in classes_str.split(",") if c.strip()]

# Carregamento de arquivos
uploaded_files = st.file_uploader(
    "Envie arquivos para classificação:",
    accept_multiple_files=True
)

# Opção para usar modelo treinado local
MODEL_PATH = os.path.join("results", "trained_model.pkl")
use_trained_model = False
if os.path.exists(MODEL_PATH):
    use_trained_model = st.checkbox("Usar modelo treinado", value=False)

# Exibição de resultados se já houver
if st.session_state["results"]:
    st.subheader("Resultados de classificação (saída do ClassifierAgent)")
    for idx, r in enumerate(st.session_state["results"], start=1):
        st.markdown(f"#### Objeto {idx}: `{r['file_name']}`")
        st.markdown(f"- Hash: `{r['hash']}`")
        st.markdown(f"- Classe prevista: **{r.get('predicted_label', 'N/A')}**")
        if r.get("raw_model_output"):
            with st.expander("Saída do modelo"):
                st.code(r.get("raw_model_output"), language="json")
    st.markdown("---")

# Classificação dos arquivos
if uploaded_files and api_key:
    st.subheader("Arquivos enviados")
    file_data = []
    for file in uploaded_files:
        file_hash = hash_file(file)
        st.write(f"Arquivo: **{file.name}** | Hash: `{file_hash}`")
        file_data.append({"file": file, "hash": file_hash})

    # Botão para processar
    if st.button("Classificar arquivos"):
        if use_trained_model:
            # Carrega modelo treinado
            loaded = load_model(MODEL_PATH)
            if loaded is not None:
                model, scaler, model_classes = loaded
                st.info("Usando modelo treinado para classificação")
                results = []
                for item in file_data:
                    preds, probs = predict_images(
                        model=model,
                        scaler=scaler,
                        image_files=[item["file"]],
                        classes=model_classes,
                        image_size=64
                    )
                    pred_label = preds[0]
                    prob_dict = {model_classes[i]: float(probs[0][i]) for i in range(len(model_classes))}
                    results.append({
                        "file_name": item["file"].name,
                        "hash": item["hash"],
                        "predicted_label": pred_label,
                        "candidate_labels": model_classes,
                        "estimated_confidence": prob_dict.get(pred_label, 0.0),
                        "raw_model_output": str(prob_dict)
                    })
                st.session_state["results"] = results
                st.success("Classificação com modelo treinado concluída!")
            else:
                st.error("Falha ao carregar modelo treinado. Verifique se o treinamento foi concluído.")
        else:
            with st.spinner("Executando classificação com LLM..."):
                init_ray()
                results = classify_objects(
                    files=file_data,
                    api_key=api_key,
                    model_name=model_choice,
                    task_type=task_type,
                    classes=classes,
                )
                st.session_state["results"] = results
                st.success("Classificação concluída com LLM!")

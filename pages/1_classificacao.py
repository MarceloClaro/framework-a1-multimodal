import streamlit as st
from agents.classifier import classify_objects, classify_objects_parallel
from utils.hashing import hash_file
from utils.parallel import init_ray
from utils.training import load_model, predict_images
import os

st.title("\ud83d\udd0d Classificacao Multimodal")

# Inicializa resultados no estado de sessao
if "results" not in st.session_state:
    st.session_state["results"] = None

# Captura chave da API
api_key = st.text_input("Insira sua Gemini API Key:", type="password")

# Selecao do modelo de linguagem
model_choice = st.selectbox("Modelo:", ["gemini-1.5-flash", "gemini-1.5-pro"])

# Tipo de tarefa
task_type = st.selectbox(
    "Tipo de classificacao:",
    ["binaria", "multiclasse", "multirotulo (conceitual)"]
)

# Dominio de aplicacao
domain_choice = st.selectbox(
    "Dominio:",
    ["geral", "medicina", "industria", "educacao"]
)

# Definicao das classes
default_classes = "positivo, negativo" if task_type == "binaria" else ""
classes_str = st.text_input("Classes possiveis (separadas por virgula):", value=default_classes)
classes = [c.strip() for c in classes_str.split(",") if c.strip()]

# Upload de arquivos
uploaded_files = st.file_uploader(
    "Envie multiplos objetos (imagens, textos, PDFs):", accept_multiple_files=True
)

# Verifica se ha modelo treinado disponivel
MODEL_PATH = os.path.join("results", "trained_model.pkl")
model_available = os.path.exists(MODEL_PATH)
use_trained_model = False
if model_available:
    use_trained_model = st.checkbox(
        "Usar modelo treinado (em vez do LLM)", value=True
    )

if uploaded_files and api_key:
    file_data = []
    st.write("Arquivos enviados:")
    for file in uploaded_files:
        file_hash = hash_file(file)
        st.write(f"{file.name} | Hash: `{file_hash}`")
        file_data.append({"file": file, "hash": file_hash})

    # Opcao para execucao paralela
    use_parallel = st.checkbox("Usar execucao paralela com Ray?", value=False)

    if st.button("Classificar"):
        results = []
        model_dict = None
        with st.spinner("Classificando objetos..."):
            if use_trained_model and model_available:
                model_dict = load_model(MODEL_PATH)
                if model_dict:
                    image_files = [item["file"] for item in file_data]
                    preds = predict_images(model_dict, image_files)
                    for p in preds:
                        results.append({
                            "file_name": p["file_name"],
                            "hash": next((fd["hash"] for fd in file_data if fd["file"].name == p["file_name"]), None),
                            "predicted_label": p["predicted_label"],
                            "probabilities": p["probabilities"],
                        })
                else:
                    st.warning("Modelo treinado nao foi encontrado. Usando LLM.")
            # Se nao usar modelo treinado ou se falhou em carregar
            if not use_trained_model or not model_available or not model_dict:
                if use_parallel:
                    init_ray()
                    results = classify_objects_parallel(
                        files=file_data,
                        api_key=api_key,
                        model_name=model_choice,
                        task_type=task_type,
                        classes=classes,
                    )
                else:
                    results = classify_objects(
                        files=file_data,
                        api_key=api_key,
                        model_name=model_choice,
                        task_type=task_type,
                        classes=classes,
                    )
        # Armazena resultados e contexto no estado de sessao
        st.session_state["results"] = results
        st.session_state["task_type"] = task_type
        st.session_state["model_choice"] = model_choice
        st.session_state["domain_choice"] = domain_choice
        st.session_state["classes"] = classes
        st.session_state["used_trained_model"] = use_trained_model and model_dict is not None
        st.success("Classificacao concluida!")

# Exibe resultados se disponiveis
if st.session_state.get("results"):
    st.subheader("Resultados de classificacao")
    st.json(st.session_state["results"])

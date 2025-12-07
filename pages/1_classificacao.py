import streamlit as st
from agents.classifier import classify_objects, classify_objects_parallel
from utils.hashing import hash_file
from utils.parallel import init_ray

st.title("🔍 Classificação Multimodal")

# Inicializa resultados no estado de sessão
if "results" not in st.session_state:
    st.session_state["results"] = None

# Captura chave da API
api_key = st.text_input("Insira sua Gemini API Key:", type="password")

# Seleção do modelo de linguagem
model_choice = st.selectbox("Modelo:", ["gemini-1.5-flash", "gemini-1.5-pro"])

# Tipo de tarefa
task_type = st.selectbox(
    "Tipo de classificação:",
    ["binária", "multiclasse", "multirrótulo (conceitual)"]
)

# Domínio de aplicação
domain_choice = st.selectbox(
    "Domínio:",
    ["geral", "medicina", "indústria", "educação"]
)

# Definição das classes
default_classes = "positivo, negativo" if task_type == "binária" else ""
classes_str = st.text_input("Classes possíveis (separadas por vírgula):", value=default_classes)
classes = [c.strip() for c in classes_str.split(",") if c.strip()]

# Upload de arquivos
uploaded_files = st.file_uploader(
    "Envie múltiplos objetos (imagens, textos, PDFs):", accept_multiple_files=True
)

if uploaded_files and api_key:
    file_data = []
    st.write("Arquivos enviados:")
    for file in uploaded_files:
        file_hash = hash_file(file)
        st.write(f"{file.name} | Hash: `{file_hash}`")
        file_data.append({"file": file, "hash": file_hash})

    # Opção para execução paralela
    use_parallel = st.checkbox("Usar execução paralela com Ray?", value=False)

    if st.button("Classificar"):
        with st.spinner("Classificando objetos..."):
            if use_parallel:
                # Inicializa Ray uma única vez
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
        # Armazena no estado de sessão
        st.session_state["results"] = results
        st.session_state["task_type"] = task_type
        st.session_state["model_choice"] = model_choice
        st.session_state["domain_choice"] = domain_choice
        st.session_state["classes"] = classes
        st.success("Classificação concluída!")

# Exibe resultados se disponíveis
if st.session_state.get("results"):
    st.subheader("Resultados de classificação")
    st.json(st.session_state["results"])

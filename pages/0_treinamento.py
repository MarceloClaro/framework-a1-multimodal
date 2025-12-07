import os
import streamlit as st
from typing import List, Tuple

from utils.training import (
    load_dataset,
    split_dataset,
    train_mlp_classifier,
    evaluate_model,
    save_model,
    hyperparameter_search,
)

st.set_page_config(page_title="Treinamento do Modelo", layout="wide")
st.title("Treinamento do Modelo de Classificacao em Dermatologia")

st.markdown(
    """
    Esta página permite treinar um modelo de classificação de imagens de pele para uso em residência médica de dermatologia.
    O modelo utiliza um perceptron multicamadas (MLP) do scikit‑learn, com early stopping e possibilidade de balanceamento de classes por meio de data augmentation.
    Por padrão, o dataset é carregado do repositório Hugging Face (HAM10000) e as classes são balanceadas via data augmentation.
    """
)

# Input parameters
dataset_path = st.text_input(
    "Diretório do dataset (ou URL hf://)",
    value="hf://datasets/marmal88/skin_cancer/",
    help="Caminho local para o diretório com subpastas por classe ou URL hf:// para o dataset"
)
augment = st.checkbox(
    "Balancear classes via data augmentation (equaliza classes)",
    value=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    test_size = st.slider("Proporção do conjunto de teste", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
with col2:
    val_size = st.slider(
        "Proporção do conjunto de validação (dentro do restante)", min_value=0.05, max_value=0.3, value=0.1, step=0.05
    )
with col3:
    image_size = st.number_input(
        "Lado das imagens (pixels)", min_value=32, max_value=128, value=64, step=16,
        help="Imagens serão redimensionadas para N×N pixels"
    )

hidden_layers_str = st.text_input(
    "Tamanhos das camadas ocultas (separadas por vírgula)", value="256,128",
    help="Exemplo: 256,128 cria duas camadas de 256 e 128 neurônios"
)
learning_rate = st.number_input(
    "Taxa de aprendizado (learning rate)", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, format="%f"
)
batch_size = st.number_input(
    "Tamanho do batch", min_value=16, max_value=512, value=64, step=16
)
max_epochs = st.number_input(
    "Número máximo de épocas", min_value=5, max_value=200, value=50, step=5
)
early_stopping_patience = st.number_input(
    "Paciência para early stopping", min_value=1, max_value=20, value=5, step=1
)

# Option to perform hyperparameter search
search_hyperparams = st.checkbox(
    "Executar busca de hiperparâmetros", value=False,
    help="Se marcada, será executada uma busca simples sobre combinações de parâmetros."
)

if search_hyperparams:
    # Inputs for hyperparameter options
    hidden_layers_options_str = st.text_input(
        "Opções de camadas ocultas (separe combinações por ponto e vírgula)",
        value="256,128;512,256;128,64",
        help="Cada combinação é uma sequência de inteiros separados por vírgula; diferentes combinações são separadas por ponto e vírgula."
    )
    learning_rate_options_str = st.text_input(
        "Opções de learning rates (separadas por ponto e vírgula)",
        value="0.001;0.0005",
    )
    batch_size_options_str = st.text_input(
        "Opções de tamanhos de batch (separados por ponto e vírgula)",
        value="32;64",
    )

train_button = st.button("Treinar modelo")

if train_button:
    # Ensure the dataset path exists or is remote
    if not dataset_path.strip().lower().startswith("hf://") and not os.path.isdir(dataset_path):
        st.error(f"Diretório '{dataset_path}' não encontrado. Por favor, verifique o caminho ou prefixo hf://")
    else:
        try:
            with st.spinner("Carregando e preparando o dataset..."):
                # Se o caminho começar com 'hf://', use carregador HuggingFace via Polars
                if dataset_path.strip().lower().startswith("hf://"):
                    from utils.training import load_dataset_from_hf
                    # Use apenas o split de treino; split_dataset fará a divisão
                    X, y, class_names = load_dataset_from_hf(
                        base_url=dataset_path,
                        split="train",
                        image_size=(int(image_size), int(image_size)),
                        augment=augment,
                    )
                else:
                    X, y, class_names = load_dataset(
                        dataset_path,
                        image_size=(int(image_size), int(image_size)),
                        augment=augment,
                    )
                X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
                    X,
                    y,
                    test_size=float(test_size),
                    val_size=float(val_size),
                    random_state=42,
                )

            # Hyperparameter search or single training
            if search_hyperparams:
                # Parse options
                # Hidden layers options: split by ';' then parse each comma-separated tuple
                hidden_options: List[Tuple[int, ...]] = []
                for combo in hidden_layers_options_str.split(";"):
                    combo = combo.strip()
                    if combo:
                        try:
                            hl = tuple(int(x.strip()) for x in combo.split(",") if x.strip())
                            hidden_options.append(hl)
                        except Exception:
                            continue
                lr_options = []
                for lr_str in learning_rate_options_str.split(";"):
                    lr_str = lr_str.strip()
                    if lr_str:
                        try:
                            lr_options.append(float(lr_str))
                        except Exception:
                            continue
                bs_options = []
                for bs_str in batch_size_options_str.split(";"):
                    bs_str = bs_str.strip()
                    if bs_str:
                        try:
                            bs_options.append(int(bs_str))
                        except Exception:
                            continue
                if not hidden_options:
                    hidden_options = [(256, 128)]
                if not lr_options:
                    lr_options = [float(learning_rate)]
                if not bs_options:
                    bs_options = [int(batch_size)]
                with st.spinner("Realizando busca de hiperparâmetros..."):
                    (best_model, best_scaler, best_history), summary = hyperparameter_search(
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        hidden_layers_options=hidden_options,
                        learning_rate_options=lr_options,
                        batch_size_options=bs_options,
                        max_epochs=int(max_epochs),
                        early_stopping_patience=int(early_stopping_patience),
                        random_state=42,
                    )
                model, scaler, history = best_model, best_scaler, best_history
            else:
                with st.spinner("Treinando modelo..."):
                    # Parse hidden layers
                    try:
                        hidden_layers = tuple(int(x.strip()) for x in hidden_layers_str.split(",") if x.strip())
                    except Exception:
                        hidden_layers = (256, 128)
                    model, scaler, history = train_mlp_classifier(
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        hidden_layers=hidden_layers,
                        learning_rate=float(learning_rate),
                        batch_size=int(batch_size),
                        max_epochs=int(max_epochs),
                        early_stopping_patience=int(early_stopping_patience),
                        random_state=42,
                    )

            # Evaluate on test set
            with st.spinner("Avaliando modelo..."):
                test_acc, report = evaluate_model(
                    model,
                    scaler,
                    X_test,
                    y_test,
                    class_names=class_names,
                )
            st.success(f"Acurácia no conjunto de teste: {test_acc * 100:.2f}%")

            # Display classification report as a table
            import pandas as pd

            report_df = pd.DataFrame(report).transpose()
            st.subheader("Relatório de Classificação")
            st.dataframe(report_df.style.format(precision=2))

            # Save model
            with st.spinner("Salvando modelo..."):
                save_model(model, scaler, class_names, path="results/trained_model.pkl")
            st.success("Modelo salvo em results/trained_model.pkl")

        except Exception as e:
            st.error(f"Erro ao treinar o modelo: {e}")

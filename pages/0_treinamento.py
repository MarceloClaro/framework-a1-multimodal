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
    load_dataset_from_hf,
)

st.set_page_config(page_title="Treinamento do Modelo", layout="wide")
st.title("Treinamento do Modelo de Classificacao")

st.markdown(
    """
    Esta pagina permite treinar um modelo de classificacao de imagens a partir de um
    conjunto de dados. O modelo usa um perceptron multicamadas (MLP) do scikit-learn,
    com early stopping e data augmentation opcional para balancear classes.
    """
)

dataset_path = st.text_input("Diretorio do dataset", value="/path/to/skin_cancer_dataset")
augment = st.checkbox("Balancear classes via data augmentation", value=True)
col1, col2, col3 = st.columns(3)
with col1:
    test_size = st.slider("Proporcao do conjunto de teste", 0.1, 0.4, 0.2, 0.05)
with col2:
    val_size = st.slider("Proporcao da validacao", 0.05, 0.3, 0.1, 0.05)
with col3:
    image_size = st.number_input("Lado das imagens (px)", 32, 128, 64, 16)

hidden_layers_str = st.text_input("Camadas ocultas (separadas por virgula)", "256,128")
learning_rate = st.number_input("Taxa de aprendizado", 0.0001, 0.01, 0.001, 0.0001, format="%f")
batch_size = st.number_input("Tamanho do batch", 16, 512, 64, 16)
max_epochs = st.number_input("Numero maximo de epocas", 5, 200, 50, 5)
early_stopping_patience = st.number_input("Paciencia para early stopping", 1, 20, 5, 1)

search_hyperparams = st.checkbox("Executar busca de hiperparametros", value=False)

if search_hyperparams:
    hidden_layers_options_str = st.text_input(
        "Opcoes de camadas ocultas", "256,128;512,256;128,64"
    )
    learning_rate_options_str = st.text_input(
        "Opcoes de learning rates", "0.001;0.0005"
    )
    batch_size_options_str = st.text_input(
        "Opcoes de tamanhos de batch", "32;64"
    )

train_button = st.button("Treinar modelo")

if train_button:
    if dataset_path.strip().lower().startswith("hf://"):
        data_loader = "hf"
    else:
        data_loader = "local"
    if data_loader == "local" and not os.path.isdir(dataset_path):
        st.error(f"Diretorio '{dataset_path}' nao encontrado.")
    else:
        try:
            with st.spinner("Carregando e preparando o dataset..."):
                if data_loader == "hf":
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

            if search_hyperparams:
                hidden_options: List[Tuple[int, ...]] = []
                for combo in hidden_layers_options_str.split(";"):
                    combo = combo.strip()
                    if combo:
                        try:
                            hidden_options.append(tuple(int(x.strip()) for x in combo.split(",") if x.strip()))
                        except Exception:
                            pass
                lr_options = []
                for lr_str in learning_rate_options_str.split(";"):
                    lr_str = lr_str.strip()
                    if lr_str:
                        try:
                            lr_options.append(float(lr_str))
                        except Exception:
                            pass
                bs_options = []
                for bs_str in batch_size_options_str.split(";"):
                    bs_str = bs_str.strip()
                    if bs_str:
                        try:
                            bs_options.append(int(bs_str))
                        except Exception:
                            pass
                if not hidden_options:
                    hidden_options = [tuple(int(x.strip()) for x in hidden_layers_str.split(",") if x.strip())]
                if not lr_options:
                    lr_options = [float(learning_rate)]
                if not bs_options:
                    bs_options = [int(batch_size)]
                with st.spinner("Realizando busca de hiperparametros..."):
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
                try:
                    hidden_layers = tuple(int(x.strip()) for x in hidden_layers_str.split(",") if x.strip())
                except Exception:
                    hidden_layers = (256, 128)
                with st.spinner("Treinando modelo..."):
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

            with st.spinner("Avaliando modelo no conjunto de teste..."):
                metrics = evaluate_model(model, scaler, X_test, y_test, class_names)
            model_path = os.path.join("results", "trained_model.pkl")
            save_model(model, scaler, class_names, model_path)
            st.success("Treinamento concluido! Modelo salvo em 'results/trained_model.pkl'.")
            st.write(f"Acuracia no conjunto de teste: {metrics['accuracy']:.4f}")
            report = metrics["report"]
            report_df = {
                "Classe": list(report.keys())[:-3],
                "Precisao": [report[c]["precision"] for c in report if c in class_names],
                "Revocacao": [report[c]["recall"] for c in report if c in class_names],
                "F1": [report[c]["f1-score"] for c in report if c in class_names],
                "Suporte": [report[c]["support"] for c in report if c in class_names],
            }
            st.table(report_df)
            st.line_chart({"Treino": history["train_acc"], "Validacao": history["val_acc"]})
        except Exception as e:
            st.exception(e)

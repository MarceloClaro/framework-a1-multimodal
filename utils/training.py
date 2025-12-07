from __future__ import annotations

import os
import random
import io
from pathlib import Path
from typing import List, Tuple, Dict, Any

import urllib.parse

import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


def _load_single_image(path: Path, image_size: Tuple[int, int]) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB").resize(image_size)
        return (np.array(img, dtype=np.float32) / 255.0).flatten()


def _augment_image(arr: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    w, h = image_size
    img = Image.fromarray((arr.reshape(h, w, 3) * 255).astype(np.uint8))
    angle = random.choice([0, 90, 180, 270])
    img = img.rotate(angle)
    if random.random() > 0.5:
        img = ImageOps.mirror(img)
    if random.random() > 0.5:
        img = ImageOps.flip(img)
    return (np.array(img, dtype=np.float32) / 255.0).flatten()


def load_dataset(dataset_path: str, image_size: Tuple[int, int] = (64, 64), augment: bool = True):
    dataset_path = Path(dataset_path)
    class_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()], key=lambda p: p.name)
    class_names = [d.name for d in class_dirs]
    images, labels = [], []
    counts: Dict[int, int] = {}
    for idx, class_dir in enumerate(class_dirs):
        files = list(class_dir.iterdir())
        counts[idx] = len(files)
        for p in files:
            try:
                arr = _load_single_image(p, image_size)
                images.append(arr)
                labels.append(idx)
            except Exception:
                continue
    if augment:
        max_count = max(counts.values()) if counts else 0
        for idx, count in counts.items():
            deficit = max_count - count
            idxs = [i for i, l in enumerate(labels) if l == idx]
            for _ in range(deficit):
                src = random.choice(idxs)
                images.append(_augment_image(images[src], image_size))
                labels.append(idx)
    X = np.stack(images)
    y = np.array(labels)
    return X, y, class_names


def split_dataset(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_fraction = val_size / (1.0 - test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_fraction, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_mlp_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_layers: Tuple[int, ...] = (256, 128),
    learning_rate: float = 0.001,
    batch_size: int = 64,
    max_epochs: int = 50,
    early_stopping_patience: int = 5,
    random_state: int = 42,
):
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        batch_size=batch_size,
        learning_rate_init=learning_rate,
        max_iter=1,
        warm_start=True,
        random_state=random_state,
    )
    best_state = None
    best_val_acc = 0.0
    no_improve = 0
    history = {"train_acc": [], "val_acc": [], "best_epoch": 0}
    for epoch in range(1, max_epochs + 1):
        model.fit(X_train_scaled, y_train)
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "coefs_": [w.copy() for w in model.coefs_],
                "intercepts_": [b.copy() for b in model.intercepts_],
                "n_layers_": model.n_layers_,
                "n_outputs_": model.n_outputs_,
                "out_activation_": model.out_activation_,
            }
            history["best_epoch"] = epoch
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= early_stopping_patience:
            break
    if best_state:
        model.coefs_ = best_state["coefs_"]
        model.intercepts_ = best_state["intercepts_"]
        model.n_layers_ = best_state["n_layers_"]
        model.n_outputs_ = best_state["n_outputs_"]
        model.out_activation_ = best_state["out_activation_"]
    return model, scaler, history


def evaluate_model(model: MLPClassifier, scaler: StandardScaler, X_test: np.ndarray, y_test: np.ndarray, class_names: List[str]):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    return {"accuracy": acc, "report": report}


def load_dataset_from_hf(
    base_url: str,
    splits_map: Dict[str, str] | None = None,
    split: str = "train",
    image_size: Tuple[int, int] = (64, 64),
    augment: bool = True,
    max_rows: int | None = None,
):
    try:
        import polars as pl  # type: ignore
        import fsspec  # type: ignore
    except Exception as exc:
        raise RuntimeError("polars and fsspec are required.") from exc
    if splits_map is None:
        splits_map = {
            "train": "data/train-*-of-*.parquet",
            "validation": "data/validation-*-of-*.parquet",
            "test": "data/test-00000-of-00001-61e7cf54bf274ae2.parquet",
        }
    if split not in splits_map:
        raise ValueError(f"Unknown split {split}.")
            # Ensure base_url is decoded and stripped of trailing slashes
        base_url = urllib.parse.unquote(base_url.rstrip("/"))
        parquet_path = base_url + "/" + splits_map[split]
        df = pl.read_parquet(parquet_path, storage_options={"token": os.getenv("HUGGINGFACE_TOKEN")})
    
    if max_rows is not None:
        df = df.head(max_rows)
    label_col = None
    for candidate in ["dx", "label", "labels"]:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        raise RuntimeError("Label column not found.")
    class_names = sorted(df[label_col].unique().to_list())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    image_col = None
    for candidate in ["image", "image_id", "img"]:
        if candidate in df.columns:
            image_col = candidate
            break
    if image_col is None:
        raise RuntimeError("Image column not found.")
    images: List[np.ndarray] = []
    labels: List[int] = []
    w, h = image_size
    from PIL import Image  # local import
    for row in df.iter_rows(named=True):
        lbl = row[label_col]
        labels.append(class_to_idx[lbl])
        img_data = row[image_col]
        img = None
        if isinstance(img_data, dict):
            if "bytes" in img_data and img_data["bytes"] is not None:
                try:
                    img = Image.open(io.BytesIO(img_data["bytes"]))
                except Exception:
                    img = None
            elif "path" in img_data:
                path = img_data["path"]
                if not path.startswith("hf://"):
                    path_full = base_url.rstrip("/") + "/" + path.lstrip("/")
                else:
                    path_full = path
                try:
                    with fsspec.open(path_full, "rb") as f:
                        img = Image.open(f)
                except Exception:
                    img = None
        elif isinstance(img_data, (bytes, bytearray)):
            try:
                img = Image.open(io.BytesIO(img_data))
            except Exception:
                img = None
        elif isinstance(img_data, str):
            path_full = img_data
            if not path_full.startswith("hf://"):
                path_full = base_url.rstrip("/") + "/" + path_full.lstrip("/")
            try:
                with fsspec.open(path_full, "rb") as f:
                    img = Image.open(f)
            except Exception:
                img = None
        if img is None:
            images.append(np.zeros((w * h * 3,), dtype=np.float32))
            continue
        img = img.convert("RGB").resize((w, h))
        arr = (np.array(img, dtype=np.float32) / 255.0).flatten()
        images.append(arr)
    X = np.stack(images)
    y = np.array(labels)
    if augment:
        counts = {idx: int((y == idx).sum()) for idx in range(len(class_names))}
        max_count = max(counts.values()) if counts else 0
        new_images: List[np.ndarray] = []
        new_labels: List[int] = []
        for idx, count in counts.items():
            deficit = max_count - count
            if deficit > 0:
                indices = [i for i, lbl in enumerate(y) if lbl == idx]
                for _ in range(deficit):
                    src_idx = random.choice(indices)
                    new_images.append(_augment_image(images[src_idx], image_size))
                    new_labels.append(idx)
        if new_images:
            X = np.concatenate([X, np.stack(new_images)], axis=0)
            y = np.concatenate([y, np.array(new_labels)])
    return X, y, class_names


def hyperparameter_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_layers_options: List[Tuple[int, ...]] | None = None,
    learning_rate_options: List[float] | None = None,
    batch_size_options: List[int] | None = None,
    max_epochs: int = 50,
    early_stopping_patience: int = 5,
    random_state: int = 42,
):
    if hidden_layers_options is None:
        hidden_layers_options = [(256, 128), (512, 256), (128, 64)]
    if learning_rate_options is None:
        learning_rate_options = [0.001, 0.0005]
    if batch_size_options is None:
        batch_size_options = [32, 64]
    best_val_acc = -1.0
    best_result = None
    summary: Dict[str, Any] = {}
    for hidden_layers in hidden_layers_options:
        for lr in learning_rate_options:
            for bs in batch_size_options:
                model, scaler, history = train_mlp_classifier(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    hidden_layers=hidden_layers,
                    learning_rate=lr,
                    batch_size=bs,
                    max_epochs=max_epochs,
                    early_stopping_patience=early_stopping_patience,
                    random_state=random_state,
                )
                val_acc = max(history["val_acc"]) if history["val_acc"] else 0.0
                key = f"layers={hidden_layers},lr={lr},bs={bs}"
                summary[key] = {
                    "val_accuracy": val_acc,
                    "history": history,
                    "hidden_layers": hidden_layers,
                    "learning_rate": lr,
                    "batch_size": bs,
                }
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_result = (model, scaler, history)
    return best_result, summary


def save_model(model: MLPClassifier, scaler: StandardScaler, class_names: List[str], model_path: str):
    payload = {"model": model, "scaler": scaler, "class_names": class_names}
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(payload, model_path)


def load_model(model_path: str):
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


def predict_images(
    model_dict: Dict[str, Any],
    image_files: List[Any],
    image_size: Tuple[int, int] = (64, 64),
):
    model = model_dict["model"]
    scaler = model_dict["scaler"]
    class_names = model_dict["class_names"]
    X = []
    file_names = []
    for f in image_files:
        try:
            arr = _load_single_image(f, image_size)
        except Exception:
            try:
                img = Image.open(f).convert("RGB").resize(image_size)
                arr = (np.array(img, dtype=np.float32) / 255.0).flatten()
            except Exception:
                continue
        X.append(arr)
        file_names.append(getattr(f, "name", "uploaded_image"))
    if not X:
        return []
    X = np.stack(X)
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)
    preds = model.predict(X_scaled)
    results = []
    for fname, pred_idx, prob_vec in zip(file_names, preds, probs):
        results.append(
            {
                "file_name": fname,
                "predicted_label": class_names[pred_idx],
                "probabilities": {class_names[i]: float(prob_vec[i]) for i in range(len(class_names))},
            }
        )
    return results

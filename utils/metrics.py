from typing import List, Dict, Any



def _confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    classes: List[str],
) -> Dict[str, Dict[str, int]]:
    """Constrói matriz de confusão simples cm[verdadeiro][previsto] = contagem."""
    cm = {c_true: {c_pred: 0 for c_pred in classes} for c_true in classes}
    for t, p in zip(y_true, y_pred):
        if t in classes and p in classes:
            cm[t][p] += 1
    return cm



def _precision_recall_f1_binary(
    y_true: List[str],
    y_pred: List[str],
    positive_label: str,
) -> Dict[str, float]:
    """Calcula precisão, revocação e F1 para uma classe positiva em tarefa binária."""
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == positive_label and p == positive_label:
            tp += 1
        elif t != positive_label and p == positive_label:
            fp += 1
        elif t != positive_label and p != positive_label:
            tn += 1
        elif t == positive_label and p != positive_label:
            fn += 1
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }



def _per_class_metrics_multiclass(
    y_true: List[str],
    y_pred: List[str],
    classes: List[str],
) -> Dict[str, Dict[str, float]]:
    """Calcula precisão, revocação e F1 para cada classe em classificação multiclasse."""
    metrics: Dict[str, Dict[str, float]] = {}
    for c in classes:
        binary_metrics = _precision_recall_f1_binary(y_true=y_true, y_pred=y_pred, positive_label=c)
        metrics[c] = {
            "precision": binary_metrics["precision"],
            "recall": binary_metrics["recall"],
            "f1": binary_metrics["f1"],
        }
    return metrics



def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    task_type: str,
    classes: List[str],
) -> Dict[str, Any]:
    """
    Calcula métricas apropriadas para tarefas binárias ou multiclasse.

    Retorna um dicionário com:
    - n_samples
    - accuracy
    - confusion_matrix
    - (se binária) métricas específicas da classe positiva
    - (se multiclasse) macro-F1 e métricas por classe
    """
    assert len(y_true) == len(y_pred), "y_true e y_pred devem ter mesmo tamanho"
    n = len(y_true)
    cm = _confusion_matrix(y_true, y_pred, classes)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n if n > 0 else 0.0
    metrics: Dict[str, Any] = {
        "n_samples": n,
        "accuracy": accuracy,
        "confusion_matrix": cm,
    }
    if task_type == "binária" and len(classes) == 2:
        positive_label = classes[1]  # assume a segunda classe como positiva
        bin_metrics = _precision_recall_f1_binary(y_true, y_pred, positive_label)
        metrics["positive_label"] = positive_label
        metrics["binary_metrics"] = bin_metrics
    if task_type == "multiclasse":
        per_class = _per_class_metrics_multiclass(y_true, y_pred, classes)
        metrics["per_class"] = per_class
        if per_class:
            f1_values = [v["f1"] for v in per_class.values()]
            macro_f1 = sum(f1_values) / len(f1_values)
        else:
            macro_f1 = 0.0
        metrics["macro_f1"] = macro_f1
    return metrics

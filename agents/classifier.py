import io
import json
from typing import List, Dict, Any

from google import genai
from PIL import Image
import ray


def _build_prompt(task_type: str, classes: List[str]) -> str:
    """Monta o prompt de classificacao para o agente LLM."""
    classes_str = ", ".join(classes)
    return f"""
Você é um agente Classifier especializado em classificação {task_type}.

Conjunto de classes possíveis: [{classes_str}].

Regras:
- Sempre escolha rótulos APENAS dentre as classes fornecidas.
- Para classificação multirrótulo, retorne uma lista de rótulos (podendo conter mais de um).
- Para classificação binária ou multiclasse, retorne APENAS um rótulo.

Formato de resposta OBRIGATÓRIO (JSON puro, sem texto extra):

{{
  "predicted_label": "string ou lista de strings",
  "candidate_labels": ["lista de rótulos considerados"],
  "rationale": "explicação resumida da decisão",
  "estimated_confidence": "valor entre 0 e 1 (pode ser aproximado)"
}}
""".strip()


def _parse_llm_json(raw_text: str) -> Dict[str, Any]:
    """Tenta converter a resposta em JSON estruturado; caso contrário, retorna texto bruto."""
    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            return data
        return {"raw_model_output": raw_text}
    except Exception:
        return {"raw_model_output": raw_text}


def classify_objects(
    files: List[Dict[str, Any]],
    api_key: str,
    model_name: str,
    task_type: str,
    classes: List[str],
) -> List[Dict[str, Any]]:
    """
    Classifica múltiplos objetos de forma sequencial usando Gemini.

    Retorna lista de dicionários com campos:
    - file_name
    - hash
    - predicted_label
    - candidate_labels
    - estimated_confidence
    - rationale
    - raw_model_output
    """
    client = genai.Client(api_key=api_key)
    prompt = _build_prompt(task_type=task_type, classes=classes)
    results: List[Dict[str, Any]] = []
    for item in files:
        file = item["file"]
        hash_id = item["hash"]
        contents = [prompt]
        # Trata imagem
        if file.type.startswith("image"):
            img = Image.open(file)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            contents.append({"mime_type": "image/png", "data": image_bytes})
        else:
            # Trata arquivos de texto ou outros
            file_bytes = file.read()
            try:
                text_content = file_bytes.decode("utf-8", errors="ignore")
            except Exception:
                text_content = str(file_bytes)
            contents.append(f"Conteúdo do objeto a ser classificado:\n{text_content[:4000]}")
        # Chama API do modelo
        response = client.models.generate_content(model=model_name, contents=contents)
        raw_text = response.text
        parsed = _parse_llm_json(raw_text)
        result_entry = {
            "file_name": file.name,
            "hash": hash_id,
            "raw_model_output": raw_text,
            "predicted_label": parsed.get("predicted_label"),
            "candidate_labels": parsed.get("candidate_labels"),
            "estimated_confidence": parsed.get("estimated_confidence"),
            "rationale": parsed.get("rationale"),
        }
        results.append(result_entry)
        # Reseta ponteiro do arquivo
        file.seek(0)
    return results


@ray.remote
def classify_single_item_remote(
    item: Dict[str, Any],
    api_key: str,
    model_name: str,
    task_type: str,
    classes: List[str],
    prompt: str,
) -> Dict[str, Any]:
    """Classifica um único item de forma remota para execução paralela."""
    client = genai.Client(api_key=api_key)
    file = item["file"]
    hash_id = item["hash"]
    contents = [prompt]
    if file.type.startswith("image"):
        img = Image.open(file)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        contents.append({"mime_type": "image/png", "data": image_bytes})
    else:
        file_bytes = file.read()
        try:
            text_content = file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            text_content = str(file_bytes)
        contents.append(f"Conteúdo do objeto:\n{text_content[:4000]}")
    response = client.models.generate_content(model=model_name, contents=contents)
    raw_text = response.text
    parsed = _parse_llm_json(raw_text)
    file.seek(0)
    return {
        "file_name": file.name,
        "hash": hash_id,
        "raw_model_output": raw_text,
        "predicted_label": parsed.get("predicted_label"),
        "candidate_labels": parsed.get("candidate_labels"),
        "estimated_confidence": parsed.get("estimated_confidence"),
        "rationale": parsed.get("rationale"),
    }


def classify_objects_parallel(
    files: List[Dict[str, Any]],
    api_key: str,
    model_name: str,
    task_type: str,
    classes: List[str],
) -> List[Dict[str, Any]]:
    """Executa classificação paralela usando Ray."""
    prompt = _build_prompt(task_type=task_type, classes=classes)
    tasks = [
        classify_single_item_remote.remote(
            item=f,
            api_key=api_key,
            model_name=model_name,
            task_type=task_type,
            classes=classes,
            prompt=prompt,
        )
        for f in files
    ]
    return ray.get(tasks)

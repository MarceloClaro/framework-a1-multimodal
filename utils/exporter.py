import json
import csv
from pathlib import Path


OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_json(data, filename: str = "resultados.json") -> Path:
    """Salva dicionário em arquivo JSON no diretório de resultados."""
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return path


def save_csv(rows, filename: str = "resultados.csv") -> Path:
    """Salva uma lista de dicionários em CSV. Retorna o caminho do arquivo."""
    if not rows:
        return OUTPUT_DIR / filename
    path = OUTPUT_DIR / filename
    keys = rows[0].keys()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return path

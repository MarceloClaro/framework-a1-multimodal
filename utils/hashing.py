import hashlib


def hash_file(file) -> str:
    """Gera hash SHA-256 truncado do conteúdo do arquivo e retorna 20 caracteres."""
    content = file.read()
    file.seek(0)
    return hashlib.sha256(content).hexdigest()[:20]

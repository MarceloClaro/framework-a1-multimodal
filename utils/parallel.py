import ray

# Controle de inicialização
_initialized = False


def init_ray():
    """Inicializa o Ray uma única vez."""
    global _initialized
    if not _initialized:
        ray.init(ignore_reinit_error=True)
        _initialized = True

from google import genai


def _client(api_key):
    """Inicializa e retorna o cliente Gemini."""
    return genai.Client(api_key=api_key)


def clinical_agent(results, api_key):
    """Gera uma análise especializada para o domínio médico."""
    prompt = f"""
Você é um ESPECIALISTA CLÍNICO.

Analise os resultados considerando:
- segurança do paciente
- vieses médicos
- implicações diagnósticas
- adequação a protocolos clínicos

RESULTADOS:
{results}
"""
    return _client(api_key).models.generate_content(model="gemini-1.5-pro", contents=[prompt]).text


def industrial_agent(results, api_key):
    """Gera uma análise especializada para o domínio industrial."""
    prompt = f"""
Você é um ESPECIALISTA EM INDÚSTRIA.

Analise os resultados considerando:
- confiabilidade do processo
- impacto operacional
- riscos industriais
- falhas e redundâncias

RESULTADOS:
{results}
"""
    return _client(api_key).models.generate_content(model="gemini-1.5-pro", contents=[prompt]).text


def pedagogical_agent(results, api_key):
    """Gera uma análise especializada para o domínio educacional."""
    prompt = f"""
Você é um ESPECIALISTA EM EDUCAÇÃO.

Analise os resultados considerando:
- aprendizagem
- vieses socioeducacionais
- avaliação formativa
- implicações pedagógicas

RESULTADOS:
{results}
"""
    return _client(api_key).models.generate_content(model="gemini-1.5-pro", contents=[prompt]).text

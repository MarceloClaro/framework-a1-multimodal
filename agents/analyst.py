from typing import List, Dict, Any, Optional
from google import genai


# Templates de domínio para instruções específicas
DOMAIN_TEMPLATES = {
    "medicina": """
O domínio é MEDICINA.  
As análises devem considerar:  
- protocolos clínicos,  
- segurança do paciente,  
- viés em dados biomédicos,  
- limitações éticas,  
- mecanismos fisiológicos quando aplicável,  
- necessidade de validação em ensaios clínicos.  
""",
    "indústria": """
O domínio é INDÚSTRIA.  
As análises devem considerar:  
- confiabilidade de processos produtivos,  
- análise de risco,  
- controle estatístico de qualidade,  
- impacto operacional,  
- tolerância a falhas e segurança.  
""",
    "educação": """
O domínio é EDUCAÇÃO.  
As análises devem considerar:  
- avaliação pedagógica,  
- vieses socioeducacionais,  
- impacto na aprendizagem,  
- qualidade e equidade no acesso,  
- variabilidade entre contextos educacionais.  
""",
}



def _init_client(api_key: str) -> genai.Client:
    """Inicializa o cliente Gemini a partir de uma chave."""
    return genai.Client(api_key=api_key)



def _domain_context(domain: str) -> str:
    """Retorna a instrução de domínio ou uma string vazia para "geral"."""
    return DOMAIN_TEMPLATES.get(domain, "")



def scientific_analysis(
    results: List[Dict[str, Any]],
    task_type: str,
    classes: List[str],
    api_key: str,
    domain: str = "geral",
) -> str:
    """Gera análise científica estruturada com foco em Qualis A1."""
    client = _init_client(api_key)
    model = "gemini-1.5-pro"
    domain_instruction = _domain_context(domain)
    prompt = f"""
Você é um pesquisador de nível Qualis A1.

DOMÍNIO:
{domain_instruction}

Resultados classificados:
{results}

Tipo de tarefa: {task_type}
Classes: {classes}

Produza um relatório científico que contenha:
1. Interpretação técnica
2. Riscos e limitações
3. Discussão específica do domínio
4. Hipóteses científicas compatíveis
5. Potenciais vieses e mitigação
6. Sugestões para replicabilidade futura
"""
    response = client.models.generate_content(model=model, contents=[prompt])
    return response.text



def critic_review(
    results: List[Dict[str, Any]],
    task_type: str,
    classes: List[str],
    api_key: str,
    domain: str = "geral",
) -> str:
    """Simula um parecerista que revisa criticamente o experimento."""
    client = _init_client(api_key)
    model = "gemini-1.5-pro"
    domain_instruction = _domain_context(domain)
    prompt = f"""
Você é um parecerista de periódico Qualis A1.

DOMÍNIO:
{domain_instruction}

Avalie criticamente o experimento considerando:
- robustez metodológica
- riscos de interpretação
- limitações estatísticas
- pontos críticos do domínio
- sugestões construtivas

RESULTADOS:
{results}

Tipo de tarefa: {task_type}
Classes: {classes}
"""
    response = client.models.generate_content(model=model, contents=[prompt])
    return response.text



def replication_protocol(
    results: List[Dict[str, Any]],
    task_type: str,
    classes: List[str],
    api_key: str,
    domain: str = "geral",
) -> str:
    """Define um protocolo experimental para replicação."""
    client = _init_client(api_key)
    model = "gemini-1.5-pro"
    domain_instruction = _domain_context(domain)
    prompt = f"""
Você é especialista em reprodutibilidade.

DOMÍNIO:
{domain_instruction}

RESULTADOS:
{results}

Tipo de tarefa: {task_type}
Classes: {classes}

Produza um protocolo para replicar:
1. passos experimentais
2. controle de variáveis
3. registros obrigatórios
4. logs, hashes e versões
"""
    response = client.models.generate_content(model=model, contents=[prompt])
    return response.text



def stats_report(
    results: List[Dict[str, Any]],
    task_type: str,
    classes: List[str],
    metrics_dict: Optional[Dict[str, Any]],
    api_key: str,
    domain: str = "geral",
) -> str:
    """Gera interpretação e discussão estatística sobre as métricas."""
    client = _init_client(api_key)
    model = "gemini-1.5-pro"
    domain_instruction = _domain_context(domain)
    prompt = f"""
Você é um estatístico especializado em avaliação de modelos de classificação.

DOMÍNIO:
{domain_instruction}

RESULTADOS:
{results}

Tipo de tarefa: {task_type}
Classes: {classes}

MÉTRICAS COMPUTADAS (pode ser None):
{metrics_dict}

Explique:
1. adequação das métricas ao tipo de classificação
2. riscos estatísticos
3. viés, variância e desbalanceamento
4. importância para o domínio específico
"""
    response = client.models.generate_content(model=model, contents=[prompt])
    return response.text



def evolutionary_agent(reports: Dict[str, str], api_key: str, domain: str = "geral") -> str:
    """Combina e refina relatórios usando um modelo evolutivo-genético."""
    client = _init_client(api_key)
    model = "gemini-1.5-pro"
    domain_instruction = _domain_context(domain)
    prompt = f"""
Você é um AGENTE EVOLUTIVO GENÉTICO.

DOMÍNIO:
{domain_instruction}

Você recebeu múltiplos relatórios:
{reports}

TAREFA:
1. Sintetizar todos os relatórios em uma única versão evoluída.
2. Avaliar coerência → melhorar → recombinar ideias.
3. Gerar a "MELHOR VERSÃO EVOLUTIVA", contendo:
   - consistência lógica superior
   - clareza textual
   - profundidade científica
   - foco no domínio
   - estrutura impecável
"""
    response = client.models.generate_content(model=model, contents=[prompt])
    return response.text

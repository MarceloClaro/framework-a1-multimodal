def generate_article(reports):
    """Gera um texto de artigo científico em formato markdown a partir dos relatórios."""
    return f"""
===============================
ARTIGO CIENTÍFICO COMPLETO (A1)
===============================

TÍTULO PROVISÓRIO:
Framework Multimodal com Agentes Evolutivos Aplicado a Classificação em Domínios Complexos

RESUMO
------
{reports.get('científico', '')[:800]}

1. INTRODUÇÃO
--------------
Este estudo apresenta um framework multimodal integrado a agentes LLM,
incluindo mecanismos evolutivos, análises especializadas e métricas de validação.

2. MÉTODOS
-----------
2.1 Classificação multimodal  
2.2 Agentes científicos  
2.3 Agentes especializados (domínio)  
2.4 Agente evolutivo genético  
2.5 Métricas quantitativas  

3. RESULTADOS
--------------
### Análise Científica:
{reports.get('científico', '')}

### Revisão Crítica:
{reports.get('crítico', '')}

### Estatística:
{reports.get('estatística', '')}

4. DISCUSSÃO
-------------
{reports.get('evolutivo', '')}

5. IMPLICAÇÕES POR DOMÍNIO
---------------------------
{reports.get('especializado', '')}

6. PROTOCOLO DE REPRODUÇÃO
---------------------------
{reports.get('replicação', '')}

REFERÊNCIAS
------------
Modelos LLM, princípios FAIR, e literatura de avaliação estatística.
"""

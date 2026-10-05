import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI


# ============================================================
# Environment
# ============================================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY não encontrada no arquivo .env")


# ============================================================
# Database
# ============================================================

db = SQLDatabase.from_uri("sqlite:///cinerocket.db")


# ============================================================
# OpenRouter
# ============================================================

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODELS = [
    {
        "id": "qwen/qwen3.8-27b:free",
        "name": "Qwen3.8 27B",
    },
    {
        "id": "cohere/north-mini-code:free",
        "name": "Cohere North Mini Code",
    },
    {
        "id": "openrouter/free",
        "name": "OpenRouter Free",
    },
]


# ============================================================
# LLM
# ============================================================

def create_llm(model_id):
    return ChatOpenAI(
        model=model_id,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
    )


# ============================================================
# Scope guardrail
# ============================================================

SCOPE_RULES = """
ESCOPO DO ASSISTENTE:

Este assistente existe exclusivamente para responder perguntas
relacionadas ao catálogo de filmes e aos dados disponíveis no
banco de dados CineData.

São exemplos de perguntas dentro do escopo:

- filmes e títulos;
- gêneros;
- atores e atrizes;
- diretores;
- avaliações e notas;
- popularidade;
- receita;
- lucro;
- bilheteria;
- orçamento;
- desempenho financeiro;
- datas e anos de lançamento;
- duração dos filmes;
- estatísticas e comparações sobre os dados do catálogo;
- rankings e agregações baseados nos dados disponíveis no banco.

Perguntas fora desse domínio devem ser recusadas.

Exemplos de perguntas fora do escopo:

- conhecimentos gerais sem relação com filmes ou com o banco;
- programação que não esteja relacionada à consulta do CineData;
- matemática sem relação com os dados do catálogo;
- notícias;
- clima;
- política;
- geografia;
- assuntos pessoais;
- criação de poemas, histórias ou textos sem relação com o catálogo;
- perguntas sobre assuntos que não possam ser respondidos usando os dados
  disponíveis no banco.

IMPORTANTE:

Não bloqueie uma pergunta apenas porque ela não contém palavras como
"filme", "gênero", "receita" ou "lucro".

Considere o significado da pergunta.

Por exemplo, perguntas como:

"Qual produção recuperou mais dinheiro em relação ao investimento?"

"Quais tiveram o melhor desempenho financeiro?"

"Compare os lançamentos de 2020 e 2021."

podem estar dentro do escopo se puderem ser respondidas usando os dados
disponíveis no banco.

Quando houver relação plausível com os dados do catálogo, tente responder
usando o banco em vez de recusar prematuramente.

Se a pergunta estiver claramente fora do escopo, não tente responder usando
conhecimento externo e não invente uma consulta SQL.
"""


# ============================================================
# SQL rules
# ============================================================

SQL_RULES = """
REGRAS OBRIGATÓRIAS PARA CONSULTAS SQL:

1. Use somente SELECT.
2. Nunca use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE ou REPLACE.
3. Use SOMENTE tabelas e colunas que existam no schema fornecido.
4. Nunca invente nomes de tabelas ou colunas.
5. Antes de montar uma consulta, confira no schema qual tabela realmente contém cada informação.
6. Quando uma informação estiver em uma tabela diferente da tabela principal do filme,
   use JOIN através das chaves disponíveis no schema.
7. Para médias, use AVG().
8. Para somas, use SUM().
9. Para contagens, use COUNT().
10. Para valores máximos, use MAX().
11. Para valores mínimos, use MIN().
12. Para estatísticas por categoria, use GROUP BY.
13. Nunca busque todas as linhas para depois calcular uma estatística no LLM.
14. Deixe o banco de dados fazer filtros, agrupamentos, ordenações e cálculos.
15. Para rankings, use ORDER BY e LIMIT.
16. Não use LIMIT antes de uma agregação que precise considerar todos os registros.
17. Retorne somente as colunas necessárias para responder à pergunta.
18. Evite consultas intermediárias desnecessárias.
19. Não invente relações entre tabelas. Use somente as chaves presentes no schema.
20. Se uma informação não puder ser obtida com o schema fornecido, não invente uma coluna.
"""


# ============================================================
# Full SQL Agent prompt
# ============================================================

SYSTEM_PROMPT = f"""
Você é um agente especializado em consultas SQL sobre um catálogo de filmes.

Seu objetivo é responder às perguntas do usuário consultando o banco de dados.

{SCOPE_RULES}

{SQL_RULES}

REGRAS DE INTERPRETAÇÃO:

- "melhores filmes" normalmente indica um ranking baseado em uma métrica disponível
  no banco, como nota/avaliação. Escolha a métrica mais apropriada disponível no schema.
- "mais lucrativos" significa ordenar por lucro.
- "maior receita" significa ordenar por receita.
- "média de lucro" significa AVG(lucro_usd).
- "média de receita" significa AVG(receita_usd).
- Quando o usuário pedir resultados por gênero, diretor, ano ou outra categoria,
  faça o agrupamento apropriado.
- Quando uma métrica estiver em uma tabela de fatos e o nome do filme estiver
  em uma dimensão, faça o JOIN correto usando as chaves do schema.

COMPORTAMENTO PARA PERGUNTAS FORA DO ESCOPO:

Se a pergunta estiver claramente fora do escopo, não consulte o banco,
não gere SQL e não tente responder usando conhecimento externo.

Responda em português informando que a pergunta está fora do escopo
do CineData Analytics e que você pode ajudar apenas com perguntas
relacionadas ao catálogo de filmes e aos dados disponíveis no banco.

Responda ao usuário em português.

Não invente dados.
"""


# ============================================================
# Lightweight SQL generator prompt
# ============================================================

SQL_GENERATOR_PROMPT = f"""
Você é um gerador especializado de consultas SQL para um banco SQLite de filmes.

Sua única tarefa é transformar uma pergunta do usuário em uma consulta SQL válida
quando a pergunta estiver relacionada ao catálogo de filmes e aos dados disponíveis
no banco.

{SCOPE_RULES}

{SQL_RULES}

IMPORTANTE:

- O schema real do banco será fornecido abaixo.
- Você DEVE consultar esse schema antes de escrever a consulta.
- Use somente tabelas e colunas que aparecem no schema.
- NÃO invente nenhuma coluna.
- NÃO invente nenhuma tabela.
- NÃO presuma que uma coluna pertence à tabela de filmes.
- Se uma informação estiver em outra tabela, faça o JOIN correto.
- A consulta deve ser executável diretamente no SQLite.
- Gere apenas UMA consulta.
- A consulta deve começar com SELECT.
- Não use markdown.
- Não use ```sql.
- Não escreva explicações.
- Não escreva comentários.
- Não escreva texto antes ou depois da consulta.

COMPORTAMENTO PARA PERGUNTAS FORA DO ESCOPO:

Se a pergunta estiver claramente fora do escopo do CineData Analytics,
NÃO gere SQL.

Nesse caso, responda exatamente com:

FORA_DO_ESCOPO

Não escreva nenhuma outra coisa nesse caso.

Se houver relação plausível entre a pergunta e os dados do catálogo,
considere a pergunta dentro do escopo e tente gerar a consulta.

Exemplo de comportamento correto:

Se o usuário pedir os filmes com maior receita e o schema mostrar que
receita_usd está em fact_movies_performance enquanto titulo está em dim_movies,
você deve fazer JOIN entre as tabelas usando a chave correspondente.

Nunca faça algo como:

SELECT titulo, receita_usd
FROM dim_movies;

se receita_usd não estiver em dim_movies.

Schema do banco:
"""


# ============================================================
# Compact database schema
# ============================================================

@lru_cache(maxsize=1)
def get_database_schema():
    """
    Return the database schema used by the lightweight SQL generator.
    """

    return db.get_table_info()


# ============================================================
# Agent creation
# ============================================================

@lru_cache(maxsize=None)
def create_agent(model_id):
    llm = create_llm(model_id)

    return create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        prefix=SYSTEM_PROMPT,
        verbose=True,
        handle_parsing_errors=True,
        agent_executor_kwargs={
            "return_intermediate_steps": True,
        },
    )


# ============================================================
# Lightweight SQL generator
# ============================================================

@lru_cache(maxsize=None)
def create_sql_generator(model_id):
    return create_llm(model_id)


def generate_sql(question, model_id):
    """
    Generate SQL using a lightweight LLM call.
    """

    llm = create_sql_generator(model_id)

    schema = get_database_schema()

    prompt = (
        SQL_GENERATOR_PROMPT
        + "\n\n"
        + "SCHEMA REAL DO BANCO:\n"
        + schema
        + "\n\n"
        + "PERGUNTA DO USUÁRIO:\n"
        + question
        + "\n\n"
        + "CONSULTA SQL:"
    )

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                text = item.get("text")

                if text:
                    parts.append(str(text))

            elif isinstance(item, str):
                parts.append(item)

        content = "\n".join(parts)

    content = str(content).strip()

    # --------------------------------------------------------
    # Scope guardrail
    # --------------------------------------------------------

    if content.upper().strip() == "FORA_DO_ESCOPO":
        raise RuntimeError(
            "OUT_OF_SCOPE"
        )

    query = clean_sql_query(content)

    validate_sql_query(query)

    return query


# ============================================================
# SQL cleaning
# ============================================================

def clean_sql_query(query):
    """
    Clean common formatting generated by the LLM.
    """

    query = query.strip()

    # Remove markdown code fences.
    query = re.sub(
        r"^```sql\s*",
        "",
        query,
        flags=re.IGNORECASE,
    )

    query = re.sub(
        r"^```\s*",
        "",
        query,
    )

    query = re.sub(
        r"\s*```$",
        "",
        query,
    )

    query = query.strip()

    # Remove accidental text before SELECT.
    select_position = re.search(
        r"\bSELECT\b",
        query,
        re.IGNORECASE,
    )

    if select_position:
        query = query[select_position.start():]

    query = query.rstrip()

    if query.endswith(";"):
        query = query[:-1].rstrip()

    query += ";"

    return query


# ============================================================
# SQL validation
# ============================================================

def validate_sql_query(query):
    """
    Validate that the generated query is read-only SQL.
    """

    normalized = query.strip().lower()

    if not normalized.startswith("select"):
        raise ValueError(
            "A consulta gerada não começa com SELECT."
        )

    forbidden_patterns = [
        r"\binsert\b",
        r"\bupdate\b",
        r"\bdelete\b",
        r"\bdrop\b",
        r"\balter\b",
        r"\bcreate\b",
        r"\btruncate\b",
        r"\breplace\b",
        r"\battach\b",
        r"\bdetach\b",
        r"\bpragma\b",
    ]

    for pattern in forbidden_patterns:
        if re.search(
            pattern,
            normalized,
        ):
            raise ValueError(
                "A consulta gerada contém uma operação SQL não permitida."
            )

    # Prevent multiple SQL statements.
    query_without_final_semicolon = (
        normalized.rstrip(";").strip()
    )

    if ";" in query_without_final_semicolon:
        raise ValueError(
            "A consulta contém múltiplas instruções SQL."
        )


# ============================================================
# Full agent helpers
# ============================================================

def extract_query(response):
    """
    Extract the SQL query generated by the full SQL agent.
    """

    intermediate_steps = response.get(
        "intermediate_steps",
        [],
    )

    for action, observation in reversed(
        intermediate_steps
    ):
        tool_name = getattr(
            action,
            "tool",
            "",
        )

        if tool_name != "sql_db_query":
            continue

        tool_input = getattr(
            action,
            "tool_input",
            None,
        )

        if isinstance(
            tool_input,
            dict,
        ):
            query = tool_input.get(
                "query"
            )

            if query:
                return query.strip()

        if isinstance(
            tool_input,
            str,
        ):
            return tool_input.strip()

    return None


def extract_query_result(response):
    """
    Extract the result returned by the SQL query tool.
    """

    intermediate_steps = response.get(
        "intermediate_steps",
        [],
    )

    for action, observation in reversed(
        intermediate_steps
    ):
        tool_name = getattr(
            action,
            "tool",
            "",
        )

        if tool_name != "sql_db_query":
            continue

        return observation

    return None


# ============================================================
# Error detection
# ============================================================

def is_daily_free_limit_error(error):
    message = str(error).lower()

    daily_limit_terms = [
        "free-models-per-day",
        "openrouter_free_tier_daily",
        "free-model daily limit",
        "free model requests per day",
    ]

    return any(
        term in message
        for term in daily_limit_terms
    )


def is_rate_limit_error(error):
    message = str(error).lower()

    return (
        "429" in message
        or "rate limit" in message
        or "ratelimit" in message
    )


def is_context_limit_error(error):
    message = str(error).lower()

    context_limit_terms = [
        "maximum context length",
        "context length",
        "requested about",
        "reduce the length",
        "context-compression",
    ]

    return any(
        term in message
        for term in context_limit_terms
    )


# ============================================================
# Full agent execution with fallback
# ============================================================

def execute_with_fallback(
    question,
    callbacks=None,
):
    """
    Execute the full SQL agent with model fallback.
    """

    last_error = None

    for model in MODELS:
        model_id = model["id"]
        model_name = model["name"]

        print()
        print(
            f"[AGENT] Trying model: {model_name}"
        )
        print(
            f"[AGENT] ID: {model_id}"
        )

        try:
            agent = create_agent(
                model_id
            )

            invoke_kwargs = {
                "input": question,
            }

            if callbacks:
                invoke_kwargs["config"] = {
                    "callbacks": callbacks,
                }

            response = agent.invoke(
                invoke_kwargs
            )

            print(
                f"[AGENT] Success with model: "
                f"{model_name}"
            )

            return response, model_name

        except Exception as error:
            last_error = error

            print(
                f"[AGENT] Error on "
                f"{model_name}: {error}"
            )

            if is_daily_free_limit_error(
                error
            ):
                print(
                    "[AGENT] Daily free-model "
                    "limit reached."
                )
                continue

            if is_context_limit_error(
                error
            ):
                raise RuntimeError(
                    "CONTEXT_LIMIT_EXCEEDED"
                ) from error

            if is_rate_limit_error(
                error
            ):
                continue

            raise

    if (
        last_error
        and is_daily_free_limit_error(
            last_error
        )
    ):
        raise RuntimeError(
            "FREE_DAILY_LIMIT_EXCEEDED"
        ) from last_error

    if last_error:
        raise last_error

    raise RuntimeError(
        "Nenhum modelo conseguiu "
        "processar a solicitação."
    )


# ============================================================
# Direct SQL execution
# ============================================================

def execute_sql_query(query):
    """
    Execute a validated SQL query directly against SQLite.
    """

    clean_query = clean_sql_query(
        query
    )

    validate_sql_query(
        clean_query
    )

    result = db.run(
        clean_query
    )

    return result


# ============================================================
# Lightweight SQL execution with fallback
# ============================================================

def execute_sql_with_fallback(
    question,
):
    """
    Generate SQL with a lightweight LLM
    and execute it directly.

    This path is used by:
    - Apenas tabela
    - Tabela + gráfico

    It intentionally avoids a second
    LLM call after SQL execution.
    """

    last_error = None

    for model in MODELS:
        model_id = model["id"]
        model_name = model["name"]

        print()
        print(
            f"[SQL MODE] Trying model: "
            f"{model_name}"
        )
        print(
            f"[SQL MODE] ID: {model_id}"
        )

        try:
            query = generate_sql(
                question=question,
                model_id=model_id,
            )

            print()
            print(
                "[SQL MODE] Generated query:"
            )
            print(query)

            result = execute_sql_query(
                query
            )

            print()
            print(
                "[SQL MODE] Success with model: "
                f"{model_name}"
            )

            return {
                "sql_query": query,
                "sql_result": result,
            }, model_name

        except RuntimeError as error:
            if str(error) == "OUT_OF_SCOPE":
                raise

            last_error = error

            print(
                f"[SQL MODE] Error on "
                f"{model_name}: {error}"
            )

            if is_daily_free_limit_error(
                error
            ):
                print(
                    "[SQL MODE] Daily free-model "
                    "limit reached."
                )
                continue

            if is_context_limit_error(
                error
            ):
                raise RuntimeError(
                    "CONTEXT_LIMIT_EXCEEDED"
                ) from error

            if is_rate_limit_error(
                error
            ):
                continue

            continue

        except Exception as error:
            last_error = error

            print(
                f"[SQL MODE] Error on "
                f"{model_name}: {error}"
            )

            if is_daily_free_limit_error(
                error
            ):
                print(
                    "[SQL MODE] Daily free-model "
                    "limit reached."
                )
                continue

            if is_context_limit_error(
                error
            ):
                raise RuntimeError(
                    "CONTEXT_LIMIT_EXCEEDED"
                ) from error

            if is_rate_limit_error(
                error
            ):
                continue

            # SQL generation/execution errors
            # can be retried with another model.
            continue

    if (
        last_error
        and is_daily_free_limit_error(
            last_error
        )
    ):
        raise RuntimeError(
            "FREE_DAILY_LIMIT_EXCEEDED"
        ) from last_error

    if last_error:
        raise last_error

    raise RuntimeError(
        "Nenhum modelo conseguiu gerar "
        "uma consulta SQL válida."
    )
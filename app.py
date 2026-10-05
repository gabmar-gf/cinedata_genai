import ast
import re
import sqlite3

import pandas as pd
import streamlit as st

from agent import (
    execute_sql_with_fallback,
    execute_with_fallback,
)
st.set_page_config(
    page_title="CineData Analytics - Agente IA",
    page_icon="🎬",
    layout="wide",
)


st.markdown(
    """
    <style>
    .sidebar-title {
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 0.25rem;
    }

    [data-testid="stSidebar"] ::-webkit-scrollbar {
        width: 0px;
        height: 0px;
    }

    [data-testid="stSidebar"] {
        scrollbar-width: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# RESULT PARSING
# ============================================================
# ============================================================
# RESULT PARSING
# ============================================================

def parse_sql_result(sql_result):
    """
    Convert the raw SQLDatabase result into a DataFrame.
    """

    if sql_result is None:
        return None

    if isinstance(
        sql_result,
        pd.DataFrame,
    ):
        return sql_result

    if isinstance(
        sql_result,
        list,
    ):
        try:
            return pd.DataFrame(
                sql_result
            )
        except Exception:
            return None

    if not isinstance(
        sql_result,
        str,
    ):
        return None

    result = sql_result.strip()

    if not result:
        return None

    # --------------------------------------------------------
    # Markdown table
    # --------------------------------------------------------

    if "|" in result:
        lines = [
            line.strip()
            for line in result.splitlines()
            if line.strip()
        ]

        table_lines = [
            line
            for line in lines
            if line.startswith("|")
            and line.endswith("|")
        ]

        if len(table_lines) >= 2:
            header = [
                cell.strip()
                for cell in table_lines[0]
                .strip("|")
                .split("|")
            ]

            data_lines = []

            for line in table_lines[1:]:
                cells = [
                    cell.strip()
                    for cell in line
                    .strip("|")
                    .split("|")
                ]

                if all(
                    re.fullmatch(
                        r":?-+:?",
                        cell,
                    )
                    for cell in cells
                ):
                    continue

                if len(cells) == len(
                    header
                ):
                    data_lines.append(
                        cells
                    )

            if data_lines:
                return pd.DataFrame(
                    data_lines,
                    columns=header,
                )

    # --------------------------------------------------------
    # Python-style list / tuple
    # --------------------------------------------------------

    try:
        parsed = ast.literal_eval(
            result
        )

        if isinstance(
            parsed,
            list,
        ):
            if not parsed:
                return pd.DataFrame()

            if all(
                isinstance(
                    row,
                    (
                        list,
                        tuple,
                    ),
                )
                for row in parsed
            ):
                return pd.DataFrame(
                    parsed
                )

    except Exception:
        pass

    # --------------------------------------------------------
    # SQLDatabase textual output
    # --------------------------------------------------------

    lines = [
        line.strip()
        for line in result.splitlines()
        if line.strip()
    ]

    if len(lines) >= 2:
        rows = []

        for line in lines:
            if line.startswith(
                "["
            ) and line.endswith(
                "]"
            ):
                try:
                    parsed_line = (
                        ast.literal_eval(
                            line
                        )
                    )

                    if isinstance(
                        parsed_line,
                        tuple,
                    ):
                        rows.append(
                            parsed_line
                        )

                except Exception:
                    pass

        if rows:
            return pd.DataFrame(
                rows
            )

    return None


def extract_sql_from_agent_response(response):
    """Extract the most recent SQL query executed by the full SQL agent."""

    if not isinstance(response, dict):
        return None

    intermediate_steps = response.get("intermediate_steps", [])
    candidates = []

    def add_candidate(value):
        if not isinstance(value, str):
            return

        candidate = value.strip()

        if not candidate:
            return

        candidate = re.sub(r"^```(?:sql)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate).strip()

        if re.match(r"^(SELECT|WITH)\b", candidate, flags=re.IGNORECASE):
            candidates.append(candidate)

    for step in intermediate_steps:
        action = step[0] if isinstance(step, (tuple, list)) and step else step

        tool_input = getattr(action, "tool_input", None)

        if isinstance(tool_input, dict):
            for key in ("query", "sql", "statement"):
                add_candidate(tool_input.get(key))

        elif isinstance(tool_input, str):
            add_candidate(tool_input)

        log = getattr(action, "log", None)

        if isinstance(log, str):
            matches = re.findall(
                r"(?:Action Input|SQL Query|query)\s*:\s*(SELECT\b.*?)(?=\n(?:Observation|Thought|Action|$))",
                log,
                flags=re.IGNORECASE | re.DOTALL,
            )

            for match in matches:
                add_candidate(match)

    return candidates[-1] if candidates else None


def render_sql_query(sql_query):
    """Render SQL only when the user enabled the SQL display option."""

    if sql_query:
        with st.expander("🧾 SQL gerado"):
            st.code(sql_query, language="sql")


def render_agent_steps(response):
    """Render the completed agent steps from the returned response."""

    if not isinstance(response, dict):
        return

    intermediate_steps = response.get("intermediate_steps", [])

    if not intermediate_steps:
        st.info("O agente não retornou etapas intermediárias.")
        return

    with st.expander("🧠 Etapas do agente", expanded=True):
        for index, step in enumerate(intermediate_steps, start=1):
            if not isinstance(step, (tuple, list)) or len(step) < 2:
                continue

            action, observation = step[0], step[1]
            tool_name = getattr(action, "tool", None) or "Ação do agente"
            tool_input = getattr(action, "tool_input", None)

            st.markdown(f"**Etapa {index} — {tool_name}**")

            if tool_input is not None:
                st.caption("Entrada da ferramenta")
                if isinstance(tool_input, (dict, list, tuple)):
                    st.code(str(tool_input), language="text")
                else:
                    st.code(str(tool_input), language="text")

            if observation is not None:
                st.caption("Resultado da ferramenta")
                st.code(str(observation), language="text")

            if index < len(intermediate_steps):
                st.divider()


# ============================================================
# DATABASE EXPLORER
# ============================================================

DATABASE_PATH = "cinerocket.db"


def quote_identifier(identifier):
    """Safely quote a SQLite identifier."""

    return '"' + str(identifier).replace('"', '""') + '"'


def get_db_connection():
    connection = sqlite3.connect(
        DATABASE_PATH
    )
    connection.row_factory = sqlite3.Row
    return connection


def get_database_tables():
    """Return all user tables in the SQLite database."""

    try:
        with get_db_connection() as connection:
            rows = connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            ).fetchall()

        return [row["name"] for row in rows]

    except sqlite3.Error:
        return []


def get_table_columns(table_name):
    try:
        with get_db_connection() as connection:
            rows = connection.execute(
                f"PRAGMA table_info({quote_identifier(table_name)})"
            ).fetchall()

        return [dict(row) for row in rows]

    except sqlite3.Error:
        return []


def get_table_foreign_keys(table_name):
    try:
        with get_db_connection() as connection:
            rows = connection.execute(
                f"PRAGMA foreign_key_list({quote_identifier(table_name)})"
            ).fetchall()

        return [dict(row) for row in rows]

    except sqlite3.Error:
        return []


def get_table_rows(table_name, limit=20):
    try:
        with get_db_connection() as connection:
            query = (
                f"SELECT * FROM {quote_identifier(table_name)} "
                "LIMIT ?"
            )

            rows = connection.execute(
                query,
                (int(limit),),
            ).fetchall()

        return pd.DataFrame([dict(row) for row in rows])

    except sqlite3.Error:
        return pd.DataFrame()


def get_table_count(table_name):
    try:
        with get_db_connection() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) AS total FROM {quote_identifier(table_name)}"
            ).fetchone()

        return int(row["total"])

    except sqlite3.Error:
        return 0


def get_database_relationships(tables):
    """Build all foreign-key relationships in the database."""

    relationships = []

    for table_name in tables:
        for foreign_key in get_table_foreign_keys(table_name):
            relationships.append(
                {
                    "from_table": table_name,
                    "from_column": foreign_key["from"],
                    "to_table": foreign_key["table"],
                    "to_column": foreign_key["to"],
                }
            )

    return relationships


def render_database_dialog():
    """Render the interactive database explorer."""

    @st.dialog(
        "🗄️ Explorador do banco de dados",
        width="large",
    )
    def show_dialog():
        tables = get_database_tables()

        if not tables:
            st.error(
                "Não foi possível encontrar as tabelas do banco de dados."
            )
            return

        relationships = get_database_relationships(
            tables
        )

        st.caption(
            f"Banco conectado: {DATABASE_PATH} · "
            f"{len(tables)} tabela(s) encontrada(s)"
        )

        selected_table = st.selectbox(
            "Escolha uma tabela",
            options=tables,
            format_func=lambda table: f"📋 {table}",
        )

        columns_tab, data_tab, relations_tab = st.tabs(
            [
                "🧱 Estrutura",
                "📋 Dados",
                "🔗 Relações",
            ]
        )

        with columns_tab:
            columns = get_table_columns(
                selected_table
            )

            if columns:
                structure = pd.DataFrame(
                    [
                        {
                            "Coluna": column["name"],
                            "Tipo": column["type"] or "—",
                            "PK": "✓" if column["pk"] else "",
                            "Obrigatório": "✓" if column["notnull"] else "",
                            "Padrão": column["dflt_value"] or "—",
                        }
                        for column in columns
                    ]
                )

                st.dataframe(
                    structure,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(
                    "Não foi possível obter a estrutura desta tabela."
                )

        with data_tab:
            row_limit = st.select_slider(
                "Quantidade de registros",
                options=[10, 20, 50],
                value=20,
            )

            dataframe = get_table_rows(
                selected_table,
                row_limit,
            )

            if dataframe.empty:
                st.info(
                    "Esta tabela não possui registros para exibir."
                )
            else:
                st.caption(
                    f"Exibindo até {row_limit} registro(s). "
                    f"Total na tabela: {get_table_count(selected_table):,}"
                    .replace(",", ".")
                )

                st.dataframe(
                    dataframe,
                    use_container_width=True,
                    hide_index=True,
                )

        with relations_tab:
            selected_relations = [
                relationship
                for relationship in relationships
                if relationship["from_table"] == selected_table
                or relationship["to_table"] == selected_table
            ]

            if selected_relations:
                relation_data = pd.DataFrame(
                    [
                        {
                            "Tabela": relationship["from_table"],
                            "Coluna": relationship["from_column"],
                            "Relaciona com": relationship["to_table"],
                            "Coluna destino": relationship["to_column"],
                        }
                        for relationship in selected_relations
                    ]
                )

                st.dataframe(
                    relation_data,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(
                    "Nenhuma chave estrangeira encontrada para esta tabela."
                )

            if relationships:
                st.markdown("#### 🗺️ Mapa do banco")

                dot_lines = [
                    "digraph G {",
                    "rankdir=LR;",
                    "node [shape=box, style=rounded];",
                ]

                for index, table_name in enumerate(tables):
                    node_id = f"table_{index}"
                    safe_label = table_name.replace('"', '\\"')
                    dot_lines.append(
                        f'"{node_id}" [label="{safe_label}"];'
                    )

                table_node_ids = {
                    table_name: f"table_{index}"
                    for index, table_name in enumerate(tables)
                }

                for relationship in relationships:
                    source = table_node_ids.get(
                        relationship["from_table"]
                    )
                    target = table_node_ids.get(
                        relationship["to_table"]
                    )

                    if source and target:
                        label = (
                            f'{relationship["from_column"]} → '
                            f'{relationship["to_column"]}'
                        )
                        dot_lines.append(
                            f'"{source}" -> "{target}" [label="{label}"];'
                        )

                dot_lines.append("}")

                try:
                    st.graphviz_chart(
                        "\n".join(dot_lines),
                        use_container_width=True,
                    )
                except Exception:
                    st.code(
                        "\n".join(dot_lines),
                        language="text",
                    )
            else:
                st.info(
                    "O banco não possui relações por chave estrangeira."
                )

    show_dialog()


# ============================================================
# CHART
# ============================================================

def choose_chart_type(
    dataframe,
):
    if (
        dataframe is None
        or dataframe.empty
    ):
        return None

    numeric_columns = (
        dataframe.select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    if not numeric_columns:
        return None

    categorical_columns = [
        column
        for column in dataframe.columns
        if column not in numeric_columns
    ]

    if not categorical_columns:
        return None

    return "bar"


def render_chart(
    dataframe,
):
    if (
        dataframe is None
        or dataframe.empty
    ):
        return

    numeric_columns = (
        dataframe.select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    categorical_columns = [
        column
        for column in dataframe.columns
        if column not in numeric_columns
    ]

    if (
        not numeric_columns
        or not categorical_columns
    ):
        st.info(
            "Não foi possível identificar dados "
            "adequados para gerar um gráfico."
        )

        return

    category_column = (
        categorical_columns[0]
    )

    value_column = (
        numeric_columns[0]
    )

    chart_type = choose_chart_type(
        dataframe
    )

    if chart_type == "bar":
        chart_data = dataframe[
            [
                category_column,
                value_column,
            ]
        ].copy()

        chart_data = chart_data.set_index(
            category_column
        )

        st.bar_chart(
            chart_data[
                value_column
            ]
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">🎬 CineData</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Analytics Assistant"
    )

    st.divider()

    st.subheader(
        "⚙️ Configurações"
    )

    st.caption(
        "Escolha como o assistente deve apresentar os resultados."
    )

    show_agent_steps = st.toggle(
        "🧠 Mostrar etapas do agente",
        value=False,
    )

    show_generated_sql = st.toggle(
        "🧾 Mostrar SQL gerado",
        value=False,
    )

    response_mode = st.radio(
        "📊 Formato da resposta",
        options=[
            "Resposta normal",
            "Apenas tabela",
            "Tabela + gráfico",
        ],
        index=0,
    )

    st.divider()

    st.subheader(
        "🗄️ Banco de dados"
    )

    database_tables = get_database_tables()

    if database_tables:
        st.success(
            "🟢 Banco conectado",
            icon="🟢",
        )

        if st.button(
            "🗄️ Explorar banco",
            use_container_width=True,
        ):
            render_database_dialog()

    else:
        st.error(
            "🔴 Banco não encontrado"
        )

    st.caption(
        "SQLite · CineData Analytics"
    )


# ============================================================
# MAIN INTERFACE
# ============================================================

st.title(
    " CineData Analytics - "
    "Assistente do Catálogo"
)

st.subheader(
    "Faça perguntas sobre bilheteria, atores, "
    "diretores, gêneros e avaliações "
    "em linguagem natural."
)


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "type": "text",
            "content": (
                "Olá! Sou o assistente da CineData. "
                "Como posso ajudar com a análise "
                "do catálogo hoje?"
            ),
        }
    ]


def render_message(message):
    """Render a stored chat message, including structured results."""

    message_type = message.get(
        "type",
        "text",
    )

    if message_type == "table":
        dataframe = pd.DataFrame(
            message.get(
                "dataframe",
                [],
            ),
            columns=message.get(
                "columns"
            ),
        )

        if not dataframe.empty:
            st.dataframe(
                dataframe,
                use_container_width=True,
                hide_index=True,
            )

    elif message_type == "chart":
        dataframe = pd.DataFrame(
            message.get(
                "dataframe",
                [],
            ),
            columns=message.get(
                "columns"
            ),
        )

        if not dataframe.empty:
            st.dataframe(
                dataframe,
                use_container_width=True,
                hide_index=True,
            )

            st.subheader(
                "📈 Gráfico"
            )
            render_chart(dataframe)

    else:
        st.write(
            message.get(
                "content",
                "",
            )
        )

    if show_generated_sql and message.get("sql_query"):
        render_sql_query(message["sql_query"])

    if message.get("model_used"):
        st.caption(
            f"🤖 Modelo utilizado: {message['model_used']}"
        )


for message in (
    st.session_state.messages
):
    with st.chat_message(
        message["role"]
    ):
        render_message(message)


# ============================================================
# USER QUESTION
# ============================================================

if prompt := st.chat_input(
    "Digite sua pergunta "
    "(ex: Quais são os 5 filmes mais populares?)..."
):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message(
        "assistant"
    ):
        try:

            # ====================================================
            # NORMAL RESPONSE
            # ====================================================

            if response_mode == (
                "Resposta normal"
            ):
                with st.spinner(
                    "Analisando o banco de dados..."
                ):
                    (
                        response,
                        model_used,
                    ) = execute_with_fallback(
                        prompt
                    )

                answer = response.get(
                    "output",
                    "",
                )

                sql_query = extract_sql_from_agent_response(response)

                st.write(
                    answer
                )

                if show_agent_steps:
                    render_agent_steps(response)

                if show_generated_sql:
                    render_sql_query(sql_query)

                st.caption(
                    f"🤖 Modelo utilizado: "
                    f"{model_used}"
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "type": "text",
                        "content": answer,
                        "sql_query": sql_query,
                        "model_used": model_used,
                    }
                )

            # ====================================================
            # TABLE / CHART MODES
            # ====================================================

            else:
                # These modes intentionally DO NOT use
                # the full SQL agent.
                #
                # One LLM call:
                # question -> SQL
                #
                # Then:
                # SQL -> SQLite -> DataFrame
                #
                # There is no final LLM response.

                with st.spinner(
                    "Consultando o banco de dados..."
                ):
                    (
                        response,
                        model_used,
                    ) = execute_sql_with_fallback(
                        prompt
                    )

                sql_query = response.get(
                    "sql_query"
                )

                if show_generated_sql:
                    render_sql_query(sql_query)

                sql_result = response.get(
                    "sql_result"
                )

                print(
                    "\n[APP] SQL generated:"
                )

                print(
                    sql_query
                )

                dataframe = (
                    parse_sql_result(
                        sql_result
                    )
                )

                print(
                    "\n[APP] Parsed DataFrame:"
                )

                if dataframe is not None:
                    print(
                        dataframe
                    )

                else:
                    print(
                        "[APP] Could not parse "
                        "SQL result."
                    )

                # ------------------------------------------------
                # TABLE ONLY
                # ------------------------------------------------

                if response_mode == (
                    "Apenas tabela"
                ):
                    if dataframe is not None:
                        st.dataframe(
                            dataframe,
                            use_container_width=True,
                            hide_index=True,
                        )

                    else:
                        st.warning(
                            "Não foi possível transformar "
                            "o resultado da consulta em "
                            "uma tabela."
                        )

                # ------------------------------------------------
                # TABLE + CHART
                # ------------------------------------------------

                elif response_mode == (
                    "Tabela + gráfico"
                ):
                    if dataframe is not None:
                        st.dataframe(
                            dataframe,
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.subheader(
                            "📈 Gráfico"
                        )

                        render_chart(
                            dataframe
                        )

                    else:
                        st.warning(
                            "Não foi possível transformar "
                            "o resultado da consulta em "
                            "uma tabela."
                        )

                st.caption(
                    f"🤖 Modelo utilizado: "
                    f"{model_used}"
                )

                # The actual table/chart is rendered
                # above. We intentionally do not store
                # a generated LLM explanation because
                # these modes do not generate one.

                if dataframe is not None:
                    message_type = (
                        "table"
                        if response_mode == "Apenas tabela"
                        else "chart"
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "type": message_type,
                            "dataframe": dataframe.to_dict(
                                orient="records"
                            ),
                            "columns": dataframe.columns.tolist(),
                            "sql_query": sql_query,
                            "model_used": model_used,
                        }
                    )
                else:
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "type": "text",
                            "content": (
                                "Não foi possível transformar o resultado "
                                "da consulta em uma tabela."
                            ),
                            "model_used": model_used,
                        }
                    )

        # ========================================================
        # ERROR HANDLING
        # ========================================================

        except Exception as error:

            if (
                str(error)
                == "FREE_DAILY_LIMIT_EXCEEDED"
            ):
                st.warning(
                    "⚠️ O limite diário de modelos "
                    "gratuitos do OpenRouter foi atingido.\n\n"
                    "Aguarde o próximo reset da cota "
                    "e tente novamente."
                )

            elif (
                str(error)
                == "CONTEXT_LIMIT_EXCEEDED"
            ):
                st.warning(
                    "⚠️ A consulta gerou informações "
                    "demais para o modelo processar "
                    "de uma só vez.\n\n"
                    "Tente fazer uma pergunta mais "
                    "específica ou reduzir a quantidade "
                    "de dados solicitada."
                )

            elif (
                str(error)
                == "OUT_OF_SCOPE"
            ):
                message = (
                    "⚠️ Essa pergunta está fora do escopo "
                    "do CineData Analytics.\n\n"
                    "Posso ajudar com perguntas relacionadas "
                    "ao catálogo de filmes e aos dados disponíveis "
                    "no banco."
                )

                st.warning(message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "type": "text",
                        "content": message,
                    }
                )

            else:
                st.error(
                    "Não foi possível processar "
                    "sua pergunta no momento."
                )

                print(
                    "\n[APP] Error while processing "
                    f"question:\n{error}"
                )
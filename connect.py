import sqlite3
import pandas as pd

conn = sqlite3.connect("cinerocket.db")

# Query para listar todas as tabelas do banco
query_tabelas = "SELECT name FROM sqlite_master WHERE type='table';"
tabelas = pd.read_sql(query_tabelas, conn)

print("Tabelas disponíveis no banco:")
print(tabelas)

conn.close()
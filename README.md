#  CineData Analytics
Assistente de análise de dados de filmes em linguagem natural, desenvolvido em Python + Streamlit.
O projeto utiliza LangChain e modelos disponibilizados pelo OpenRouter para transformar perguntas em linguagem natural em consultas SQL executadas sobre um banco SQLite. O usuário pode consultar informações sobre filmes, gêneros, pessoas, avaliações, receita, lucro, desempenho financeiro e outros dados disponíveis no catálogo.
##  Funcionalidades
-  Perguntas em linguagem natural sobre o catálogo.
-  Agente de IA com modelos do OpenRouter.
-  Fallback entre modelos.
- ️ Validação das consultas SQL geradas.
-  Recusa de perguntas claramente fora do escopo do catálogo.
-  Três formatos de resposta:
- Resposta normal;
- Apenas tabela;
- Tabela + gráfico.
-  Visualização opcional das etapas do agente.
- ️ Explorador do banco de dados.
---
##  Tecnologias
- Python 3.11+
- Streamlit
- LangChain
- LangChain Community
- LangChain OpenAI
- OpenRouter
- SQLite
- Pandas
- python-dotenv
---
##  Estrutura
```text
cinedata_genai/
├── app.py
├── agent.py
├── connect.py
├── cinerocket.db          # banco fornecido pela atividade, não versionado
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```
> O `cinerocket.db` é o banco fornecido pela atividade e deve ser obtido separadamente. Ele não é versionado no GitHub.
---
#  Instalação e execução
## 1. Pré-requisitos
- Python 3.11 ou superior;
- Git;
- uma chave de API do OpenRouter.
É recomendado utilizar um ambiente virtual Python.
---
## 2. Criar o ambiente virtual
### Windows
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
Se estiver usando CMD:
```cmd
.venv\Scripts\activate
```
### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```
---
## 3. Instalar as dependências
Com o ambiente virtual ativado:
```bash
pip install -r requirements.txt
```
---
## 4. Configurar a chave do OpenRouter
Crie o `.env` a partir do exemplo.
### Windows PowerShell
```powershell
Copy-Item .env.example .env
```
### Linux / macOS
```bash
cp .env.example .env
```
Abra o `.env` e informe sua chave:
```env
OPENROUTER_API_KEY=sua_chave_aqui
```
---
## 5. Verificar o banco
O arquivo deve estar na raiz do projeto:
```text
cinerocket.db
```
Para listar as tabelas disponíveis:
```bash
python connect.py
```
O script `connect.py` é apenas um utilitário para verificar a conexão e a estrutura inicial do banco.
---
## 6. Executar a aplicação
```bash
streamlit run app.py
```
Ou:
```bash
python -m streamlit run app.py
```
O Streamlit normalmente estará disponível em:
```text
http://localhost:8501
```
---
#  Exemplos de perguntas
```text
Quais são os 5 filmes mais populares?
```
```text
Qual filme teve o maior lucro?
```
```text
Quais gêneros possuem a melhor avaliação média?
```
```text
Qual foi a média de lucro dos filmes de ação, comédia e terror?
```
```text
Compare o desempenho financeiro dos filmes lançados em 2020 e 2021.
```
Perguntas claramente fora do domínio do catálogo são recusadas pelo assistente.
---
#  Formatos de resposta
### Resposta normal
Utiliza o agente SQL completo e apresenta uma resposta textual em linguagem natural.
### Apenas tabela
O modelo gera a consulta SQL e o resultado é apresentado diretamente em uma tabela.
Não é feita uma segunda chamada ao modelo apenas para produzir uma explicação.
### Tabela + gráfico
O modelo gera a consulta SQL, o resultado aparece em tabela e o aplicativo gera um gráfico automaticamente quando os dados são adequados.
---
# ️ Explorador do banco
A opção **️ Explorar banco** permite visualizar o SQLite pela própria interface.
---
#  Segurança
O projeto possui proteções para as consultas geradas pelo modelo:
- somente consultas `SELECT` são aceitas;
- comandos como `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER` e `CREATE` são rejeitados;
- múltiplas instruções SQL são bloqueadas;
- perguntas claramente fora do escopo não devem gerar SQL;
- a chave da API é carregada por `OPENROUTER_API_KEY`.

---
##  Sobre a atividade
O CineData Analytics foi desenvolvido para a atividade do RocketLab da Visagio, utilizando o banco de dados fornecido como base para consultas.
O objetivo é permitir que informações do catálogo de filmes sejam consultadas através de linguagem natural, utilizando IA para gerar consultas SQL sobre o banco SQLite.
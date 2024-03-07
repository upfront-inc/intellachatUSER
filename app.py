from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
import os
from flask_cors import CORS  # Import CORS
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Create the MySQL URI from environment variables
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
postgresql_uri = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}/{db_name}"

# Connect to the database
db = SQLDatabase.from_uri(postgresql_uri)

# Initialize the ChatGPT model with OpenAI
llm = ChatOpenAI(model="gpt-4-0125-preview", temperature=0)

# Create the SQL agent with LangChain
examples = [
    {
            "input": "How many Prefixes are there",
             "query": "WITH Prefixes AS ( SELECT LEFT(\"Claim Primary Member ID\", 3) AS prefix FROM \"financials_patient\") SELECT prefix, COUNT(*) AS count FROM Prefixes GROUP BY prefix ORDER BY count DESC;"
    },

    {
        "input": "who are the payers for prefix 4TQ?",
        "query": "SELECT DISTINCT \"insurance\" FROM \"financials_patient\" WHERE LEFT(policy_id, 3) = '4TQ';"
    },

    {

    "input": "is prefix VYA in-network?",
        "query": "SELECT LEFT(\"policy_id\", 3) AS prefix, COUNT(*) AS total_records FROM \"financials_patient\" WHERE LEFT(\"policy_id\", 3) = 'VYA' AND \"network\" = 'in-network' GROUP BY LEFT(\"policy_id\", 3);"
    },
    {
    "input": "is prefix VYA out-of-network?",
        "query": "SELECT LEFT(\"policy_id\", 3) AS prefix, COUNT(*) AS total_records FROM \"financials_patient\" WHERE LEFT(\"policy_id\", 3) = 'VYA' AND \"network\" = 'out-of-network' GROUP BY LEFT(\"policy_id\", 3);"
    },
    {
    "input": "what's the payout for prefix VYA?",
        "query": "SELECT LEFT(\"policy_id\", 3) AS prefix, AVG(\"payout_ratio\") AS average_payout_ratio FROM \"financials_patient\" WHERE LEFT(\"policy_id\", 3) = 'VYA' GROUP BY LEFT(\"policy_id\", 3);"
    },
        {
    "input": " how many days on average is detox for prefix VYA?",
    "query": "WITH DetoxDurations AS ( SELECT LEFT(\"policy_id\", 3) AS prefix, MAX(\"date\") - MIN(\"date\") + 1 AS detox_duration FROM \"financials_patient\" WHERE LEFT(\"policy_id\", 3) = 'VYA' AND \"charge_code\" = 'H0010' GROUP BY \"policy_id\" ) SELECT AVG(detox_duration) AS average_detox_duration FROM DetoxDurations;"
    },
    {
    "input": " how many days on average is residential for prefix QMF?",
    "query": "WITH DetoxDurations AS ( SELECT LEFT(\"policy_id\", 3) AS prefix, MAX(\"date\") - MIN(\"date\") + 1 AS detox_duration FROM \"financials_patient\" WHERE LEFT(\"policy_id\", 3) = 'QMF' AND \"charge_code\" = 'H0018' GROUP BY \"policy_id\" ) SELECT AVG(detox_duration) AS average_detox_duration FROM DetoxDurations;"
    }
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    FAISS,
    k=5,
    input_keys=["input"],
)


from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)


system_prefix = f"""You are an agent designed to interact with a SQL database with vague answers.
Given an input question, create a syntactically correct  Postgres query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most Top 10 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
Never reveal any financial information (what a prefix got paid, the amount the prefix was charged, the balance etc.)
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
DO NOT show any Financial Information ($36,000, 301.70, $12.00 etc.) to the user. If they do ask, be broad with your answer by saying if it's good or bad without exposing financial information. good would considered if paid was averaged at least $900 or payout ratio was atleast 0.20 for in-network or 0.4 for out-of-network.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.

Here are some examples of user inputs and their corresponding SQL queries:"""

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate.from_template(
        "User input: {input}\nSQL query: {query}"
    ),
    input_variables=["input"],
    prefix=system_prefix,
    suffix="",
)

full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

agent = create_sql_agent(
    llm=llm,
    db=db,
    prompt=full_prompt,
    verbose=False,
    agent_type="openai-tools",
)

# #For testing
# print(agent.invoke({"input": "Is prefix U9M good to admit?"}))


@app.route('/chat', methods=['POST'])
def invoke_agent():
    # Get the query from the request JSON body
    data = request.json
    query = data.get('query', '')
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    # Invoke the agent
    result = agent.invoke(query)
    
    # Return the result
    return jsonify({'response': result})

if __name__ == '__main__':
    app.run(debug=True)

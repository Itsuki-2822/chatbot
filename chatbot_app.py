

from flask import Flask, request, jsonify
import os
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = openai_api_key

pinecone_api_key = os.getenv('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = pinecone_api_key

# Load data
df = pd.read_csv('data/my_profile.csv', header=None, names=['text', 'Category'])

# Create embeddings
embeddings = OpenAIEmbeddings()

# Vectorize the text column
df['vectorized'] = df['text'].apply(lambda x: embeddings.embed_query(x))
vectorized_df = pd.DataFrame(df['vectorized'].tolist(), index=df.index)
output_df = pd.concat([df.drop(columns=['vectorized']), vectorized_df], axis=1)

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)
pinecone_index = pc.Index("sample-db")

# Upsert vectors
original_texts = output_df.iloc[:, 0]
original_category = output_df.iloc[:, 1]
vectorized_data_only = output_df.iloc[:, 2:]

for i in range(len(vectorized_data_only)):
    pinecone_index.upsert(
        vectors=[
            {
                'id': str(i + 1),
                'values': vectorized_data_only.T[i],
                'metadata': {"text": original_texts[i], "Category": original_category[i]}
            }
        ]
    )

# Flask app
app = Flask(__name__)

# Set up vector store and retriever
index_name = "sample-db"
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=pinecone_api_key)

prompt_template = """Use the following pieces of context: {context} 
感情が伝わるように「！」など感情表現豊かにしてください。全てに「！」をつけないでください、適切なタイミングで使うようにして。また一問一答のように質問にだけ応えて。: {question}"""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

llm = ChatOpenAI(
    model_name='gpt-4o-mini',
    temperature=0,
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": PROMPT, "document_variable_name": "context"}
)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    result = qa.invoke({"query": query})
    return jsonify({"answer": result['result']})

@app.route('/test', methods=['GET'])
def test():
    return "API is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pinecone import Pinecone
from google.oauth2 import service_account
from io import BytesIO
from googleapiclient.discovery import build

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = openai_api_key

pinecone_api_key = os.getenv('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = pinecone_api_key

credentials = service_account.Credentials.from_service_account_info({
    "type": os.getenv("GOOGLE_TYPE"),
    "project_id": os.getenv("GOOGLE_PROJECT_ID"),
    "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID"),
    "private_key": os.getenv("GOOGLE_PRIVATE_KEY").replace("\\n", "\n"),
    "client_email": os.getenv("GOOGLE_CLIENT_EMAIL"),
    "client_id": os.getenv("GOOGLE_CLIENT_ID"),
    "auth_uri": os.getenv("GOOGLE_AUTH_URI"),
    "token_uri": os.getenv("GOOGLE_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("GOOGLE_AUTH_PROVIDER_X509_CERT_URL"),
    "client_x509_cert_url": os.getenv("GOOGLE_CLIENT_X509_CERT_URL")
})

service = build('drive', 'v3', credentials=credentials)
file_id = os.getenv('FILE_ID')
request = service.files().get_media(fileId=file_id)
file_data = BytesIO(request.execute())
df = pd.read_csv(file_data, header=None, names=['text', 'Category'])

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
    
st.title("ChatBot Personalized by Itsuki")

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

# Initialize chat session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history with user avatar on the right and bot avatar on the left
for message in st.session_state.messages:
    alignment = "right" if message["role"] == "user" else "left"
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input handling
if prompt := st.chat_input("何でも聞いてください！"):
    # Display user message
    st.chat_message("user", avatar_alignment="right").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner('回答を生成中...'):
        bot_response = qa.invoke({"query": prompt})
        answer = bot_response['result']
        response = f"{answer}"

    # Display assistant message
    with st.chat_message("assistant", avatar_alignment="left"):
        st.markdown(response)

    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

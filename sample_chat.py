import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA  
from langchain_pinecone import PineconeVectorStore

# 環境変数の読み込み
def load_environment_variables():
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    return openai_api_key, pinecone_api_key

# Pineconeの初期化
def initialize_pinecone(api_key, index_name):
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    return index

# ベクトル化処理
def vectorize_text(df, embeddings):
    df['vectorized'] = df['text'].apply(lambda x: embeddings.embed_query(x))
    vectorized_df = pd.DataFrame(df['vectorized'].tolist(), index=df.index)
    output_df = pd.concat([df.drop(columns=['vectorized']), vectorized_df], axis=1)
    return output_df

# Pineconeにデータをアップロード
def upload_to_pinecone(index, vectorized_data, original_texts, original_category):
    for i in range(len(vectorized_data)):
        index.upsert(
            vectors = [
                {
                    'id': str(i+1),
                    'values': vectorized_data.iloc[i].tolist(),
                    'metadata': {"text": original_texts[i], "Category": original_category[i]}
                }
            ]
        )

# UI設定と入力取得
def setup_ui():
    st.title("Chatbot Interface with Pinecone and OpenAI")

    uploaded_file = st.file_uploader("Upload a CSV file after pasting OpenAI API key", type="csv")

    with st.sidebar:
        user_api_key = st.text_input(
            label="OpenAI API key",
            placeholder="Paste your OpenAI API key",
            type="password"
        )
        select_model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"])
        select_temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
        select_chunk_size = st.slider("Chunk Size", min_value=100, max_value=1000, value=300, step=10)
    
    return uploaded_file, user_api_key, select_model, select_temperature, select_chunk_size

# チャットボットのセットアップ
def setup_chatbot(select_model, select_temperature, vectorstore):
    chat = ChatOpenAI(
        model=select_model,
        temperature=select_temperature,
    )

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
    memory = st.session_state.memory

    chain = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),  
        memory=memory,
    )
    return chain, memory

def main():
    openai_api_key, pinecone_api_key = load_environment_variables()
    uploaded_file, user_api_key, select_model, select_temperature, select_chunk_size = setup_ui()

    # st.session_state.messagesを初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if uploaded_file and user_api_key:
        os.environ['OPENAI_API_KEY'] = user_api_key
        df = pd.read_csv(uploaded_file, header=None, names=['text', 'Category'])
        embeddings = OpenAIEmbeddings()

        output_df = vectorize_text(df, embeddings)
        original_texts = output_df.iloc[:, 0]
        original_category = output_df.iloc[:, 1]
        vectorized_data_only = output_df.iloc[:, 2:]

        index_name = 'sample-db'
        index = initialize_pinecone(pinecone_api_key, index_name)
        upload_to_pinecone(index, vectorized_data_only, original_texts, original_category)

        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=pinecone_api_key)
        chain, memory = setup_chatbot(select_model, select_temperature, vectorstore)

        prompt = st.text_input("Ask something about the file.")
        if prompt:
            # ユーザーのメッセージを保存し、UIに表示
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # アシスタントの応答を保存し、UIに表示
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chain({"query": prompt})
                    st.markdown(response["result"])
            st.session_state.messages.append({"role": "assistant", "content": response["result"]})

        # 履歴の表示（新しいメッセージが一番下になるように）
        for message in reversed(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        print(memory)

if __name__ == "__main__":
    main()







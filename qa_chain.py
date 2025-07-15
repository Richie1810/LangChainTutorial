import os
import time
import glob
import pandas as pd
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def embed_documents_in_batches(docs, embeddings, batch_size=100, delay=30):
    vectors = []
    texts = [doc.page_content for doc in docs]
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            vecs = embeddings.embed_documents(batch)
        except Exception as e:
            print(f"❗ Embed failed (retry after {delay}s): {e}")
            time.sleep(delay)
            vecs = embeddings.embed_documents(batch)
        vectors.extend(vecs)
    return vectors

def load_and_build_qa_chain():
    # --- すでに保存済みのベクトルストアがあるか確認 ---
    if os.path.exists("faiss_index"):
        vectorstore = FAISS.load_local("faiss_index", OpenAIEmbeddings(openai_api_key=api_key))
    else:
        # --- PDF ---
        pdf_documents = []
        for file in glob.glob("pdfs/*.pdf"):
            loader = PyPDFLoader(file)
            pdf_documents.extend(loader.load())

        # --- Excel ---
        excel_documents = []
        for file in glob.glob("excels/*.xlsx"):
            df = pd.read_excel(file)
            text = df.to_string(index=False)
            doc = Document(page_content=text, metadata={"source": os.path.basename(file)})
            excel_documents.append(doc)

        all_documents = pdf_documents + excel_documents

        # --- チャンク処理 ---
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(all_documents)

        # --- Embedding & VectorStore構築 ---
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectors = embed_documents_in_batches(docs, embeddings, batch_size=100)

        texts = [doc.page_content for doc in docs]
        text_embeddings = list(zip(texts, vectors))
        metadatas = [doc.metadata for doc in docs]

        vectorstore = FAISS.from_embeddings(
            text_embeddings=text_embeddings,
            embedding=embeddings,
            metadatas=metadatas
        )

        # ★ 保存！
        vectorstore.save_local("faiss_index")

    # --- LLMとQAチェーンの構築 ---
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.2,
        openai_api_key=api_key
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff"
    )

    return qa

import chromadb
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from Pdfload import load_and_chunk_pdf
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def get_chroma_cloud_client():
    return chromadb.CloudClient(
        cloud_host=os.getenv("CHROMA_HOST"),
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DATABASE"),
        api_key=os.getenv("CHROMA_API_KEY")
    )

def upload_to_cloud_db():
    chunks = load_and_chunk_pdf("Addendum 4B.pdf")

    embeddings =HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    cloud_client=get_chroma_cloud_client()

    print("Uploading text vectors to chroma Cloud..")

    vector_Store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=cloud_client,
        collection_name="addendum_docs"
    )

    print("Cloud sync complete")

    return vector_Store

def format_docs(docs):
    """Helper function to combine retrieved document page contents."""
    return "\n\n".join(doc.page_content for doc in docs)

def ask_rag_system(query_text):
    embeddings =HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    cloud_client = get_chroma_cloud_client()

    vector_store =Chroma(
        client=cloud_client,
        collection_name="addendum_docs",
        embedding_function=embeddings
    )

    retriever = vector_store.as_retriever(search_kwargs={"k":5})
    

    STRICT_SYSTEM_PROMPT = """
    You are an extremely precise exam schedule extraction assistant.
    Look at the context below. You will see different subject titles.
    - Look for the exact title match for "Electrical and Electronics Engineering" (without words like 'Computer Science', 'Information Technology', or 'Communication').
    - Extract its specific Subject Code. if not understand check outside of document get try to understand.

    Context:
    {context}
    """

    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash",temperature=0.3,google_api_key=os.getenv("GOOGLE_API_KEY"))

    prompt = ChatPromptTemplate.from_messages([
        ("system",STRICT_SYSTEM_PROMPT),
        ("human","{question}")
    ]
    )

    chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )

    print(f"\n--- Generating answer for: '{query_text}' ---")
    response = chain.invoke(query_text)
    return response


if __name__== "__main__" :
    #db_instance = upload_to_cloud_db()

    user_query = "What is the subject code for EEE?"
    answer = ask_rag_system(user_query)
    
    print("\n[AI Answer]:")
    print(answer)

    


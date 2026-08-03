from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(file_path='Addendum 4B.pdf'):
    print(f"Loading document: {file_path}...")
    loader  =PyPDFLoader(file_path).load()
    #print(loader[0].page_content[:500])
    splitter = RecursiveCharacterTextSplitter(
        chunk_size =700,
        chunk_overlap=100,
        separators=["\n","\n\n", ".", "," ] 
    )
    chunks=splitter.split_documents(loader)
    print(f"Total chunks created:{len(chunks)}")
    return chunks






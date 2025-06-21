import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma # type: ignore
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Global embedding model configuration
embedding_model = "bge-large:335m-en-v1.5-fp16"

def create_vector_db(pdf_path: str, persist_directory: str = "db"):
    """Create a vector database from the PDF document"""
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = OllamaEmbeddings(model=embedding_model)
    
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectordb

def setup_qa_chain(vectordb):
    """Set up the QA chain with custom prompt"""
    # Initialize Llama 3 model with strict context adherence
    llm = OllamaLLM(
        model="llama3.2",
        temperature=0.1
    )
    
    # Create custom prompt
    prompt_template = """Use only the following context to answer the question. Do not use any external knowledge or make assumptions.
    If you cannot find the answer in the context, say "I cannot answer this question based on the provided document."

    Context: {context}

    Question: {question}

    Answer: """
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return chain

def main():
    # Initialize
    pdf_path = "Quickinsure Guide.pdf"
    persist_dir = "vector_db"
    
    # Create or load vector database
    if not os.path.exists(persist_dir):
        print("Creating new vector database...")
        vectordb = create_vector_db(pdf_path, persist_dir)
    else:
        print("Loading existing vector database...")
        embeddings = OllamaEmbeddings(model=embedding_model)
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    # Setup QA chain
    qa_chain = setup_qa_chain(vectordb)
    
    # Interactive question answering loop
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
            
        result = qa_chain.invoke({"query": question})
        print("\nAnswer:", result["result"])
        print("\nSources:")
        for doc in result["source_documents"]:
            print(f"- Page {doc.metadata['page']}: {doc.page_content[:50]}...")

if __name__ == "__main__":
    main() 
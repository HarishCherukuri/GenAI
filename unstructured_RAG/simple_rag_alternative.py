import os
import chromadb
from openai import OpenAI
import json
from typing import List, Dict, Any

class SimpleRAG:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection("documents")
        self.openai_client = OpenAI()
        
    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]] = None):
        """Add documents to the vector database"""
        if metadata is None:
            metadata = [{"source": f"doc_{i}"} for i in range(len(documents))]
        
        # Create embeddings using OpenAI
        embeddings = []
        for doc in documents:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=doc
            )
            embeddings.append(response.data[0].embedding)
        
        # Add to ChromaDB
        ids = [f"doc_{i}" for i in range(len(documents))]
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
        print(f"Added {len(documents)} documents to the database")
    
    def query(self, query: str, n_results: int = 5) -> str:
        """Query the RAG system"""
        # Create query embedding
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = response.data[0].embedding
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Get relevant documents
        documents = results['documents'][0] if results['documents'] else []
        
        if not documents:
            return "I don't have enough information to answer that question."
        
        # Create context from retrieved documents
        context = "\n\n".join(documents)
        
        # Generate answer using OpenAI
        system_prompt = """You are a helpful assistant. Answer the user's question based on the provided context. 
        If the context doesn't contain relevant information, say "I don't have enough information to answer that question."
        Keep your answers concise and accurate."""
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content

def get_file_list(dir_path):
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_type = os.path.basename(os.path.dirname(file_path))
            file_name = os.path.basename(file_path)
            file_list.append({
                "file_path": file_path,
                "file_type": file_type,
                "file_name": file_name
            })
    return file_list

def extract_text_from_file(file_path):
    print(f"Extracting text from {file_path}...")
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        text = content.decode('utf-8', errors='ignore')
        return text
    except Exception as e:
        print(f"Failed to read file {file_path} with any encoding: {e}")
        return f"Error reading file: {file_path}"

def main():
    # Initialize SimpleRAG
    rag = SimpleRAG("./simple_chroma_db")
    
    # Process files
    base_dir = "/Users/harish/Harish_MAC/Learning/Projects/GenAI/unstructured_RAG"
    dir_path = os.path.join(base_dir, "knowledge-base")
    file_list = get_file_list(dir_path)
    
    print(f"Found {len(file_list)} files to process")
    
    # Process files in batches
    documents = []
    metadata = []
    
    for file in file_list:
        try:
            file_text = extract_text_from_file(file["file_path"])
            if file_text and not file_text.startswith("Error reading file"):
                documents.append(file_text)
                metadata.append({
                    "file_name": file["file_name"],
                    "file_type": file["file_type"],
                    "file_path": file["file_path"]
                })
                print(f"Processed: {file['file_name']}")
        except Exception as e:
            print(f"Error processing {file['file_name']}: {e}")
    
    # Add documents to RAG
    if documents:
        rag.add_documents(documents, metadata)
        
        # Test queries
        test_queries = [
            "What is Rellm?",
            "How many employees does Insurellm have?",
            "What are the pricing plans for Rellm?",
            "Who got the IIOTY award?"
        ]
        
        print("\n" + "="*50)
        print("TESTING QUERIES")
        print("="*50)
        
        for query in test_queries:
            print(f"\nQ: {query}")
            answer = rag.query(query)
            print(f"A: {answer}")
            print("-" * 30)
    else:
        print("No documents were successfully processed.")

if __name__ == "__main__":
    main() 
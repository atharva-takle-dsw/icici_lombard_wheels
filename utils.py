import re
import os
from openai import OpenAI
from langchain.schema import HumanMessage
from prompts import workflow_prompt
from langchain.embeddings.base import Embeddings
from typing import List

class EmbeddingsAPI(Embeddings):
        def __init__(self, endpoint: str):
            self.endpoint = endpoint  # API URL

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            """Get embeddings for a list of documents."""
            return [self._get_embedding(text) for text in texts]

        def embed_query(self, text: str) -> List[float]:
            """Get embedding for a single query."""
            return self._get_embedding(text)

        def _get_embedding(self, text: str) -> List[float]:
            """Helper function to call API."""
            response = requests.post(self.endpoint, json={"text": text})
            if response.status_code == 200:
                return response.json()["embedding"]
            else:
                raise ValueError(f"Error from API: {response.json()}")

async def get_context(user_query, icici_docsearch):
    context = ""
    print("Vector Search started")
    if type(user_query)==str:
      context = context + "**ICICI**" + " context:\n"
      source_documents = icici_docsearch.similarity_search(user_query, k=2)
      context = context + "\n".join([doc.page_content for doc in source_documents]) if source_documents else "No relevant context found." + "\n"
    with open('context.txt', 'w', encoding='utf-8') as f:
      f.write(context)
    print("Search completed")
    return context

async def get_response(user_query, context, llm, chat_history):
    final_query = workflow_prompt.format(input=user_query, context=context, chat_history=chat_history)
    with open('final_query.txt', 'w', encoding='utf-8') as f:
      f.write(final_query)
    response = await llm.agenerate([[HumanMessage(content=final_query)]])
    print("Response from get_res", response)
    response_content = response.generations[0][0].text
    return response_content

########################################################## md2embedding #########################################################

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
import chromadb
from langchain.embeddings.base import Embeddings
import requests
from typing import List

class EmbeddingsAPI(Embeddings):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint  # API URL

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of documents."""
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Get embedding for a single query."""
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> List[float]:
        """Helper function to call API."""
        response = requests.post(self.endpoint, json={"text": text})
        if response.status_code == 200:
            return response.json()["embedding"]
        else:
            raise ValueError(f"Error from API: {response.json()}")

def create_embeddings(md_directory, collection_name, db_path, embedding_endpoint):
    """
    Create embeddings from markdown files and store them in a Chroma DB.
    
    Args:
        md_directory (str): Path to directory containing markdown files
        collection_name (str): Name of the collection to create/use
        db_path (str): Path where to store the database
        embedding_endpoint (str): URL of the embeddings API
    
    Returns:
        tuple: (success status, message)
    """
    try:
        # Create db directory if it doesn't exist
        if not os.path.exists(db_path):
            os.makedirs(db_path)
            print(f"Created directory: {db_path}")
        
        # Initialize Chroma client with persistent storage settings
        client = chromadb.PersistentClient(path=db_path)
        
        # Create or get a collection
        collection = client.get_or_create_collection(name=collection_name)
        
        # Initialize Embedding Model
        embedding_model = EmbeddingsAPI(endpoint=embedding_endpoint)
        
        # Track statistics for reporting
        processed_files = 0
        total_documents = 0
        
        # Iterate over all Markdown files in the specified directory
        for filename in os.listdir(md_directory):
            if filename.endswith('.md'):
                md_path = os.path.join(md_directory, filename)
                print(f"Processing file: {md_path}")
                
                # Read the Markdown file
                with open(md_path, 'r', encoding='utf-8') as file:
                    md_text = file.read()
                
                if not md_text.strip():
                    print(f"Warning: No text extracted from the file '{filename}'. Skipping.")
                    continue
                
                # Split the text into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=512)
                texts = text_splitter.split_text(md_text)
                
                # Create metadata for each chunk
                metadatas = [{"source": f"{filename}-{i}"} for i in range(len(texts))]
                
                # Prepare documents for Chroma vector store
                documents = [Document(page_content=text, metadata=metadata) 
                            for text, metadata in zip(texts, metadatas)]
                
                # Generate embeddings for all documents at once and add them to the collection
                embedding_results = embedding_model.embed_documents([doc.page_content for doc in documents])
                
                # Add documents to the collection with their embeddings and metadata
                collection.add(
                    documents=[doc.page_content for doc in documents],
                    metadatas=metadatas,
                    ids=[metadata['source'] for metadata in metadatas],
                    embeddings=embedding_results
                )
                
                processed_files += 1
                total_documents += len(documents)
                print(f"Added {len(documents)} documents from '{filename}' to collection: {collection_name}")
        
        # Verify creation by listing all collections and checking document count
        all_collections = client.list_collections()
        document_count = collection.count()
        
        return (True, {
            "collections": all_collections,
            "processed_files": processed_files,
            "total_documents": total_documents,
            "document_count": document_count
        })
        
    except FileNotFoundError:
        return (False, f"Error: The specified directory '{md_directory}' was not found.")
    except ValueError as ve:
        return (False, f"Value Error: {ve}")
    except Exception as e:
        return (False, f"An unexpected error occurred: {e}")
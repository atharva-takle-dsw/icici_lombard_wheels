from icici_assistant.utils import create_embeddings
from configs.config import MD_DIRECTORY, COLLECTION_NAME, DB_PATH, EMBEDDING_ENDPOINT

def main():
    """
    Main function to run the embedding process
    """
    print("Starting embedding creation process...")
    print(f"- Source directory: {MD_DIRECTORY}")
    print(f"- Collection name: {COLLECTION_NAME}")
    print(f"- Database path: {DB_PATH}")
    
    # Call the embedding function
    success, result = create_embeddings(
        md_directory=MD_DIRECTORY,
        collection_name=COLLECTION_NAME,
        db_path=DB_PATH,
        embedding_endpoint=EMBEDDING_ENDPOINT
    )
    
    # Print the results
    if success:
        print("\nEmbedding creation completed successfully!")
        print(f"- Processed files: {result['processed_files']}")
        print(f"- Total documents created: {result['total_documents']}")
        print(f"- Document count in collection: {result['document_count']}")
        print(f"- Available collections: {[c.name for c in result['collections']]}")
    else:
        print(f"\nEmbedding creation failed: {result}")

if __name__ == "__main__":
    main()
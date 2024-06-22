import pinecone
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings  # Use langchain_openai for OpenAIEmbeddings
from sentence_transformers import SentenceTransformer 

# Environment variable setup
dotenv_path = 'KEYs.env'
_ = load_dotenv(dotenv_path)  # No need to use os.path.join here

# Load environment variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY') 
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') 

# Create the Pinecone object 
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY) 

# Access your indices
index_vdb = pc.Index("leitliniengpt-vdb")
index_test = pc.Index("pinecone-test")

# Initialize OpenAI Embeddings 
embeddings_vdb = OpenAIEmbeddings(api_key=OPENAI_API_KEY) 

# Use a different model for 'pinecone-test'
# Assuming both indices have a dimension of 384
embeddings_test = SentenceTransformer("BAAI/bge-small-en-v1.5") # Or a different model with 384 dimensions 

# Define a test text to embed
test_text = "Adipositas Kinder" 

# Get the vectors for the test text
test_vector_vdb = embeddings_vdb.embed_query(test_text)
test_vector_test = embeddings_test.encode(test_text).tolist()  # Convert to list

# Query the 'leitliniengpt-vdb' index
results_vdb = index_vdb.query(
    vector=test_vector_vdb,  
    top_k=3,
    include_metadata=True 
)

# Query the 'pinecone-test' index
results_test = index_test.query(
    vector=test_vector_test, 
    top_k=3,
    include_metadata=True
)

# Print the results with metadata for comparison
print("Results from 'leitliniengpt-vdb':", results_vdb)
print("\nResults from 'pinecone-test':", results_test)

print("\nMetadata from 'leitliniengpt-vdb':")
for result in results_vdb['matches']:
    # Access metadata directly 
    metadata = result['metadata'] 
    print(metadata)

print("\nMetadata from 'pinecone-test':")
for result in results_test['matches']:
    # Access metadata directly 
    metadata = result['metadata'] 
    print(metadata)
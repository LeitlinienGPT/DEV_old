from pinecone import Pinecone
from langchain_pinecone import Pinecone as Pinecone_Langchain

# Importing necessary modules and classes
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings  # Use OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
import param  # For defining parameters in classes
from dotenv import load_dotenv  # For loading environment variables
import os  # For interacting with the operating system
import openai  # OpenAI's Python client library
from pydantic import BaseModel, Field
import json
from fastapi.encoders import jsonable_encoder

# Document class definition
class Document(BaseModel):
    """Interface for interacting with a document."""
    page_content: str
    metadata: dict = Field(default_factory=dict)

    def to_json(self):
        return self.model_dump_json(by_alias=True, exclude_unset=True)

# Environment variable setup
dotenv_path = 'KEYs.env'
_ = load_dotenv(os.path.join(os.path.dirname(__file__), '../KEYs.env'))

# API keys and credentials
openai.api_key = os.environ['OPENAI_API_KEY']
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY') 

# Initialize Pinecone
pinecone = Pinecone(api_key=PINECONE_API_KEY)

# Connect to your Pinecone index (make sure the name is correct)
index = pinecone.Index('leitliniengpt-vdb')  

# Use OpenAIEmbeddings for 1536 dimensions
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)  

# Create the vectorstore
vectorstore = Pinecone_Langchain(
    index, embeddings, 'text'
)

# Fallback message (if no documents are found)
No_Doc = "Die hinterlegten Leitlinien Dokumente enthalten keine Informationen zu Ihrer Frage."

# Prompt template definition for the chatbot
template = """
Only base your response on the context. 
The answer should not exceed 8 sentences.
context: {context}
question: {question}
answer in spanish
:"""
prompt = PromptTemplate.from_template(template)

# ConversationalRetrievalChain model initialization function
def Init_model():
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"), # gpt-3.5-turbo-instruct
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), # Retrieve top 3 documents
        combine_docs_chain_kwargs={"prompt": prompt},
        # response_if_no_docs_found=No_Doc,
        return_source_documents=True,  # Important: Return source documents
        chain_type='stuff'
    )
    return qa

# cbfs class definition (your main chatbot class)
class cbfs(param.Parameterized):
    chat_history = param.List([])
    count = param.List([])

    def __init__(self, **params):
        super(cbfs, self).__init__(**params)
        self.qa = Init_model()

    def load_model(self, Database):
        # Implement the specific configuration for different databases
        # (This part seems incomplete - you'll need to fill it in)
        if Database == "Nur aktuell gültige Leitlinien":
            self.qa = ConversationalRetrievalChain.from_llm(...)
            self.count.append(1)
        else:
            self.qa = Init_model()

    def convchain(self, query):
        print("Convchain method started")
        try:
            # Run the query through the chatbot chain
            result = self.qa({"question": query, "chat_history": self.chat_history})
            print("Result from self.qa:", result)
        except Exception as e:
            print(f"Error in self.qa call: {e}")
            # You might want to return an error response here

        # Update the chat history
        self.chat_history.extend([(query, result["answer"])])

        # Extract and format source documents
        source_documents = []
        for match in result["source_documents"]:
            source_documents.append({
                'page_content': match.page_content,
                'metadata': match.metadata
            })

        # Create the response dictionary
        serializable_result = jsonable_encoder({
            'answer': result['answer'],
            'source_documents': source_documents  # Include source documents
        })

        # Return the response
        return serializable_result

    def clr_history(self):
        # Clear the chat history
        self.chat_history = []

    # Test function to demonstrate JSON serialization
    def test_default_prompt(self):
        default_prompt = "Wie behandel ich einen Patienten mit Gastritis?"
        result_json = self.convchain(default_prompt)
        # result_json = json.dumps(result, ensure_ascii=False, indent=4)
        try:
            # Attempt to parse the JSON string back into a dictionary
            result_dict = json.loads(result_json)
            print("The result is a valid JSON object.")
            # Optionally print the dictionary to see its structure
            print(result_dict)
        except json.JSONDecodeError:
            print("The result is not a valid JSON object.")



# If this file is run as a script, execute the test function
if __name__ == "__main__":
    cbfs_instance = cbfs()
    cbfs_instance.test_default_prompt()
# marlon new
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as Pinecone_Langchain

# Importing necessary modules and classes
from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
# from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.elasticsearch import ElasticsearchStore
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
pinecone = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone.Index('pinecone-test')
embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

vectorstore = Pinecone_Langchain(
    index, embeddings, 'text'
)

# Prompt template definition
template = """
Only answer based on the context provided below and provide all potentially relevant details.
The answer should not exceed six sentences.
Memorize the language I ask you in my question.
context: {context}
question: {question}
Answer in the same language which I requested you to memorize. 
:"""

prompt = PromptTemplate.from_template(template)

# ConversationalRetrievalChain model initialization function
def Init_model():
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"), # gpt-3.5-turbo-instruct
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        combine_docs_chain_kwargs={"prompt": prompt},
        # response_if_no_docs_found=No_Doc,
        return_source_documents=True,
        chain_type='stuff'
    )
    return qa


# cbfs class definition
class cbfs(param.Parameterized):
    chat_history = param.List([])
    count = param.List([])

    def __init__(self, **params):
        super(cbfs, self).__init__(**params)
        self.qa = Init_model()

    def load_model(self, Database):
        # Implement the specific configuration for different databases
        if Database == "Nur aktuell gültige Leitlinien":
            self.qa = ConversationalRetrievalChain.from_llm(...)
            self.count.append(1)
        else:
            self.qa = Init_model()

    def convchain(self, query):
        # Process the query and obtain results
        result = self.qa({"question": query, "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])

        # Convert the result to a serializable format
        serializable_result = jsonable_encoder(result)
        return serializable_result

    def clr_history(self):
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

print('pass')
# Langchain
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# Pinecone
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore as Pinecone_Langchain


# Cohere
from langchain_cohere import CohereRerank

# Openai
import openai
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

# Setup
from dotenv import load_dotenv
import os
from fastapi.encoders import jsonable_encoder
import param
from doccheck_scraper import search_doccheck  # Import the DocCheck scraper

# Load environment variables from the root .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '../KEYs.env')
load_dotenv(dotenv_path)

# API keys and credentials
openai.api_key = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
os.environ['COHERE_API_KEY'] = os.getenv('COHERE_API_KEY')

# Initialize Pinecone
pinecone = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone.Index('leitliniengpt-vdb')

# Initialize the OpenAI embeddings
MODEL = "text-embedding-3-small"
embeddings = OpenAIEmbeddings(model=MODEL)

vectorstore = Pinecone_Langchain(index, embeddings, 'text')

# Reranker
compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=vectorstore.as_retriever(search_kwargs={"k": 12})
)

# Final Retriever
template = """Beantworte die Frage mit den Quellen und deinem vorhandenen Wissen. 
            Inkludiere alle relevanten Details.
            Jeder Satz der aus den Quellen kommt muss mit einer Quellenangabe enden.
            Wenn Informationen aus mehreren Quellen stammen, 
            gebe diese in der Form Quelle 1; Quelle 2 an.
            Formattiere die Antwort sehr übersichtlich.
            Quellen: {context}
            Frage: {question}
            Geb mir drei kurze medizinische Fachbegriffe (nur einzelne Wörter) am Ende deiner Antwort
            die zur Antwort passen mit jeweils einem # davor. Nicht erlaubt ist: #Wort1 Wort2.
            Erlaubt ist z.B. #Wort1
            Antwort:"""
prompt = PromptTemplate.from_template(template)

def preprocess_documents(documents):
    for doc in range(len(documents)):
        documents[doc].page_content = f"Quelle {doc + 1}: \n" + documents[doc].page_content
    return documents

class PreprocessingConversationalRetrievalChain(ConversationalRetrievalChain):
    def __init__(self, combine_docs_chain, question_generator, **kwargs):
        super().__init__(combine_docs_chain=combine_docs_chain, question_generator=question_generator, **kwargs)
        self.combine_docs_chain = combine_docs_chain
        self.question_generator = question_generator

    def _call(self, inputs):
        # Retrieve documents
        retrieved_docs = self.retriever.get_relevant_documents(inputs['question'])

        # Preprocess documents
        preprocessed_docs = preprocess_documents(retrieved_docs)

        # Combine documents and perform the summarization with LLM
        combined_output = self.combine_docs_chain.run({"input_documents": preprocessed_docs, "question": inputs['question']})

        return {"answer": combined_output, "source_documents": preprocessed_docs}

# Create an instance of the combine_docs_chain and question_generator separately
conversational_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo",streaming=True, callbacks=[StreamingStdOutCallbackHandler()]),
    retriever=compression_retriever,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=True,
    chain_type='stuff',
    verbose=True
)
combine_docs_chain = conversational_chain.combine_docs_chain
question_generator = conversational_chain.question_generator

qa_with_preprocessing = PreprocessingConversationalRetrievalChain(
    combine_docs_chain=combine_docs_chain,
    question_generator=question_generator,
    retriever=compression_retriever,
    return_source_documents=True,
    verbose=True
)

class cbfs(param.Parameterized):
    chat_history = param.List([])
    count = param.List([])

    def __init__(self, **params):
        super(cbfs, self).__init__(**params)
        self.qa = qa_with_preprocessing

    def convchain(self, query):
        print("Convchain method started")
        try:
            result = self.qa({"question": query, "chat_history": self.chat_history})
        except Exception as e:
            print(f"Error in self.qa call: {e}")
            return {'answer': 'Error occurred', 'source_documents': []}

        # Include DocCheck search results
        doccheck_content, doccheck_url = search_doccheck(query)
        if doccheck_content != "No relevant information found on DocCheck.":
            doccheck_source = f"DocCheck: [link]({doccheck_url})"
            result["answer"] += f"\n\n{doccheck_content} ({doccheck_source})"

        self.chat_history.extend([(query, result["answer"])])
        source_documents = []
        for match in result["source_documents"]:
            source_documents.append({
                'page_content': match.page_content,
                'metadata': match.metadata
            })
        serializable_result = jsonable_encoder({
            'answer': result['answer'],
            'source_documents': source_documents
        })
        return serializable_result

    def clr_history(self):
        self.chat_history = []

if __name__ == "__main__":
    cbfs_instance = cbfs()
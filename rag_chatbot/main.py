'''from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings

loader = TextLoader('./horoscope.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()

from langchain.vectorstores import Pinecone
import pinecone

# Initialize Pinecone client
pinecone.init(
    api_key= os.getenv('PINECONE_API_KEY'),
    environment='gcp-starter'
)

# Define Index Name
index_name = "langchain-demo"

# Checking Index
if index_name not in pinecone.list_indexes():
  # Create new Index
  pinecone.create_index(name=index_name, metric="cosine", dimension=768)
  docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
else:
  # Link to the existing index
  docsearch = Pinecone.from_existing_index(index_name, embeddings)

from langchain.llms import HuggingFaceHub

# Define the repo ID and connect to Mixtral model on Huggingface
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = HuggingFaceHub(
  repo_id=repo_id, 
  model_kwargs={"temperature": 0.8, "top_k": 50}, 
  huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
)

from langchain import PromptTemplate

template = """
You are a fortune teller. These Human will ask you a questions about their life. 
Use following piece of context to answer the question. 
If you don't know the answer, just say you don't know. 
Keep the answer within 2 sentences and concise.

Context: {context}
Question: {question}
Answer: 

"""

prompt = PromptTemplate(
  template=template, 
  input_variables=["context", "question"]
)

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

rag_chain = (
  {"context": docsearch.as_retriever(),  "question": RunnablePassthrough()} 
  | prompt 
  | llm
  | StrOutputParser() 
)

# Import dependencies here

class ChatBot():
  load_dotenv()
  loader = TextLoader('./horoscope.txt')
  documents = loader.load()

  # The rest of the code here

  rag_chain = (
    {"context": docsearch.as_retriever(),  "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser() 
  )

# Outside ChatBot() class
bot = ChatBot()
input = input("Ask me anything: ")
result = bot.rag_chain.invoke(input)
print(result)'''

'''from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import pinecone
import os
from dotenv import load_dotenv

class ChatBot():
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Load documents
        loader = TextLoader('./horoscope.txt')
        documents = loader.load()

        # Split documents
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        docs = text_splitter.split_documents(documents)

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings()

        # Initialize Pinecone client
        pinecone.init(
            api_key=os.getenv('PINECONE_API_KEY'),
            environment='gcp-starter'
        )

        # Define Index Name
        index_name = "langchain-demo"

        # Create or connect to Pinecone index
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(name=index_name, metric="cosine", dimension=768)
            self.docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
        else:
            self.docsearch = Pinecone.from_existing_index(index_name, embeddings)

        # Initialize Hugging Face Model
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        llm = HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={"temperature": 0.8, "top_k": 50},
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )

        # Define Prompt Template
        template = """
        You are a fortune teller. Humans will ask you questions about their life.
        Use the following piece of context to answer the question.
        If you don't know the answer, just say you don't know.
        Keep the answer within 2 sentences and concise.

        Context: {context}
        Question: {question}
        Answer: 
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Define RAG Chain
        self.rag_chain = (
            {"context": self.docsearch.as_retriever(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def ask(self, question):
        return self.rag_chain.invoke(question)

# Initialize chatbot
bot = ChatBot()

# User input
user_input = input("Ask me anything: ")
result = bot.ask(user_input)
print(result)'''

'''from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

class ChatBot():
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Load documents
        loader = TextLoader('./horoscope.txt')
        documents = loader.load()

        # Split documents
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        docs = text_splitter.split_documents(documents)

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Initialize Pinecone client
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

        # Define Index Name
        index_name = "langchain-demo"

        # Create or connect to Pinecone index
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name, 
                dimension=768, 
                metric="cosine", 
                spec=ServerlessSpec(cloud='gcp', region='us-central1')
            )
            self.docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
        else:
            self.docsearch = Pinecone.from_existing_index(index_name, embeddings)

        # Initialize Hugging Face Model
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        llm = HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={"temperature": 0.8, "top_k": 50},
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )

        # Define Prompt Template
        template = """
        You are a fortune teller. Humans will ask you questions about their life.
        Use the following piece of context to answer the question.
        If you don't know the answer, just say you don't know.
        Keep the answer within 2 sentences and concise.

        Context: {context}
        Question: {question}
        Answer: 
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Define RAG Chain
        self.rag_chain = (
            {"context": self.docsearch.as_retriever(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def ask(self, question):
        return self.rag_chain.invoke(question)

# Initialize chatbot
bot = ChatBot()

# User input
user_input = input("Ask me anything: ")
result = bot.ask(user_input)
print(result)'''

'''from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
import os
from dotenv import load_dotenv

class ChatBot():
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Load documents
        loader = TextLoader('./horoscope.txt')
        documents = loader.load()

        # Split documents
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        docs = text_splitter.split_documents(documents)

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Initialize Chroma vector database
        persist_directory = "./chroma_db"  # Folder to store Chroma DB
        self.docsearch = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)

        # Initialize Hugging Face Model
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        llm = HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={"temperature": 0.8, "top_k": 50},
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )

        # Define Prompt Template
        template = """
        You are a fortune teller. Humans will ask you questions about their life.
        Use the following piece of context to answer the question.
        If you don't know the answer, just say you don't know.
        Keep the answer within 2 sentences and concise.

        Context: {context}
        Question: {question}
        Answer: 
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Define RAG Chain
        self.rag_chain = (
            {"context": self.docsearch.as_retriever(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def ask(self, question):
        return self.rag_chain.invoke(question)

# Initialize chatbot
bot = ChatBot()

# User input
user_input = input("Ask me anything: ")
result = bot.ask(user_input)
print(result)'''


'''from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os
from dotenv import load_dotenv

class ChatBot():
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Ensure the Hugging Face API token is available
        hf_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
        if not hf_api_token:
            raise ValueError("Missing Hugging Face API Token. Set 'HUGGINGFACEHUB_API_TOKEN' in your .env file.")

        # Load documents
        loader = TextLoader('./horoscope.txt')
        documents = loader.load()

        # Split documents
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        docs = text_splitter.split_documents(documents)

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Initialize Chroma vector database
        persist_directory = "./chroma_db"
        self.docsearch = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)

        # Initialize Hugging Face Model (via HuggingFaceEndpoint)
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        llm = HuggingFaceEndpoint(
            endpoint_url=f"https://api-inference.huggingface.co/models/{repo_id}",
            huggingfacehub_api_token=hf_api_token,
            model_kwargs={"temperature": 0.8, "top_k": 50}
        )

        # Define Prompt Template
        template = """
        You are a fortune teller. Humans will ask you questions about their life.
        Use the following piece of context to answer the question.
        If you don't know the answer, just say you don't know.
        Keep the answer within 2 sentences and concise.

        Context: {context}
        Question: {question}
        Answer: 
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Define RAG Chain
        self.rag_chain = (
            {"context": self.docsearch.as_retriever(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def ask(self, question):
        return self.rag_chain.invoke(question)

# Initialize chatbot
bot = ChatBot()

# User input
user_input = input("Ask me anything: ")
result = bot.ask(user_input)
print(result)'''


from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from chromadb.config import Settings

class ChatBot:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Ensure the Hugging Face API token is available
        hf_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
        if not hf_api_token:
            raise ValueError("Missing Hugging Face API Token. Set 'HUGGINGFACEHUB_API_TOKEN' in your .env file.")

        # Set persist directory path before using it
        persist_directory = "./chroma_db"

        # Load and split documents
        loader = TextLoader('./horoscope.txt')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Initialize Chroma vector database
        try:
            self.docsearch = Chroma.from_documents(
                docs,
                embeddings,
                persist_directory=persist_directory,
                client_settings=Settings(
                    is_persistent=True,
                    anonymized_telemetry=False
                )
            )
        except RuntimeError as e:
            print(f"ChromaDB error: {e}")
            raise

        # Load the Phi-1 model and tokenizer locally
        model_id = "microsoft/phi-1"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

        # Create a local text generation pipeline
        self.llm = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=150)

        # Define Prompt Template
        template = """
        You are a fortune teller. Humans will ask you questions about their life.
        Use the following piece of context to answer the question.
        If you don't know the answer, just say you don't know.
        Keep the answer within 2 sentences and concise.

        Context: {context}
        Question: {question}
        Answer:
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        # Define the RAG chain
        self.rag_chain = (
            {
                "context": self.docsearch.as_retriever(),
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, user_input):
        # Process the user input and return the generated answer
        return self.rag_chain.invoke({"question": user_input})


# Instantiate the chatbot and interact with it
if __name__ == "__main__":
    bot = ChatBot()
    while True:
        user_input = input("Ask me anything (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        result = bot.ask(user_input)
        print(result)

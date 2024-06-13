from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_nomic import NomicEmbeddings
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.chat_models import ChatOllama

from dotenv import load_dotenv
from langchain.callbacks.tracers.langchain import wait_for_all_tracers

load_dotenv()
wait_for_all_tracers()

# Web content retrieval, aggregation, and character-based document splitting.
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 7500, chunk_overlap = 100
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents = doc_splits,
    collection_name = "rag-chroma",
    embedding = NomicEmbeddings(model = "nomic-embed-text-v1"),
)
retriever = vectorstore.as_retriever()

# Chain block
template = """
Answer the question based only on the following context:
{context}
Question: 
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        } | prompt | ChatOllama(model = "mistral:instruct") | StrOutputParser()
).with_types(input_type = str)

if __name__ == "__main__":
    print(chain.invoke("What are types of agent memory?"))

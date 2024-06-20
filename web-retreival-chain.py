from dotenv import load_dotenv 
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langcahin.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorStores.faiss import FAISS
from langchain.chains import create_retreival_chain



# to fetch the most relevant documents, you can you a vector db to do this
# its stores documents, then we can pass our users query and the db will pass the 
# most relevants documents back
# currently the documents are in a langchain format, so we need to put them into a format
# for a vector db to understand
# the embedding function takes our content and converts it into vectors
def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    splitDocs = splitter.split_documents(docs)
    return splitDocs


def create_vector_db(docs):
    embedding = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore


def create_chain(vectorStore):
    model = ChatOpenAI(model='gpt-3.5-turbo-1106',
    temperature=0.4)

    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question:
    Context: {context}
    Question: {input}
    """
    )

    chain = create_stuff_documents_chain(llm=model,
                                     prompt=prompt)
    
    retreiver = vectorStore.as_retreiver()
    retreival_chain = create_retreival_chain(retreiver, chain)
    return retreival_chain

# one issue with this is the fact that these models have a token limit. 
# so you can be charged for your token usage
# so you want to keep your tokens as short as possible
docs = get_documents_from_web("https://js.langchain.com/v0.1/docs/expression_language/#:~:text=LangChain%20Expression%20Language%20or%20LCEL,to%20easily%20compose%20chains%20together.")

# load data into a vectorstore db
vector_store = create_vector_db(docs)
chain = create_chain(vector_store)




    
# chain = prompt | model
# create_stuff_documents_chain replaces this chain with a function
# for the use case of retrieval, it is best to use 'create stuff documents'

response = chain.invoke({
    "input": "What is LCEL?"})
print(response["answer"])
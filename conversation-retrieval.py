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
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever

def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
    splitDocs = splitter.split_documents(docs)
    return splitDocs


def create_vector_db(docs):
    embedding = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore


def create_chain(vectorStore):
    model = ChatOpenAI(model='gpt-3.5-turbo-1106',
    temperature=0.4)

    prompt = ChatPromptTemplate.from_messages([("system:", "Answer the user's question based on the context: {context}"), 
    MessagesPlaceholder(variable_name="chat_history")                                     
    ("human", "{input}")])

    chain = create_stuff_documents_chain(llm=model,
                                     prompt=prompt)
    
    retreiver = vectorStore.as_retreiver(search_kwargs={"k": 3})

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    history_aware_retriever = create_history_aware_retriever(llm=model, retriever=retreiver, prompt=retriever_prompt)
    retreival_chain = create_retreival_chain(history_aware_retriever , chain)
    return retreival_chain


def process_chat(chain, question, chat_history):
    response = chain.invoke({
    "input": question,
    "chat_history": chat_history
    })
    return (response["answer"])

if __name__ == '__main__':
    docs = get_documents_from_web("https://js.langchain.com/v0.1/docs/expression_language/#:~:text=LangChain%20Expression%20Language%20or%20LCEL,to%20easily%20compose%20chains%20together.")
    vector_store = create_vector_db(docs)
    chain = create_chain(vector_store)

    # you need to add the convo of the human and the ai with a specific schema, that langchain can provide
    chat_history =[]

    while True:
    # grab the input from terminal
    # currently when you do this, and ask tell your chat bot your name, it wont remember it,
    # because there is no conversation history
        user_input = input("You:")
        if user_input.lower() == 'exit':
            break
        response = process_chat(chain, user_input)
        # this will dynamically build up our chat bots history to remember the questions 
        # you ask it 
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print("Assistant:", response)

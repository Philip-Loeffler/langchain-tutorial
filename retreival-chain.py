from dotenv import load_dotenv 
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain

docA = Document(page_content="LangChain Expression Language, or LCEL, is a declarative way to easily compose chains together. LCEL was designed from day 1 to support putting prototypes in production, with no code changes, from the simplest “prompt + LLM” chain to the most complex chains.")


model = ChatOpenAI(model='gpt-3.5-turbo-1106',
temperature=0.4)


prompt = ChatPromptTemplate.from_template("""
Answer the user's question:
Context: {context}
Question: {input}
"""
)
    
# chain = prompt | model
# create_stuff_documents_chain replaces this chain with a function
# for the use case of retrieval, it is best to use 'create stuff documents'
chain = create_stuff_documents_chain(llm=model,
                                     prompt=prompt)
response = chain.invoke({
    "input": "What is LCEL?", "context": [docA]})
print(response)
from dotenv import load_dotenv 
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate



#instantiate model
llm = ChatOpenAI(model='gpt-3.5-turbo-1106',
temperature=0.1,
max_tokens=1000,
verbose=True)

#prompt template
from_template_prompt = ChatPromptTemplate.from_template("tell me a joke about a ${subject}")


# from message is a way to prime your model, such as what type of role you want 
# the model to take, and what kind of responses you want
from_message_prompt = ChatPromptTemplate.from_messages([("system", 'You are an AI chef, create a unique recipe based on the following ingredients'), ("human", "{input}")])

# chains. chains allow us to combine multiple objects together
# prompt is passed into the llm
template_chain = from_template_prompt | llm;

message_chain = from_message_prompt | llm;
from_template_response = template_chain.invoke({"subject": "dog"})
print(from_template_response)


from_message_chain = message_chain.invoke({"input": "tomoatoes"})
print(from_message_chain)

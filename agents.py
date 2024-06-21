# agents = you can give them an instruction and it will determine 
# what actions to take, and in which sequence. They are able to act within 
# their enviroment, using tools. 

from dotenv import load_dotenv 
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage


model = ChatOpenAI(
    model='gpt-3.5-turbo-1106',
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a friendly assist called max"),
        (MessagesPlaceholder(variable_name="chat_history"))

    ("human", "{input}"), 
    (MessagesPlaceholder(variable_name="agent_scratchpad"))
])

# by using tavily we are able to search the internet for answers in our invoke
search = TavilySearchResults()
tools = [search]

# this function defines our agent
agent = create_openai_functions_agent(
    llm=model,
    prompt=prompt,
    tools=tools
)

# this will invoke our agent
agentExecutor = AgentExecutor(agent=agent, tools=tools)

def process_chat(agentExecutor, user_input, chat_history):
    response = agentExecutor.invoke({"input": user_input, "chat_history": chat_history})
    return response["output"]



if __name__ == '__main__':


    # you need to add the convo of the human and the ai with a specific schema, that langchain can provide
    chat_history =[]

    while True:
    # grab the input from terminal
    # currently when you do this, and ask tell your chat bot your name, it wont remember it,
    # because there is no conversation history
        user_input = input("You:")
        if user_input.lower() == 'exit':
            break
        response = process_chat(agentExecutor, user_input)
        # this will dynamically build up our chat bots history to remember the questions 
        # you ask it 
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print("Assistant:", response)

print(response)
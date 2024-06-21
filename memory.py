from dotenv import load_dotenv 
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langhchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory

UPSTASH_URL = 'url'
UPSTASH_TOKEN = 'token'

history = UpstashRedisChatMessageHistory(
    url=UPSTASH_URL,
    token=UPSTASH_TOKEN,
    # this is the unique id for the conversation
    session_id="chat1",
    # this means the conversation will not expire
    ttl=0
)
model = ChatOpenAI(model='gpt-3.5-turbo-1106',
temperature=0.7,
max_tokens=1000,
verbose=True)


prompt = ChatPromptTemplate.from_messages("system", "you are a friendly AI assistant"),
MessagesPlaceholder(variable_name="chat_history")
("human", "{input}")

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory="history"
)

# chain = prompt | model

# need to use this LLMChain because the normal way of handling chains does not work with memory
chain = LLMChain(
    llm=model,
    prompt=prompt,
    memory=memory,
    verbose=True
)

msg1 = {
    "input": "My name is leon"
}

response1 = chain.invoke(msg1)
print(response1)


msg2 = {
    "input": "what is my name"
}

# upstash is a database website that is being used as example for hosting our memory responses
# im not signing up for any thing like this currently so this memory example works locally with what 
# is seen here. Will add other code though

# pip install upstash.redis
response2 = chain.invoke(msg2)
print(response2)


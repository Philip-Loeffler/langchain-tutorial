from dotenv import load_dotenv 
load_dotenv()

from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model='gpt-3.5-turbo',
temperature=0.1,
max_tokens=1000,
verbose=True)
# verbose allows you to debug output from the model

# question response
response = llm.invoke("hello how are you")

# allows list of values, so multiple prompts, and the are run in parrael
batch = llm.batch(["hello how are you," "you are a model"])

# stream, you receive response back in chunks
stream = llm.stream("hello how are you");

for chunk in stream:
    print(chunk.content, end="", flush=True)

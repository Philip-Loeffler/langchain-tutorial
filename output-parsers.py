from dotenv import load_dotenv 
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field


model = ChatOpenAI(model='gpt-3.5-turbo-1106',
temperature=0.7,
max_tokens=1000,
verbose=True)


def call_string_output_parser():
    prompt = ChatPromptTemplate.from_messages([("system", 'tell me a joke about the following subject'), ("human", "{input}")])
    
    # by using the parser, we are not going to recieve an object in our terminal, now it will be a string
    parser = StrOutputParser()
    chain =  prompt | model | parser
    return chain.invoke({"input": "dog"})



# by using this parser we will return a list 
def call_list_output_parser():
    prompt = ChatPromptTemplate.from_messages([("system", 'generate list of ten synonyms for the following word return with comma seperated list'), ("human", "{input}")])
    parser = CommaSeparatedListOutputParser()
    chain = prompt | model | parser
    return chain.invoke({"input": "happy"})


def call_json_output_parser():
        prompt = ChatPromptTemplate.from_messages([("system", 'extract info from the following phase.\nFormmating Instructions: {format_instructions}'), ("human", "{phrase}")])
        class Person(BaseModel):
            name: str = Field(description="the name of the person")
            age: int = Field(description="the age of the person")
            
        parser = JsonOutputParser(pydantic_object=Person)
        chain = prompt | model | parser
        return chain.invoke({"phrase": "Max is 30 years old",
        "format_instructions": parser.get_format_instructions()})


print(call_string_output_parser())
print(call_list_output_parser())
print(call_json_output_parser())
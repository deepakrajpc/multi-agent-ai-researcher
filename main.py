
from dotenv import load_dotenv 
from langchain_groq import ChatGroq
import os 


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model= "openai/gpt-oss-20b"  , 
    temperature=0.7,
    api_key=groq_api_key)

result = llm.invoke("hii")
print(result.content)
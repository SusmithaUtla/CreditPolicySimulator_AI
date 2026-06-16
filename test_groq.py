from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

response = llm.invoke(
    "Explain credit risk in one paragraph."
)

print(response.content)
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

def llm_inference(context, query):
    load_dotenv("keys.env")
    if "GROQ_API_KEY" not in os.environ:
        print("No api keyyyy")
        return None
    else:

        llm = ChatGroq(model="llama3-8b-8192")

        system_template = "Answer the {query} based on the {context}"

        prompt_template = ChatPromptTemplate(messages=[("system", system_template), ("user", "{query}")])

        prompt = prompt_template.invoke({"context":f"{context}", "query":f"{query}"})

        output = llm.invoke(prompt)
        return output.content

from dotenv import load_dotenv
load_dotenv()
import os
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


llm = ChatOpenAI(model="openai/gpt-4o", api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1"    )
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    response = agent.invoke({"messages":HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?")})
    print("Agent response:", response)


if __name__ == "__main__":
    main()

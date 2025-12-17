from dotenv import load_dotenv
import os
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()


@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' times 'y'."""
    return x * y


if __name__ == "__main__":
    print("Hello Tool Calling")

    tools = [TavilySearchResults(), multiply]

    llm = ChatOpenAI(
        model="openai/gpt-4o",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="you're a helpful assistant",
    )

    res = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "what is the weather in Dubai right now? compare it with San Francisco, output should be in Celsius"}
            ]
        }
    )

    print(res) 

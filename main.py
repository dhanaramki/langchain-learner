from dotenv import load_dotenv

load_dotenv()

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from operator import itemgetter

print("Initializing variables.....")

embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",  # or any OpenRouter embedding model
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

llm = ChatOpenAI(
    model="openai/gpt-4-turbo",  # or any OpenRouter chat model
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"),
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """ Answer the question based only on the following context:
        
        {context} \n\n 
        
        Question: {question} 
        
        Provide a detailed answer:"""
)


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join([doc.page_content for doc in docs])


def retrieval_qa_chain_without_lcel(query: str) -> str:
    """Perform retrieval-augmented generation for a given query."""
    # Retrieve relevant documents
    docs = retriever.invoke(query)

    # Format the documents
    context = format_docs(docs)

    # Create the prompt
    prompt = prompt_template.format_messages(context=context, question=query)

    # Get the answer from the LLM
    response = llm.invoke(prompt)

    return response.content


def retrieval_qa_chain_with_lcel():
    """
    Create a retrieval-augmented generation chain using LCEL.
    Returns a chain that can be invoked with a query.
    """

    retrieval_qa_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_qa_chain


if __name__ == "__main__":

    print("Retrieval-Augmented Generation Chatbot")

    Query = "What is pinecode in machine learning?"

    # =============================================================================
    # Option 0: Raw invocation without RAG
    # =============================================================================

    print("#" + "=" * 70)
    print("# Option 0: Raw invocation without RAG")
    print("#" + "=" * 70)

    result_raw = llm.invoke([HumanMessage(content=Query)])
    print("\nAnswer without RAG:\n")
    print(result_raw.content)

    # =============================================================================
    # Option 1: Use implemenation without LCEL
    # =============================================================================

    print("#" + "=" * 70)
    print("# Option 1: Use implementation without LCEL")
    print("#" + "=" * 70)

    result_without_lcel = retrieval_qa_chain_without_lcel(Query)
    print("\nAnswer without LCEL:\n")
    print(result_without_lcel)


   # =============================================================================
    # Option 2: Use implemenation with LCEL
    # =============================================================================

    print("#" + "=" * 70)
    print("# Option 2: Use implementation with LCEL")
    print("#" + "=" * 70)

    chain_with_lcel = retrieval_qa_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": Query})
    print("\nAnswer with LCEL:\n")
    print(result_with_lcel)

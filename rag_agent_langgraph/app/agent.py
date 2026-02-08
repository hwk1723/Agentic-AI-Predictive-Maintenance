import os
from datetime import date
from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from langchain_openai import OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma

memory = MemorySaver()

class AvailabilityToolInput(BaseModel):
    """Input query for the tool."""

    query: str = Field(
        ...,
        description=(
            "The query should specifies the predicted condition. "
            "A natural language description of what predicted condition information to retrieve."
        ),
    )

@tool(args_schema=AvailabilityToolInput)
def maintenance_docs_RAG(query: str) -> str:
    """
    maintenance_docs_RAG_tool

    Retrieves and summarizes relevant maintenance, troubleshooting, or repair procedures
    for a machine based on its predicted condition or reported symptoms.

    This tool uses a Retrieval-Augmented Generation (RAG) approach to search and extract
    information from a collection of maintenance manuals, standard operating procedures (SOPs),
    and technical documentation.

    Use this tool to recommend appropriate actions for technicians or engineers.

    Inputs:
        query (str):
            - The query should specifies the predicted condition.
            - A natural language description of what predicted condition information to retrieve.

    Output:
        str:
            - A structured summary containing: 
                Immediate actions, or
                Diagnostic steps, or
                Corrective actions.
            - If not specified, provide all 3 types of recommendations.

    Example Response:
        "recommendations": "1. The Immediate actions are to...\n"
                        "2. Diagnostic steps are to...\n"
                        "3. Corrective actions are to..."
    Usage Notes:
        - Do not attempt to generate maintenance steps without retrieval.
        - Always base the query on the predicted condition or diagnostic result.
        - Responses should guide the user toward safe and verified maintenance practices.
        - If multiple procedures are retrieved, summarize them into a clear, actionable plan.
    """

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    persist_directory = r"/Users/howaikit/Documents/GitHub/Agentic-AI-Predictive-Maintenance/rag_agent_langgraph/app/vectorstore"

    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    try:
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory,
            collection_name="maintenance_and_sop_collection"
        )
        print(f"Vector store loaded.")
    except Exception as e:
        print(f"Error loading vector store: {e}")
        raise

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}\n")
    return "\n\n".join(results)

class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str

class RagAgent:
    """RagAgent - a specialized assistant for scheduling."""

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    SYSTEM_INSTRUCTION = (
        "You are maintenance assistant. "
        "Your sole purpose is to use the 'maintenance_docs_RAG' tool to answer questions about maintenance procedures for machines. "
        "If the user asks about anything other than maintenance procedures for machines, "
        "politely state that you cannot help with that topic and can only assist with maintenance queries. "
        "Do not attempt to answer unrelated questions or use tools for other purposes."
        "Set response status to input_required if the user needs to provide more information."
        "Set response status to error if there is an error while processing the request."
        "Set response status to completed if the request is complete."
    )

    def __init__(self):
        self.model = ChatOllama(
            model="qwen3:4b",
            temperature=0,
            streaming=False,
        )
        self.tools = [maintenance_docs_RAG]

        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            debug=False,
        )

    def invoke(self, query, context_id):
        config: RunnableConfig = {"configurable": {"thread_id": context_id}}
        today_str = f"Today's date is {date.today().strftime('%Y-%m-%d')}."
        augmented_query = f"{today_str}\n\nUser query: {query}"
        result = self.graph.invoke(
            {"messages": [("user", augmented_query)]},
            config,
        )
        return self.get_agent_response(config)

    def get_agent_response(self, config):
        current_state = self.graph.get_state(config)
        messages = current_state.values.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, AIMessage):
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": last_msg.content,
                }

        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": (
                "We are unable to process your request at the moment. "
                "Please try again."
            ),
        }

    async def stream(self, query, context_id):
        """
        Compatibility shim for a2a.

        a2a expects agents to expose `stream()`, but we intentionally
        do NOT stream LangGraph events. This method executes the agent
        synchronously via invoke() and yields exactly one final result.
        """
        result = self.invoke(query, context_id)
        yield result
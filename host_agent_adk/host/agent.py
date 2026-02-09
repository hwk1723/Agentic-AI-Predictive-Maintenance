import asyncio
import json
import uuid
from typing import Any, AsyncIterable, List

import httpx
import nest_asyncio
from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
)
from dotenv import load_dotenv
from google.adk import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from .tools import (
    generate_explanation_tool,
    evidence_collection_tool,
)
from .remote_agent_connection import RemoteAgentConnections

load_dotenv()
nest_asyncio.apply()


class HostAgent:
    """The Host agent."""

    def __init__(
        self,
    ):
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ""
        self._agent = self.create_agent()
        self._user_id = "host_agent"
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    async def _async_init_components(self, remote_agent_addresses: List[str]):
        async with httpx.AsyncClient(timeout=30) as client:
            for address in remote_agent_addresses:
                card_resolver = A2ACardResolver(client, address)
                try:
                    card = await card_resolver.get_agent_card()
                    remote_connection = RemoteAgentConnections(
                        agent_card=card, agent_url=address
                    )
                    self.remote_agent_connections[card.name] = remote_connection
                    self.cards[card.name] = card
                except httpx.ConnectError as e:
                    print(f"ERROR: Failed to get agent card from {address}: {e}")
                except Exception as e:
                    print(f"ERROR: Failed to initialize connection for {address}: {e}")

        agent_info = [
            json.dumps({"name": card.name, "description": card.description})
            for card in self.cards.values()
        ]
        print("agent_info:", agent_info)
        self.agents = "\n".join(agent_info) if agent_info else "No friends found"

    @classmethod
    async def create(
        cls,
        remote_agent_addresses: List[str],
    ):
        instance = cls()
        await instance._async_init_components(remote_agent_addresses)
        return instance

    def create_agent(self) -> Agent:
        return Agent(
            model="gemini-2.5-pro",
            name="Host_Agent",
            instruction=self.root_instruction,
            description="This Host agent orchestrates maintenance tasks across multiple sub-agents.",
            tools=[
                self.send_message,
                evidence_collection_tool,
                generate_explanation_tool
            ],
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        return f"""
        **Role:**
        You are the Host Agent for an industrial agentic predictive maintenance system.
        Your role is to orchestrate specialized agents, collect and persist evidence,
        and deliver accurate, evidence-backed machine information to engineers.

        You are NOT a data source, predictor, or procedural expert.
        You do not invent, infer, or assume.
        All knowledge must come from agents and be stored as evidence before use.

        ---

        ## Primary Objective

        Determine the user’s intent, retrieve the required information using agents,
        store all new information as evidence, and respond appropriately based on the
        type of request.

        Not all requests require prediction, SOP retrieval, or explanation.

        ---

        ## Step 0: Intent Classification (MANDATORY)

        Before calling any agent, classify the user request into ONE of the following:

        ### A. Informational (Fact-Only)
        Examples:
        • “What is the current status of machine X?”

        Characteristics:
        • Requires factual data only
        • No prediction, diagnosis, or recommendation
        • No SOP or procedural explanation required

        ---

        ### B. Analytical / Diagnostic
        Examples:
        • “What is the predicted failure for machine X?”
        • “What SOP applies to the predicted failure of product Z?”
        • “Why is machine Y likely to fail?”

        Characteristics:
        • Requires interpretation, prediction, or diagnosis
        • May require SOP/manual alignment
        • Requires explanation

        ---

        Your orchestration strategy MUST depend on this classification.

        ---

        ## Evidence-First Rule (ALWAYS APPLIES)

        • Any information returned by any agent is considered UNSTORED
        until `evidence_collection_tool` has been called.
        • You MUST store evidence immediately after receiving it.
        • You may NOT reason, summarize, or respond using unstored information.

        This rule applies to ALL request types.

        ---

        ## Execution Paths

        ### Path A: Informational (Fact-Only)

        Execution steps:
        1. Call the Database Agent
        2. Store returned data using `evidence_collection_tool`
        3. Respond directly to the user using the stored evidence
        4. Explains the queried data factually without interpretation

        Rules:
        • Do NOT call the Predictive Maintenance Agent
        • Do NOT call the RAG Agent
        • Do NOT call `generate_explanation_tool`
        • Respond concisely and factually
        • No interpretation, diagnosis, or recommendations

        This path TERMINATES after step 4.

        ---

        ### Path B: Analytical / Diagnostic

        Execution steps:
        1. Call the Database Agent (if data is required)
        2. Store returned data using `evidence_collection_tool`
        3. Call the Predictive Maintenance Agent (if prediction is required)
        4. Store predictions using `evidence_collection_tool`
        5. Call the RAG Agent (if SOP/manual alignment is required)
        6. Store retrieved documentation using `evidence_collection_tool`
        7. Before calling evidence_collection_tool, you MUST convert all agent responses into structured key–value data. Never pass free-form text into the tool.
        8. Call `generate_explanation_tool`
        9. Respond to the user using ONLY the generated explanation

        Rules:
        • You MUST NOT generate your own explanation
        • You MUST NOT skip evidence collection
        • You MUST follow this sequence unless evidence already exists

        ---

        ## No Early Termination Rule

        • Calling an agent NEVER completes a task by itself
        • The task is complete ONLY when the selected execution path
        reaches its defined termination step

        Stopping early outside the defined termination points is NOT allowed.

        ---

        ## Completion Criteria

        ### Informational Requests
        • Required agent(s) called
        • All returned data stored as evidence
        • User response contains only factual data

        ### Analytical / Diagnostic Requests
        • All relevant agents queried
        • All returned data stored as evidence
        • `generate_explanation_tool` has been called
        • Response uses only the generated explanation

        ---

        ## Pre-Response Self-Check (Silent)

        Before responding, verify:
        • Have I classified the intent correctly?
        • Have I stored all returned data as evidence?
        • Am I calling unnecessary agents?
        • Am I explaining something without `generate_explanation_tool`?

        If any answer is “no”, continue orchestration.

        ---

        <Available Agents>
        {self.agents}
        </Available Agents>
        """

    async def stream(
        self, query: str, session_id: str
    ) -> AsyncIterable[dict[str, Any]]:
        """
        Streams the agent's response to a given query.
        """
        session = await self._runner.session_service.get_session(
            app_name=self._agent.name,
            user_id=self._user_id,
            session_id=session_id,
        )
        content = types.Content(role="user", parts=[types.Part.from_text(text=query)])
        if session is None:
            session = await self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                state={},
                session_id=session_id,
            )
        async for event in self._runner.run_async(
            user_id=self._user_id, session_id=session.id, new_message=content
        ):
            if event.is_final_response():
                response = ""
                if (
                    event.content
                    and event.content.parts
                    and event.content.parts[0].text
                ):
                    response = "\n".join(
                        [p.text for p in event.content.parts if p.text]
                    )
                yield {
                    "is_task_complete": True,
                    "content": response,
                }
            else:
                yield {
                    "is_task_complete": False,
                    "updates": "The host agent is thinking...",
                }

    async def send_message(self, agent_name: str, task: str, tool_context: ToolContext):
        """Sends a task to a sub agent."""
        if agent_name not in self.remote_agent_connections:
            raise ValueError(f"Agent {agent_name} not found")
        client = self.remote_agent_connections[agent_name]

        if not client:
            raise ValueError(f"Client not available for {agent_name}")

        # Simplified task and context ID management
        state = tool_context.state
        task_id = state.get("task_id", str(uuid.uuid4()))
        context_id = state.get("context_id", str(uuid.uuid4()))
        message_id = str(uuid.uuid4())

        payload = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": task}],
                "messageId": message_id,
                "taskId": task_id,
                "contextId": context_id,
            },
        }

        message_request = SendMessageRequest(
            id=message_id, params=MessageSendParams.model_validate(payload)
        )
        send_response: SendMessageResponse = await client.send_message(message_request)
        print("send_response", send_response)

        if not isinstance(
            send_response.root, SendMessageSuccessResponse
        ) or not isinstance(send_response.root.result, Task):
            print("Received a non-success or non-task response. Cannot proceed.")
            return

        response_content = send_response.root.model_dump_json(exclude_none=True)
        json_content = json.loads(response_content)

        resp = []
        if json_content.get("result", {}).get("artifacts"):
            for artifact in json_content["result"]["artifacts"]:
                if artifact.get("parts"):
                    resp.extend(artifact["parts"])
        return resp


def _get_initialized_host_agent_sync():
    """Synchronously creates and initializes the HostAgent."""

    async def _async_main():
        # Hardcoded URLs for the friend agents
        friend_agent_urls = [
            "http://localhost:10002",  # Database Agent
            "http://localhost:10003",  # Prediction Agent
            "http://localhost:10004",  # RAG Agent
        ]

        print("initializing host agent")
        hosting_agent_instance = await HostAgent.create(
            remote_agent_addresses=friend_agent_urls
        )
        print("HostAgent initialized")
        return hosting_agent_instance.create_agent()

    try:
        return asyncio.run(_async_main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            print(
                f"Warning: Could not initialize HostAgent with asyncio.run(): {e}. "
                "This can happen if an event loop is already running (e.g., in Jupyter). "
                "Consider initializing HostAgent within an async function in your application."
            )
        else:
            raise


root_agent = _get_initialized_host_agent_sync()

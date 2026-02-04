import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils.errors import ServerError
from app.agent import RagAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RagAgentExecutor(AgentExecutor):
    """RagAgentExecutor - a specialized executor for the RagAgent."""

    def __init__(self):
        self.agent = RagAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        if not context.task_id or not context.context_id:
            raise ValueError("RequestContext must have task_id and context_id")
        if not context.message:
            raise ValueError("RequestContext must have a message")

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            await updater.submit()
        await updater.start_work()

        query = context.get_user_input()
        try:
            async for item in self.agent.stream(query, context.context_id):
                is_task_complete = item["is_task_complete"]
                require_user_input = item["require_user_input"]
                parts = [Part(root=TextPart(text=item["content"]))]

                if not is_task_complete and not require_user_input:
                    await updater.update_status(
                        TaskState.working,
                        message=updater.new_agent_message(parts),
                    )
                elif require_user_input:
                    await updater.update_status(
                        TaskState.input_required,
                        message=updater.new_agent_message(parts),
                    )
                    break
                else:
                    await updater.add_artifact(
                        parts,
                        name="scheduling_result",
                    )
                    await updater.complete()
                    break

        except Exception as e:
            logger.error(f"An error occurred while streaming the response: {e}")
            raise ServerError(error=InternalError()) from e

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())

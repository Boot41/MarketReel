"""Sample Google ADK agent.

Replace the greeting tool and agent definition with your own logic.
Run standalone:  adk run app.agent
Integrate via:   POST /api/v1/agent/run
"""

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


def greet(name: str) -> dict[str, str]:
    """Return a greeting for the given name."""
    return {"greeting": f"Hello, {name}! Welcome to App Scaffold."}


root_agent = Agent(
    name="scaffold_agent",
    model="gemini-2.0-flash",
    description="A starter agent for the scaffold project.",
    instruction="You are a helpful assistant. Use the greet tool when asked to greet someone.",
    tools=[greet],
)


async def run_agent(user_message: str, user_id: str = "user") -> str:
    """Run the agent with a single user message and return the final text response."""
    session_service = InMemorySessionService()  # type: ignore[no-untyped-call]
    session = await session_service.create_session(
        app_name="scaffold_agent",
        user_id=user_id,
    )

    runner = Runner(agent=root_agent, app_name="scaffold_agent", session_service=session_service)

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    final_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text or ""

    return final_text

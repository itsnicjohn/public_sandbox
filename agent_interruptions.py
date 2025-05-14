from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.agents.llm import function_tool
from livekit.agents.job import get_job_context
from livekit.plugins import (
    openai,
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")

    @function_tool()
    async def end_call(self) -> None:
        """Use this function to end the call upon the user's request."""
        ctx = get_job_context()
        await ctx.delete_room()


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(voice="224126de-034c-429b-9fde-71031fba9a59"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )

    async def print_chat_ctx():
        print("\nChat Context:")
        for chat_item in session._chat_ctx.items:
            print(f"\t{chat_item.role}{' (INTERRUPTED)' if chat_item.interrupted else ''}: \n\t\t: {chat_item.text_content}")
        print("\n")

    ctx.add_shutdown_callback(print_chat_ctx)



if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
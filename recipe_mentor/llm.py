"""
The real LLM-backed judge, built on Google's Agent Development Kit (ADK) --
one of the hackathon's four accepted "Google Agent Framework" options
(ADK, GenAI SDK, Antigravity SDK, GenKit). Isolated in its own module so
the offline path never requires `google-adk` to be installed.

This replaces an earlier LangChain-based implementation
(`langchain_google_vertexai.ChatVertexAI`) that worked but didn't satisfy
the hackathon's own mandatory technology requirement -- LangChain isn't on
the accepted-framework list. ADK is a strictly better fit anyway: it's a
purpose-built agent framework (stronger "Architectural Discipline" story
than a raw SDK call), it uses the same GenAI SDK under the hood so Vertex
routing is unchanged, and its response events give plain text directly --
no need for the block-parsing `_response_text()` shim LangChain's
Gemini 3.5 integration required.

GCP state (see docs/ADR_Recipe_Mentor_Vertex_Backend_2026-08-29.md for the
full history): project `neofix-676da`, model `gemini-3.5-flash`,
location "global" (confirmed live -- "us-central1" 404s on this project's
Gemini 3.5 access).
"""
from __future__ import annotations

import asyncio
import os

#: Environment-configurable so anyone forking this example points it at
#: their own GCP project rather than the one it was built against. The
#: literal fallbacks are that project (neofix-676da) purely for provenance
#: -- they carry no special access and won't work against your own project.
DEFAULT_PROJECT = os.environ.get("RECIPE_MENTOR_GCP_PROJECT", "neofix-676da")
#: "global" is deliberate, not a placeholder -- location="us-central1" 404s
#: on gemini-3.5-flash access (confirmed live, see the ADR). Try "global"
#: first on a new project too before assuming a regional endpoint works.
DEFAULT_LOCATION = os.environ.get("RECIPE_MENTOR_GCP_LOCATION", "global")
DEFAULT_MODEL = os.environ.get("RECIPE_MENTOR_MODEL", "gemini-3.5-flash")
APP_NAME = "recipe_mentor"

_JUDGE_INSTRUCTION = (
    "You are a strict but fair reviewer of one step of a twelve-step ML "
    "production recipe. You judge whether a builder's free-text answer "
    "actually satisfies the stated rule for that step -- a plausible-sounding "
    "answer that dodges the specific rule does not pass. Reply in exactly "
    "the format the user's prompt asks for, nothing else."
)


def _ensure_vertex_env(project: str, location: str) -> None:
    """
    ADK (via the GenAI SDK it wraps) reads Vertex routing from environment
    variables, not constructor kwargs -- confirmed against ADK's own docs,
    not assumed. `setdefault` so a caller's own env (e.g. a different
    project for local testing) isn't silently overridden.
    """
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project)
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", location)


class AdkJudge:
    """
    A single ADK agent + session, reused across every step of one mentor
    session. Reusing one session (rather than a fresh one per step) is
    deliberate: it lets the agent see the whole recipe walk's prior
    exchanges in context, not just the current step in isolation.
    """

    def __init__(
        self,
        *,
        project: str = DEFAULT_PROJECT,
        location: str = DEFAULT_LOCATION,
        model: str = DEFAULT_MODEL,
        user_id: str = "sozo_lab_demo",
    ):
        _ensure_vertex_env(project, location)
        # Imported lazily so the offline backend never requires google-adk
        # to be installed.
        from google.adk.agents import LlmAgent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService

        self.model = model
        self.user_id = user_id
        self.session_id = f"{user_id}_session"
        self._agent = LlmAgent(model=model, name="recipe_mentor_judge", instruction=_JUDGE_INSTRUCTION)
        self._session_service = InMemorySessionService()
        self._runner = Runner(agent=self._agent, app_name=APP_NAME, session_service=self._session_service)
        self._session_ready = False

    async def _ensure_session(self) -> None:
        if not self._session_ready:
            await self._session_service.create_session(
                app_name=APP_NAME, user_id=self.user_id, session_id=self.session_id,
            )
            self._session_ready = True

    async def _ask_async(self, prompt: str) -> str:
        from google.genai.types import Content, Part

        await self._ensure_session()
        message = Content(parts=[Part(text=prompt)], role="user")
        final_text = ""
        async for event in self._runner.run_async(
            user_id=self.user_id, session_id=self.session_id, new_message=message,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_text = event.content.parts[0].text or final_text
        return final_text

    def ask(self, prompt: str) -> str:
        """Synchronous entry point -- mentor.py's session loop stays plain
        sync code; this is the one place that bridges into asyncio."""
        return asyncio.run(self._ask_async(prompt))


def build_judge(**kwargs) -> AdkJudge:
    return AdkJudge(**kwargs)

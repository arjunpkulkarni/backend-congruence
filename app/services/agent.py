"""
Congruence Ops Agent — Iteration 2

Uses LangChain's create_agent (v1.2+) with real tools backed by the
data access layer.  The agent reads actual session data, transcripts,
clinical notes, and practice analytics from disk.

Key changes from Iteration 1:
  - Stub tools replaced with real implementations (agent_tools.py)
  - Tools call into data_access.py which reads data/sessions/ on disk
  - Agent uses langchain.agents.create_agent with tool-calling loop
  - Per-user conversation history kept in memory
  - Role-based tool filtering still enforced
"""

import os
import logging
from typing import Dict, List

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from app.models.schemas import AgentChatRequest, AgentChatResponse, AgentAction
from app.services.agent_tools import ALL_TOOLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Role → permitted tool names
# ---------------------------------------------------------------------------
ROLE_PERMISSIONS: Dict[str, List[str]] = {
    "clinician": [
        "find_patient",
        "list_all_patients",
        "get_patient_record",
        "get_session_transcript_tool",
        "generate_clinical_note",
        "suggest_icd10_codes",
    ],
    "admin": [
        "find_patient",
        "list_all_patients",
        "get_patient_record",
        "generate_insurance_packet",
        "send_intake_form",
        "check_claim_status",
    ],
    "practice_owner": [
        "find_patient",
        "list_all_patients",
        "get_patient_record",
        "get_session_transcript_tool",
        "generate_clinical_note",
        "generate_insurance_packet",
        "suggest_icd10_codes",
        "check_claim_status",
        "schedule_appointment",
        "send_intake_form",
        "get_practice_analytics",
    ],
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are the Congruence Ops Agent, an AI assistant for therapy practice operations.

You have access to REAL patient session data.  When a user asks about a patient,
session, or clinical note you MUST call the appropriate tool to retrieve the data
rather than making anything up.

You help clinicians, administrators, and practice owners with:
- Retrieving and summarising patient records and session history
- Viewing transcripts and clinical notes from therapy sessions
- Analysing emotional congruence scores and patterns
- Suggesting ICD-10 codes based on clinical observations
- Generating insurance authorisation documentation
- Practice-wide analytics and metrics

Workflow guidelines:
1. **When the user mentions a patient by NAME** (e.g., "Rob Wazowski", "demo patient"):
   - ALWAYS call find_patient first to get the patient_id
   - Then use that patient_id for subsequent tool calls
2. **When the user provides a patient_id** (UUID format):
   - Use it directly in tool calls
3. If the user asks about a session but doesn't specify which, default
   to the latest session for that patient.
4. Always cite specific data (congruence scores, timestamps, quotes)
   when summarising clinical information.
5. Be professional and HIPAA-compliant at all times.
6. When uncertain, ask clarifying questions.

IMPORTANT: Users prefer to use patient names, not IDs. Always try find_patient first
when a name is mentioned.
"""


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------
class CongruenceOpsAgent:
    def __init__(self):
        self.llm = self._initialize_llm()
        # Per-user conversation history: user_id -> list of messages
        self._histories: Dict[str, list] = {}

    # -- LLM --
    def _initialize_llm(self) -> ChatOpenAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not configured")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=0.1,
            max_tokens=2000,
        )

    # -- History --
    def _get_history(self, user_id: str) -> list:
        if user_id not in self._histories:
            self._histories[user_id] = []
        return self._histories[user_id]

    def _trim_history(self, user_id: str, max_turns: int = 10) -> None:
        """Keep only the last max_turns pairs of messages."""
        hist = self._histories.get(user_id, [])
        if len(hist) > max_turns * 2:
            self._histories[user_id] = hist[-(max_turns * 2):]

    # -- Permission helpers --
    @staticmethod
    def _tools_for_role(role: str) -> list:
        allowed_names = ROLE_PERMISSIONS.get(role, [])
        return [t for t in ALL_TOOLS if t.name in allowed_names]

    @staticmethod
    def check_permission(user_role: str, tool_name: str) -> bool:
        return tool_name in ROLE_PERMISSIONS.get(user_role, [])

    # -- Build agent graph per request --
    def _build_agent(self, role: str):
        """Build a compiled agent graph with role-filtered tools."""
        tools = self._tools_for_role(role)

        agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
        )
        return agent

    # -- Main entry point --
    async def process_message(self, request: AgentChatRequest) -> AgentChatResponse:
        try:
            agent = self._build_agent(request.role)
            history = self._get_history(request.user_id)

            # Build input messages: history + new user message
            input_messages = list(history) + [HumanMessage(content=request.message)]

            # Invoke the agent
            result = await agent.ainvoke({"messages": input_messages})

            # Extract the response and tools used
            output_messages = result.get("messages", [])
            tools_used: List[str] = []
            response_text = ""

            for msg in output_messages:
                # Tool calls show up as ToolMessage or AIMessage with tool_calls
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tools_used.append(tc.get("name", "unknown"))
                # The final AI response
                if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                    response_text = msg.content

            # If we didn't find a clean final response, take the last AIMessage content
            if not response_text:
                for msg in reversed(output_messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        response_text = msg.content
                        break

            # Update conversation history
            history.append(HumanMessage(content=request.message))
            history.append(AIMessage(content=response_text))
            self._trim_history(request.user_id)

            # Generate contextual actions
            actions = self._generate_actions(response_text, request.role)

            return AgentChatResponse(
                response=response_text,
                actions=actions,
                tools_used=list(set(tools_used)),  # deduplicate
                context=request.context or {},
                metadata={
                    "model_used": self.llm.model_name,
                    "tools_available": [t.name for t in self._tools_for_role(request.role)],
                },
            )

        except Exception as exc:
            logger.exception("Agent processing error: %s", exc)
            return AgentChatResponse(
                response=(
                    "I apologize, but I encountered an error processing your request. "
                    "Please try again."
                ),
                actions=[],
                tools_used=[],
                context=request.context or {},
                metadata={"error": str(exc)},
            )

    # -- Contextual actions --
    def _generate_actions(self, response: str, user_role: str) -> List[AgentAction]:
        actions: List[AgentAction] = []
        lower = response.lower()

        if "patient" in lower:
            actions.append(AgentAction(type="select_patient", label="Select Patient", data={}))
        if "session" in lower:
            actions.append(AgentAction(type="view_sessions", label="View Today's Sessions", data={}))

        if user_role == "clinician":
            if any(kw in lower for kw in ("note", "soap", "documentation")):
                actions.append(AgentAction(type="generate_note", label="Generate Clinical Note", data={}))
            if any(kw in lower for kw in ("icd", "diagnostic", "code")):
                actions.append(AgentAction(type="suggest_codes", label="Suggest ICD-10 Codes", data={}))

        elif user_role == "admin":
            if "intake" in lower:
                actions.append(AgentAction(type="manage_intake", label="Manage Intake Forms", data={}))
            if any(kw in lower for kw in ("insurance", "authorization", "claim")):
                actions.append(AgentAction(type="insurance_packet", label="Generate Insurance Packet", data={}))

        elif user_role == "practice_owner":
            if any(kw in lower for kw in ("analytics", "metrics", "overview")):
                actions.append(AgentAction(type="view_analytics", label="View Practice Analytics", data={}))

        return actions


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_agent_instance = None


def get_agent() -> CongruenceOpsAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = CongruenceOpsAgent()
    return _agent_instance

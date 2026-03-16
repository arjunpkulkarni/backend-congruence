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
from app.services.agent_intent import classify_intent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Role → permitted tool names
# ---------------------------------------------------------------------------
ROLE_PERMISSIONS: Dict[str, List[str]] = {
    "clinician": [
        "search_clinical_evidence",  # NEW - Evidence mode
        "find_patient",
        "list_all_patients",
        "get_patient_record",
        "get_session_transcript_tool",
        "generate_clinical_note",
        "suggest_icd10_codes",
    ],
    "admin": [
        "search_clinical_evidence",  # NEW - Evidence mode
        "find_patient",
        "list_all_patients",
        "get_patient_record",
        "generate_insurance_packet",
        "send_intake_form",
        "check_claim_status",
    ],
    "practice_owner": [
        "search_clinical_evidence",  # NEW - Evidence mode
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
SYSTEM_PROMPT_EVIDENCE = """\
You are the Congruence Ops Agent in EVIDENCE MODE.

The user is asking for PROOF, EVIDENCE, or specific MENTIONS from patient data.

CRITICAL RULES:
1. ALWAYS call search_clinical_evidence FIRST with the user's search terms
2. DO NOT provide general medical knowledge or explanations
3. ONLY cite what is found in the actual patient data
4. If no evidence found, say "No evidence found in patient records"
5. Format: Show quotes/data FIRST, then brief interpretation

When the user mentions a patient name, call find_patient first to get patient_id,
then pass it to search_clinical_evidence.

Example flow:
User: "Show me notes that suggest OCD"
You: Call search_clinical_evidence("OCD") -> Return exact quotes from notes

DO NOT write essays. DO NOT provide general information. ONLY show what's in the data.
"""

SYSTEM_PROMPT_SUMMARY = """\
You are the Congruence Ops Agent in SUMMARY MODE.

The user wants a high-level overview or summary of patient data.

CRITICAL RULES:
1. Call the appropriate tool to fetch the data (notes, transcript, patient record)
2. Provide a concise summary highlighting key points
3. Include specific metrics (congruence scores, timestamps) when available
4. DO NOT make up information - only summarize what the tools return

When the user mentions a patient name, call find_patient first to get patient_id.

Example flow:
User: "Summarize Rob's latest session"
You: find_patient("Rob") -> get_patient_record(patient_id) -> Summarize key points

Keep summaries concise and clinically relevant.
"""

SYSTEM_PROMPT_ACTION = """\
You are the Congruence Ops Agent in ACTION MODE.

The user wants to execute a workflow or generate documentation.

CRITICAL RULES:
1. Validate you have all required context (patient_id, session_id, etc.)
2. If missing info, ask for it before calling the tool
3. Call the appropriate action tool (generate_clinical_note, generate_insurance_packet, etc.)
4. Confirm the action was completed successfully

When the user mentions a patient name, call find_patient first to get patient_id.

Example flow:
User: "Generate SOAP note for Rob"
You: find_patient("Rob") -> generate_clinical_note(patient_id) -> Confirm completion

Always confirm what action was taken and what was generated.
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
    def _build_agent(self, role: str, mode: str = "summary"):
        """Build a compiled agent graph with role-filtered tools and mode-specific prompt."""
        tools = self._tools_for_role(role)
        
        # Select system prompt based on mode
        if mode == "evidence":
            system_prompt = SYSTEM_PROMPT_EVIDENCE
        elif mode == "action":
            system_prompt = SYSTEM_PROMPT_ACTION
        else:
            system_prompt = SYSTEM_PROMPT_SUMMARY

        agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=system_prompt,
        )
        return agent

    # -- Main entry point --
    async def process_message(self, request: AgentChatRequest) -> AgentChatResponse:
        try:
            # STEP 1: INTENT CLASSIFICATION
            intent = classify_intent(request.message)
            logger.info(f"Intent classified: {intent.mode} (confidence: {intent.confidence:.2f})")
            
            # STEP 2: BUILD AGENT WITH MODE-SPECIFIC PROMPT
            agent = self._build_agent(request.role, mode=intent.mode)
            history = self._get_history(request.user_id)

            # STEP 3: ADD MODE-SPECIFIC CONTEXT TO MESSAGE
            # For evidence mode, emphasize the search terms
            if intent.mode == "evidence":
                enhanced_message = f"{request.message}\n\n[SYSTEM: This is an EVIDENCE REQUEST. Search terms: {', '.join(intent.search_terms)}. Call search_clinical_evidence first.]"
            else:
                enhanced_message = request.message

            # Build input messages: history + new user message
            input_messages = list(history) + [HumanMessage(content=enhanced_message)]

            # STEP 4: INVOKE AGENT
            result = await agent.ainvoke({"messages": input_messages})

            # STEP 5: EXTRACT RESPONSE AND TOOLS
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

            # Update conversation history (with original message, not enhanced)
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
                    "intent_mode": intent.mode,  # NEW - expose mode to frontend
                    "intent_confidence": intent.confidence,
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

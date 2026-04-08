"""
Congruence Ops Agent — Iteration 3

Uses LangChain's create_agent (v1.2+) with real tools backed by Supabase.
The agent reads actual session data, transcripts, clinical notes, and 
practice analytics from the cloud database.

Key features:
  - All data stored in and retrieved from Supabase (cloud-first architecture)
  - Tools call into data_access.py which queries Supabase tables
  - Agent uses langchain.agents.create_agent with tool-calling loop
  - Conversation history persisted in Supabase database
  - Role-based tool filtering enforced
  - Intent classification (evidence/summary/action modes)
  - Multi-session summary support for queries like "last 3 sessions"
"""

import os
import logging
from typing import Dict, List, Optional

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from app.models.schemas import AgentChatRequest, AgentChatResponse, AgentAction
from app.services.agent_tools import ALL_TOOLS
from app.services.agent_intent import classify_intent
from app.services.database import get_conversation_db
from app.models.conversation import MessageCreate

logger = logging.getLogger(__name__)

ROLE_PERMISSIONS: Dict[str, List[str]] = {
    "clinician": [
        "search_clinical_evidence",  
        "find_patient",
        "list_all_patients",
        "get_patient_record",
        "get_session_transcript_tool",
        "generate_clinical_note",
        "get_multiple_sessions_summary",  # NEW - Multi-session summary
        "suggest_icd10_codes",
    ],
    "admin": [
        "search_clinical_evidence", 
        "find_patient",
        "list_all_patients",
        "get_patient_record",
        "get_multiple_sessions_summary", 
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
        "get_multiple_sessions_summary",  # NEW - Multi-session summary
        "generate_insurance_packet",
        "suggest_icd10_codes",
        "check_claim_status",
        "schedule_appointment",
        "send_intake_form",
        "get_practice_analytics",
    ],
}

SYSTEM_PROMPT_EVIDENCE = """\
You are the Congruence Ops Agent in EVIDENCE MODE.

The user is asking for PROOF, EVIDENCE, or specific MENTIONS from patient data.

CRITICAL RULES:
1. When the user mentions a specific session (e.g., "OCD session", "anxiety session"):
   - First call find_patient to get the patient_id
   - Then call search_clinical_evidence with the search terms and patient_id
   - The search will automatically prioritize sessions with matching titles
2. ALWAYS call search_clinical_evidence with the user's search terms
3. DO NOT provide general medical knowledge or explanations
4. ONLY cite what is found in the actual patient data
5. If no evidence found, say "No evidence found in patient records"
6. Format: Show quotes/data FIRST, then brief interpretation
7. When multiple sessions match, evidence from ALL matching sessions will be shown

When the user mentions a patient name, call find_patient first to get patient_id,
then pass it to search_clinical_evidence.

Example flow:
User: "Tell me about Sophia's OCD session"
You: 
  1. find_patient("Sophia") -> get patient_id
  2. search_clinical_evidence("OCD patient_id") -> Returns evidence from OCD session

User: "Show me notes that suggest OCD"
You: search_clinical_evidence("OCD") -> Return exact quotes from all patients

DO NOT write essays. DO NOT provide general information. ONLY show what's in the data.
The search tool now searches ALL sessions and prioritizes sessions with matching titles.
"""

SYSTEM_PROMPT_SUMMARY = """\
You are the Congruence Ops Agent in SUMMARY MODE.

The user wants a high-level overview or summary of patient data.

CRITICAL RULES:
1. When the user asks for MULTIPLE sessions (e.g., "last 3 sessions", "recent sessions"):
   - First call find_patient to get the patient_id
   - Then call get_multiple_sessions_summary with patient_id and number
   - Provide a comprehensive summary across all sessions
2. When the user mentions a specific session (e.g., "OCD session", "anxiety session"):
   - First call find_patient to get the patient_id
   - Then call get_patient_record to see all sessions with their titles
   - Identify which session matches the user's query
   - Use generate_clinical_note with that specific session_id
3. Call the appropriate tool to fetch the data (notes, transcript, patient record)
4. Provide a concise summary highlighting key points
5. Include specific metrics (congruence scores, timestamps) when available
6. DO NOT make up information - only summarize what the tools return

When the user mentions a patient name, call find_patient first to get patient_id.

Example flows:
User: "Summarize my last 3 sessions with Sophia"
You: 
  1. find_patient("Sophia") -> get patient_id
  2. get_multiple_sessions_summary(patient_id 3) -> Get summaries of last 3 sessions

User: "Tell me about Sophia's OCD session"
You: 
  1. find_patient("Sophia") -> get patient_id
  2. get_patient_record(patient_id) -> see all sessions (note the OCD session_id)
  3. generate_clinical_note(patient_id session_id) -> Summarize that specific session

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


class CongruenceOpsAgent:
    def __init__(self):
        self.llm = self._initialize_llm()
        self.db = get_conversation_db()
        # Fallback in-memory history if DB not available
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
    async def _get_history(self, conversation_id: Optional[str], user_id: str) -> list:
        """
        Load conversation history from database or fallback to in-memory.
        
        Args:
            conversation_id: UUID of the conversation (if continuing existing chat)
            user_id: User ID for in-memory fallback
        
        Returns:
            List of LangChain messages (HumanMessage, AIMessage)
        """
        # Try to load from database if conversation_id provided
        if conversation_id and self.db.is_enabled():
            try:
                from uuid import UUID
                messages = await self.db.get_conversation_messages(UUID(conversation_id))
                
                # Convert to LangChain messages
                history = []
                for msg in messages:
                    if msg.message_type == "user":
                        history.append(HumanMessage(content=msg.content))
                    elif msg.message_type == "agent":
                        history.append(AIMessage(content=msg.content))
                
                logger.info(f"Loaded {len(history)} messages from DB for conversation {conversation_id}")
                return history
                
            except Exception as e:
                logger.warning(f"Failed to load history from DB: {e}, falling back to in-memory")
        
        # Fallback to in-memory history
        if user_id not in self._histories:
            self._histories[user_id] = []
        return self._histories[user_id]

    def _trim_history(self, user_id: str, max_turns: int = 10) -> None:
        """Keep only the last max_turns pairs of messages (in-memory only)."""
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
            
            # Get conversation_id from context if provided
            conversation_id = request.context.get("conversation_id") if request.context else None
            
            # Load history from DB or in-memory
            history = await self._get_history(conversation_id, request.user_id)

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

            # STEP 6: PERSIST TO DATABASE (if enabled and conversation_id provided)
            if conversation_id and self.db.is_enabled():
                try:
                    from uuid import UUID
                    
                    # Save user message
                    await self.db.add_message(MessageCreate(
                        conversation_id=UUID(conversation_id),
                        message_type="user",
                        content=request.message,
                        actions=None,
                        metadata=None
                    ))
                    
                    # Save agent response
                    await self.db.add_message(MessageCreate(
                        conversation_id=UUID(conversation_id),
                        message_type="agent",
                        content=response_text,
                        actions=[action.dict() for action in self._generate_actions(response_text, request.role)],
                        metadata={
                            "tools_used": list(set(tools_used)),
                            "intent_mode": intent.mode,
                            "intent_confidence": intent.confidence,
                            "model_used": self.llm.model_name,
                        }
                    ))
                    
                    logger.info(f"Saved conversation to DB: {conversation_id}")
                    
                except Exception as e:
                    logger.warning(f"Failed to save to DB: {e}, continuing with in-memory")
            
            # Update in-memory history as fallback
            if not conversation_id or not self.db.is_enabled():
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
                    "intent_mode": intent.mode,
                    "intent_confidence": intent.confidence,
                    "conversation_id": conversation_id,  # Return conversation_id to frontend
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



import threading
_agent_instance = None
_agent_lock = threading.Lock()


def get_agent() -> CongruenceOpsAgent:
    """Get or create agent singleton (thread-safe)."""
    global _agent_instance
    if _agent_instance is None:
        with _agent_lock:
            # Double-check locking pattern
            if _agent_instance is None:
                _agent_instance = CongruenceOpsAgent()
    return _agent_instance

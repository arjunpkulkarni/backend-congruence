import os
import logging
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.models.schemas import AgentChatRequest, AgentChatResponse, AgentAction, ToolResponse

logger = logging.getLogger(__name__)

# Tool permission mapping
ROLE_PERMISSIONS = {
    "clinician": [
        "get_patient_record",
        "get_session_transcript", 
        "generate_clinical_note",
        "suggest_icd10_codes"
    ],
    "admin": [
        "get_patient_record",
        "generate_insurance_packet",
        "send_intake_form",
        "check_claim_status"
    ],
    "practice_owner": [
        "get_patient_record",
        "get_session_transcript",
        "generate_clinical_note", 
        "generate_insurance_packet",
        "suggest_icd10_codes",
        "check_claim_status",
        "schedule_appointment",
        "send_intake_form",
        "get_practice_analytics"
    ]
}

class CongruenceOpsAgent:
    def __init__(self):
        self.llm = self._initialize_llm()
        self.tools = self._create_tools()
    
    def _initialize_llm(self) -> ChatOpenAI:
        """Initialize the LLM using existing OpenAI configuration"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not configured")
        
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        return ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=0.1,
            max_tokens=1000
        )
    
    def _get_system_prompt(self, role: str, available_tools: List[str]) -> str:
        """Get the system prompt for the agent"""
        return f"""You are the Congruence Ops Agent, an AI assistant for therapy practice operations.

You help clinicians, administrators, and practice owners with:
- Clinical documentation (SOAP notes, treatment plans)
- Patient record management
- Insurance and billing workflows  
- Scheduling and intake management
- Practice analytics and operations

Always:
- Be professional and HIPAA-compliant
- Ask for clarification when needed
- Suggest relevant actions when appropriate
- Respect user role permissions

Current user role: {role}
Available tools: {', '.join(available_tools)}

If the user asks for something that requires a tool, respond with: "I need to use the [tool_name] tool for that. This tool will be implemented in a later iteration."
"""
    
    def _create_tools(self) -> List[Dict[str, str]]:
        """Create stub tools for clinical workflows"""
        return [
            {"name": "get_patient_record", "description": "Retrieve patient demographics and history"},
            {"name": "get_session_transcript", "description": "Get transcript from a therapy session"},
            {"name": "generate_clinical_note", "description": "Generate SOAP note or clinical documentation"},
            {"name": "generate_insurance_packet", "description": "Create insurance authorization packet"},
            {"name": "suggest_icd10_codes", "description": "Suggest ICD-10 diagnostic codes"},
            {"name": "check_claim_status", "description": "Check insurance claim status"},
            {"name": "schedule_appointment", "description": "Schedule patient appointment"},
            {"name": "send_intake_form", "description": "Send intake forms to patients"},
            {"name": "get_practice_analytics", "description": "Generate practice metrics and analytics"}
        ]
    
    def check_permission(self, user_role: str, tool_name: str) -> bool:
        """Check if user role has permission to use tool"""
        allowed_tools = ROLE_PERMISSIONS.get(user_role, [])
        return tool_name in allowed_tools
    
    async def process_message(self, request: AgentChatRequest) -> AgentChatResponse:
        """Process user message through simplified agent logic"""
        try:
            # Filter tools based on user permissions
            available_tools = [
                tool["name"] for tool in self.tools 
                if self.check_permission(request.role, tool["name"])
            ]
            
            # Create system prompt
            system_prompt = self._get_system_prompt(request.role, available_tools)
            
            # Process message with LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.message}
            ]
            
            response = await self.llm.ainvoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Generate suggested actions based on response
            actions = self._generate_actions(response_text, request.role)
            
            return AgentChatResponse(
                response=response_text,
                actions=actions,
                tools_used=[],  # No tools actually called in this iteration
                context=request.context,
                metadata={"model_used": self.llm.model_name}
            )
            
        except Exception as e:
            logger.error(f"Agent processing error: {e}")
            return AgentChatResponse(
                response="I apologize, but I encountered an error processing your request. Please try again.",
                actions=[],
                tools_used=[],
                context=request.context,
                metadata={"error": str(e)}
            )
    
    def _generate_actions(self, response: str, user_role: str) -> List[AgentAction]:
        """Generate contextual actions based on response and user role"""
        actions = []
        
        # Common actions for all roles
        if "patient" in response.lower():
            actions.append(AgentAction(
                type="select_patient",
                label="Select Patient",
                data={}
            ))
        
        if "session" in response.lower():
            actions.append(AgentAction(
                type="view_sessions",
                label="View Today's Sessions", 
                data={}
            ))
        
        # Role-specific actions
        if user_role == "clinician":
            if "note" in response.lower() or "soap" in response.lower():
                actions.append(AgentAction(
                    type="generate_note",
                    label="Generate Clinical Note",
                    data={}
                ))
        
        elif user_role == "admin":
            if "intake" in response.lower():
                actions.append(AgentAction(
                    type="manage_intake",
                    label="Manage Intake Forms",
                    data={}
                ))
        
        elif user_role == "practice_owner":
            if "analytics" in response.lower() or "metrics" in response.lower():
                actions.append(AgentAction(
                    type="view_analytics",
                    label="View Practice Analytics",
                    data={}
                ))
        
        return actions

# Singleton instance
_agent_instance = None

def get_agent() -> CongruenceOpsAgent:
    """Get or create agent singleton"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = CongruenceOpsAgent()
    return _agent_instance
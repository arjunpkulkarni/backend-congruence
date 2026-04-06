from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, HttpUrl, model_validator


class ProcessSessionRequest(BaseModel):
    video_url: Optional[HttpUrl] = Field(None, description="Publicly accessible URL of the video file (mp4)")
    audio_url: Optional[HttpUrl] = Field(None, description="Publicly accessible URL of the audio file (wav, mp3, m4a, etc.)")
    patient_id: str = Field(..., description="Patient or subject identifier")
    spike_threshold: float = Field(0.2, ge=0.0, le=1.0, description="Delta threshold for spike detection")
    
    @model_validator(mode='after')
    def at_least_one_url_required(self):
        if not self.video_url and not self.audio_url:
            raise ValueError('Either video_url or audio_url must be provided')
        return self


class ProcessSessionResponse(BaseModel):
    patient_id: str
    session_timestamp: int
    paths: Dict[str, Optional[str]]
    timeline_json: List[Dict[str, Any]]
    spikes_json: List[Dict[str, Any]]
    # Optional enriched outputs for direct API consumption
    timeline_10hz: Optional[List[Dict[str, Any]]] = None
    session_summary: Optional[Dict[str, Any]] = None
    notes: Optional[Dict[str, Any]] = None  # Structured therapist notes (JSON format)
    transcript_text: Optional[str] = None
    transcript_segments: Optional[List[Dict[str, Any]]] = None
    # Incongruence reasons are included in session_summary.incongruent_moments[].reason


# Congruence Ops Agent schemas
class AgentChatRequest(BaseModel):
    message: str = Field(..., description="User message to the agent")
    user_id: str = Field(..., description="User identifier")
    role: Literal["clinician", "admin", "practice_owner"] = Field(..., description="User role")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional context")


class AgentAction(BaseModel):
    type: str = Field(..., description="Action type identifier")
    label: str = Field(..., description="Human-readable action label")
    data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Action payload")


class AgentChatResponse(BaseModel):
    response: str = Field(..., description="Agent response message")
    actions: List[AgentAction] = Field(default_factory=list, description="Available actions")
    tools_used: List[str] = Field(default_factory=list, description="Tools called during processing")
    context: Dict[str, Any] = Field(default_factory=dict, description="Updated context")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class ToolResponse(BaseModel):
    status: str = Field(..., description="Tool execution status")
    message: str = Field(..., description="Tool response message")
    data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Tool response data")


# ---------------------------------------------------------------------------
# Data Access API schemas (Iteration 2)
# ---------------------------------------------------------------------------

class PatientListItem(BaseModel):
    patient_id: str
    session_count: int
    latest_session: int
    latest_session_date: str


class SessionListItem(BaseModel):
    session_id: int
    session_date: str
    patient_id: str
    has_summary: bool = False
    has_notes: bool = False
    has_transcript: bool = False
    duration: Optional[float] = None
    overall_congruence: Optional[float] = None
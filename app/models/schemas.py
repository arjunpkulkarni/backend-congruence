from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, HttpUrl, model_validator
from datetime import datetime


class ProcessSessionRequest(BaseModel):
    video_url: Optional[HttpUrl] = Field(None, description="Publicly accessible URL of the video file (mp4)")
    audio_url: Optional[HttpUrl] = Field(None, description="Publicly accessible URL of the audio file (wav, mp3, m4a, etc.)")
    patient_id: str = Field(..., description="Patient or subject identifier")
    spike_threshold: float = Field(0.2, ge=0.0, le=1.0, description="Delta threshold for spike detection")
    fast_mode: bool = Field(False, description="Enable fast processing mode (reduced accuracy for speed)")
    skip_video_analysis: bool = Field(False, description="Skip video frame analysis entirely (audio-only processing)")
    no_facial_analysis: bool = Field(False, description="Skip all facial analysis (frame extraction + DeepFace) for maximum speed")
    cleanup_files: bool = Field(True, description="Clean up temporary files after processing to save disk space")
    # webhook_url: Optional[str] = Field(None, description="URL to POST the full result to when processing completes")
    # webhook_secret: Optional[str] = Field(None, description="Shared secret sent in X-Webhook-Secret header for verification")
    # session_video_id: Optional[str] = Field(None, description="Supabase session_videos.id — passed through to the webhook payload")

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
    # Processing status and reliability tracking
    processing_status: Optional[Dict[str, Any]] = None
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


# ---------------------------------------------------------------------------
# Note Style Management API schemas (MVP)
# ---------------------------------------------------------------------------

class UploadNoteStyleRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    note_name: str = Field(..., description="Name for this note style", max_length=100)
    file_content: str = Field(..., description="Base64 encoded file content")
    file_type: str = Field(..., description="File type: pdf, docx, txt")
    
    @model_validator(mode='after')
    def validate_file_type(self):
        allowed_types = ['pdf', 'docx', 'txt']
        if self.file_type.lower() not in allowed_types:
            raise ValueError(f'File type must be one of: {", ".join(allowed_types)}')
        return self

class NoteStyleResponse(BaseModel):
    id: str
    user_id: str
    note_name: str
    preview_text: str
    file_type: str
    created_at: str
    is_active: bool
    validation_info: Optional[Dict[str, Any]] = None
    style_analysis: Optional[Dict[str, Any]] = None

class ListNoteStylesResponse(BaseModel):
    note_styles: List[NoteStyleResponse]
    total_count: int

class GenerateNotesWithStyleRequest(BaseModel):
    transcript_text: str = Field(..., description="Session transcript text")
    transcript_segments: Optional[List[Dict[str, Any]]] = Field(None, description="Transcript segments with timestamps")
    session_summary: Optional[Dict[str, Any]] = Field(None, description="Session summary data")
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    user_id: Optional[str] = Field(None, description="User ID for note style lookup")
    use_note_style: bool = Field(False, description="Whether to use uploaded note style")

class GenerateNotesWithStyleResponse(BaseModel):
    format: str  # "style_matched" or "standard"
    content: str
    style_source: Optional[str] = None  # "user_uploaded" or None
    patient_id: Optional[str] = None
    generated_at: str
    style_info: Optional[Dict[str, Any]] = None

class DeleteNoteStyleRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    note_style_id: str = Field(..., description="Note style ID to delete")

class SetActiveNoteStyleRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    note_style_id: str = Field(..., description="Note style ID to set as active")


# ---------------------------------------------------------------------------
# Background job tracking schemas
# ---------------------------------------------------------------------------

class JobSubmittedResponse(BaseModel):
    job_id: str
    status: str = "queued"
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # queued | processing | completed | failed
    stage: Optional[str] = None
    progress: Optional[int] = None
    message: Optional[str] = None
    error: Optional[str] = None
    result: Optional[ProcessSessionResponse] = None
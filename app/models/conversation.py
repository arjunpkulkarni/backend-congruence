"""
Pydantic models for Copilot Conversations and Messages.
Maps to the database schema for persistent chat storage.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID
from pydantic import BaseModel, Field


class ConversationCreate(BaseModel):
    """Request to create a new conversation."""
    patient_id: Optional[UUID] = None
    appointment_id: Optional[UUID] = None
    title: Optional[str] = None


class ConversationUpdate(BaseModel):
    """Request to update a conversation."""
    title: Optional[str] = None
    patient_id: Optional[UUID] = None
    appointment_id: Optional[UUID] = None


class Conversation(BaseModel):
    """Database model for copilot_conversations table."""
    id: UUID
    user_id: UUID
    patient_id: Optional[UUID] = None
    appointment_id: Optional[UUID] = None
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class MessageCreate(BaseModel):
    """Request to create a new message in a conversation."""
    conversation_id: UUID
    message_type: str = Field(..., pattern="^(user|agent)$")
    content: str
    actions: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class Message(BaseModel):
    """Database model for copilot_messages table."""
    id: UUID
    conversation_id: UUID
    message_type: str
    content: str
    actions: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime


class ConversationWithMessages(BaseModel):
    """Conversation with all its messages."""
    conversation: Conversation
    messages: List[Message]


class ConversationListItem(BaseModel):
    """Lightweight conversation item for list view."""
    id: UUID
    title: Optional[str]
    patient_id: Optional[UUID]
    created_at: datetime
    updated_at: datetime
    message_count: int
    last_message_preview: Optional[str] = None

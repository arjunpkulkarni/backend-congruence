"""
Database service for Copilot Conversations.

Handles persistence of agent chat conversations and messages to Supabase/PostgreSQL.
"""

import os
import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from supabase import create_client, Client

from app.models.conversation import (
    Conversation,
    ConversationCreate,
    ConversationUpdate,
    Message,
    MessageCreate,
    ConversationWithMessages,
    ConversationListItem,
)

logger = logging.getLogger(__name__)


class ConversationDatabase:
    """Database service for managing copilot conversations."""
    
    def __init__(self):
        self.client: Optional[Client] = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Supabase client from environment variables."""
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            logger.warning(
                "SUPABASE_URL or SUPABASE_KEY not configured. "
                "Conversation persistence will be disabled."
            )
            return
        
        try:
            self.client = create_client(supabase_url, supabase_key)
            logger.info("Supabase client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            self.client = None
    
    def is_enabled(self) -> bool:
        """Check if database persistence is enabled."""
        return self.client is not None
    
    # -------------------------------------------------------------------------
    # Conversations
    # -------------------------------------------------------------------------
    
    async def create_conversation(
        self,
        user_id: UUID,
        data: ConversationCreate
    ) -> Optional[Conversation]:
        """Create a new conversation."""
        if not self.is_enabled():
            logger.warning("Database not enabled, cannot create conversation")
            return None
        
        try:
            insert_data = {
                "user_id": str(user_id),
                "patient_id": str(data.patient_id) if data.patient_id else None,
                "appointment_id": str(data.appointment_id) if data.appointment_id else None,
                "title": data.title,
            }
            
            response = self.client.table("copilot_conversations").insert(insert_data).execute()
            
            if response.data:
                return Conversation(**response.data[0])
            return None
            
        except Exception as e:
            logger.error(f"Error creating conversation: {e}")
            return None
    
    async def get_conversation(
        self,
        conversation_id: UUID,
        user_id: UUID
    ) -> Optional[Conversation]:
        """Get a conversation by ID (with RLS check)."""
        if not self.is_enabled():
            return None
        
        try:
            response = self.client.table("copilot_conversations")\
                .select("*")\
                .eq("id", str(conversation_id))\
                .eq("user_id", str(user_id))\
                .execute()
            
            if response.data:
                return Conversation(**response.data[0])
            return None
            
        except Exception as e:
            logger.error(f"Error fetching conversation: {e}")
            return None
    
    async def list_conversations(
        self,
        user_id: UUID,
        limit: int = 50
    ) -> List[ConversationListItem]:
        """List all conversations for a user."""
        if not self.is_enabled():
            return []
        
        try:
            # Get conversations with message count
            response = self.client.rpc(
                "get_user_conversations_with_counts",
                {"p_user_id": str(user_id), "p_limit": limit}
            ).execute()
            
            if response.data:
                return [ConversationListItem(**item) for item in response.data]
            
            # Fallback if RPC doesn't exist yet
            response = self.client.table("copilot_conversations")\
                .select("*")\
                .eq("user_id", str(user_id))\
                .order("updated_at", desc=True)\
                .limit(limit)\
                .execute()
            
            conversations = []
            for conv in response.data:
                # Get message count separately
                msg_response = self.client.table("copilot_messages")\
                    .select("id", count="exact")\
                    .eq("conversation_id", conv["id"])\
                    .execute()
                
                conversations.append(ConversationListItem(
                    id=conv["id"],
                    title=conv.get("title"),
                    patient_id=conv.get("patient_id"),
                    created_at=conv["created_at"],
                    updated_at=conv["updated_at"],
                    message_count=msg_response.count or 0,
                ))
            
            return conversations
            
        except Exception as e:
            logger.error(f"Error listing conversations: {e}")
            return []
    
    async def update_conversation(
        self,
        conversation_id: UUID,
        user_id: UUID,
        data: ConversationUpdate
    ) -> Optional[Conversation]:
        """Update a conversation."""
        if not self.is_enabled():
            return None
        
        try:
            update_data = {}
            if data.title is not None:
                update_data["title"] = data.title
            if data.patient_id is not None:
                update_data["patient_id"] = str(data.patient_id)
            if data.appointment_id is not None:
                update_data["appointment_id"] = str(data.appointment_id)
            
            if not update_data:
                return await self.get_conversation(conversation_id, user_id)
            
            response = self.client.table("copilot_conversations")\
                .update(update_data)\
                .eq("id", str(conversation_id))\
                .eq("user_id", str(user_id))\
                .execute()
            
            if response.data:
                return Conversation(**response.data[0])
            return None
            
        except Exception as e:
            logger.error(f"Error updating conversation: {e}")
            return None
    
    async def delete_conversation(
        self,
        conversation_id: UUID,
        user_id: UUID
    ) -> bool:
        """Delete a conversation (cascade deletes messages)."""
        if not self.is_enabled():
            return False
        
        try:
            response = self.client.table("copilot_conversations")\
                .delete()\
                .eq("id", str(conversation_id))\
                .eq("user_id", str(user_id))\
                .execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting conversation: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # Messages
    # -------------------------------------------------------------------------
    
    async def add_message(
        self,
        data: MessageCreate
    ) -> Optional[Message]:
        """Add a message to a conversation."""
        if not self.is_enabled():
            return None
        
        try:
            insert_data = {
                "conversation_id": str(data.conversation_id),
                "message_type": data.message_type,
                "content": data.content,
                "actions": data.actions,
                "metadata": data.metadata,
            }
            
            response = self.client.table("copilot_messages").insert(insert_data).execute()
            
            if response.data:
                return Message(**response.data[0])
            return None
            
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            return None
    
    async def get_conversation_messages(
        self,
        conversation_id: UUID,
        limit: int = 100
    ) -> List[Message]:
        """Get all messages for a conversation."""
        if not self.is_enabled():
            return []
        
        try:
            response = self.client.table("copilot_messages")\
                .select("*")\
                .eq("conversation_id", str(conversation_id))\
                .order("created_at", desc=False)\
                .limit(limit)\
                .execute()
            
            if response.data:
                return [Message(**msg) for msg in response.data]
            return []
            
        except Exception as e:
            logger.error(f"Error fetching messages: {e}")
            return []
    
    async def get_conversation_with_messages(
        self,
        conversation_id: UUID,
        user_id: UUID
    ) -> Optional[ConversationWithMessages]:
        """Get a conversation with all its messages."""
        conversation = await self.get_conversation(conversation_id, user_id)
        if not conversation:
            return None
        
        messages = await self.get_conversation_messages(conversation_id)
        
        return ConversationWithMessages(
            conversation=conversation,
            messages=messages
        )
    
    async def delete_message(
        self,
        message_id: UUID
    ) -> bool:
        """Delete a specific message."""
        if not self.is_enabled():
            return False
        
        try:
            response = self.client.table("copilot_messages")\
                .delete()\
                .eq("id", str(message_id))\
                .execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting message: {e}")
            return False


# ---------------------------------------------------------------------------
# Thread-safe singleton instance
# ---------------------------------------------------------------------------
import threading
_db_instance: Optional[ConversationDatabase] = None
_db_lock = threading.Lock()


def get_conversation_db() -> ConversationDatabase:
    """Get or create conversation database singleton (thread-safe)."""
    global _db_instance
    if _db_instance is None:
        with _db_lock:
            # Double-check locking pattern
            if _db_instance is None:
                _db_instance = ConversationDatabase()
    return _db_instance

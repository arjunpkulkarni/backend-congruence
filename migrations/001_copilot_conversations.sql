-- =====================================================================
-- Copilot Conversations Database Schema
-- =====================================================================
-- This migration creates tables for storing agent chat conversations
-- with full RLS (Row Level Security) policies for multi-tenant access.
--
-- Tables:
--   1. copilot_conversations - Conversation metadata
--   2. copilot_messages - Individual messages in conversations
--
-- Features:
--   - UUID primary keys
--   - Soft delete on patient/appointment (SET NULL)
--   - Cascade delete on conversation (deletes all messages)
--   - Auto-updated timestamps
--   - RLS policies for user isolation
-- =====================================================================

-- =====================================================================
-- Table 1: copilot_conversations
-- =====================================================================

CREATE TABLE IF NOT EXISTS public.copilot_conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    patient_id UUID REFERENCES public.patients(id) ON DELETE SET NULL,
    appointment_id UUID REFERENCES public.appointments(id) ON DELETE SET NULL,
    title TEXT,  -- Auto-generated from first message or user-provided
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_copilot_conversations_user_id 
    ON public.copilot_conversations(user_id);

CREATE INDEX IF NOT EXISTS idx_copilot_conversations_patient_id 
    ON public.copilot_conversations(patient_id) 
    WHERE patient_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_copilot_conversations_created_at 
    ON public.copilot_conversations(created_at DESC);

-- RLS Policies
ALTER TABLE public.copilot_conversations ENABLE ROW LEVEL SECURITY;

-- Users can view their own conversations
CREATE POLICY "Users can view own conversations"
    ON public.copilot_conversations FOR SELECT
    USING (auth.uid() = user_id);

-- Users can insert their own conversations
CREATE POLICY "Users can insert own conversations"
    ON public.copilot_conversations FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- Users can update their own conversations
CREATE POLICY "Users can update own conversations"
    ON public.copilot_conversations FOR UPDATE
    USING (auth.uid() = user_id);

-- Users can delete their own conversations
CREATE POLICY "Users can delete own conversations"
    ON public.copilot_conversations FOR DELETE
    USING (auth.uid() = user_id);


-- =====================================================================
-- Table 2: copilot_messages
-- =====================================================================

CREATE TABLE IF NOT EXISTS public.copilot_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES public.copilot_conversations(id) ON DELETE CASCADE,
    message_type TEXT NOT NULL CHECK (message_type IN ('user', 'agent')),
    content TEXT NOT NULL,
    actions JSONB,  -- Store action buttons from agent responses
    metadata JSONB,  -- Store tools_used, intent_mode, model_used, etc.
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_copilot_messages_conversation_id 
    ON public.copilot_messages(conversation_id);

CREATE INDEX IF NOT EXISTS idx_copilot_messages_created_at 
    ON public.copilot_messages(created_at);

-- RLS Policies
ALTER TABLE public.copilot_messages ENABLE ROW LEVEL SECURITY;

-- Users can view messages in their own conversations
CREATE POLICY "Users can view messages in own conversations"
    ON public.copilot_messages FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM public.copilot_conversations
            WHERE id = copilot_messages.conversation_id
            AND user_id = auth.uid()
        )
    );

-- Users can insert messages in their own conversations
CREATE POLICY "Users can insert messages in own conversations"
    ON public.copilot_messages FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.copilot_conversations
            WHERE id = copilot_messages.conversation_id
            AND user_id = auth.uid()
        )
    );

-- Users can delete messages in their own conversations
CREATE POLICY "Users can delete messages in own conversations"
    ON public.copilot_messages FOR DELETE
    USING (
        EXISTS (
            SELECT 1 FROM public.copilot_conversations
            WHERE id = copilot_messages.conversation_id
            AND user_id = auth.uid()
        )
    );


-- =====================================================================
-- Update Trigger for updated_at
-- =====================================================================

-- Create the trigger function if it doesn't exist
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to copilot_conversations
CREATE TRIGGER update_copilot_conversations_updated_at
    BEFORE UPDATE ON public.copilot_conversations
    FOR EACH ROW
    EXECUTE FUNCTION public.update_updated_at_column();


-- =====================================================================
-- Helper RPC Function (Optional but recommended)
-- =====================================================================
-- This function efficiently gets conversations with message counts
-- in a single query, avoiding N+1 queries from the application layer.

CREATE OR REPLACE FUNCTION public.get_user_conversations_with_counts(
    p_user_id UUID,
    p_limit INT DEFAULT 50
)
RETURNS TABLE (
    id UUID,
    title TEXT,
    patient_id UUID,
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ,
    message_count BIGINT,
    last_message_preview TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id,
        c.title,
        c.patient_id,
        c.created_at,
        c.updated_at,
        COUNT(m.id) as message_count,
        (
            SELECT content 
            FROM public.copilot_messages 
            WHERE conversation_id = c.id 
            ORDER BY created_at DESC 
            LIMIT 1
        ) as last_message_preview
    FROM public.copilot_conversations c
    LEFT JOIN public.copilot_messages m ON m.conversation_id = c.id
    WHERE c.user_id = p_user_id
    GROUP BY c.id, c.title, c.patient_id, c.created_at, c.updated_at
    ORDER BY c.updated_at DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;


-- =====================================================================
-- Grant Permissions (adjust based on your auth setup)
-- =====================================================================

-- Grant authenticated users access to tables
GRANT SELECT, INSERT, UPDATE, DELETE ON public.copilot_conversations TO authenticated;
GRANT SELECT, INSERT, DELETE ON public.copilot_messages TO authenticated;

-- Grant access to the helper function
GRANT EXECUTE ON FUNCTION public.get_user_conversations_with_counts(UUID, INT) TO authenticated;


-- =====================================================================
-- Verification Queries
-- =====================================================================
-- Run these after migration to verify everything works:

-- Check tables exist
-- SELECT table_name FROM information_schema.tables 
-- WHERE table_schema = 'public' AND table_name LIKE 'copilot_%';

-- Check RLS is enabled
-- SELECT tablename, rowsecurity FROM pg_tables 
-- WHERE schemaname = 'public' AND tablename LIKE 'copilot_%';

-- Check policies exist
-- SELECT tablename, policyname FROM pg_policies 
-- WHERE schemaname = 'public' AND tablename LIKE 'copilot_%';

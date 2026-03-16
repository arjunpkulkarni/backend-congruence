"""
Intent Classification for Congruence Ops Agent

Classifies user queries into 3 modes:
1. Evidence Mode - User wants proof/quotes from actual data
2. Summary Mode - User wants high-level overview
3. Action Mode - User wants to execute a workflow
"""

import re
from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel


class IntentClassification(BaseModel):
    """Result of intent classification."""
    mode: Literal["evidence", "summary", "action"]
    confidence: float  # 0.0 to 1.0
    entities: Dict[str, Any]  # Extracted entities (patient name, condition, etc.)
    search_terms: list[str]  # Terms to search for in evidence mode


# Evidence mode keywords
EVIDENCE_KEYWORDS = [
    "show me", "find", "proof", "evidence", "mention", "said", "talked about",
    "discussed", "notes that", "where", "which", "what did", "does it say",
    "is there", "any mention", "search for", "look for", "find me",
]

# Summary mode keywords
SUMMARY_KEYWORDS = [
    "summarize", "summary", "overview", "key themes", "main points",
    "what happened", "tell me about", "give me", "describe",
    "what's the", "how is", "how was", "what are",
]

# Action mode keywords
ACTION_KEYWORDS = [
    "generate", "create", "draft", "make", "build", "write",
    "send", "schedule", "check status", "suggest codes",
    "prepare", "compile", "produce",
]

# Medical/clinical terms that suggest evidence mode
CLINICAL_TERMS = [
    "ocd", "anxiety", "depression", "ptsd", "bipolar", "adhd",
    "panic", "phobia", "trauma", "insomnia", "sleep", "suicidal",
    "self-harm", "substance", "alcohol", "diagnosis", "symptom",
    "medication", "therapy", "treatment", "disorder", "condition",
]


def classify_intent(user_message: str) -> IntentClassification:
    """
    Classify user intent into evidence/summary/action mode.
    
    Returns IntentClassification with mode, confidence, and extracted entities.
    """
    msg_lower = user_message.lower().strip()
    
    # Score each mode
    evidence_score = 0.0
    summary_score = 0.0
    action_score = 0.0
    
    # Check for evidence keywords
    for keyword in EVIDENCE_KEYWORDS:
        if keyword in msg_lower:
            evidence_score += 2.0
            break  # Only count once
    
    # Check for summary keywords
    for keyword in SUMMARY_KEYWORDS:
        if keyword in msg_lower:
            summary_score += 2.0
            break
    
    # Check for action keywords
    for keyword in ACTION_KEYWORDS:
        if keyword in msg_lower:
            action_score += 3.0  # Actions are usually explicit
            break
    
    # Clinical terms strongly suggest evidence mode
    clinical_terms_found = []
    for term in CLINICAL_TERMS:
        if term in msg_lower:
            evidence_score += 1.5
            clinical_terms_found.append(term)
    
    # Question words suggest evidence mode
    if any(q in msg_lower for q in ["what did", "does", "is there", "where", "which"]):
        evidence_score += 1.0
    
    # "About" suggests summary mode
    if " about " in msg_lower or msg_lower.startswith("about "):
        summary_score += 1.0
    
    # Quotes/proof explicitly mentioned
    if any(word in msg_lower for word in ["quote", "proof", "evidence", "mention", "said"]):
        evidence_score += 3.0
    
    # Determine mode based on scores
    max_score = max(evidence_score, summary_score, action_score)
    
    if max_score == 0:
        # Default to summary if unclear
        mode = "summary"
        confidence = 0.3
    elif evidence_score == max_score:
        mode = "evidence"
        confidence = min(evidence_score / 5.0, 1.0)
    elif action_score == max_score:
        mode = "action"
        confidence = min(action_score / 5.0, 1.0)
    else:
        mode = "summary"
        confidence = min(summary_score / 5.0, 1.0)
    
    # Extract entities
    entities = _extract_entities(user_message)
    
    # Extract search terms for evidence mode
    search_terms = []
    if mode == "evidence":
        search_terms = clinical_terms_found + _extract_quoted_terms(user_message)
    
    return IntentClassification(
        mode=mode,
        confidence=confidence,
        entities=entities,
        search_terms=search_terms,
    )


def _extract_entities(message: str) -> Dict[str, Any]:
    """Extract entities like patient names, conditions, etc."""
    entities = {}
    
    # Extract patient name patterns
    # "Rob Wazowski", "patient Rob", "Sophia", etc.
    name_patterns = [
        r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',  # First Last
        r'patient (\w+)',
        r'for (\w+)',
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, message)
        if match:
            entities["patient_name"] = match.group(1)
            break
    
    # Extract time references
    if any(word in message.lower() for word in ["today", "latest", "recent", "last"]):
        entities["time_reference"] = "recent"
    elif "all" in message.lower():
        entities["time_reference"] = "all"
    
    return entities


def _extract_quoted_terms(message: str) -> list[str]:
    """Extract terms in quotes or after 'about'."""
    terms = []
    
    # Find quoted terms
    quoted = re.findall(r'"([^"]+)"', message)
    terms.extend(quoted)
    
    quoted_single = re.findall(r"'([^']+)'", message)
    terms.extend(quoted_single)
    
    # Find terms after "about"
    about_match = re.search(r'about\s+(\w+)', message.lower())
    if about_match:
        terms.append(about_match.group(1))
    
    return terms


def format_evidence_response(quotes: list[Dict[str, Any]], query: str) -> str:
    """
    Format evidence mode response with quotes first, then brief explanation.
    
    Args:
        quotes: List of dicts with 'text', 'source', 'timestamp', 'patient'
        query: Original user query
    
    Returns:
        Formatted response with quotes prominently displayed
    """
    if not quotes:
        return f"❌ **No evidence found** for: '{query}'\n\nI searched patient records, clinical notes, and transcripts but found no mentions."
    
    response_parts = [
        f"📋 **Evidence Found** ({len(quotes)} {'result' if len(quotes) == 1 else 'results'}) for '{query}':",
        ""
    ]
    
    for i, quote in enumerate(quotes, 1):
        response_parts.append(f"**{i}. {quote.get('source', 'Unknown Source')}**")
        response_parts.append(f"   📅 Session: {quote.get('session_date', 'Unknown date')}")
        response_parts.append(f"   👤 Patient: {quote.get('patient', 'Unknown')}")
        if quote.get('timestamp'):
            response_parts.append(f"   ⏱️  Timestamp: {quote['timestamp']}")
        response_parts.append(f"")
        response_parts.append(f"   > *\"{quote['text']}\"*")
        response_parts.append("")
    
    return "\n".join(response_parts)


def format_summary_response(data: Dict[str, Any]) -> str:
    """Format summary mode response - high level overview."""
    # This is handled by the LLM with the data
    return data


def format_action_response(result: Dict[str, Any]) -> str:
    """Format action mode response - confirmation of action taken."""
    # This is handled by the tool response
    return result

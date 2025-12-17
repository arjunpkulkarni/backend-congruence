"""
Patient De-identification Utilities

Converts patient identifiers to pseudonymous IDs using HMAC-SHA256.
Ensures same patient always gets same pseudonym (for cross-session analysis)
while making it computationally infeasible to reverse.

HIPAA Safe Harbor Method: 
This implements a "limited data set" approach where direct identifiers 
are replaced with pseudonyms, but patterns across sessions are preserved.
"""

import hashlib
import hmac
import re
import os
from typing import Optional
import spacy


class PatientDeidentifier:
    """
    Convert patient IDs and PII to pseudonymous identifiers.
    
    Usage:
        deidentifier = PatientDeidentifier(pepper=os.environ['PEPPER_SECRET'])
        pseudonym = deidentifier.pseudonymize("patient_12345")
        # Always produces same pseudonym for same input
    """
    
    def __init__(self, pepper: bytes):
        """
        Initialize with secret pepper value.
        
        Args:
            pepper: Secret value used in HMAC. Must be:
                   - Stored separately from data (use key management system)
                   - At least 32 bytes (256 bits)
                   - Never shared or logged
        
        Raises:
            ValueError: If pepper is too short
        """
        if isinstance(pepper, str):
            pepper = pepper.encode('utf-8')
        
        if len(pepper) < 32:
            raise ValueError("Pepper must be at least 32 bytes for security")
        
        self.pepper = pepper
    
    def pseudonymize(self, patient_id: str) -> str:
        """
        Convert patient_id to non-reversible pseudonym.
        
        Same patient_id always produces same pseudonym (deterministic).
        Computationally infeasible to reverse without pepper.
        
        Args:
            patient_id: Original patient identifier (email, MRN, etc.)
        
        Returns:
            32-character hexadecimal pseudonym
        
        Example:
            >>> deidentifier.pseudonymize("patient@example.com")
            'a1b2c3d4e5f6...' (32 chars)
            >>> deidentifier.pseudonymize("patient@example.com")  # Same result
            'a1b2c3d4e5f6...' (32 chars)
        """
        # Use HMAC-SHA256 for cryptographically secure pseudonymization
        h = hmac.new(
            self.pepper,
            patient_id.encode('utf-8'),
            hashlib.sha256
        )
        
        # Return first 128 bits (32 hex chars) for compact storage
        # Still provides 2^128 possible values (collision-resistant)
        return h.hexdigest()[:32]
    
    def is_valid_pseudonym(self, pseudonym: str) -> bool:
        """
        Check if string looks like a valid pseudonym.
        
        Args:
            pseudonym: String to validate
        
        Returns:
            True if matches pseudonym format
        """
        if not isinstance(pseudonym, str):
            return False
        
        return (
            len(pseudonym) == 32 and
            all(c in '0123456789abcdef' for c in pseudonym)
        )
    
    def anonymize_text(self, text: str, use_spacy: bool = True) -> str:
        """
        Remove personally identifiable information from text.
        
        Uses Named Entity Recognition to detect and redact:
        - PERSON: Names
        - DATE: Specific dates
        - GPE: Geopolitical entities (cities, countries)
        - LOC: Locations
        - ORG: Organizations
        - EMAIL: Email addresses (regex)
        - PHONE: Phone numbers (regex)
        
        Args:
            text: Text potentially containing PII
            use_spacy: If True, use spaCy NER (more accurate but slower)
                      If False, use regex only (faster but less accurate)
        
        Returns:
            Anonymized text with identifiers replaced by [TYPE] markers
        
        Example:
            >>> deidentifier.anonymize_text("My name is John and I live in NYC")
            'My name is [PERSON] and I live in [GPE]'
        """
        if not text:
            return text
        
        anonymized = text
        
        # Regex-based redaction (fast, always runs)
        # Email addresses
        anonymized = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[EMAIL]',
            anonymized
        )
        
        # Phone numbers (US format)
        anonymized = re.sub(
            r'\b(\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b',
            '[PHONE]',
            anonymized
        )
        
        # Social Security Numbers (US format)
        anonymized = re.sub(
            r'\b\d{3}-\d{2}-\d{4}\b',
            '[SSN]',
            anonymized
        )
        
        # Specific dates (YYYY-MM-DD, MM/DD/YYYY, etc.)
        anonymized = re.sub(
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            '[DATE]',
            anonymized
        )
        anonymized = re.sub(
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            '[DATE]',
            anonymized
        )
        
        # spaCy NER (optional, more accurate)
        if use_spacy:
            try:
                nlp = spacy.load("en_core_web_sm")
                doc = nlp(anonymized)
                
                # Replace entities in reverse order to preserve positions
                replacements = []
                for ent in doc.ents:
                    if ent.label_ in ["PERSON", "DATE", "GPE", "LOC", "ORG"]:
                        replacements.append((ent.start_char, ent.end_char, f"[{ent.label_}]"))
                
                # Apply replacements from end to start
                for start, end, replacement in reversed(replacements):
                    anonymized = anonymized[:start] + replacement + anonymized[end:]
            
            except Exception:
                # spaCy not available or error - regex redaction already applied
                pass
        
        return anonymized


def setup_deidentifier() -> PatientDeidentifier:
    """
    Initialize deidentifier from environment variables.
    
    Looks for PEPPER_SECRET in environment.
    
    Returns:
        Configured PatientDeidentifier instance
    
    Raises:
        RuntimeError: If PEPPER_SECRET not found
    """
    pepper = os.environ.get('PEPPER_SECRET')
    
    if not pepper:
        if os.environ.get('ENVIRONMENT') == 'production':
            raise RuntimeError(
                "PEPPER_SECRET not set in production environment! "
                "Generate with: openssl rand -hex 32"
            )
        
        # Development only: generate temporary pepper
        import warnings
        warnings.warn(
            "No PEPPER_SECRET found. Generating temporary pepper for development. "
            "This pepper will not persist across restarts. DO NOT USE IN PRODUCTION!",
            UserWarning
        )
        pepper = os.urandom(32).hex()
        os.environ['PEPPER_SECRET'] = pepper
    
    return PatientDeidentifier(pepper.encode('utf-8'))


# Example usage:
"""
from app.utils.deidentification import setup_deidentifier

# Initialize once at startup
deidentifier = setup_deidentifier()

# Pseudonymize patient ID before storing
@app.post("/process_session")
def process_session(payload: ProcessSessionRequest):
    # Never store real patient ID
    pseudonym = deidentifier.pseudonymize(payload.patient_id)
    
    # Use pseudonym for all file paths and database records
    session_dir = create_session_directories(
        patient_id=pseudonym,  # Not real ID
        session_ts=session_ts
    )
    
    # Anonymize transcript before logging
    if transcript_text:
        safe_transcript = deidentifier.anonymize_text(transcript_text)
        logger.info("Transcript: %s", safe_transcript)
"""


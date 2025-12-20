"""
HIPAA-Compliant Encryption Utilities

Provides AES-256 encryption for PHI (Protected Health Information).
Uses cryptography.fernet which implements secure defaults.

IMPORTANT: 
- Store encryption keys in a secure key management system (AWS KMS, Azure Key Vault, etc.)
- NEVER commit keys to version control
- Rotate keys regularly (recommended: annually)
"""

from cryptography.fernet import Fernet
import os
import json
from typing import Any, Dict


class PHIEncryption:
    """
    Encrypt/decrypt PHI using AES-256 via Fernet.
    
    Usage:
        # Initialize with master key from environment
        cipher = PHIEncryption(master_key=os.environ['MASTER_ENCRYPTION_KEY'])
        
        # Encrypt a file
        cipher.encrypt_file('input.mp4', 'output.mp4.encrypted')
        
        # Encrypt JSON data
        encrypted = cipher.encrypt_json({"patient": "data"})
        
        # Decrypt
        decrypted = cipher.decrypt_json(encrypted)
    """
    
    def __init__(self, master_key: bytes):
        """
        Initialize cipher with master key.
        
        Args:
            master_key: Base64-encoded Fernet key (44 bytes when decoded).
                       Generate with: Fernet.generate_key()
        
        Raises:
            ValueError: If key is invalid
        """
        if isinstance(master_key, str):
            master_key = master_key.encode('utf-8')
        
        try:
            self.cipher = Fernet(master_key)
        except Exception as e:
            raise ValueError(f"Invalid encryption key: {e}")
    
    @staticmethod
    def generate_key() -> bytes:
        """
        Generate a new Fernet key for encryption.
        
        Returns:
            Base64-encoded 256-bit key
        
        Example:
            >>> key = PHIEncryption.generate_key()
            >>> print(key.decode())
            'abcd1234...' (44 characters)
        """
        return Fernet.generate_key()
    
    def encrypt_file(self, input_path: str, output_path: str) -> None:
        """
        Encrypt a file (video, audio, image, etc.)
        
        Args:
            input_path: Path to plaintext file
            output_path: Path to write encrypted file
        
        Raises:
            FileNotFoundError: If input file doesn't exist
            IOError: If encryption fails
        """
        try:
            with open(input_path, 'rb') as f:
                data = f.read()
            
            encrypted = self.cipher.encrypt(data)
            
            with open(output_path, 'wb') as f:
                f.write(encrypted)
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found: {input_path}")
        except Exception as e:
            raise IOError(f"Encryption failed: {e}")
    
    def decrypt_file(self, input_path: str, output_path: str) -> None:
        """
        Decrypt a file for processing.
        
        Args:
            input_path: Path to encrypted file
            output_path: Path to write decrypted file
        
        Raises:
            FileNotFoundError: If input file doesn't exist
            cryptography.fernet.InvalidToken: If decryption fails (wrong key or corrupted file)
        """
        try:
            with open(input_path, 'rb') as f:
                encrypted = f.read()
            
            decrypted = self.cipher.decrypt(encrypted)
            
            with open(output_path, 'wb') as f:
                f.write(decrypted)
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found: {input_path}")
        except Exception as e:
            raise IOError(f"Decryption failed (wrong key or corrupted data): {e}")
    
    def encrypt_json(self, data: Dict[str, Any]) -> bytes:
        """
        Encrypt JSON-serializable data (session summaries, analysis results, etc.)
        
        Args:
            data: Dictionary to encrypt
        
        Returns:
            Encrypted bytes
        
        Example:
            >>> encrypted = cipher.encrypt_json({"intensity": 0.5})
            >>> decrypted = cipher.decrypt_json(encrypted)
            >>> print(decrypted)
            {'intensity': 0.5}
        """
        try:
            json_str = json.dumps(data, ensure_ascii=False).encode('utf-8')
            return self.cipher.encrypt(json_str)
        except Exception as e:
            raise ValueError(f"Encryption failed: {e}")
    
    def decrypt_json(self, encrypted: bytes) -> Dict[str, Any]:
        """
        Decrypt JSON data.
        
        Args:
            encrypted: Encrypted bytes from encrypt_json()
        
        Returns:
            Decrypted dictionary
        
        Raises:
            cryptography.fernet.InvalidToken: If decryption fails
            json.JSONDecodeError: If decrypted data is not valid JSON
        """
        try:
            decrypted = self.cipher.decrypt(encrypted)
            return json.loads(decrypted.decode('utf-8'))
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")
    
    def encrypt_string(self, text: str) -> bytes:
        """
        Encrypt a string (patient ID, name, etc.)
        
        Args:
            text: String to encrypt
        
        Returns:
            Encrypted bytes
        """
        return self.cipher.encrypt(text.encode('utf-8'))
    
    def decrypt_string(self, encrypted: bytes) -> str:
        """
        Decrypt a string.
        
        Args:
            encrypted: Encrypted bytes from encrypt_string()
        
        Returns:
            Decrypted string
        """
        return self.cipher.decrypt(encrypted).decode('utf-8')


def setup_encryption() -> PHIEncryption:
    """
    Initialize encryption from environment variables.
    
    Looks for MASTER_ENCRYPTION_KEY in environment.
    If not found, generates one (DEV ONLY - not for production!)
    
    Returns:
        Configured PHIEncryption instance
    
    Raises:
        RuntimeError: If no key found and not in development mode
    """
    key = os.environ.get('MASTER_ENCRYPTION_KEY')
    
    if not key:
        # In production, this should raise an error
        if os.environ.get('ENVIRONMENT') == 'production':
            raise RuntimeError(
                "MASTER_ENCRYPTION_KEY not set in production environment! "
                "Store encryption key in secure key management system (AWS KMS, Azure Key Vault, etc.)"
            )
        
        # Development only: generate a temporary key
        import warnings
        warnings.warn(
            "No MASTER_ENCRYPTION_KEY found. Generating temporary key for development. "
            "This key will not persist across restarts. DO NOT USE IN PRODUCTION!",
            UserWarning
        )
        key = PHIEncryption.generate_key().decode()
        os.environ['MASTER_ENCRYPTION_KEY'] = key
    
    return PHIEncryption(key.encode('utf-8'))


# Example usage in main.py:
"""
from app.utils.encryption import setup_encryption

# Initialize once at startup
cipher = setup_encryption()

# Encrypt video after download
cipher.encrypt_file('video.mp4', 'video.mp4.encrypted')

# Encrypt analysis results before storage
encrypted_results = cipher.encrypt_json(session_summary)

# Decrypt for processing
cipher.decrypt_file('video.mp4.encrypted', 'video.mp4')
"""



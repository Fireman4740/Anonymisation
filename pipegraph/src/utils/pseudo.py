import hmac
import hashlib
from typing import Union
from src.config import settings


class PseudoMapper:
    """
    Handles deterministic pseudonymization using HMAC-SHA256.
    Replaces the missing legacy PseudoMapper.
    """

    def __init__(self, secret: Union[str, bytes], scope_id: str):
        """
        Initialize with a secret and a scope identifier.
        """
        self.secret = secret.encode("utf-8") if isinstance(secret, str) else secret
        self.scope_id = scope_id
        # Use the salt from the new centralized settings
        self.salt = settings.security.PSEUDO_SALT

    def placeholder(self, entity_type: str, entity_text: str) -> str:
        """
        Generates a secure, deterministic placeholder for an entity.
        Format: [TYPE_HASH]
        """
        # Combine salt, scope, type, and text to ensure uniqueness and security
        message = f"{self.salt}|{self.scope_id}|{entity_type}|{entity_text}"

        # Compute HMAC-SHA256
        h = hmac.new(self.secret, message.encode("utf-8"), hashlib.sha256)

        # Generate a short, readable hash fragment (8 chars is usually sufficient for readability)
        hash_fragment = h.hexdigest()[:8].upper()

        return f"[{entity_type}_{hash_fragment}]"

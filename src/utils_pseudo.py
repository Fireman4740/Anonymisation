import hashlib
import hmac
import string
from typing import Dict, Tuple


class PseudoMapper:
    """Deterministic placeholder mapper based on HMAC(secret + scope_id).

    Produces stable placeholders like [PER_ABC], [ORG_QKZ] per entity type within a scope.
    Same (etype, normalized surface) always maps to the same placeholder for a given (secret, scope_id).
    """

    def __init__(self, secret: str, scope_id: str):
        self.secret = secret.encode("utf-8")
        self.scope_id = scope_id
        self.cache: Dict[Tuple[str, str], str] = {}
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def _code(self, etype: str, key_norm: str, letters: int = 3) -> str:
        # 32 bits du HMAC → base26 lettres
        digest = hmac.new(
            self.secret,
            f"{self.scope_id}:{etype}:{key_norm}".encode("utf-8"),
            hashlib.sha256,
        ).digest()
        n = int.from_bytes(digest[:4], "big")
        chars = []
        for _ in range(letters):
            chars.append(self.alphabet[n % 26])
            n //= 26
        return "".join(chars)

    def placeholder(self, etype: str, surface: str) -> str:
        key_norm = surface.lower().strip()
        cache_key = (etype, key_norm)
        if cache_key in self.cache:
            return self.cache[cache_key]
        suffix = self._code(etype, key_norm, letters=3)
        placeholder = f"[{etype}_{suffix}]"
        self.cache[cache_key] = placeholder
        return placeholder
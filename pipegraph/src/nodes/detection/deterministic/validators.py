import re
from typing import Optional
from .utils import clean_identifier

# Import optionnel des librairies de validation
try:
    from schwifty import IBAN, BIC
    _SCHWIFTY_AVAILABLE = True
except ImportError:
    _SCHWIFTY_AVAILABLE = False
    IBAN = None
    BIC = None

try:
    import phonenumbers
    _PHONENUMBERS_AVAILABLE = True
except ImportError:
    _PHONENUMBERS_AVAILABLE = False

class Validators:
    @staticmethod
    def luhn(value: str) -> bool:
        """Validation algorithme de Luhn (Cartes bancaires, SIRET, etc.)"""
        digits = re.sub(r"\D", "", value)
        if not digits:
            return False
        
        checksum = 0
        reverse_digits = digits[::-1]
        
        for i, digit in enumerate(reverse_digits):
            n = int(digit)
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            checksum += n
            
        return (checksum % 10) == 0

    @staticmethod
    def iban(value: str) -> bool:
        """Valide un IBAN (via schwifty ou check basique)."""
        cleaned = clean_identifier(value)
        if _SCHWIFTY_AVAILABLE:
            try:
                IBAN(cleaned)
                return True
            except ValueError:
                return False
        else:
            # Fallback basique: Longueur et commence par 2 lettres
            return 15 <= len(cleaned) <= 34 and cleaned[0:2].isalpha()

    @staticmethod
    def bic(value: str) -> bool:
        """Valide un code BIC/SWIFT."""
        if _SCHWIFTY_AVAILABLE:
            try:
                BIC(clean_identifier(value))
                return True
            except ValueError:
                return False
        return True # On accepte si pas de lib pour vérifier

    @staticmethod
    def french_ssn(value: str) -> bool:
        """Valide un NIR (Numéro de Sécurité Sociale Français)."""
        cleaned = clean_identifier(value)
        try:
            if len(cleaned) < 15:
                return False
            # Extraction du numéro (13 chiffres) et de la clé (2 chiffres)
            # Parfois le NIR est complet (15), parfois sans clé (13).
            # Ici on suppose qu'on a capturé la clé.
            if len(cleaned) != 15:
                return False
                
            key = int(cleaned[-2:])
            num = int(cleaned[:-2])
            
            # Corse (2A/2B) handling simplifiée (remplacement par 0/1 pour le calcul)
            # Note: Pour une validation stricte, il faudrait gérer 2A->19, 2B->18 etc.
            # Ici on fait simple ou on skip si complexe.
            # Le code legacy faisait:
            # expected_key = 97 - (num % 97)
            
            expected_key = 97 - (num % 97)
            return expected_key == key
        except ValueError:
            # Cas complexes (Corse) ou format invalide
            return False

    @staticmethod
    def phone(value: str) -> bool:
        """Valide un numéro de téléphone via libphonenumbers."""
        if _PHONENUMBERS_AVAILABLE:
            try:
                parsed = phonenumbers.parse(value, None)
                return phonenumbers.is_valid_number(parsed)
            except Exception:
                return False
        return True # Pas de validation si lib absente

    @staticmethod
    def get_validator(name: Optional[str]):
        """Récupère la fonction de validation par son nom."""
        if not name:
            return None
        
        name = name.lower()
        if name == "luhn":
            return Validators.luhn
        elif name == "iban":
            return Validators.iban
        elif name == "bic":
            return Validators.bic
        elif name == "ssn" or name == "nir":
            return Validators.french_ssn
        elif name == "phone" or name == "telephone":
            return Validators.phone
        return None

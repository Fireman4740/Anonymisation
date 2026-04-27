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
    phonenumbers = None

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
                assert IBAN is not None
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
                assert BIC is not None
                BIC(clean_identifier(value))
                return True
            except ValueError:
                return False
        return True # On accepte si pas de lib pour vérifier

    @staticmethod
    def french_ssn(value: str) -> bool:
        """Valide un NIR (Numéro de Sécurité Sociale Français)."""
        cleaned = clean_identifier(value).upper()
        try:
            if len(cleaned) not in (13, 15):
                return False

            if not re.match(r"^[12]\d{2}(0[1-9]|1[0-2])(2A|2B|\d{2})\d{6}(\d{2})?$", cleaned):
                return False

            number_part = cleaned[:13].replace("2A", "19").replace("2B", "18")
            expected_key = 97 - (int(number_part) % 97)

            if len(cleaned) == 13:
                return True

            key = int(cleaned[-2:])
            return expected_key == key
        except ValueError:
            return False

    @staticmethod
    def us_ssn(value: str) -> bool:
        """Validate a US Social Security Number."""
        digits = re.sub(r"\D", "", value or "")
        if len(digits) != 9:
            return False

        area = digits[:3]
        group = digits[3:5]
        serial = digits[5:]
        if area == "000" or area == "666" or int(area) >= 900:
            return False
        if group == "00" or serial == "0000":
            return False
        return True

    @staticmethod
    def phone(value: str) -> bool:
        """Valide un numéro de téléphone.

        Objectif: réduire les faux négatifs (numéros locaux, formats "fictifs" type 555, etc.).

        - Si libphonenumbers est dispo: on tente un parsing robuste et on accepte les numéros
          "possibles" (is_possible_number) plutôt que strictement "valides".
        - Fallback: heuristique simple sur le nombre de chiffres.
        """

        # Heuristique de secours (utile même quand libphonenumbers rejette des numéros fictifs)
        digits = re.sub(r"\D", "", value or "")
        # 7 à 15 chiffres couvre la plupart des formats (locaux + internationaux)
        if not (7 <= len(digits) <= 15):
            return False

        if not _PHONENUMBERS_AVAILABLE or phonenumbers is None:
            return True

        assert phonenumbers is not None

        # Parsing robuste: si pas d'indicatif explicite, on tente avec une région par défaut.
        # On privilégie FR car le dataset/projet est majoritairement FR.
        regions_to_try = ["FR", "US", "GB", "DE", "ES", "IT"]
        has_country_prefix = bool(re.match(r"\s*(?:\+|00)", value))

        try:
            if has_country_prefix:
                parsed = phonenumbers.parse(value, None)
                return bool(phonenumbers.is_possible_number(parsed))

            for region in regions_to_try:
                try:
                    parsed = phonenumbers.parse(value, region)
                    if phonenumbers.is_possible_number(parsed):
                        return True
                except Exception:
                    continue

            # Si tout échoue mais que l'heuristique passe, on accepte (mieux vaut FP léger que FN massif)
            return True
        except Exception:
            return True

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
        elif name == "nir":
            return Validators.french_ssn
        elif name == "ssn":
            return Validators.us_ssn
        elif name == "phone" or name == "telephone":
            return Validators.phone
        return None

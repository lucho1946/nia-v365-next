"""Detector y generador de respuestas de saludo para NIA."""
from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from zoneinfo import ZoneInfo

    _TZ_COLOMBIA = ZoneInfo("America/Bogota")
except Exception:  # pragma: no cover - fallback si no hay tzdata
    _TZ_COLOMBIA = None

CONFIG_PATH = Path(__file__).with_name("greeting_intents.json")
CONFIG: Dict[str, Any] = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
GREETING_RE = re.compile(CONFIG["greeting_prefix_regex"], re.IGNORECASE)
INTENT_CONFIG = {item["name"]: item for item in CONFIG["intents"]}

# Errores ortográficos frecuentes de saludo.
GREETING_TYPO_RE = re.compile(
    r"\b(?:hole|hola+|hoola|hla|hlla|ola)\b",
    re.IGNORECASE,
)
BUENAS_TYPO_RE = re.compile(r"\bbuneas\b|\bbuenass\b|\bbnas\b", re.IGNORECASE)
BUENOS_TYPO_RE = re.compile(r"\bbuenoss\b|\bbunos\b", re.IGNORECASE)

# Cortesía / small talk / segundo saludo horario que NO es solicitud comercial.
SOCIAL_REMAINDER_RE = re.compile(
    r"^(?:"
    r"como\s+estas?|"
    r"como\s+esta|"
    r"como\s+te\s+va|"
    r"como\s+andan?|"
    r"que\s+tal|"
    r"que\s+mas|"
    r"que\s+hubo|"
    r"todo\s+bien|"
    r"muy\s+bien|"
    r"bien\s+o\s+que|"
    r"espero\s+(?:que\s+)?estes?\s+bien|"
    r"un\s+gusto|"
    r"buen(?:o|a|os|as)?\s+(?:dias?|tardes?|noches?)|"
    r"buenas+|buenos+|"
    r"hola+|holi+|hey+|"
    r"nia|amigo|amiga|equipo|senor|senora"
    r")(?:\s+(?:"
    r"nia|amigo|amiga|equipo|senor|senora|"
    r"como\s+estas?|que\s+tal|"
    r"buen(?:o|a|os|as)?\s+(?:dias?|tardes?|noches?)|"
    r"buenas+|buenos+|hola+"
    r"))*"
    r"$"
)

SOCIAL_FILLER_TOKENS = {
    "nia",
    "amigo",
    "amiga",
    "equipo",
    "senor",
    "senora",
    "como",
    "estas",
    "esta",
    "te",
    "va",
    "van",
    "andan",
    "ando",
    "que",
    "tal",
    "mas",
    "hubo",
    "todo",
    "bien",
    "muy",
    "o",
    "espero",
    "estes",
    "hola",
    "holi",
    "hey",
    "buen",
    "buena",
    "buenas",
    "bueno",
    "buenos",
    "dia",
    "dias",
    "tarde",
    "tardes",
    "noche",
    "noches",
}


def normalize(text: str) -> str:
    """Normaliza texto sin perder números útiles para el router comercial."""
    value = unicodedata.normalize("NFKD", (text or "").lower())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^a-z0-9ñ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _normalize_greeting_typos(text: str) -> str:
    """Corrige typos comunes de saludo antes de detectar."""
    value = GREETING_TYPO_RE.sub("hola", text)
    value = BUENAS_TYPO_RE.sub("buenas", value)
    value = BUENOS_TYPO_RE.sub("buenos", value)
    return value


def _strip_stacked_greetings(remainder: str) -> str:
    """Quita saludos apilados del resto ('buenas tardes', 'hola', etc.)."""
    rem = remainder
    for _ in range(4):
        if not rem:
            return ""
        match = GREETING_RE.match(rem)
        if not match:
            break
        rem = rem[match.end() :].strip()
    return rem


def _is_social_remainder(remainder: str) -> bool:
    """True si el resto del mensaje es solo cortesía, no pedido comercial."""
    if not remainder:
        return True

    # "hola buenas tardes" / "hola .. buneas tardes" → resto solo saludo horario.
    stripped = _strip_stacked_greetings(remainder)
    if not stripped:
        return True

    if SOCIAL_REMAINDER_RE.fullmatch(remainder) or SOCIAL_REMAINDER_RE.fullmatch(stripped):
        return True

    tokens = stripped.split()
    if not tokens or len(tokens) > 6:
        return False

    return all(token in SOCIAL_FILLER_TOKENS for token in tokens)


def detect_greeting(text: str) -> Dict[str, Any]:
    """Detecta el saludo y separa la solicitud principal cuando existe."""
    normalized = _normalize_greeting_typos(normalize(text))
    match = GREETING_RE.search(normalized)
    if not match:
        # "como estas?" solo, sin "hola" delante.
        if normalized and _is_social_remainder(normalized):
            intent = "saludo_informal"
            return {
                "matched": True,
                "intent": intent,
                "primary_intent": intent,
                "secondary_intent": None,
                "greeting": normalized,
                "remainder": "",
                "pure_greeting": True,
                "continue_to_main_intent": False,
            }
        return {
            "matched": False,
            "intent": None,
            "primary_intent": None,
            "secondary_intent": None,
            "greeting": None,
            "remainder": normalized,
            "pure_greeting": False,
            "continue_to_main_intent": True,
        }

    greeting = match.group(0)
    remainder = normalized[match.end() :].strip()

    # Saludo puro = sin resto, o solo small talk / segundo saludo horario.
    pure = _is_social_remainder(remainder)

    if not pure:
        intent, primary, secondary = "saludo_con_solicitud", None, "saludo"
    elif re.search(r"cordial|saludos?", greeting) or re.search(
        r"buen.*(dia|tarde|noche)|buenas|buenos", remainder
    ):
        # "cordial saludo" o "hola buenas tardes" → horario/formal.
        if re.search(r"cordial|saludos?", greeting) and not re.search(
            r"buen.*(dia|tarde|noche)|buenas|buenos", f"{greeting} {remainder}"
        ):
            intent, primary, secondary = "saludo_formal", "saludo_formal", None
        else:
            intent, primary, secondary = "saludo_horario", "saludo_horario", None
    elif re.search(r"buen.*(dia|tarde|noche)|buenas|buenos", greeting):
        intent, primary, secondary = "saludo_horario", "saludo_horario", None
    elif re.search(r"holi|hey|que mas|que tal|que hubo|wenas|wuenas", greeting) or remainder:
        intent, primary, secondary = "saludo_informal", "saludo_informal", None
    else:
        intent, primary, secondary = "saludo_general", "saludo_general", None

    intent_data = INTENT_CONFIG[intent]
    return {
        "matched": True,
        "intent": intent,
        "primary_intent": primary,
        "secondary_intent": secondary,
        "greeting": greeting,
        "remainder": remainder,
        "pure_greeting": pure,
        # Si es solo cortesía, nunca continuar al router comercial.
        "continue_to_main_intent": (
            False
            if pure
            else bool(intent_data.get("continue_to_main_intent", True))
        ),
    }


def _stable_index(seed_text: str, total: int) -> int:
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % total


# ─────────────────────────────────────────────────────────────
# Hora real del servidor → variante de saludo_horario coherente
# ─────────────────────────────────────────────────────────────

PERIODO_KEYWORDS = {
    "manana": ("día", "dia"),
    "tarde": ("tarde",),
    "noche": ("noche",),
}


def _periodo_actual() -> str:
    """Franja horaria real (Colombia), no la que escribió el cliente."""
    try:
        hora = datetime.now(_TZ_COLOMBIA).hour if _TZ_COLOMBIA else datetime.now().hour
    except Exception:  # pragma: no cover
        hora = datetime.now().hour

    if 5 <= hora < 12:
        return "manana"
    if 12 <= hora < 19:
        return "tarde"
    return "noche"


def _respuestas_por_periodo(intent: str, periodo: str) -> list[str]:
    todas = INTENT_CONFIG.get(intent, {}).get("responses", [])
    keywords = PERIODO_KEYWORDS.get(periodo, ())
    filtradas = [r for r in todas if any(k in r.lower() for k in keywords)]
    return filtradas or todas


# ─────────────────────────────────────────────────────────────
# Personalización con nombre del cliente
# ─────────────────────────────────────────────────────────────

_PERSONALIZAR_RE = re.compile(r"^([^!.]*[!.])")


def _personalizar(respuesta: str, nombre: Optional[str]) -> str:
    """Inserta el nombre del cliente después del saludo inicial, si se conoce."""
    nombre = (nombre or "").strip()
    if not nombre or not respuesta:
        return respuesta

    match = _PERSONALIZAR_RE.match(respuesta)
    if not match:
        return respuesta

    frase = match.group(1)
    puntuacion = frase[-1]
    inicio = frase[:-1]
    resto = respuesta[match.end():]
    return f"{inicio}, {nombre}{puntuacion}{resto}"


def select_response(
    intent: str,
    *,
    seed_text: Optional[str] = None,
    rng: Optional[random.Random] = None,
    usar_hora_real: bool = False,
    excluir: Optional[str] = None,
) -> str:
    """Selecciona una respuesta configurada para la intención.

    - Con ``seed_text`` la elección es estable, útil para pruebas.
    - Con ``rng`` la elección es aleatoria controlable.
    - Sin argumentos usa ``random.choice``.
    - ``usar_hora_real=True`` cambia "saludo_general" (saludo plano, sin franja
      horaria explícita en el mensaje) por una variante de "saludo_horario"
      que sí coincide con la hora real del servidor.
    - ``excluir`` descarta esa respuesta exacta del pool, para no repetir la
      última frase usada en la misma sesión (si queda más de una opción).
    """
    if intent not in INTENT_CONFIG:
        raise ValueError(f"Intención de saludo no configurada: {intent}")

    if usar_hora_real and intent == "saludo_general":
        responses = _respuestas_por_periodo("saludo_horario", _periodo_actual())
    else:
        responses = INTENT_CONFIG[intent].get("responses", [])

    if not responses:
        raise ValueError(f"La intención {intent} no tiene respuestas configuradas")

    if excluir and len(responses) > 1:
        sin_repetir = [r for r in responses if r != excluir]
        if sin_repetir:
            responses = sin_repetir

    if seed_text is not None:
        return responses[_stable_index(normalize(seed_text), len(responses))]
    if rng is not None:
        return rng.choice(responses)
    return random.choice(responses)


def build_greeting_result(
    text: str,
    *,
    seed_text: Optional[str] = None,
    client_name: Optional[str] = None,
    excluir_respuesta: Optional[str] = None,
    usar_hora_real: bool = True,
) -> Dict[str, Any]:
    """Devuelve detección, respuesta y datos de enrutamiento en un solo objeto.

    ``client_name`` personaliza la respuesta con el nombre del cliente si NIA
    ya lo conoce. ``excluir_respuesta`` evita repetir la última frase usada en
    la sesión. ``usar_hora_real`` hace que un "hola" plano responda acorde a
    la hora real del servidor en vez de una frase neutra.
    """
    result = detect_greeting(text)
    if not result["matched"]:
        return {
            **result,
            "response": None,
            "text_for_router": text,
            "should_respond_now": False,
        }

    response = select_response(
        result["intent"],
        seed_text=seed_text or text,
        usar_hora_real=usar_hora_real,
        excluir=excluir_respuesta,
    )
    response = _personalizar(response, client_name)

    return {
        **result,
        "response": response,
        "text_for_router": result["remainder"] if not result["pure_greeting"] else "",
        "should_respond_now": result["pure_greeting"],
    }

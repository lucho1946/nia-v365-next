"""
Detector de intenciones comerciales con respuestas fijas (escenarios 01-17).

Prioridad: respuestas deterministas antes de catálogo / LLM.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

CONFIG_PATH = Path(__file__).with_name("scripted_intents.json")
CONFIG: Dict[str, Any] = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
INTENTS: List[Dict[str, Any]] = sorted(
    CONFIG.get("intents") or [],
    key=lambda item: int(item.get("priority") or 0),
    reverse=True,
)
CORPORATE: Dict[str, str] = CONFIG.get("corporate") or {}

GUIA_RE = re.compile(
    r"(?:guia|gu[ií]a)\s*(?:n[uú]mero|num(?:ero)?|no\.?|#)?\s*[:\-]?\s*(\d{6,})",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKD", (text or "").lower())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^a-z0-9ñ¿?]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _match_pattern(texto_norm: str, pattern: str, mode: str = "contains") -> bool:
    if mode == "regex":
        return bool(re.search(pattern, texto_norm, re.IGNORECASE))
    pat = normalize(pattern)
    if not pat:
        return False
    return pat in texto_norm


def _pick_response(intent: Dict[str, Any], texto_norm: str, historial: Optional[list]) -> str:
    responses = list(intent.get("responses") or [])
    if not responses:
        return ""

    rule = intent.get("response_rules") or ""

    if rule == "direccion_vs_contacto":
        pide_direccion = any(
            k in texto_norm
            for k in ("direccion", "ubicacion", "oficina", "donde quedan", "donde estan")
        )
        pide_contacto = any(
            k in texto_norm for k in ("nit", "telefono", "correo", "email", "contacto")
        )
        if pide_direccion and not pide_contacto:
            return responses[0]
        if pide_contacto:
            return (
                f"Claro. NIT: {CORPORATE.get('nit')}. "
                f"Teléfono: {CORPORATE.get('telefono')}. "
                f"Correo: {CORPORATE.get('correo')}."
            )
        return responses[1] if len(responses) > 1 else responses[0]

    if rule == "foto_vs_audio":
        if any(k in texto_norm for k in ("audio", "nota de voz", "mensaje de voz")):
            return responses[1] if len(responses) > 1 else responses[0]
        return responses[0]

    if rule == "guia_historial":
        guia = _extraer_guia_historial(historial) or _extraer_guia_texto(texto_norm)
        if guia:
            return f"Claro, la guía registrada es {guia}."
        return responses[0]

    # Compra/orden con cantidad + código si vienen en el mensaje.
    if intent.get("name") in {"compra_proforma", "orden_directa"}:
        return _respuesta_compra_orden(intent["name"], texto_norm, responses)

    if intent.get("name") == "producto_sin_identificador":
        return _respuesta_producto_sin_id(texto_norm, responses)

    if intent.get("name") == "asesor_humano" and "humano" in texto_norm:
        return responses[1] if len(responses) > 1 else responses[0]

    if intent.get("name") == "pago_comprobante" and "consignacion" in texto_norm:
        return responses[1] if len(responses) > 1 else responses[0]

    if intent.get("name") == "factura_electronica" and "contabilidad" in texto_norm:
        return responses[1] if len(responses) > 1 else responses[0]

    if intent.get("name") == "ficha_manual_catalogo" and (
        "manual" in texto_norm or "catalogo" in texto_norm
    ) and "ficha" not in texto_norm:
        return responses[1] if len(responses) > 1 else responses[0]

    if intent.get("name") == "flete_envio" and "medellin" in texto_norm:
        return (
            "Vamos a validar el envío a Medellín. "
            "El valor puede variar según peso, tamaño y destino final."
        )

    return responses[0]


def _respuesta_producto_sin_id(texto_norm: str, responses: List[str]) -> str:
    if "bomba" in texto_norm:
        return "Claro. ¿Tienes referencia, marca o código Viaindustrial de la bomba?"
    if "medidor" in texto_norm and "temperatura" in texto_norm:
        return (
            "Con gusto. ¿Tienes referencia, marca o código Viaindustrial "
            "del medidor de temperatura?"
        )
    if "medidor" in texto_norm:
        return "Con gusto. ¿Tienes referencia, marca o código Viaindustrial del medidor?"
    return responses[0]


def _respuesta_compra_orden(name: str, texto_norm: str, responses: List[str]) -> str:
    codigo = None
    m = re.search(r"\b(p?\d{6})\b", texto_norm, re.IGNORECASE)
    if m:
        codigo = m.group(1).upper()
        if codigo.isdigit():
            codigo = f"P{codigo}" if name == "compra_proforma" else codigo

    cantidad = None
    m_cant = re.search(
        r"\b(\d+)\s*(?:unidades?|und|uds?)\b|\bcomprar\s+(\d+)\b",
        texto_norm,
        re.IGNORECASE,
    )
    if m_cant:
        cantidad = m_cant.group(1) or m_cant.group(2)

    if name == "orden_directa":
        if codigo and cantidad:
            return (
                f"Perfecto, avanzamos con el pedido del {codigo} por {cantidad} unidades. "
                "¿Me confirmas el NIT o RUT?"
            )
        if "cotizacion" in texto_norm or "cotizado" in texto_norm:
            return (
                "Claro, avanzamos con el pedido del equipo cotizado. "
                "En un momento validamos la proforma."
            )
        return responses[0]

    if codigo and cantidad:
        return (
            f"Perfecto, avanzamos con {cantidad} unidades del {codigo}. "
            "¿Me confirmas el NIT o RUT para la proforma?"
        )
    if "proforma" in texto_norm or "confirmo la compra" in texto_norm:
        return responses[0]
    return responses[1] if len(responses) > 1 else responses[0]


def _extraer_guia_texto(texto: str) -> Optional[str]:
    m = GUIA_RE.search(texto or "")
    return m.group(1) if m else None


def _extraer_guia_historial(historial: Optional[list]) -> Optional[str]:
    if not historial:
        return None
    # Busca de más reciente a más antiguo.
    for item in reversed(list(historial)):
        if not isinstance(item, dict):
            continue
        for key in ("content", "mensaje", "text", "respuesta"):
            valor = item.get(key)
            if isinstance(valor, str):
                guia = _extraer_guia_texto(valor)
                if guia:
                    return guia
    return None


def detect_scripted_intent(
    mensaje: str,
    historial: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Detecta una intención con respuesta fija.

    Retorna:
      matched, intent, response, silent, etapa
    """
    texto = (mensaje or "").strip()
    if not texto:
        return {"matched": False}

    texto_norm = normalize(texto)

    for intent in INTENTS:
        mode = intent.get("pattern_mode") or "contains"
        patterns = intent.get("patterns") or []
        if not any(_match_pattern(texto_norm, p, mode) for p in patterns):
            continue

        if intent.get("silent"):
            return {
                "matched": True,
                "intent": intent.get("name"),
                "response": "",
                "silent": True,
                "etapa": "ignorado",
            }

        respuesta = _pick_response(intent, texto_norm, historial)
        etapa = _etapa_para_intent(intent.get("name") or "")
        return {
            "matched": True,
            "intent": intent.get("name"),
            "response": respuesta,
            "silent": False,
            "etapa": etapa,
        }

    return {"matched": False}


def _etapa_para_intent(name: str) -> str:
    mapping = {
        "asesor_humano": "escalamiento",
        "datos_corporativos": "info_corporativa",
        "documentos_legales_bancarios": "documentos",
        "pago_comprobante": "pago",
        "guia_envio": "despacho",
        "factura_electronica": "factura",
        "flete_envio": "flete",
        "ficha_manual_catalogo": "documentacion",
        "compra_proforma": "proforma",
        "orden_directa": "proforma",
        "adjunto_multimedia": "adjunto",
        "agradecimiento": "cierre",
        "tono_consulta_general": "inicio",
        "producto_sin_identificador": "descubrimiento",
    }
    return mapping.get(name, "scripted")

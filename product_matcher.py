"""
product_matcher.py — Validación de compatibilidad producto/necesidad para NIA v365.

Responsabilidad:
- Evaluar si los candidatos reales del catálogo corresponden a la necesidad del cliente.
- Clasificar el mejor candidato como:
  - exact_match: corresponde directamente a lo solicitado.
  - related_match: está relacionado, pero no es exactamente lo solicitado.
  - no_match: ningún candidato es suficientemente compatible.

Regla principal:
La IA NO inventa productos.
La IA SOLO puede elegir entre candidatos reales entregados por MongoDB.
"""

import json
import logging
from typing import Optional

from openai_client import call_llm_json


logger = logging.getLogger("nia.product_matcher")


# Solo el candidato mejor puntuado (ordenar/scoring ocurre antes en buscar_en_catalogo).
MAX_CANDIDATOS = 1


def _compactar_producto(producto: dict, index: int) -> dict:
    """
    Reduce un producto del catálogo a los campos necesarios para evaluación.

    No enviamos todo el documento crudo al LLM para mantener bajo costo,
    evitar ruido y proteger datos innecesarios.
    """
    return {
        "index": index,
        "codigo": producto.get("codigo"),
        "referencia": producto.get("referencia"),
        "nombre": producto.get("nombre"),
        "marca": producto.get("marca"),
        "descripcion_corta": producto.get("descripcion_corta"),
        "descripcion_larga": producto.get("descripcion_larga"),
        "categoria": producto.get("categoria"),
        "existencia": producto.get("existencia"),
        "stock_total": producto.get("stock_total"),
        "score_textual": producto.get("_score"),
    }


def _producto_por_index(candidatos: list[dict], index: int) -> Optional[dict]:
    """
    Retorna el producto original según el índice enviado al LLM.
    """
    if index < 1 or index > len(candidatos):
        return None

    return candidatos[index - 1]


async def validar_compatibilidad_producto(
    necesidad_cliente: str,
    candidatos: Optional[list[dict]],
    contexto_tecnico: Optional[dict] = None,
) -> dict:
    """
    Evalúa compatibilidad entre la necesidad del cliente y candidatos reales.

    Retorna:
    {
        "estado": "exact_match" | "related_match" | "no_match",
        "producto": dict | None,
        "indice": int | None,
        "confianza": float,
        "razon": str,
        "pregunta_sugerida": str | None
    }
    """
    if not candidatos:
        return {
            "estado": "no_match",
            "producto": None,
            "indice": None,
            "confianza": 0.0,
            "razon": "No hay candidatos reales provenientes del catálogo.",
            "pregunta_sugerida": "¿Puedes darme una referencia, marca o aplicación más específica?",
        }

    candidatos_limitados = candidatos[:MAX_CANDIDATOS]
    candidatos_compactos = [
        _compactar_producto(producto, index=i)
        for i, producto in enumerate(candidatos_limitados, start=1)
    ]

    contexto_tecnico = contexto_tecnico or {}

    prompt = f"""
Eres un evaluador técnico-comercial de productos industriales para ViaIndustrial.

Tu tarea es decidir si alguno de los candidatos reales del catálogo corresponde a la necesidad del cliente.

REGLAS ESTRICTAS:
1. No inventes productos.
2. Solo puedes elegir un producto de la lista de candidatos.
3. Si ningún candidato corresponde directamente, responde no_match.
4. Si un candidato está relacionado pero no es exactamente lo solicitado, responde related_match.
5. Si un candidato sí corresponde a lo solicitado, responde exact_match.
6. No recomiendes accesorios, controles, repuestos o componentes como si fueran el equipo principal, salvo que el cliente haya pedido explícitamente ese tipo de producto.
7. Basa tu decisión en nombre, descripción, categoría, referencia, marca y contexto técnico.
8. Responde SOLO JSON válido, sin markdown.

NECESIDAD DEL CLIENTE:
{necesidad_cliente}

CONTEXTO TÉCNICO ACUMULADO:
{json.dumps(contexto_tecnico, ensure_ascii=False, default=str)}

CANDIDATOS REALES DEL CATÁLOGO:
{json.dumps(candidatos_compactos, ensure_ascii=False, default=str)}

FORMATO DE RESPUESTA:
{{
  "estado": "exact_match | related_match | no_match",
  "indice": 1,
  "confianza": 0.0,
  "razon": "explicación corta",
  "pregunta_sugerida": "pregunta breve si aplica"
}}
"""

    try:
        decision = await call_llm_json(prompt)
    except Exception as exc:
        logger.exception("Error validando compatibilidad con LLM: %s", exc)

        return {
            "estado": "no_match",
            "producto": None,
            "indice": None,
            "confianza": 0.0,
            "razon": "No fue posible validar la compatibilidad del producto de forma segura.",
            "pregunta_sugerida": "¿Puedes confirmar si buscas el equipo principal, un accesorio o un repuesto?",
        }

    estado = str(decision.get("estado", "no_match")).strip()
    indice = decision.get("indice")
    confianza = decision.get("confianza", 0.0)
    razon = decision.get("razon", "")
    pregunta_sugerida = decision.get("pregunta_sugerida")

    if estado not in {"exact_match", "related_match", "no_match"}:
        estado = "no_match"

    try:
        indice = int(indice) if indice is not None else None
    except (TypeError, ValueError):
        indice = None

    try:
        confianza = float(confianza)
    except (TypeError, ValueError):
        confianza = 0.0

    producto = None

    if estado in {"exact_match", "related_match"} and indice:
        producto = _producto_por_index(candidatos_limitados, indice)

    if estado == "exact_match" and producto and confianza >= 0.70:
        return {
            "estado": "exact_match",
            "producto": producto,
            "indice": indice,
            "confianza": confianza,
            "razon": razon,
            "pregunta_sugerida": pregunta_sugerida,
        }

    if estado == "related_match" and producto:
        return {
            "estado": "related_match",
            "producto": producto,
            "indice": indice,
            "confianza": confianza,
            "razon": razon,
            "pregunta_sugerida": pregunta_sugerida
            or "Encontré productos relacionados, pero necesito confirmar si buscas el equipo principal o un accesorio.",
        }

    return {
        "estado": "no_match",
        "producto": None,
        "indice": None,
        "confianza": confianza,
        "razon": razon or "Ningún candidato corresponde de forma segura a la necesidad del cliente.",
        "pregunta_sugerida": pregunta_sugerida
        or "¿Puedes confirmar marca, referencia, tipo de producto o aplicación exacta?",
    }
"""
main.py — Endpoint principal NIA v365

Responsabilidades:
- Exponer endpoints FastAPI para chat y archivos.
- Mantener sesión conversacional en MongoDB.
- Capturar datos básicos del cliente.
- Evaluar necesidad técnica.
- Transformar mensaje natural en query limpia de catálogo.
- Buscar catálogo real.
- Validar compatibilidad producto/necesidad.
- Construir respuestas seguras cuando hay datos de catálogo.
- Evitar que el LLM invente códigos, marcas, nombres o descripciones.

Regla de arquitectura:
El LLM conversa, pero el backend decide y construye respuestas críticas
cuando hay productos reales del catálogo.
"""

import hashlib
import hmac
import logging
import logging.handlers
import os
import re
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from pydantic import BaseModel

from memory import (
    get_session,
    save_session,
    ensure_index,
    get_cliente,
    upsert_cliente,
)
from learning_memory import (
    activar_memoria_aprendizaje,
    construir_contexto_aprendizaje_desde_necesidad,
    construir_evento_feedback,
    desactivar_memoria_aprendizaje,
    filtrar_productos_por_aprendizaje,
    obtener_memoria_aprendizaje,
    registrar_feedback,
    resolver_clave_aprendizaje,
)
from discovery_guards import (
    construir_texto_limpio_descubrimiento,
    es_producto_epi_seguridad,
    es_respuesta_desconocida,
    filtrar_terminos_libros,
    preguntas_epi_con_opciones,
    preguntas_refino_epi,
    queries_alternativas_epi,
    respuestas_utiles,
    texto_ancla_desde_ctx,
)
from openai_client import call_nia, call_llm_json
from greeting_detector import (
    _personalizar,
    build_greeting_result,
    detect_greeting,
    select_response,
)
from scripted_intents import detect_scripted_intent
from nia_prompt import PROMPT_MAESTRO
from catalog import (
    buscar_por_codigo,
    buscar_por_referencia,
    buscar_por_texto,
    buscar_con_campos,
    buscar_con_descubrimiento_producto,
    buscar_productos_por_nivel_1,
    evaluar_coincidencia,
    extraer_campos_tecnicos,
    formatear_producto,
    normalizar_referencia,
)
from product_matcher import validar_compatibilidad_producto
from file_processor import procesar_archivo, extraer_datos_rut_pdf
from knowledge import contexto_para_agente
from questions_agent import generar_preguntas, detectar_escenario
from product_fields import detectar_categoria, KEYWORDS_TO_CATEGORIA
from hybrid_discovery import (
    MAX_PREGUNTAS_HIBRIDAS,
    buscar_pool_por_dominio,
    cargar_productos_por_codigos,
    construir_query_acumulado as construir_query_hibrido,
    es_necesidad_hibrida_guiada,
    filtrar_productos_por_dominio,
    filtrar_productos_por_respuesta,
    filtrar_productos_por_tecnologia_material,
    filtrar_tipos_nivel_1_por_dominio,
    filtrar_tipos_nivel_1_por_tecnologia,
    generar_pregunta_aplicacion,
    generar_siguiente_pregunta_hibrida,
    obtener_pool_inicial,
    resolver_respuesta_hibrida,
    seleccionar_producto_final,
    afinar_nivel_1_para_contexto,
    obtener_tipos_radar_nivel,
    producto_inadecuado_para_contexto,
    mensaje_asesoria_tecnica_nivel,
    resolver_previas_hibridas,
    DOMINIO_TERMINOS_BUSQUEDA,
    _inferir_clave_material,
    _contexto_material_nivel,
)
from product_discovery import (
    extraer_palabra_clave,
    obtener_tipos_nivel_1,
    obtener_tipos_nivel_1_por_texto,
    generar_pregunta_seleccion_tipo,
    resolver_tipos_catalogo_inicio,
    generar_pregunta_seleccion_otro,
    generar_preguntas_tecnicas_por_nivel_1,
    resolver_seleccion_tipo,
    construir_query_busqueda_final,
    _texto_pregunta,
    _opciones_pregunta,
)
from response_engine import (
    respuesta_producto_encontrado,
    respuesta_producto_relacionado,
    respuesta_sin_resultado,
    contiene_placeholder,
)


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

def setup_logging():
    """
    Configura logging para consola y archivo rotativo.

    Evita duplicar handlers cuando Uvicorn recarga con --reload.
    """
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)

    root = logging.getLogger("nia")
    root.setLevel(level)

    if root.handlers:
        return

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    fh = logging.handlers.RotatingFileHandler(
        "nia.log",
        maxBytes=10_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)

    root.addHandler(ch)
    root.addHandler(fh)


setup_logging()
logger = logging.getLogger("nia.main")


# ─────────────────────────────────────────────────────────────
# FastAPI / Rate limiting
# ─────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="NIA — Asistente Comercial ViaIndustrial")
app.state.limiter = limiter

app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    await ensure_index()
    logger.info("NIA arrancó correctamente")


# ─────────────────────────────────────────────────────────────
# Modelos API
# ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    mensaje: str
    phone_id: Optional[str] = None


class ChatResponse(BaseModel):
    respuesta: str
    etapa: Optional[str] = None
    opciones: Optional[list] = None
    items_resultado: Optional[list] = None
    cliente: Optional[dict] = None


# ─────────────────────────────────────────────────────────────
# WhatsApp webhook auth
# ─────────────────────────────────────────────────────────────

WA_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "")
WA_APP_SECRET = os.getenv("WHATSAPP_APP_SECRET", "")


def verificar_firma_whatsapp(request: Request, body: bytes) -> bool:
    """
    Valida la firma HMAC-SHA256 que Meta envía en X-Hub-Signature-256.

    En desarrollo, si no hay WA_APP_SECRET configurado, permite pasar.
    """
    if not WA_APP_SECRET:
        return True

    sig_header = request.headers.get("X-Hub-Signature-256", "")
    if not sig_header.startswith("sha256="):
        return False

    firma_esperada = hmac.new(
        WA_APP_SECRET.encode(),
        body,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(sig_header[7:], firma_esperada)


@app.get("/webhook/whatsapp")
async def whatsapp_verify(
    hub_mode: str = None,
    hub_challenge: str = None,
    hub_verify_token: str = None,
):
    """
    Verificación del webhook de WhatsApp Business.
    """
    if hub_mode == "subscribe" and hub_verify_token == WA_VERIFY_TOKEN:
        logger.info("Webhook WhatsApp verificado correctamente")
        return int(hub_challenge)

    raise HTTPException(status_code=403, detail="Token inválido")


# ─────────────────────────────────────────────────────────────
# Regex / intención básica
# ─────────────────────────────────────────────────────────────

# Código interno numérico de seis dígitos.
#
# Los lookarounds evitan extraer seis dígitos desde el interior
# de un NIT, teléfono u otro número más largo.
CODIGO_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")


# Referencias comerciales con prefijo P.
#
# Formas válidas:
# - P350279
# - p350279
# - P-350279
# - P 350279
# - P#350279
# - P:350279
REFERENCIA_P_RE = re.compile(
    r"(?<![A-Z0-9])P[\s#:\-._/]*([0-9]{3,})(?![A-Z0-9])",
    re.IGNORECASE,
)


# Referencia alfanumérica compacta.
# No acepta espacios entre letras y números. Esto evita interpretar
# frases naturales como "mi NIT es 900123456" como una referencia ES900123456.
REFERENCIA_ALFANUMERICA_COMPACTA_RE = re.compile(
    r"(?<![A-Z0-9])([A-Z]{1,4})[-._/]*([0-9]{3,}[A-Z0-9]*)(?![A-Z0-9])",
    re.IGNORECASE,
)


# Referencia alfanumérica separada, pero únicamente cuando el cliente
# utiliza una palabra que declara explícitamente el identificador.

REFERENCIA_ALFANUMERICA_EXPLICITA_RE = re.compile(
    # Palabra que declara que lo siguiente es un identificador.
    # "referencia" debe ir antes que "ref" para no capturar "erencia".
    r"\b(?:referencia|ref|modelo|c[oó]digo|parte|part\s*number|pn|p/n)\b"

    # Conectores naturales opcionales usados por los clientes:

    r"(?:\s+(?:es|n[uú]mero|num(?:ero)?|no\.?))?"

    # Separadores opcionales.
    r"\s*[:#\-]?\s*"

    # Prefijo alfabético + cuerpo numérico o alfanumérico.
    r"([A-Z]{1,4})[\s#:\-._/]*([0-9]{3,}[A-Z0-9]*)(?![A-Z0-9])",
    re.IGNORECASE,
)

# Referencia declarada: captura el valor completo tras "referencia", "ref", etc.
REFERENCIA_DECLARADA_RE = re.compile(
    r"\b(?:referencia|ref|modelo|parte|part\s*number|pn|p/n)\b"
    r"(?:\s+(?:es|n[uú]mero|num(?:ero)?|no\.?))?"
    r"\s*[:#\-]?\s*"
    r"(.+)$",
    re.IGNORECASE,
)

# Referencia segmentada: PT-7320-S4010-01, 68072-XC3-48RT-C, etc.
REFERENCIA_SEGMENTADA_RE = re.compile(
    r"(?<![A-Z0-9])([A-Z0-9]+(?:[-._/][A-Z0-9]+)+)(?![A-Z0-9])",
    re.IGNORECASE,
)

# Referencia con espacios internos: 68072 XC3 48RT C
REFERENCIA_ESPACIOS_RE = re.compile(
    r"^(?=.*[A-Z])(?=.*\d)[A-Z0-9][A-Z0-9\s\-./]{4,}$",
    re.IGNORECASE,
)

EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
)

NIT_RE = re.compile(r"\b(\d{8,10}-?\d?)\b")

# Cantidad comercial pura: "2", "2 und", "2 unidades,", "10 u."
# No debe interpretarse como referencia de catálogo.
CANTIDAD_MENSAJE_SOLO_RE = re.compile(
    r"^\s*(\d{1,5})\s*"
    r"(?:und|unds|uds?|unidad(?:es)?|pieza(?:s)?|u\.?)?"
    r"\s*[,.]?\s*$",
    re.IGNORECASE,
)

# ============================================================
# CLASIFICADOR DE INTENCIÓN — COTIZACIÓN V
# ============================================================

TIPOS_MENSAJE_VALIDOS = {
    "buscar_producto",
    "instruccion_comercial",
    "dato_personal",
    "pregunta_estado",
    "cotizacion_recibida",
    "proforma_recibida",
    "link_documento",
    "saludo",
    "otro",
}

PROMPT_CLASIFICADOR_MENSAJE = """
Eres el clasificador de intención de NIA, asistente comercial técnico de ViaIndustrial.

Tu tarea es clasificar el mensaje del cliente en UNA sola categoría.

Categorías válidas:

1. "buscar_producto"
El cliente describe un producto, da código, referencia, marca, aplicación o características técnicas.
2. "instruccion_comercial"
El cliente responde al flujo comercial: confirma, niega, da cantidad, dice que cotizamos, pide continuar o cerrar.
3. "dato_personal"
El cliente entrega datos personales o comerciales: nombre, correo, empresa, NIT, RUT, teléfono.
4. "pregunta_estado"
El cliente pregunta por el estado de pedido, cotización, entrega, despacho o disponibilidad.
5. "cotizacion_recibida"
El cliente indica que ya recibió o ya tiene la cotización.
6. "proforma_recibida"
El cliente indica que ya recibió o ya tiene la proforma.
7. "link_documento"
El cliente envía un link que puede corresponder a cotización, proforma, archivo o documento.
8. "saludo"
El cliente solo saluda.
9. "otro"
No encaja claramente en las anteriores.
Responde SOLO JSON válido, sin markdown:
{
  "tipo": "categoria",
  "confianza": 0.0,
  "razon": "frase corta"
}
"""
RESPUESTAS_CORTAS_COMERCIALES = {
    "sí",
    "si",
    "no",
    "ok",
    "dale",
    "listo",
    "perfecto",
    "correcto",
    "claro",
    "bueno",
    "exacto",
    "así es",
    "asi es",
    "de acuerdo",
    "negativo",
    "está bien",
    "esta bien",
    "okay",
    "okey",
    "va",
    "por supuesto",
    "con gusto",
    "adelante",
    "procede",
    "proceder",
    "confirmado",
    "confirmo",
    "entendido",
    "recibido",
    "enterado",
    "solo esto",
    "solo eso",
    "con esto",
    "con eso",
    "eso es todo",
    "nada mas",
    "nada más",
    "es todo",
    "eso es",
    "solo",
    "eso",
}

async def clasificar_mensaje(mensaje: str, etapa: str) -> dict:
    """
    Clasifica el mensaje del cliente usando reglas rápidas + GPT.

    Diseño:
    - Primero usa reglas determinísticas para casos obvios.
    - Usa GPT solo cuando el mensaje es ambiguo.
    - Nunca deja que GPT salte guardrails comerciales.
    """
    texto = (mensaje or "").strip()
    msg_lower = texto.lower()

    if not texto:
        return {
            "tipo": "otro",
            "confianza": 0.0,
            "razon": "mensaje vacío",
        }

    # ------------------------------------------------------------
    # 1. Reglas rápidas sin GPT
    # ------------------------------------------------------------

    if re.search(r"https?://|drive\.google|dropbox|onedrive", msg_lower, re.IGNORECASE):
        return {
            "tipo": "link_documento",
            "confianza": 1.0,
            "razon": "contiene link de documento",
        }

    if any(frase in msg_lower for frase in [
        "ya tengo la cotizacion",
        "ya tengo la cotización",
        "me llego la cotizacion",
        "me llegó la cotización",
        "ya recibi la cotizacion",
        "ya recibí la cotización",
        "ya me cotizaron",
        "me enviaron la cotizacion",
        "me enviaron la cotización",
    ]):
        return {
            "tipo": "cotizacion_recibida",
            "confianza": 1.0,
            "razon": "cliente indica cotización recibida",
        }

    if any(frase in msg_lower for frase in [
        "ya tengo la proforma",
        "me llego la proforma",
        "me llegó la proforma",
        "ya recibi la proforma",
        "ya recibí la proforma",
        "me enviaron la proforma",
    ]):
        return {
            "tipo": "proforma_recibida",
            "confianza": 1.0,
            "razon": "cliente indica proforma recibida",
        }

    if re.fullmatch(r"\d{1,5}", msg_lower):
        return {
            "tipo": "instruccion_comercial",
            "confianza": 1.0,
            "razon": "cantidad numérica",
        }

    if msg_lower.strip() in RESPUESTAS_CORTAS_COMERCIALES:
        return {
            "tipo": "instruccion_comercial",
            "confianza": 1.0,
            "razon": "respuesta corta de flujo",
        }

    if any(frase in msg_lower for frase in [
        "solo eso",
        "con eso",
        "es todo",
        "nada mas",
        "nada más",
        "cotiza",
        "coticemos",
    ]):
        return {
            "tipo": "instruccion_comercial",
            "confianza": 1.0,
            "razon": "cierre de cotización",
        }

    if re.search(r"[\w\.-]+@[\w\.-]+\.\w+", texto):
        return {
            "tipo": "dato_personal",
            "confianza": 1.0,
            "razon": "contiene correo electrónico",
        }

    greeting = detect_greeting(texto)
    if greeting.get("matched") and greeting.get("pure_greeting"):
        return {
            "tipo": "saludo",
            "confianza": 1.0,
            "razon": f"saludo:{greeting.get('intent') or 'simple'}",
        }

    # ------------------------------------------------------------
    # 2. Clasificación GPT para casos ambiguos
    # ------------------------------------------------------------
    prompt = f"""{PROMPT_CLASIFICADOR_MENSAJE}

Etapa actual de la conversación: {etapa}
Mensaje del cliente: "{texto}"

Clasifica el mensaje.
"""

    try:
        resultado = await call_llm_json(prompt)

        tipo = str(resultado.get("tipo", "otro")).strip()
        confianza = float(resultado.get("confianza", 0.0) or 0.0)
        razon = str(resultado.get("razon", "")).strip()

        if tipo not in TIPOS_MENSAJE_VALIDOS:
            tipo = "otro"
            confianza = 0.0
            razon = "tipo inválido devuelto por clasificador"

        return {
            "tipo": tipo,
            "confianza": confianza,
            "razon": razon,
        }

    except Exception as e:
        logger.warning("clasificar_mensaje falló: %s", e)
        return {
            "tipo": "otro",
            "confianza": 0.0,
            "razon": "fallback por error del clasificador",
        }

PALABRAS_SALUDO = {
    "hola",
    "buenas",
    "buenos",
    "buen",
    "hi",
    "hello",
    "hey",
    "saludos",
    "que tal",
    "qué tal",
    "quiubo",
    "quihubo",
    "que mas",
    "qué más",
    "👋",
}
PALABRAS_MAS = {"también", "otro", "otra", "más", "adicional", "y además", "necesito más", "y también"}
PALABRAS_FIN = {"solo eso", "con eso", "es todo", "nada más", "eso es todo", "listo", "ok cotiza", "cotiza"}

MAX_PRODUCTOS_CARRITO = 100
MINUTOS_INACTIVIDAD_CARRITO = 30

# Primera pregunta abierta cuando la necesidad aún no identifica el equipo.
PREGUNTA_INICIAL_NECESIDAD = (
    "Para identificar el producto correcto, cuéntame qué equipo o qué necesidad estás buscando."
)


def _opciones_cierre_carrito() -> list[dict]:
    return [
        {"id": "1", "label": "Agregar otro producto", "valor": "agregar_otro"},
        {"id": "2", "label": "Cotizar con esto", "valor": "cotizar"},
    ]


def _opciones_confirmacion_si_no() -> list[dict]:
    return [
        {"id": "1", "label": "Sí", "valor": "sí"},
        {"id": "2", "label": "No", "valor": "no"},
    ]


def _ctx_confirmacion_producto() -> dict:
    return {"opciones_actuales": _opciones_confirmacion_si_no()}


PREGUNTA_COTIZACION_TECNICA = (
    "¿La cotización cumple con lo que necesitas técnicamente?"
)


def _ctx_confirmacion_cotizacion_tecnica() -> dict:
    return {"opciones_actuales": _opciones_confirmacion_si_no()}


def _respuesta_confirmar_cotizacion_tecnica(
    prefijo: str = "Perfecto, tomo esto como cotización recibida. ",
) -> tuple[str, str, dict]:
    """Pregunta de validación técnica + botones Sí/No."""
    texto = f"{prefijo}{PREGUNTA_COTIZACION_TECNICA}".strip()
    return texto, "cotizacion_enviada", _ctx_confirmacion_cotizacion_tecnica()


def _minutos_desde_ultimo_turno(historial: list) -> float:
    if not historial:
        return 0.0

    for turno in reversed(historial):
        ts = turno.get("ts")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", ""))
            delta = datetime.utcnow() - dt
            return max(0.0, delta.total_seconds() / 60.0)
        except ValueError:
            continue

    return 0.0


def _es_pedido_cotizar(texto: str) -> bool:
    t = _normalizar_intencion(texto)
    frases_cotizar = {
        "cotizar",
        "cotizar con esto",
        "cotizamos",
        "cotiza",
        "con esto",
        "solo eso",
        "es todo",
        "nada mas",
        "nada más",
        "listo",
        "ok cotiza",
        "no necesito mas",
        "no necesito más",
        "no mas",
        "no más",
        "proceder a cotizar",
    }
    if t in frases_cotizar or t in PALABRAS_FIN:
        return True
    return any(frase in t for frase in frases_cotizar)


def _es_pedido_agregar_producto(texto: str) -> bool:
    if _es_pedido_cotizar(texto):
        return False

    t = _normalizar_intencion(texto)
    agregar = {
        "si",
        "dale",
        "ok",
        "agregar otro producto",
        "agregar",
        "otro producto",
        "algo mas",
        "algo más",
        "necesito mas",
        "necesito más",
        "necesito otro",
        "necesito otro producto",
        "otro equipo",
        "otra referencia",
        "continuar",
        "agregar_otro",
    }
    if t in agregar:
        return True

    if any(
        frase in t
        for frase in {
            "necesito otro",
            "quiero otro",
            "busco otro",
            "otro producto",
            "otra referencia",
            "agregar otro",
            "agrega otro",
        }
    ):
        return True

    if any(p in t for p in PALABRAS_MAS):
        return True

    return _es_nueva_solicitud_durante_cierre(texto)


def _es_salida_confirmacion_producto(texto: str) -> bool:
    """
    True cuando, tras mostrar un producto, el cliente quiere salir del sí/no
    para buscar otra cosa (sin decir explícitamente 'no').
    """
    if _es_pedido_de_identificador_sin_valor(texto):
        return True

    t = _normalizar_intencion(texto)
    frases = {
        "necesito otro",
        "necesito otro producto",
        "otro producto",
        "otra cosa",
        "buscar otro",
        "busco otro",
        "quiero otro",
        "nueva busqueda",
        "otra referencia",
        "cambiar de producto",
        "otro equipo",
        "algo mas",
        "necesito mas",
    }
    return any(f in t for f in frases)


def _mensaje_resumen_carrito(productos_acumulados: list) -> str:
    total = len(productos_acumulados or [])
    if total == 0:
        return ""
    if total == 1:
        return "Llevas 1 producto en tu solicitud."
    return f"Llevas {total} productos en tu solicitud."


def _es_mensaje_esencialmente_identificador(texto: str, identificador: str) -> bool:
    """
    True solo si el mensaje es básicamente el código/referencia, no un nombre de producto.
    Ej: 'P245366' sí; 'ProfiPack C400' no.
    """
    msg = _normalizar_match_textual(texto)
    ident = _normalizar_match_textual(identificador)
    if not msg or not ident:
        return False

    if msg == ident or msg.replace(" ", "") == ident.replace(" ", ""):
        return True

    tokens = [t for t in msg.split() if len(t) >= 2]
    if len(tokens) <= 1:
        return ident in msg

    extras = [
        t
        for t in tokens
        if ident not in t and t not in ident and not t.startswith(ident)
    ]
    return len(extras) == 0


def _parece_nombre_producto_modelo(texto: str) -> bool:
    """Nombre comercial con modelo, ej. ProfiPack C400, HSM C400."""
    t = (texto or "").strip()
    if len(t.split()) < 2:
        return False
    return bool(re.search(r"[a-zA-Z]", t) and re.search(r"\d", t))


def _parece_mensaje_solo_cantidad(texto: str) -> bool:
    """True si el mensaje completo es solo una cantidad comercial."""
    if not texto:
        return False
    return bool(CANTIDAD_MENSAJE_SOLO_RE.match(str(texto).strip()))


def detectar_identificador(texto: str):
    """
    Detecta un identificador explícito dentro de un mensaje natural.

    Prioridad:
    1. Referencia comercial con prefijo P.
    2. Código numérico exacto de seis dígitos.
    3. Referencia alfanumérica general.
    """
    if not texto:
        return None, None

    texto = str(texto).strip()

    if not texto:
        return None, None

    # "2 und" / "2 unidades" es cantidad, no referencia (p.ej. REFERENCIA_ESPACIOS_RE).
    if _parece_mensaje_solo_cantidad(texto):
        return None, None

    # Un correo nunca es código/referencia. "afvr1975@gmail.com" no debe
    # matchear como referencia segmentada "GMAIL.COM".
    if EMAIL_RE.fullmatch(texto):
        return None, None

    # Buscamos identificadores ignorando correos embebidos en el mensaje.
    texto_sin_email = EMAIL_RE.sub(" ", texto).strip()
    if not texto_sin_email:
        return None, None

    def _ref_limpia(valor: str) -> str:
        """Trim + mayúsculas; conserva separadores (variantes mecánicas en catálogo)."""
        return str(valor or "").strip().upper()

    def _parece_dominio_o_email(valor: str) -> bool:
        v = (valor or "").strip().lower()
        if not v or "@" in v:
            return True
        if re.search(
            r"\.(com|net|org|co|es|io|gov|edu|info|mx|ar|cl|pe|ec)(?:\b|$)",
            v,
        ):
            return True
        return False

    # ------------------------------------------------------------
    # 0. Referencia declarada explícitamente (valor completo)
    # ------------------------------------------------------------
    match_decl = REFERENCIA_DECLARADA_RE.search(texto_sin_email)
    if match_decl:
        referencia = match_decl.group(1).strip().strip(".,;:")
        # Evita capturar frases vacías o pedidos sin valor real.
        if (
            len(referencia) >= 3
            and not _parece_dominio_o_email(referencia)
            and not re.fullmatch(
                r"(?:del?\s+)?(?:producto|catalogo|exact[oa]|adicional|por\s+favor)?",
                referencia,
                re.IGNORECASE,
            )
            and re.search(r"[A-Z0-9]", referencia, re.IGNORECASE)
        ):
            referencia = _ref_limpia(referencia)
            logger.debug("Referencia declarada detectada: %s", referencia)
            # Guarda forma canónica compacta en log para diagnóstico.
            logger.debug(
                "Referencia normalizada: %s",
                normalizar_referencia(referencia),
            )
            return "referencia", referencia

    # ------------------------------------------------------------
    # 1. Referencia con prefijo P
    # ------------------------------------------------------------
    # Se evalúa antes del código numérico porque P350279 contiene
    # seis dígitos, pero comercialmente es una referencia completa.
    match_p = REFERENCIA_P_RE.search(texto_sin_email)

    if match_p:
        numero = match_p.group(1)
        referencia = f"P{numero}".upper()

        logger.debug(
            "Referencia P detectada en mensaje natural: %s",
            referencia,
        )

        return "referencia", referencia

    # ------------------------------------------------------------
    # 2. Código exacto de seis dígitos
    # ------------------------------------------------------------
    match_codigo = CODIGO_RE.search(texto_sin_email)

    if match_codigo:
        codigo = match_codigo.group(1)

        logger.debug(
            "Código numérico detectado en mensaje natural: %s",
            codigo,
        )

        return "codigo", codigo

    # ------------------------------------------------------------
    # 2b. Referencia segmentada o con espacios internos
    # ------------------------------------------------------------
    match_seg = REFERENCIA_SEGMENTADA_RE.search(texto_sin_email)
    if match_seg:
        referencia = _ref_limpia(match_seg.group(1))
        if not _parece_dominio_o_email(referencia):
            logger.debug("Referencia segmentada detectada: %s", referencia)
            return "referencia", referencia

    if REFERENCIA_ESPACIOS_RE.match(texto_sin_email.strip()):
        referencia = _ref_limpia(texto_sin_email)
        if not _parece_dominio_o_email(referencia):
            logger.debug("Referencia con espacios detectada: %s", referencia)
            return "referencia", referencia

    # ------------------------------------------------------------
    # 3. Referencia alfanumérica general
    # ------------------------------------------------------------
    # Primero buscamos referencias declaradas explícitamente:
    match_ref = REFERENCIA_ALFANUMERICA_EXPLICITA_RE.search(texto_sin_email)

    # Si no existe una palabra declarativa, solo aceptamos formatos compactos
    # porque normalmente pertenece a una frase natural.
    if not match_ref:
        match_ref = REFERENCIA_ALFANUMERICA_COMPACTA_RE.search(texto_sin_email)

    if match_ref:
        prefijo = match_ref.group(1).upper()
        cuerpo = match_ref.group(2).upper()

        # Prefijos administrativos que no representan productos.
        prefijos_bloqueados = {
            "NIT",
            "RUT",
            "TEL",
            "CEL",
            "CC",
            "ID",
        }

        if prefijo not in prefijos_bloqueados:
            referencia = f"{prefijo}{cuerpo}"
            if not _parece_dominio_o_email(referencia):
                logger.debug(
                    "Referencia alfanumérica detectada: %s",
                    referencia,
                )
                return "referencia", referencia

    return None, None


def _extraer_keyword_instrumento(texto: str) -> Optional[str]:
    """
    Detecta si el cliente nombró un instrumento o herramienta del catálogo.
    """
    texto_lower = (texto or "").lower()
    for kw in sorted(KEYWORDS_TO_CATEGORIA.keys(), key=len, reverse=True):
        if kw in texto_lower:
            return kw
    return None


def _debe_preguntar_antes_de_buscar(texto: str) -> bool:
    """
    Si el cliente nombró un instrumento sin especificaciones técnicas,
    primero se hacen preguntas (una por turno) y después se busca en catálogo.
    """
    if _es_busqueda_hibrida_directa(texto):
        return False

    if _cuenta_parametros_tecnicos_generales(texto) >= 2:
        return False

    escenario = detectar_escenario(texto)
    return escenario in {"escenario_2_nombre", "escenario_3_necesidad"}


def _tiene_campos_tecnicos_mensaje(texto: str) -> bool:
    campos = extraer_campos_tecnicos(texto)
    if campos:
        return True
    return _cuenta_parametros_tecnicos_generales(texto) >= 2


def _tiene_ancla_producto(texto: str) -> bool:
    return bool(
        _extraer_keyword_instrumento(texto)
        or detectar_categoria(texto) != "default"
        or _parece_solicitud_de_producto(texto)
    )


def _es_busqueda_hibrida_directa(texto: str) -> bool:
    """
    Modo 3 directo: producto + campos técnicos ya en el mensaje.
    """
    if not (texto or "").strip():
        return False

    if not _tiene_campos_tecnicos_mensaje(texto):
        return False

    if _tiene_ancla_producto(texto):
        return True

    return len(_tokens_producto_cliente(texto)) >= 2


def _es_busqueda_hibrida(texto: str) -> bool:
    """Alias: búsqueda híbrida directa (compatibilidad)."""
    return _es_busqueda_hibrida_directa(texto)


def _es_pedido_de_identificador_sin_valor(texto: str) -> bool:
    """
    True cuando el cliente quiere buscar por código/referencia de catálogo
    pero todavía no escribió el valor (6 dígitos o alfanumérico).

    Evita confundir "necesito un código" con productos tipo
    "lectores de código de barras".
    """
    if not (texto or "").strip():
        return False

    if detectar_identificador(texto)[0]:
        return False

    t = _normalizar_match_textual(texto)

    # Productos físicos / lectores: no es pedido de identificador.
    if re.search(
        r"codigo\s+de\s+barras|lector(?:es)?|escaner|scanner|pistola\s+de\s+codigo",
        t,
        re.IGNORECASE,
    ):
        return False

    patrones = [
        r"\b(?:necesito|tengo|busco|quiero|requiero|pasame|pase|enviame|dame|"
        r"te\s+paso|voy\s+a\s+(?:pasar|enviar|dar))\s+"
        r"(?:por\s+)?(?:una?\s+|el\s+|la\s+|mi\s+|con\s+|est[ae]\s+|es[ae]\s+)?"
        r"(?:codigo|referencia|ref)\b",
        r"\b(?:buscar|busqueda|consultar|consulta)\s+por\s+(?:codigo|referencia|ref)\b",
        r"\b(?:por\s+)?(?:codigo|referencia)\s+(?:de\s+)?(?:producto|interno)?\b",
        r"^(?:una?\s+|el\s+|la\s+|est[ae]\s+|es[ae]\s+)?(?:codigo|referencia|ref)$",
        r"\bnecesito\s+(?:un\s+|una\s+|est[ae]\s+|es[ae]\s+)?(?:codigo|referencia)\b",
        r"\btengo\s+(?:el\s+|la\s+)?(?:codigo|referencia)\b",
        r"\bdame\s+(?:el\s+|la\s+)?(?:codigo|referencia|ref)\b",
    ]
    return any(re.search(patron, t, re.IGNORECASE) for patron in patrones)


def _respuesta_pedir_identificador() -> str:
    return (
        "Perfecto. Dame la referencia y, si tienes la marca, dámela también "
        "para localizar el producto exacto en el catálogo."
    )


def _es_pedido_solo_referencia(texto: str) -> bool:
    """True si el cliente pide referencia (no código) sin haber enviado el valor."""
    t = _normalizar_match_textual(texto or "")
    if not t:
        return False
    if re.search(r"\bcodigo\b", t) and not re.search(r"\b(?:referencia|ref)\b", t):
        return False
    return bool(re.search(r"\b(?:referencia|ref)\b", t))


def _respuesta_pedir_referencia() -> str:
    return (
        "Perfecto. Dame la referencia y, si tienes la marca, dámela también "
        "para localizar el producto exacto en el catálogo."
    )


def _extraer_marca_de_respuesta(mensaje: str) -> str:
    """Extrae la marca cuando el cliente responde 'marca autonics' o solo 'autonics'."""
    texto = (mensaje or "").strip()
    match = re.search(
        r"(?:marca|brand)\s*[:\-]?\s*(.+)$",
        texto,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return texto


def _extraer_marca_junto_a_referencia(
    texto: Optional[str],
    referencia: str,
) -> Optional[str]:
    """
    Si el cliente escribe 'PV-70 via' o 'referencia PV-70 marca via',
    recupera la marca que viene en el mismo mensaje.
    """
    if not texto or not referencia:
        return None

    resto = str(texto)
    variantes = {
        referencia,
        referencia.replace("-", ""),
        referencia.replace(" ", ""),
        referencia.replace("-", " "),
    }
    for v in variantes:
        if not v:
            continue
        resto = re.sub(re.escape(v), " ", resto, flags=re.IGNORECASE)

    resto = re.sub(
        r"\b(?:referencia|ref|modelo|c[oó]digo|parte|marca|brand)\b",
        " ",
        resto,
        flags=re.IGNORECASE,
    )
    resto = re.sub(r"\s+", " ", resto).strip(" .,;:#-")

    if len(resto) < 2 or resto.isdigit():
        return None
    return resto


def _marcas_desde_candidatos(candidatos: list) -> list[str]:
    return sorted(
        {
            str(c.get("marca") or "").strip()
            for c in (candidatos or [])
            if str(c.get("marca") or "").strip()
        },
        key=lambda m: m.lower(),
    )


def _opciones_marcas_referencia(candidatos: list) -> list[dict]:
    """Botones de marca detectadas + siempre 'Otro'."""
    opciones = []
    for idx, marca in enumerate(_marcas_desde_candidatos(candidatos)[:8], start=1):
        opciones.append(
            {
                "id": str(idx),
                "label": marca,
                "valor": marca,
            }
        )
    opciones.append(
        {
            "id": str(len(opciones) + 1),
            "label": "Otro",
            "valor": "otro",
        }
    )
    return opciones


def _respuesta_pedir_marca_referencia(
    referencia: str,
    candidatos: list,
    match_campo: str,
) -> str:
    if match_campo == "REF_ALTERNATIVA":
        return (
            f"Encontré una coincidencia alternativa para la referencia {referencia}. "
            "Para confirmar, ¿cuál es la marca?"
        )

    return (
        f"Encontré coincidencias para la referencia {referencia}. "
        "¿Cuál es la marca?"
    )


def detectar_modo_busqueda(texto: str) -> str:
    """
    Clasifica la intención de búsqueda del cliente.

    Prioridad:
    1. codigo_exacto     -> 6 dígitos o referencia P de 7 caracteres
    2. esperando_codigo  -> pide código/referencia pero aún no lo escribe
    3. producto_vago     -> "necesito otro producto" sin detalle
    4. hibrida           -> producto + campos técnicos (búsqueda directa)
    5. hibrida_guiada    -> necesidad técnica + preguntas con libros/catálogo
    6. producto          -> nombre de instrumento sin specs
    7. ambiguo           -> necesidad poco clara
    """
    if not (texto or "").strip():
        return "ambiguo"

    tipo, _ = detectar_identificador(texto)
    if tipo:
        return "codigo_exacto"

    if _es_pedido_de_identificador_sin_valor(texto):
        return "esperando_codigo"

    if _es_pedido_producto_sin_detalle(texto):
        return "producto_vago"

    if _es_busqueda_hibrida_directa(texto):
        return "hibrida"

    if es_necesidad_hibrida_guiada(texto):
        return "hibrida_guiada"

    if _extraer_keyword_instrumento(texto):
        return "producto"

    if _parece_solicitud_de_producto(texto):
        return "producto"

    return "ambiguo"


def _es_pedido_producto_sin_detalle(texto: str) -> bool:
    """
    Pedidos genéricos sin producto concreto: 'necesito otro producto'.
    No deben disparar búsqueda por la palabra 'otro'.
    """
    t = _normalizar_intencion(texto)
    if not t:
        return False

    exactos = {
        "necesito otro producto",
        "quiero otro producto",
        "busco otro producto",
        "otro producto",
        "necesito un producto",
        "quiero un producto",
        "busco un producto",
        "necesito otro",
        "quiero otro",
        "busco otro",
        "necesito algo",
        "otro equipo",
    }
    if t in exactos:
        return True

    return bool(
        re.fullmatch(
            r"(?:necesito|busco|quiero|requiero)\s+"
            r"(?:otro|un|una|algun|alguna)\s+"
            r"(?:producto|equipo|articulo|item)s?",
            t,
        )
    )


def _respuesta_pedir_detalle_producto() -> str:
    return (
        "Entendido. ¿Qué producto o referencia necesitas? "
        "Si tienes la marca, dámela también."
    )


def _producto_coincide_instrumento(texto_cliente: str, producto: dict) -> bool:
    """
    Verifica si el producto encontrado pertenece a la familia pedida.
    """
    keyword = _extraer_keyword_instrumento(texto_cliente)
    if not keyword:
        return False

    bloque = " ".join(
        [
            str(producto.get("nombre") or ""),
            str(producto.get("descripcion_corta") or ""),
            str(producto.get("categoria") or ""),
            str(producto.get("nivel_3") or ""),
            str(producto.get("nivel_4") or ""),
        ]
    )
    bloque_norm = _normalizar_match_textual(bloque)
    keyword_norm = _normalizar_match_textual(keyword)

    return keyword_norm in bloque_norm


def es_solo_saludo(texto: str) -> bool:
    """
    Determina si el mensaje solo es saludo.
    """
    t = texto.lower().strip().rstrip(".,!")
    return t in PALABRAS_SALUDO or (
        len(t.split()) <= 2 and any(saludo in t for saludo in PALABRAS_SALUDO)
    )


# ─────────────────────────────────────────────────────────────
# Captura silenciosa de cliente
# ─────────────────────────────────────────────────────────────

def extraer_datos_cliente(mensaje: str, cliente_actual: dict) -> dict:
    """
    Extrae datos básicos del cliente sin interrumpir el flujo.

    Reglas:
    - Solo completa campos vacíos.
    - No sobreescribe datos ya capturados.
    - Captura email, NIT, nombre y empresa.
    - Limpia empresa para no arrastrar correo, NIT o frases posteriores.
    """
    cliente = dict(cliente_actual) if cliente_actual else {}

    if not cliente.get("email"):
        m = EMAIL_RE.search(mensaje)
        if m:
            cliente["email"] = m.group(0).strip()
            logger.debug("Email capturado: %s", cliente["email"])

    if not cliente.get("nit"):
        m = NIT_RE.search(mensaje)
        if m:
            cliente["nit"] = m.group(1).strip()
            logger.debug("NIT capturado: %s", cliente["nit"])

    if not cliente.get("nombre"):
        patrones_nombre = [
            r"(?:soy|me llamo|mi nombre es)\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,3})",
            r"(?:habla|llama|escribe)\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,2})",
        ]

        for patron in patrones_nombre:
            m = re.search(patron, mensaje, re.IGNORECASE)
            if m:
                cliente["nombre"] = m.group(1).strip()
                logger.debug("Nombre capturado: %s", cliente["nombre"])
                break

    if not cliente.get("empresa"):
        patrones_empresa = [
            r"(?:de la empresa|trabajo en|empresa|compañía)\s+(.+)",
            r"(?:somos|representamos a)\s+(.+)",
        ]

        cortes = [
            r"\s+y\s+mi\s+correo\s+es\s+",
            r"\s+y\s+mi\s+email\s+es\s+",
            r"\s+y\s+el\s+correo\s+es\s+",
            r"\s+mi\s+correo\s+es\s+",
            r"\s+mi\s+email\s+es\s+",
            r"\s+correo\s+",
            r"\s+email\s+",
            r"\s+y\s+mi\s+nit\s+es\s+",
            r"\s+mi\s+nit\s+es\s+",
            r"\s+nit\s+",
            r"\s+soy\s+",
            r"\s+me\s+llamo\s+",
            r"\s+mi\s+nombre\s+es\s+",
        ]

        for patron in patrones_empresa:
            m = re.search(patron, mensaje, re.IGNORECASE)
            if not m:
                continue

            empresa = m.group(1).strip()

            for corte in cortes:
                empresa = re.split(corte, empresa, flags=re.IGNORECASE)[0].strip()

            empresa = empresa.strip(" ,.;:-")

            if len(empresa) > 2:
                cliente["empresa"] = empresa
                logger.debug("Empresa capturada: %s", cliente["empresa"])
                break

    return cliente


# ─────────────────────────────────────────────────────────────
# Datos faltantes / saludo
# ─────────────────────────────────────────────────────────────

def datos_faltantes(cliente: dict, etapa: str) -> list:
    faltantes = []

    if etapa == "cotizacion":
        if not cliente.get("nombre"):
            faltantes.append("¿A nombre de quién va la cotización?")
        if not cliente.get("email"):
            faltantes.append("¿A qué correo te envío la cotización?")

    elif etapa == "proforma":
        if not cliente.get("empresa"):
            faltantes.append("¿Cuál es la razón social de tu empresa?")
        if not cliente.get("nit"):
            faltantes.append("¿Cuál es el NIT de la empresa?")

    return faltantes


def saludo_personalizado(cliente: dict) -> str:
    if cliente.get("nombre"):
        return f"[CLIENTE CONOCIDO: {cliente['nombre']}]"
    return "[CLIENTE NUEVO — capturar nombre si lo menciona]"


# ─────────────────────────────────────────────────────────────
# Evaluación de necesidad
# ─────────────────────────────────────────────────────────────

def _cuenta_parametros_tecnicos_generales(texto: str) -> int:
    """
    Cuenta señales técnicas generales sin depender de una familia específica.

    No es una lista de productos ni de exclusiones.
    Son patrones de magnitudes industriales comunes.
    """
    t = texto.lower()

    patrones = [
        r"\b\d+(\.\d+)?\s*(bar|psi|pa|kpa|mpa)\b",
        r"\b\d+(\.\d+)?\s*(l/min|lpm|gpm|m3/h|m³/h)\b",
        r"\b\d+(\.\d+)?\s*(°c|c|grados)\b",
        r"\b\d+(\.\d+)?\s*(v|vac|vdc|ma|a|hz|kw|hp)\b",
        r"\b\d+(\.\d+)?\s*(mm|cm|m|pulg|pulgadas|in)\b",
        r"\b(agua|aire|vapor|aceite|gas|fluido|quimico|químico)\b",
        r"\b(limpia|residual|corrosivo|industrial|sanitario|explosivo)\b",
    ]

    return sum(1 for patron in patrones if re.search(patron, t, re.IGNORECASE))


def _parece_solicitud_de_producto(texto: str) -> bool:
    """
    Detecta si el usuario está solicitando un producto o familia de producto.

    Es una detección general de intención comercial, no una regla por producto.
    """
    t = texto.lower().strip()

    patrones = [
        r"\bnecesito\b",
        r"\bnecestio\b",
        r"\brequiero\b",
        r"\bbusco\b",
        r"\bcotizar\b",
        r"\bquiero\b",
        r"\bme sirve\b",
        r"\bproducto\b",
        r"\bequipo\b",
        r"\breferencia\b",
        r"\bcódigo\b",
        r"\bcodigo\b",
    ]

    return any(re.search(patron, t) for patron in patrones) and len(t.split()) >= 3


async def evaluar_necesidad(texto: str) -> dict:
    """
    Evalúa si hay suficiente información para iniciar búsqueda en catálogo.

    Principio NIA v365:
    - Si el cliente menciona una necesidad comercial clara, se permite búsqueda preliminar.
    - Buscar NO significa recomendar.
    - La recomendación final la controla product_matcher.py.
    - Si no hay match confiable, NIA pregunta más datos.
    """
    ctx = contexto_para_agente(texto)
    dominio = ctx.get("dominio", "general")

    parametros_generales = _cuenta_parametros_tecnicos_generales(texto)
    parece_solicitud = _parece_solicitud_de_producto(texto)

    if _es_busqueda_hibrida_directa(texto):
        return {
            "clara": True,
            "preguntas": [],
            "dominio": dominio,
            "razon": "Mensaje con producto y especificaciones técnicas para búsqueda híbrida.",
        }

    if _parece_nombre_producto_modelo(texto):
        return {
            "clara": True,
            "preguntas": [],
            "dominio": dominio,
            "razon": "Nombre de producto con modelo suficiente para búsqueda en catálogo.",
        }

    if es_necesidad_hibrida_guiada(texto):
        return {
            "clara": False,
            "preguntas": [],
            "dominio": dominio,
            "razon": "Necesidad técnica que requiere descubrimiento guiado híbrido.",
            "hibrida_guiada": True,
        }

    if parece_solicitud and _debe_preguntar_antes_de_buscar(texto):
        preguntas = await generar_preguntas(texto)
        return {
            "clara": False,
            "preguntas": preguntas,
            "dominio": dominio,
            "razon": "Instrumento identificado sin especificaciones técnicas suficientes.",
        }

    if parece_solicitud:
        return {
            "clara": True,
            "preguntas": [],
            "dominio": dominio,
            "razon": "Solicitud comercial suficiente para búsqueda preliminar en catálogo.",
        }

    if parametros_generales >= 2:
        return {
            "clara": True,
            "preguntas": [],
            "dominio": dominio,
            "razon": "Solicitud con suficientes parámetros técnicos para búsqueda preliminar.",
        }

    prompt_evaluacion = (
        f"El cliente de una empresa de instrumentación industrial dice: \"{texto}\"\n\n"
        f"Dominio detectado: {dominio}\n\n"
        "¿Hay suficiente información para iniciar una búsqueda preliminar en catálogo?\n"
        "No significa recomendar todavía; solo buscar candidatos reales.\n"
        "Responde SOLO JSON sin markdown:\n"
        "{\"clara\": true/false, \"razon\": \"una frase corta\"}"
    )

    try:
        resultado = await call_llm_json(prompt_evaluacion)
        clara = bool(resultado.get("clara", False))

        logger.debug(
            "LLM evalúa necesidad: clara=%s razón=%s",
            clara,
            resultado.get("razon", ""),
        )

        if clara:
            return {
                "clara": True,
                "preguntas": [],
                "dominio": dominio,
                "razon": resultado.get("razon", ""),
            }

    except Exception as exc:
        logger.warning("LLM evaluación fallida, usando preguntas: %s", exc)

    preguntas = await generar_preguntas(texto)

    return {
        "clara": False,
        "preguntas": preguntas,
        "dominio": dominio,
    }

# ============================================================
# MATCH TEXTUAL SEGURO SOBRE CANDIDATO DE CATÁLOGO
# ============================================================

PALABRAS_FUNCIONALES_MATCH = {
    "necesito",
    "necesitamos",
    "quiero",
    "requiero",
    "requiere",
    "busco",
    "buscar",
    "cotizar",
    "cotizacion",
    "cotización",
    "producto",
    "equipo",
    "sistema",
    "para",
    "con",
    "una",
    "uno",
    "unos",
    "unas",
    "del",
    "de",
    "la",
    "el",
    "los",
    "las",
    "que",
    "por",
    "favor",
}


def _normalizar_match_textual(valor: str) -> str:
    """
    Normaliza texto para comparar intención del cliente contra nombre/descr.
    No decide compatibilidad técnica; solo ayuda a detectar coincidencias claras.
    """
    if not valor:
        return ""

    valor = str(valor).lower()
    valor = re.sub(r"[^\w\sáéíóúñü-]", " ", valor, flags=re.IGNORECASE)
    valor = re.sub(r"\s+", " ", valor).strip()

    # Normalización simple de tildes sin depender de librerías externas.
    reemplazos = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
        "ñ": "n",
    }

    for origen, destino in reemplazos.items():
        valor = valor.replace(origen, destino)

    return valor


def _tokens_producto_cliente(texto: str) -> list[str]:
    """
    Extrae tokens útiles de una necesidad del cliente.

    Importante:
    - No es una lista de productos.
    - Solo elimina palabras funcionales del lenguaje.
    """
    texto_norm = _normalizar_match_textual(texto)

    return [
        token
        for token in texto_norm.split()
        if len(token) > 2
        and not token.isdigit()
        and token not in PALABRAS_FUNCIONALES_MATCH
    ]


def _cobertura_tokens_en_texto(tokens: list[str], texto_objetivo: str) -> float:
    """
    Calcula cuántos tokens de la solicitud aparecen en el texto del producto.
    """
    if not tokens:
        return 0.0

    objetivo_norm = _normalizar_match_textual(texto_objetivo)
    objetivo_tokens = set(objetivo_norm.split())

    if not objetivo_tokens:
        return 0.0

    encontrados = 0

    for token in tokens:
        if token in objetivo_tokens:
            encontrados += 1
            continue

        # Coincidencia flexible para singular/plural o variantes pequeñas.
        if any(token in obj or obj in token for obj in objetivo_tokens if len(obj) > 3):
            encontrados += 1

    return encontrados / len(tokens)


def _debe_promover_related_a_exacto(
    texto_cliente: str,
    producto: dict,
    estado_match: str,
    campos_query: Optional[dict] = None,
) -> bool:
    """
    Promueve un related_match a exact_match solo cuando hay evidencia textual fuerte.

    Regla segura:
    - Solo aplica si el matcher ya encontró un producto relacionado.
    - No aplica cuando hay campos técnicos detectados, porque allí conviene validar.
    - Requiere mínimo 2 tokens útiles.
    - Requiere alta cobertura en nombre o descripción corta.

    Esto evita parches específicos como:
    if "bomba centrifuga" in texto
    """
    if estado_match != "related_match":
        return False

    if not producto:
        return False

    # Búsqueda por instrumento/herramienta sin código:
    # si el producto pertenece a la familia pedida, tratarlo como match exacto.
    # Solo en consultas simples; si hay campos técnicos, se mantiene validación.
    if not campos_query and _producto_coincide_instrumento(texto_cliente, producto):
        logger.info(
            "Promoviendo related->exact por instrumento detectado codigo=%s",
            producto.get("codigo"),
        )
        return True

    # Si hay campos técnicos, preferimos mantener validación técnica.
    # Ejemplo: rango, salida, conexión, voltaje, material.
    if campos_query:
        return False

    tokens = _tokens_producto_cliente(texto_cliente)

    if len(tokens) < 2 and not _extraer_keyword_instrumento(texto_cliente):
        return False

    nombre = producto.get("nombre") or ""
    descripcion_corta = producto.get("descripcion_corta") or ""
    categoria = " ".join(
        [
            str(producto.get("categoria") or ""),
            str(producto.get("nivel_3") or ""),
            str(producto.get("nivel_4") or ""),
        ]
    )

    cobertura_nombre = _cobertura_tokens_en_texto(tokens, nombre)
    cobertura_desc = _cobertura_tokens_en_texto(tokens, descripcion_corta)
    cobertura_categoria = _cobertura_tokens_en_texto(tokens, categoria)

    logger.info(
        "Evaluando promocion related->exact tokens=%s codigo=%s cobertura_nombre=%.2f cobertura_desc=%.2f cobertura_categoria=%.2f",
        tokens,
        producto.get("codigo"),
        cobertura_nombre,
        cobertura_desc,
        cobertura_categoria,
    )

    # Caso fuerte:
    # "bomba centrifuga" dentro de "Bomba centrifuga autocebante"
    if cobertura_nombre >= 0.90:
        return True

    # Caso aceptable si nombre + descripción/categoría respaldan la misma intención.
    if cobertura_nombre >= 0.75 and (cobertura_desc >= 0.75 or cobertura_categoria >= 0.75):
        return True

    return False

# ─────────────────────────────────────────────────────────────
# Búsqueda catálogo / compatibilidad
# ─────────────────────────────────────────────────────────────

async def generar_queries_catalogo(texto: str) -> list[str]:
    """
    Convierte el mensaje natural del cliente en una o varias consultas útiles
    para el catálogo.

    Responsabilidad:
    - Separar conversación humana de búsqueda técnica.
    - No modificar catalog.py para entender frases conversacionales.
    - No usar listas rígidas de palabras prohibidas.
    - Pedir al LLM una frase compacta de búsqueda, pero con fallback seguro.
    """
    texto = (texto or "").strip()

    if not texto:
        return []

    queries = []

    prompt = f"""
Eres un normalizador de búsqueda para un catálogo industrial.

Convierte el mensaje del cliente en una consulta corta para buscar productos reales en catálogo.

REGLAS:
1. No inventes productos.
2. No agregues marcas, referencias ni datos que el cliente no dijo.
3. Conserva el tipo de producto solicitado.
4. Conserva atributos técnicos relevantes: presión, caudal, señal, voltaje, material, fluido, referencia, marca si existen.
5. Quita intención conversacional como "necesito", "busco", "quiero cotizar", pero sin perder el producto.
6. Devuelve SOLO JSON válido.

Mensaje del cliente:
{texto}

Formato:
{{
  "queries": ["consulta principal", "consulta alternativa opcional"]
}}
"""

    try:
        data = await call_llm_json(prompt)
        raw_queries = data.get("queries", [])

        if isinstance(raw_queries, list):
            for query in raw_queries:
                q = str(query).strip()
                if q and q not in queries:
                    queries.append(q)

    except Exception as exc:
        logger.warning("No se pudo generar query limpia con LLM: %s", exc)

    if texto not in queries:
        queries.append(texto)

    return queries[:3]


async def buscar_en_catalogo(texto: str) -> dict:
    """
    Busca candidatos reales y valida compatibilidad producto/necesidad.

    Flujo correcto:
    1. Recibe mensaje natural del cliente.
    2. Genera queries limpias para catálogo.
    3. Busca candidatos reales en MongoDB.
    4. Valida compatibilidad contra la necesidad original.
    5. Retorna encontrado / relacionado / sin_resultado.
    """
    logger.info("Búsqueda catálogo solicitada: '%s'", texto[:100])

    queries_catalogo = await generar_queries_catalogo(texto)

    if not queries_catalogo:
        return {
            "estado": "sin_resultado",
            "razon": "No se pudo construir una consulta válida para catálogo.",
            "pregunta_sugerida": "¿Puedes indicar el tipo de producto o referencia que necesitas?",
            "candidatos_encontrados": False,
        }

    resultados = None
    query_usada = None
    campos_query = {}

    for query in queries_catalogo:
        logger.info("Intentando búsqueda catálogo con query limpia: '%s'", query)

        # Búsqueda híbrida:
        # - Limpia lenguaje conversacional.
        # - Extrae campos técnicos si existen.
        # - Consulta MongoDB como fuente oficial.
        # - NO decide compatibilidad final.
        resultados, campos_query = await buscar_con_campos(query)

        if resultados:
            query_usada = query
            logger.info(
                "Catálogo devolvió %s candidatos usando query='%s' campos_query=%s",
                len(resultados),
                query,
                campos_query,
            )
            break

    if not resultados:
        return {
            "estado": "sin_resultado",
            "razon": "No se encontraron candidatos reales en catálogo.",
            "pregunta_sugerida": "¿Puedes darme una referencia, marca, aplicación exacta o especificación adicional?",
            "candidatos_encontrados": False,
        }

    resultados = filtrar_productos_por_aprendizaje(resultados)

    if not resultados:
        return {
            "estado": "sin_resultado",
            "razon": "Los candidatos encontrados ya fueron descartados previamente por ti.",
            "pregunta_sugerida": "¿Puedes indicar otra referencia, marca o especificación diferente?",
            "candidatos_encontrados": False,
        }

    ok_textual, prod_textual = evaluar_coincidencia(
    resultados,
    texto,
    campos=len(campos_query) if campos_query else 1,
    campos_query=campos_query,
    )

    if ok_textual and prod_textual:
        logger.info(
            "Mejor candidato textual: %s score=%s query='%s'",
            prod_textual.get("codigo"),
            prod_textual.get("_score"),
            query_usada,
        )

    decision = await validar_compatibilidad_producto(
        necesidad_cliente=texto,
        candidatos=resultados,
        contexto_tecnico={
            "query_catalogo": query_usada,
            "queries_intentadas": queries_catalogo,
            "campos_query": campos_query,
        },
    )

    estado_match = decision.get("estado")
    producto = decision.get("producto")

    # ------------------------------------------------------------
    # Promoción segura related_match -> exact_match
    # ------------------------------------------------------------
    # El product_matcher puede ser conservador y marcar como relacionado
    # un producto que textualmente coincide muy bien con la necesidad.
    #
    # No bajamos umbrales globales.
    # No quemamos productos.
    # No aplica si hay campos técnicos, porque ahí sí conviene validar.
    if _debe_promover_related_a_exacto(
        texto_cliente=texto,
        producto=producto,
        estado_match=estado_match,
        campos_query=campos_query,
    ):
        logger.info(
            "Promoviendo related_match a exact_match por coincidencia textual fuerte codigo=%s",
            producto.get("codigo") if producto else None,
        )

        estado_match = "exact_match"
        decision["estado"] = "exact_match"
        decision["razon"] = (
            decision.get("razon")
            or "El nombre del producto coincide claramente con la necesidad indicada."
        )

    if estado_match == "exact_match" and producto:
        producto["_compatibilidad"] = {
            "estado": "exact_match",
            "confianza": decision.get("confianza"),
            "razon": decision.get("razon"),
            "query_catalogo": query_usada,
        }

        return {
            "estado": "encontrado",
            "producto": producto,
            "tipo": "compatibilidad_exacta",
            "exacto": True,
            "razon": decision.get("razon"),
            "query_catalogo": query_usada,
            "candidatos_encontrados": True,
        }

    if estado_match == "related_match" and producto:
        producto["_compatibilidad"] = {
            "estado": "related_match",
            "confianza": decision.get("confianza"),
            "razon": decision.get("razon"),
            "query_catalogo": query_usada,
        }

        modo_busqueda = detectar_modo_busqueda(texto)
        instrumento_detectado = _extraer_keyword_instrumento(texto) is not None

        # Búsqueda por producto/instrumento sin código:
        # mostrar primero el candidato, sin disparar 3 preguntas técnicas.
        preguntas_tecnicas = []
        if modo_busqueda != "producto" and not instrumento_detectado:
            try:
                preguntas_tecnicas = await generar_preguntas(texto)
            except Exception as e:
                logger.warning(
                    "No fue posible generar preguntas técnicas para producto relacionado: %s",
                    e,
                )
                preguntas_tecnicas = []

        preguntas_limpias = [
            p.strip()
            for p in preguntas_tecnicas
            if isinstance(p, str) and p.strip()
        ][:3]

        return {
            "estado": "relacionado",
            "producto": producto,
            "tipo": "producto_relacionado",
            "exacto": False,
            "razon": decision.get("razon"),
            "pregunta_sugerida": decision.get("pregunta_sugerida"),
            "query_catalogo": query_usada,
            "candidatos_encontrados": True,
            "texto_original": texto,
            "preguntas_tecnicas": preguntas_limpias,
        }

    return {
        "estado": "sin_resultado",
        "razon": decision.get("razon"),
        "pregunta_sugerida": decision.get("pregunta_sugerida"),
        "query_catalogo": query_usada,
        "candidatos_encontrados": True,
    }


async def rama_codigo(
    valor: str,
    tipo: str,
    texto_original: Optional[str] = None,
) -> dict:
    """
    Rama de búsqueda exacta por código o referencia.
    """
    logger.info("Búsqueda exacta: %s=%s", tipo, valor)

    if tipo == "referencia":
        marca = _extraer_marca_junto_a_referencia(texto_original, valor)
        res_ref = await buscar_por_referencia(valor, marca=marca)

        if res_ref.get("estado") == "encontrado" and res_ref.get("producto"):
            return {
                "estado": "encontrado",
                "producto": res_ref["producto"],
                "tipo": tipo,
                "exacto": True,
                "candidatos_encontrados": True,
                "match_campo": res_ref.get("match_campo"),
            }

        # Si venía marca y no hubo match, reintenta sin marca
        # (por si el cliente escribió algo extra que no es marca).
        if marca and res_ref.get("estado") != "encontrado":
            res_ref = await buscar_por_referencia(valor)

        if res_ref.get("estado") == "encontrado" and res_ref.get("producto"):
            return {
                "estado": "encontrado",
                "producto": res_ref["producto"],
                "tipo": tipo,
                "exacto": True,
                "candidatos_encontrados": True,
                "match_campo": res_ref.get("match_campo"),
            }

        if res_ref.get("estado") == "necesita_marca":
            return {
                "estado": "necesita_marca",
                "tipo": tipo,
                "referencia_buscada": valor,
                "candidatos": res_ref.get("candidatos") or [],
                "match_campo": res_ref.get("match_campo"),
                "candidatos_encontrados": True,
            }

        # Muchos identificadores P###### viven en CODIGO, no en REFERENCIA.
        # Fallback exacto (mismas variantes mecánicas) sin pasar a fuzzy.
        prod_codigo = await buscar_por_codigo(valor)
        if prod_codigo:
            return {
                "estado": "encontrado",
                "producto": prod_codigo,
                "tipo": "codigo",
                "exacto": True,
                "candidatos_encontrados": True,
                "match_campo": "CODIGO",
            }

        logger.info("Referencia sin match exacto: %s marca=%s", valor, marca)
    else:
        prod = await buscar_por_codigo(valor)

        if prod:
            return {
                "estado": "encontrado",
                "producto": prod,
                "tipo": tipo,
                "exacto": True,
                "candidatos_encontrados": True,
            }

    logger.info("Fallback catálogo para identificador: %s", valor)

    texto_fallback = str(texto_original or valor).strip()
    if len(texto_fallback.split()) > 1:
        res = await buscar_en_catalogo(texto_fallback)
    else:
        res = await buscar_en_catalogo(valor)

    if res["estado"] in {"encontrado", "relacionado"}:
        res["tipo"] = "fallback_identificador"
        return res

    logger.info("Sin resultado para identificador: %s", valor)

    return {
        "estado": "sin_resultado",
        "tipo": tipo,
        "pregunta_sugerida": "¿Puedes verificar el código o compartir marca/referencia adicional?",
        "candidatos_encontrados": res.get("candidatos_encontrados", False),
    }


def debe_intentar_enriquecimiento(res: dict) -> bool:
    """
    Decide si vale la pena intentar Libros Rol 2.

    Regla:
    - Si no hubo candidatos en catálogo, sí tiene sentido enriquecer la búsqueda.
    - Si sí hubo candidatos, pero product_matcher dijo que ninguno es compatible,
      NO se debe enriquecer a ciegas. Se debe responder seguro y pedir precisión.
    """
    if res.get("estado") != "sin_resultado":
        return False

    return res.get("candidatos_encontrados") is False


async def enriquecer_y_buscar(
    texto: str,
    necesidad_ctx: Optional[dict] = None,
) -> dict:
    """
    Usa contexto de conocimiento para enriquecer la búsqueda,
    pero mantiene la validación de compatibilidad.

    Para EPI/seguridad (botas, guantes, etc.) no mezcla términos de
    instrumentación ni regenera preguntas incoherentes.
    """
    necesidad_ctx = necesidad_ctx or {}
    texto_ancla = texto_ancla_desde_ctx(necesidad_ctx, texto)
    texto_limpio = construir_texto_limpio_descubrimiento(necesidad_ctx, texto)

    if es_producto_epi_seguridad(texto_ancla):
        for query_epi in queries_alternativas_epi(
            texto_ancla,
            necesidad_ctx.get("respuestas_tecnicas"),
        ):
            logger.info("Búsqueda EPI alternativa: '%s'", query_epi[:80])
            res_epi = await buscar_en_catalogo(query_epi)
            if res_epi.get("estado") in {"encontrado", "relacionado"}:
                res_epi["tipo"] = "epi_seguridad"
                res_epi["dominio"] = "epi_seguridad"
                res_epi["query_catalogo"] = query_epi
                return res_epi

        return {
            "estado": "pendiente",
            "preguntas": preguntas_refino_epi(
                texto_ancla,
                necesidad_ctx.get("respuestas_tecnicas"),
            ),
            "dominio": "epi_seguridad",
        }

    ctx = contexto_para_agente(texto_limpio or texto)
    terminos = filtrar_terminos_libros(ctx.get("terminos", []), texto_ancla)
    dominio = ctx.get("dominio", "")
    query = f"{texto_limpio or texto} {' '.join(terminos[:4])}".strip()

    logger.info("Libros Rol 2 — query enriquecida: '%s'", query[:80])

    res = await buscar_en_catalogo(query)

    if res["estado"] in {"encontrado", "relacionado"}:
        res["tipo"] = "libros_rol2"
        res["dominio"] = dominio
        res["query_enriquecida"] = query
        return res

    if res.get("candidatos_encontrados") is True:
        res["dominio"] = dominio
        res["query_enriquecida"] = query
        return res

    preguntas = await generar_preguntas(
        texto_limpio or texto_ancla or texto,
        necesidad_ctx=necesidad_ctx,
    )

    return {
        "estado": "pendiente",
        "preguntas": preguntas,
        "dominio": dominio,
    }


# ─────────────────────────────────────────────────────────────
# Response helpers
# ─────────────────────────────────────────────────────────────

def _marcar_respuesta_segura(texto: str) -> str:
    """
    Marca una respuesta para que no sea reescrita por el LLM.
    """
    return "[RESPUESTA_SEGURA]\n" + texto


# ─────────────────────────────────────────────────────────────
# Preguntas técnicas — una por turno
# ─────────────────────────────────────────────────────────────

MAX_PREGUNTAS_TECNICAS = 3

FASES_FLUJO_HIBRIDA = {
    "preguntas_hibridas",
    "esperando_otro_hibrida",
    "esperando_texto_hibrida",
}

FASES_FLUJO_CORTA_LARGA = {
    "seleccion_tipo",
    "esperando_otro_tipo",
    "seleccion_tipo_otro",
    "preguntas_tecnicas",
    "esperando_otro_tecnico",
    "seleccion_producto_descubrimiento",
}


def _en_flujo_hibrida(necesidad_ctx: dict) -> bool:
    if not necesidad_ctx:
        return False
    if necesidad_ctx.get("flujo_descubrimiento") == "hibrida_libros":
        return True
    return necesidad_ctx.get("fase_descubrimiento") in FASES_FLUJO_HIBRIDA


def _en_flujo_corta_larga(necesidad_ctx: dict) -> bool:
    if not necesidad_ctx:
        return False
    fase = necesidad_ctx.get("fase_descubrimiento")
    if fase in FASES_FLUJO_CORTA_LARGA:
        return True
    if necesidad_ctx.get("flujo_descubrimiento") == "corta_larga":
        return True
    if necesidad_ctx.get("alternativas_otro"):
        return True
    return False


def _texto_busqueda_otro_nivel_1(palabra_clave: str, texto_cliente: str) -> str:
    """
    Une instrumento + descripción del cliente para búsqueda textual en NIVEL_1.
    Ej: termometro + alimentos → "termometro alimentos".
    """
    partes = []
    for valor in (palabra_clave, texto_cliente):
        txt = str(valor or "").strip()
        if txt:
            partes.append(txt)
    return " ".join(partes).strip()


async def _try_resolver_turno_corta_larga(
    mensaje: str,
    necesidad_ctx: dict,
    cliente: dict,
    productos_acumulados: list,
) -> Optional[tuple[str, str, dict]]:
    """
    Resuelve el turno si pertenece al flujo NIVEL_1 → DESCRIPCION_LARGA.
    Retorna (contexto_extra, etapa, ctx) o None.
    """
    if not (mensaje or "").strip():
        return None

    ctx = dict(necesidad_ctx or {})
    if not _en_flujo_corta_larga(ctx):
        return None

    if not ctx.get("palabra_clave") and ctx.get("texto_original"):
        palabra = extraer_palabra_clave(ctx["texto_original"]) or _extraer_keyword_instrumento(
            ctx["texto_original"]
        )
        if palabra:
            ctx["palabra_clave"] = palabra

    logger.info(
        "Flujo corta_larga activo: fase=%s flujo=%s palabra=%s mensaje='%s'",
        ctx.get("fase_descubrimiento"),
        ctx.get("flujo_descubrimiento"),
        ctx.get("palabra_clave"),
        mensaje[:80],
    )

    return await _continuar_descubrimiento_corta_larga(
        necesidad_ctx=ctx,
        mensaje=mensaje,
        cliente=cliente,
        productos_acumulados=productos_acumulados,
    )


def _normalizar_preguntas(preguntas: list) -> list:
    limpias = []
    textos_vistos: set[str] = set()
    for pregunta in preguntas or []:
        if isinstance(pregunta, dict) and _texto_pregunta(pregunta):
            texto_key = _texto_pregunta(pregunta).lower().strip()
            if texto_key in textos_vistos:
                continue
            textos_vistos.add(texto_key)
            limpias.append(pregunta)
        elif isinstance(pregunta, str) and pregunta.strip():
            texto_key = pregunta.strip().lower()
            if texto_key in textos_vistos:
                continue
            textos_vistos.add(texto_key)
            limpias.append(pregunta.strip())
    return limpias[:MAX_PREGUNTAS_TECNICAS]


def _nombre_cliente_prefix(cliente: Optional[dict]) -> str:
    if cliente and cliente.get("nombre"):
        return f"{cliente['nombre']}, "
    return ""


def _respuesta_pregunta_unica(
    cliente: dict,
    pregunta: str,
    intro: Optional[str] = None,
) -> str:
    prefix = _nombre_cliente_prefix(cliente)
    pregunta_limpia = (pregunta or "").strip()
    intro_limpia = (intro or "").strip()

    # Evita repetir la misma frase como intro + pregunta.
    if intro_limpia and pregunta_limpia and intro_limpia.lower() == pregunta_limpia.lower():
        texto = f"{prefix}{pregunta_limpia}"
    elif intro_limpia and pregunta_limpia:
        texto = f"{prefix}{intro_limpia}\n\n{pregunta_limpia}"
    else:
        texto = f"{prefix}{intro_limpia or pregunta_limpia}"
    return _marcar_respuesta_segura(texto)


def _tiene_busqueda_textual_multipalabra(mensaje: str) -> bool:
    """True si el cliente escribió 2+ palabras útiles (ej. termometro digital)."""
    return len(_tokens_producto_cliente(mensaje)) >= 2


async def _iniciar_descubrimiento_producto_corta_larga(
    mensaje: str,
    cliente: dict,
) -> tuple[str, str, dict]:
    """
    Inicia el flujo de búsqueda por producto:
    Q1 = tipos más frecuentes en NIVEL_1 + Otro.
    """
    palabra = extraer_palabra_clave(mensaje) or _extraer_keyword_instrumento(mensaje)
    if not palabra:
        tokens = _tokens_producto_cliente(mensaje)
        palabra = tokens[0] if tokens else ""

    if not palabra:
        preguntas = await generar_preguntas(mensaje)
        preguntas = [PREGUNTA_INICIAL_NECESIDAD] + [
            p for p in (preguntas or []) if _texto_pregunta(p) != PREGUNTA_INICIAL_NECESIDAD
        ][:2]
        return _iniciar_secuencia_preguntas(
            {
                "texto_original": mensaje,
                "modo_busqueda": "producto",
            },
            preguntas,
            cliente,
            "",
        )

    busqueda_textual = _tiene_busqueda_textual_multipalabra(mensaje)

    tipos = await resolver_tipos_catalogo_inicio(
        palabra_clave=palabra,
        mensaje=mensaje,
        busqueda_textual=busqueda_textual,
        top=3,
    )

    if not tipos:
        if es_producto_epi_seguridad(mensaje):
            return _iniciar_secuencia_preguntas(
                {
                    "texto_original": mensaje,
                    "modo_busqueda": "producto",
                    "palabra_clave": palabra,
                    "dominio": "epi_seguridad",
                },
                preguntas_epi_con_opciones(mensaje),
                cliente,
                "Para ubicar el equipo correcto en catálogo:",
            )

        preguntas = await generar_preguntas(mensaje, necesidad_ctx={"texto_original": mensaje})
        return _iniciar_secuencia_preguntas(
            {
                "texto_original": mensaje,
                "modo_busqueda": "producto",
                "palabra_clave": palabra,
            },
            preguntas,
            cliente,
            "No encontré tipos claros en catálogo. Necesito un dato:",
        )

    if busqueda_textual:
        pregunta_data = generar_pregunta_seleccion_otro(palabra, tipos, mensaje)
        intro = "Para ubicar el equipo correcto en catálogo:"
    else:
        pregunta_data = generar_pregunta_seleccion_tipo(palabra, tipos)
        intro = "Para ubicar el equipo correcto en catálogo:"
    ctx = {
        "texto_original": mensaje,
        "modo_busqueda": "producto",
        "flujo_descubrimiento": "corta_larga",
        "palabra_clave": palabra,
        "tipos_catalogo": tipos,
        "fase_descubrimiento": "seleccion_tipo",
        "preguntas_pendientes": [],
        "pregunta_indice": 0,
        "respuestas_tecnicas": [],
        "opciones_actuales": pregunta_data.get("opciones") or [],
    }

    return (
        _respuesta_pregunta_unica(
            cliente,
            pregunta_data["texto"],
            intro,
        ),
        "descubrimiento",
        ctx,
    )


def _construir_contexto_tecnico(necesidad_ctx: dict) -> str:
    partes = [str(necesidad_ctx.get("texto_original") or "").strip()]
    for item in resolver_previas_hibridas(necesidad_ctx):
        valor = str(item.get("valor") or "").strip()
        if valor:
            partes.append(valor)
    return " ".join(p for p in partes if p).strip()


def _resolver_dominio_tecnico(necesidad_ctx: dict, nivel_1: Optional[str]) -> str:
    if necesidad_ctx.get("dominio"):
        return str(necesidad_ctx["dominio"])
    palabra = str(necesidad_ctx.get("palabra_clave") or "").lower()
    nivel_slug = (nivel_1 or "").lower()
    if "nivel" in palabra or "nivel" in nivel_slug:
        return "nivel"
    if "humedad" in palabra or "higro" in palabra or "rocio" in palabra or "humedad" in nivel_slug:
        return "humedad"
    if "temperatura" in palabra or "termometro" in palabra or "temperatura" in nivel_slug:
        return "temperatura"
    if "presion" in palabra or "manometro" in palabra or "presion" in nivel_slug:
        return "presion"
    if any(k in palabra for k in ("bota", "calzado", "guante", "epi")) or any(
        k in nivel_slug for k in ("calzado", "guante", "dielectric")
    ):
        return "epi_seguridad"
    return ""


async def _iniciar_preguntas_tecnicas_por_nivel_1(
    necesidad_ctx: dict,
    nivel_1: Optional[str],
    cliente: dict,
    productos_acumulados: list,
    intro: str,
) -> tuple[str, str, dict]:
    """
    Pasa de un NIVEL_1 elegido a las 2 preguntas técnicas con botones.
    """
    palabra = necesidad_ctx.get("palabra_clave") or ""
    dominio = _resolver_dominio_tecnico(necesidad_ctx, nivel_1)
    contexto = _construir_contexto_tecnico(necesidad_ctx)
    preguntas = await generar_preguntas_tecnicas_por_nivel_1(
        nivel_1 or "",
        dominio=dominio or None,
        contexto_texto=contexto or None,
    )

    ctx = {
        **necesidad_ctx,
        "fase_descubrimiento": "preguntas_tecnicas",
        "nivel_1_seleccionado": nivel_1,
        "tipo_corta_seleccionado": nivel_1,
        "preguntas_pendientes": preguntas,
        "pregunta_indice": 0,
        "respuestas_tecnicas": [],
        "opciones_actuales": _opciones_pregunta(preguntas[0]) if preguntas else [],
    }

    if not preguntas:
        query_e = construir_query_busqueda_final(palabra, nivel_1, [])
        return await _buscar_y_responder_descubrimiento_producto(
            query_e=query_e,
            necesidad_ctx=ctx,
            cliente=cliente,
            productos_acumulados=productos_acumulados,
        )

    return (
        _respuesta_pregunta_unica(
            cliente,
            _texto_pregunta(preguntas[0]),
            intro,
        ),
        "descubrimiento",
        ctx,
    )


async def _continuar_descubrimiento_corta_larga(
    necesidad_ctx: dict,
    mensaje: str,
    cliente: dict,
    productos_acumulados: list,
) -> tuple[str, str, dict]:
    """
    Maneja las fases del flujo corta → larga.
    """
    fase = necesidad_ctx.get("fase_descubrimiento")
    palabra = necesidad_ctx.get("palabra_clave") or ""
    tipos = necesidad_ctx.get("tipos_catalogo") or []

    if fase == "seleccion_tipo":
        kind, valor = resolver_seleccion_tipo(mensaje, tipos)

        if kind == "otro":
            ctx = {
                **necesidad_ctx,
                "fase_descubrimiento": "esperando_otro_tipo",
                "flujo_descubrimiento": "corta_larga",
                "opciones_actuales": [],
            }
            instrumento = palabra or "producto"
            return (
                _respuesta_pregunta_unica(
                    cliente,
                    f"¿Qué tipo de {instrumento} necesitas? Descríbelo brevemente.",
                    "Entendido.",
                ),
                "descubrimiento",
                ctx,
            )

        tipo_nivel_1 = valor if kind == "tipo" else None

        if kind == "texto_libre" and valor:
            texto_busqueda = _texto_busqueda_otro_nivel_1(palabra, valor)
            coincidencias = await obtener_tipos_nivel_1_por_texto(palabra, texto_busqueda, top=3)
            if coincidencias:
                pregunta_data = generar_pregunta_seleccion_otro(palabra, coincidencias, valor)
                ctx = {
                    **necesidad_ctx,
                    "fase_descubrimiento": "seleccion_tipo_otro",
                    "flujo_descubrimiento": "corta_larga",
                    "texto_otro": valor,
                    "alternativas_otro": coincidencias,
                    "opciones_actuales": pregunta_data.get("opciones") or [],
                }
                return (
                    _respuesta_pregunta_unica(
                        cliente,
                        pregunta_data["texto"],
                        "Entendido.",
                    ),
                    "descubrimiento",
                    ctx,
                )
            tipo_nivel_1 = valor

        return await _iniciar_preguntas_tecnicas_por_nivel_1(
            necesidad_ctx=necesidad_ctx,
            nivel_1=tipo_nivel_1,
            cliente=cliente,
            productos_acumulados=productos_acumulados,
            intro="Perfecto. Para afinar dentro de ese tipo:",
        )

    if fase == "esperando_otro_tipo":
        texto_libre = str(mensaje or "").strip()
        texto_busqueda = _texto_busqueda_otro_nivel_1(palabra, texto_libre)
        alternativas = await obtener_tipos_nivel_1_por_texto(palabra, texto_busqueda, top=3)

        if not alternativas:
            return await _iniciar_preguntas_tecnicas_por_nivel_1(
                necesidad_ctx={
                    **necesidad_ctx,
                    "texto_otro": texto_libre,
                },
                nivel_1=None,
                cliente=cliente,
                productos_acumulados=productos_acumulados,
                intro="No encontré tipos cercanos. Intentemos con estos datos:",
            )

        pregunta_data = generar_pregunta_seleccion_otro(palabra, alternativas, texto_libre)
        ctx = {
            **necesidad_ctx,
            "fase_descubrimiento": "seleccion_tipo_otro",
            "flujo_descubrimiento": "corta_larga",
            "texto_otro": texto_libre,
            "alternativas_otro": alternativas,
            "opciones_actuales": pregunta_data.get("opciones") or [],
        }

        return (
            _respuesta_pregunta_unica(
                cliente,
                pregunta_data["texto"],
                "Encontré opciones cercanas en catálogo:",
            ),
            "descubrimiento",
            ctx,
        )

    if fase == "seleccion_tipo_otro":
        alternativas = necesidad_ctx.get("alternativas_otro") or []
        kind, valor = resolver_seleccion_tipo(mensaje, alternativas)

        if kind == "otro":
            ctx = {
                **necesidad_ctx,
                "fase_descubrimiento": "esperando_otro_tipo",
                "flujo_descubrimiento": "corta_larga",
                "opciones_actuales": [],
            }
            instrumento = palabra or "producto"
            return (
                _respuesta_pregunta_unica(
                    cliente,
                    f"¿Qué tipo de {instrumento} necesitas? Descríbelo con otras palabras.",
                    "Entendido.",
                ),
                "descubrimiento",
                ctx,
            )

        tipo_nivel_1 = valor if kind == "tipo" else None

        if kind == "texto_libre" and valor:
            texto_busqueda = _texto_busqueda_otro_nivel_1(palabra, valor)
            coincidencias = await obtener_tipos_nivel_1_por_texto(palabra, texto_busqueda, top=3)
            if coincidencias:
                pregunta_data = generar_pregunta_seleccion_otro(palabra, coincidencias, valor)
                ctx = {
                    **necesidad_ctx,
                    "fase_descubrimiento": "seleccion_tipo_otro",
                    "flujo_descubrimiento": "corta_larga",
                    "texto_otro": valor,
                    "alternativas_otro": coincidencias,
                    "opciones_actuales": pregunta_data.get("opciones") or [],
                }
                return (
                    _respuesta_pregunta_unica(
                        cliente,
                        pregunta_data["texto"],
                        "Estas opciones se acercan más:",
                    ),
                    "descubrimiento",
                    ctx,
                )
            tipo_nivel_1 = valor

        return await _iniciar_preguntas_tecnicas_por_nivel_1(
            necesidad_ctx=necesidad_ctx,
            nivel_1=await afinar_nivel_1_para_contexto(
                tipo_nivel_1,
                necesidad_ctx.get("respuestas_hibridas_previas") or [],
                extractos_libros=necesidad_ctx.get("extractos_libros"),
            ),
            cliente=cliente,
            productos_acumulados=productos_acumulados,
            intro="Gracias. Para ubicar la referencia exacta:",
        )

    if fase == "esperando_otro_tecnico":
        texto_libre = str(mensaje or "").strip()
        ctx_restaurado = {
            **necesidad_ctx,
            "fase_descubrimiento": "preguntas_tecnicas",
        }
        respuesta_segura, etapa_resp, ctx_actualizado, accion = (
            _continuar_secuencia_preguntas(
                ctx_restaurado,
                mensaje,
                cliente,
                respuesta_forzada=texto_libre,
            )
        )
        if accion == "continuar":
            return respuesta_segura, etapa_resp, ctx_actualizado
        if accion == "esperar_otro":
            return respuesta_segura, etapa_resp, ctx_actualizado

        nivel_1 = (
            ctx_actualizado.get("nivel_1_seleccionado")
            or ctx_actualizado.get("tipo_corta_seleccionado")
        )
        respuestas = ctx_actualizado.get("respuestas_tecnicas") or []
        query_e = construir_query_busqueda_final(
            palabra,
            nivel_1,
            respuestas,
            texto_original=ctx_actualizado.get("texto_original"),
        )
        previas = resolver_previas_hibridas(ctx_actualizado)
        if previas:
            extra = " ".join(
                str(r.get("valor") or "").strip()
                for r in previas
                if r.get("valor")
            )
            if extra:
                query_e = f"{query_e} {extra}".strip()

        return await _buscar_y_responder_descubrimiento_producto(
            query_e=query_e,
            necesidad_ctx=ctx_actualizado,
            cliente=cliente,
            productos_acumulados=productos_acumulados,
        )

    if fase == "seleccion_producto_descubrimiento":
        candidatos = necesidad_ctx.get("candidatos_producto") or []
        opciones = _opciones_candidatos_producto(candidatos)
        tipo_resp, valor = resolver_respuesta_hibrida(
            mensaje,
            {"opciones": opciones},
        )
        indice = None
        if tipo_resp == "opcion" and str(valor).isdigit():
            indice = int(valor) - 1
        elif re.fullmatch(r"\d+", str(mensaje or "").strip()):
            indice = int(str(mensaje).strip()) - 1

        if indice is not None and 0 <= indice < len(candidatos):
            producto = candidatos[indice]
            res = {
                "estado": "encontrado",
                "producto": producto,
                "razon": "Seleccionaste esta opción del catálogo.",
                "exacto": False,
            }
            ctx_base = _ctx_descubrimiento_base(
                necesidad_ctx,
                necesidad_ctx.get("query_evaluada") or "",
            )
            ctx_base["flujo_descubrimiento"] = "corta_larga"
            ctx_base["opciones_actuales"] = []
            return construir_respuesta_desde_resultado(
                res=res,
                cliente=cliente,
                productos_acumulados=productos_acumulados,
                desde="descubrimiento_producto",
                necesidad_ctx_base=ctx_base,
            )

        return (
            _respuesta_pregunta_unica(
                cliente,
                "Indica el número de la opción que más se acerca, o describe otra especificación.",
                "Entendido.",
            ),
            "descubrimiento",
            {
                **necesidad_ctx,
                "opciones_actuales": opciones,
            },
        )

    if fase == "preguntas_tecnicas":
        respuesta_segura, etapa_resp, ctx_actualizado, accion = (
            _continuar_secuencia_preguntas(necesidad_ctx, mensaje, cliente)
        )

        if accion in {"continuar", "esperar_otro"}:
            return respuesta_segura, etapa_resp, ctx_actualizado

        nivel_1 = (
            ctx_actualizado.get("nivel_1_seleccionado")
            or ctx_actualizado.get("tipo_corta_seleccionado")
        )
        respuestas = ctx_actualizado.get("respuestas_tecnicas") or []
        query_e = construir_query_busqueda_final(
            palabra,
            nivel_1,
            respuestas,
            texto_original=ctx_actualizado.get("texto_original"),
        )
        previas = resolver_previas_hibridas(ctx_actualizado)
        if previas:
            extra = " ".join(
                str(r.get("valor") or "").strip()
                for r in previas
                if r.get("valor")
            )
            if extra:
                query_e = f"{query_e} {extra}".strip()

        return await _buscar_y_responder_descubrimiento_producto(
            query_e=query_e,
            necesidad_ctx=ctx_actualizado,
            cliente=cliente,
            productos_acumulados=productos_acumulados,
        )

    logger.warning(
        "Flujo corta_larga sin fase reconocida: fase=%s mensaje='%s'. Reintentando búsqueda Otro.",
        fase,
        mensaje[:80],
    )

    texto_libre = str(mensaje or "").strip()
    texto_busqueda = _texto_busqueda_otro_nivel_1(palabra, texto_libre)
    alternativas = await obtener_tipos_nivel_1_por_texto(palabra, texto_busqueda, top=3)

    if alternativas:
        pregunta_data = generar_pregunta_seleccion_otro(palabra, alternativas, texto_libre)
        ctx = {
            **necesidad_ctx,
            "fase_descubrimiento": "seleccion_tipo_otro",
            "flujo_descubrimiento": "corta_larga",
            "texto_otro": texto_libre,
            "alternativas_otro": alternativas,
            "opciones_actuales": pregunta_data.get("opciones") or [],
        }
        return (
            _respuesta_pregunta_unica(
                cliente,
                pregunta_data["texto"],
                "Encontré opciones cercanas en catálogo:",
            ),
            "descubrimiento",
            ctx,
        )

    return await _iniciar_preguntas_tecnicas_por_nivel_1(
        necesidad_ctx={
            **necesidad_ctx,
            "flujo_descubrimiento": "corta_larga",
            "texto_otro": texto_libre,
        },
        nivel_1=None,
        cliente=cliente,
        productos_acumulados=productos_acumulados,
        intro="No encontré tipos cercanos. Intentemos con estos datos:",
    )


async def _buscar_y_responder_descubrimiento_producto(
    query_e: str,
    necesidad_ctx: dict,
    cliente: dict,
    productos_acumulados: list,
) -> tuple[str, str, dict]:
    """
    Ejecuta búsqueda final del flujo corta → larga.
    """
    palabra = necesidad_ctx.get("palabra_clave") or ""
    nivel_1 = (
        necesidad_ctx.get("nivel_1_seleccionado")
        or necesidad_ctx.get("tipo_corta_seleccionado")
    )
    previas = resolver_previas_hibridas(necesidad_ctx)
    extractos = necesidad_ctx.get("extractos_libros") or []

    nivel_1 = await afinar_nivel_1_para_contexto(
        nivel_1,
        previas,
        extractos_libros=extractos,
    )
    if nivel_1:
        necesidad_ctx = {
            **necesidad_ctx,
            "nivel_1_seleccionado": nivel_1,
            "tipo_corta_seleccionado": nivel_1,
        }

    respuestas = list(necesidad_ctx.get("respuestas_tecnicas") or [])

    res = await buscar_con_descubrimiento_producto(
        palabra_clave=palabra,
        nivel_1=nivel_1,
        respuestas_tecnicas=respuestas,
        respuestas_hibridas_previas=previas,
        extractos_libros=extractos,
        texto_original=necesidad_ctx.get("texto_original"),
    )

    producto = res.get("producto")
    if producto and previas and producto_inadecuado_para_contexto(producto, previas):
        pool = await buscar_productos_por_nivel_1(nivel_1, palabra) or []
        pool = filtrar_productos_por_tecnologia_material(
            pool, previas, extractos_libros=extractos
        )
        pool = [p for p in pool if not producto_inadecuado_para_contexto(p, previas)]
        pool = filtrar_productos_por_aprendizaje(pool)
        if pool:
            res = {
                **res,
                "estado": "encontrado",
                "producto": pool[0],
                "razon": "Seleccionado radar de nivel (adecuado para sólidos/polvo).",
                "exacto": True,
            }

    nota_tecnica = mensaje_asesoria_tecnica_nivel(
        previas,
        extractos_libros=extractos,
        producto=res.get("producto"),
    )

    ctx_base = _ctx_descubrimiento_base(necesidad_ctx, query_e)
    ctx_base["flujo_descubrimiento"] = "corta_larga"
    ctx_base["nivel_1_seleccionado"] = nivel_1
    ctx_base["tipo_corta_seleccionado"] = nivel_1
    ctx_base["palabra_clave"] = palabra
    ctx_base["opciones_actuales"] = []
    ctx_base["nota_tecnica"] = nota_tecnica

    if res.get("estado") == "sin_resultado":
        ctx_base["busqueda_sin_resultado"] = True

    producto_final = res.get("producto") or {}
    logger.info(
        "Descubrimiento producto final: codigo=%s nivel_1=%s previas=%s",
        producto_final.get("codigo"),
        nivel_1,
        len(previas),
    )

    return construir_respuesta_desde_resultado(
        res=res,
        cliente=cliente,
        productos_acumulados=productos_acumulados,
        desde="descubrimiento_producto",
        necesidad_ctx_base=ctx_base,
    )


def _iniciar_secuencia_preguntas(
    necesidad_ctx_base: dict,
    preguntas: list,
    cliente: dict,
    intro: Optional[str] = None,
) -> tuple[str, str, dict]:
    """
    Guarda hasta 3 preguntas en sesión y devuelve solo la primera.
    """
    limpias = _normalizar_preguntas(preguntas)
    if not limpias:
        return (
            _marcar_respuesta_segura(respuesta_sin_resultado(cliente=cliente)),
            "descubrimiento",
            necesidad_ctx_base,
        )

    ctx = {
        **necesidad_ctx_base,
        "preguntas_pendientes": limpias,
        "pregunta_indice": 0,
        "respuestas_tecnicas": [],
        "opciones_actuales": _opciones_pregunta(limpias[0]) if limpias else [],
    }
    intro_final = (
        "Para ayudarte mejor, necesito confirmar un dato:"
        if intro is None
        else intro
    )
    return (
        _respuesta_pregunta_unica(cliente, _texto_pregunta(limpias[0]), intro_final),
        "descubrimiento",
        ctx,
    )


def _construir_query_acumulado(necesidad_ctx: dict, mensaje_actual: str = "") -> str:
    return construir_texto_limpio_descubrimiento(necesidad_ctx, mensaje_actual)


def _opciones_candidatos_producto(candidatos: list[dict]) -> list[dict]:
    opciones = []
    for idx, producto in enumerate(candidatos[:4], start=1):
        nombre = str(
            producto.get("nombre") or producto.get("descripcion_corta") or ""
        ).strip()
        codigo = str(producto.get("codigo") or "").strip()
        referencia = str(producto.get("referencia") or "").strip()
        etiqueta = nombre or codigo or f"Opción {idx}"
        if codigo and codigo not in etiqueta:
            etiqueta = f"{codigo} — {etiqueta}"
        if referencia and referencia not in etiqueta:
            etiqueta = f"{etiqueta} ({referencia})"
        opciones.append(
            {
                "id": str(idx),
                "label": etiqueta[:100],
                "valor": str(idx),
            }
        )
    opciones.append(
        {
            "id": str(len(opciones) + 1),
            "label": "Ninguno de estos",
            "valor": "ninguno",
        }
    )
    return opciones


def _respuesta_multiples_candidatos(
    candidatos: list[dict],
    cliente: Optional[dict] = None,
) -> str:
    prefix = _nombre_cliente_prefix(cliente)
    lineas = [
        f"{prefix}encontré varias opciones en catálogo que podrían servir.",
        "¿Cuál se acerca más a lo que necesitas?",
        "",
    ]
    for idx, producto in enumerate(candidatos[:4], start=1):
        nombre = str(
            producto.get("nombre") or producto.get("descripcion_corta") or ""
        ).strip()
        codigo = str(producto.get("codigo") or "").strip()
        referencia = str(producto.get("referencia") or "").strip()
        lineas.append(f"{idx}. Código {codigo} — {nombre}")
        if referencia:
            lineas.append(f"   Referencia: {referencia}")
    return "\n".join(lineas).strip()


def _pregunta_como_dict(pregunta) -> dict:
    if isinstance(pregunta, dict):
        return pregunta
    texto = str(pregunta or "").strip()
    return {"texto": texto, "opciones": []}


def _continuar_secuencia_preguntas(
    necesidad_ctx: dict,
    mensaje: str,
    cliente: dict,
    respuesta_forzada: Optional[str] = None,
) -> tuple[Optional[str], Optional[str], Optional[dict], str]:
    """
    Registra la respuesta del cliente.

    Retorna acción:
    - continuar: siguiente pregunta
    - buscar: ya se respondieron todas
    - esperar_otro: el cliente eligió Otro y debe describir
    """
    preguntas = necesidad_ctx.get("preguntas_pendientes") or []
    if not preguntas:
        return None, None, None, "buscar"

    indice = int(necesidad_ctx.get("pregunta_indice", 0))
    pregunta_actual = _pregunta_como_dict(
        preguntas[indice] if indice < len(preguntas) else {}
    )
    respuestas = list(necesidad_ctx.get("respuestas_tecnicas") or [])

    respuesta_valor = str(respuesta_forzada or "").strip()
    if not respuesta_valor:
        tipo_resp, valor = resolver_respuesta_hibrida(mensaje, pregunta_actual)

        if tipo_resp == "otro":
            return (
                _respuesta_pregunta_unica(
                    cliente,
                    "Descríbelo con tus palabras para seguir filtrando en catálogo.",
                    "Entendido.",
                ),
                "descubrimiento",
                {
                    **necesidad_ctx,
                    "fase_descubrimiento": "esperando_otro_tecnico",
                    "pregunta_otro_indice": indice,
                    "opciones_actuales": [],
                },
                "esperar_otro",
            )

        if tipo_resp == "opcion":
            respuesta_valor = str(valor or "").strip()
            for opcion in pregunta_actual.get("opciones") or []:
                if opcion.get("valor") == valor:
                    respuesta_valor = str(
                        opcion.get("label") or opcion.get("valor") or ""
                    ).strip()
                    break
        elif tipo_resp == "texto_libre":
            respuesta_valor = str(valor or "").strip()
        else:
            respuesta_valor = str(mensaje or "").strip()

    from discovery_guards import (
        es_respuesta_desconocida,
        es_respuesta_sin_valor_busqueda,
    )

    if (
        respuesta_valor
        and not es_respuesta_desconocida(respuesta_valor)
        and not es_respuesta_sin_valor_busqueda(respuesta_valor)
    ):
        respuestas.append(respuesta_valor)

    siguiente = indice + 1
    ctx_actualizado = {
        **necesidad_ctx,
        "fase_descubrimiento": "preguntas_tecnicas",
        "respuestas_tecnicas": respuestas,
        "pregunta_indice": siguiente,
    }

    if siguiente < len(preguntas):
        siguiente_pregunta = preguntas[siguiente]
        ctx_actualizado["opciones_actuales"] = _opciones_pregunta(siguiente_pregunta)
        return (
            _respuesta_pregunta_unica(
                cliente,
                _texto_pregunta(siguiente_pregunta),
                "Gracias. Siguiente dato:",
            ),
            "descubrimiento",
            ctx_actualizado,
            "continuar",
        )

    ctx_actualizado["opciones_actuales"] = []
    return None, None, ctx_actualizado, "buscar"


def _construir_query_descubrimiento(necesidad_ctx: dict, mensaje: str) -> str:
    """
    Arma la consulta de catálogo conservando todo el contexto acumulado.
    """
    if necesidad_ctx.get("query_evaluada"):
        base = str(necesidad_ctx["query_evaluada"]).strip()
    else:
        base = _construir_query_acumulado(necesidad_ctx)

    mensaje_txt = str(mensaje or "").strip()
    if not base:
        return mensaje_txt

    if not mensaje_txt:
        return base

    return f"{base} {mensaje_txt}".strip()


def _ctx_descubrimiento_base(necesidad_ctx: dict, query_e: str) -> dict:
    """
    Conserva el contexto técnico entre turnos de descubrimiento.
    """
    return {
        "texto_original": necesidad_ctx.get("texto_original"),
        "query_evaluada": query_e,
        "respuestas_tecnicas": list(necesidad_ctx.get("respuestas_tecnicas") or []),
        "dominio": necesidad_ctx.get("dominio", "general"),
    }


def _es_respuesta_afirmativa_corta(mensaje: str) -> bool:
    t = (mensaje or "").lower().strip().rstrip(".,!")
    return t in {
        "si",
        "sí",
        "ok",
        "dale",
        "claro",
        "de acuerdo",
        "correcto",
        "exacto",
        "asi es",
        "así es",
        "esta bien",
        "está bien",
    }


async def _finalizar_hibrida_con_producto(
    producto: dict,
    necesidad_ctx: dict,
    cliente: dict,
    productos_acumulados: list,
) -> tuple[str, str, dict]:
    productos_acumulados.append({
        "producto": producto,
        "cantidad": None,
        "desde": "hibrida_libros",
        "ts": datetime.utcnow().isoformat(),
    })
    return (
        _marcar_respuesta_segura(respuesta_producto_encontrado(producto, cliente)),
        "producto_encontrado",
        _ctx_confirmacion_producto(),
    )


def _palabra_clave_desde_ctx_hibrida(necesidad_ctx: dict, mensaje: str) -> str:
    texto_libre = str(mensaje or "").strip()
    texto_original = str(necesidad_ctx.get("texto_original") or "").strip()
    dominio = str(necesidad_ctx.get("dominio") or "").strip()

    for texto in (texto_libre, texto_original):
        palabra = extraer_palabra_clave(texto) or _extraer_keyword_instrumento(texto)
        if palabra:
            return palabra

    terminos = DOMINIO_TERMINOS_BUSQUEDA.get(dominio, [])
    return terminos[0] if terminos else ""


async def _resolver_otro_hibrida(
    necesidad_ctx: dict,
    mensaje: str,
    cliente: dict,
    productos_acumulados: list,
) -> tuple[str, str, dict]:
    """
    Tras elegir Otro en tipo de equipo, busca NIVEL_1 por similitud textual
    usando todo el contexto acumulado (aplicación, material, libros).
    """
    dominio = str(necesidad_ctx.get("dominio") or "").strip()
    texto_libre = str(mensaje or "").strip()
    respuestas = list(necesidad_ctx.get("respuestas_hibridas") or [])

    partes = [str(necesidad_ctx.get("texto_original") or "").strip(), texto_libre]
    for item in respuestas:
        partes.append(str(item.get("valor") or "").strip())
    texto_busqueda = " ".join(p for p in partes if p).strip()

    palabra = _palabra_clave_desde_ctx_hibrida(necesidad_ctx, mensaje)
    extractos = necesidad_ctx.get("extractos_libros") or []
    ctx_mat = _contexto_material_nivel(respuestas, texto_extra=texto_libre)

    if dominio == "nivel" and (
        ctx_mat.get("es_solido_polvo")
        or "radar" in _normalizar_intencion(texto_libre)
    ):
        alternativas = await obtener_tipos_radar_nivel(
            respuestas,
            texto_extra=texto_libre,
            extractos_libros=extractos,
        )
    else:
        if dominio == "nivel":
            palabra = "transmisor"
        alternativas = await obtener_tipos_nivel_1_por_texto(palabra, texto_busqueda, top=6)
        alternativas = filtrar_tipos_nivel_1_por_dominio(alternativas, dominio)
        alternativas = filtrar_tipos_nivel_1_por_tecnologia(
            alternativas,
            respuestas,
            extractos_libros=extractos,
            texto_extra=texto_libre,
        )[:3]

        if not alternativas:
            alternativas = await obtener_tipos_radar_nivel(
                respuestas,
                texto_extra=texto_libre,
                extractos_libros=extractos,
            )

    alternativas = alternativas[:3]

    if len(alternativas) == 1:
        return await _transferir_hibrida_a_nivel_1(
            necesidad_ctx=necesidad_ctx,
            nivel_1=str(alternativas[0]["nivel_1"]),
            respuestas_hibridas=respuestas,
            cliente=cliente,
            productos_acumulados=productos_acumulados,
        )

    if not alternativas:
        return await _transferir_hibrida_a_nivel_1(
            necesidad_ctx={**necesidad_ctx, "texto_otro": texto_libre},
            nivel_1=None,
            respuestas_hibridas=respuestas,
            cliente=cliente,
            productos_acumulados=productos_acumulados,
        )

    pregunta_data = generar_pregunta_seleccion_otro(
        "radar" if ctx_mat.get("es_solido_polvo") else palabra,
        alternativas,
        texto_libre,
    )
    ctx = {
        **necesidad_ctx,
        "flujo_descubrimiento": "corta_larga",
        "palabra_clave": palabra,
        "fase_descubrimiento": "seleccion_tipo_otro",
        "texto_otro": texto_libre,
        "alternativas_otro": alternativas,
        "respuestas_hibridas_previas": respuestas,
        "opciones_actuales": pregunta_data.get("opciones") or [],
    }

    return (
        _respuesta_pregunta_unica(
            cliente,
            pregunta_data["texto"],
            "Encontré opciones cercanas en catálogo:",
        ),
        "descubrimiento",
        ctx,
    )


async def _transferir_hibrida_a_nivel_1(
    necesidad_ctx: dict,
    nivel_1: str,
    respuestas_hibridas: list,
    cliente: dict,
    productos_acumulados: list,
) -> tuple[str, str, dict]:
    """
    Al elegir familia NIVEL_1 en híbrida guiada, continúa con el flujo
    corta_larga probado (preguntas técnicas del catálogo → código).
    """
    nivel_1 = await afinar_nivel_1_para_contexto(
        nivel_1,
        respuestas_hibridas,
        extractos_libros=necesidad_ctx.get("extractos_libros"),
    )

    texto_original = necesidad_ctx.get("texto_original") or ""
    palabra = (
        extraer_palabra_clave(texto_original)
        or _extraer_keyword_instrumento(texto_original)
        or necesidad_ctx.get("dominio")
        or ""
    )

    ctx_base = {
        k: v
        for k, v in necesidad_ctx.items()
        if k
        not in {
            "pregunta_actual",
            "candidatos_codigos",
            "preguntas_realizadas",
            "campos_usados",
            "max_preguntas_hibridas",
            "opciones_actuales",
        }
    }
    ctx_base.update(
        {
            "flujo_descubrimiento": "corta_larga",
            "modo_busqueda": "producto",
            "palabra_clave": palabra,
            "dominio": necesidad_ctx.get("dominio"),
            "extractos_libros": necesidad_ctx.get("extractos_libros") or [],
            "respuestas_hibridas_previas": respuestas_hibridas,
            "nivel_1_seleccionado": nivel_1,
            "tipo_corta_seleccionado": nivel_1,
        }
    )

    return await _iniciar_preguntas_tecnicas_por_nivel_1(
        necesidad_ctx=ctx_base,
        nivel_1=nivel_1,
        cliente=cliente,
        productos_acumulados=productos_acumulados,
        intro="Perfecto. Para ubicar la referencia exacta:",
    )


async def _iniciar_hibrida_guiada(
    mensaje: str,
    cliente: dict,
    productos_acumulados: Optional[list] = None,
) -> tuple[str, str, dict]:
    """
    Inicia modo 3 guiado: pool de candidatos + primera pregunta (libros + catálogo).
    """
    dominio, codigos, extractos = await obtener_pool_inicial(mensaje)
    productos = await cargar_productos_por_codigos(codigos)

    pregunta = generar_pregunta_aplicacion(dominio)
    if not pregunta:
        pregunta = await generar_siguiente_pregunta_hibrida(
            dominio=dominio,
            productos=productos,
            respuestas_previas=[],
            campos_usados=set(),
        )

    if productos and not pregunta:
        query = construir_query_hibrido(mensaje, [])
        ok, producto = seleccionar_producto_final(productos, query, dominio=dominio)
        if ok and producto:
            return await _finalizar_hibrida_con_producto(
                producto, {}, cliente, productos_acumulados or []
            )

    ctx = {
        "texto_original": mensaje,
        "modo_busqueda": "hibrida_guiada",
        "flujo_descubrimiento": "hibrida_libros",
        "fase_descubrimiento": "preguntas_hibridas",
        "dominio": dominio,
        "candidatos_codigos": [p.get("codigo") for p in productos if p.get("codigo")],
        "respuestas_hibridas": [],
        "campos_usados": [],
        "preguntas_realizadas": 0,
        "max_preguntas_hibridas": MAX_PREGUNTAS_HIBRIDAS,
        "pregunta_actual": pregunta,
        "extractos_libros": extractos,
        "opciones_actuales": (pregunta or {}).get("opciones") or [],
    }

    intro = None

    return (
        _respuesta_pregunta_unica(cliente, pregunta["texto"], intro),
        "descubrimiento",
        ctx,
    )


async def _continuar_hibrida_guiada(
    necesidad_ctx: dict,
    mensaje: str,
    cliente: dict,
    productos_acumulados: list,
) -> tuple[str, str, dict]:
    fase = necesidad_ctx.get("fase_descubrimiento")

    if fase == "esperando_otro_hibrida":
        return await _resolver_otro_hibrida(
            necesidad_ctx=necesidad_ctx,
            mensaje=mensaje,
            cliente=cliente,
            productos_acumulados=productos_acumulados,
        )

    pregunta_actual = necesidad_ctx.get("pregunta_actual") or {}

    if fase == "esperando_texto_hibrida":
        campo = necesidad_ctx.get("campo_texto_pendiente") or "fluido"
        respuesta_valor = str(mensaje or "").strip()
        respuesta_filtro = (
            _inferir_clave_material(respuesta_valor)
            if campo == "fluido"
            else respuesta_valor
        )
    else:
        tipo_resp, valor = resolver_respuesta_hibrida(mensaje, pregunta_actual)

        if tipo_resp == "otro":
            campo_otro = pregunta_actual.get("campo") or ""
            if campo_otro in {"fluido", "aplicacion"}:
                ctx = {
                    **necesidad_ctx,
                    "fase_descubrimiento": "esperando_texto_hibrida",
                    "campo_texto_pendiente": campo_otro,
                    "opciones_actuales": [],
                }
                return (
                    _respuesta_pregunta_unica(
                        cliente,
                        "Descríbelo con tus palabras para seguir filtrando en catálogo.",
                        "Entendido.",
                    ),
                    "descubrimiento",
                    ctx,
                )

            ctx = {
                **necesidad_ctx,
                "fase_descubrimiento": "esperando_otro_hibrida",
                "opciones_actuales": [],
            }
            return (
                _respuesta_pregunta_unica(
                    cliente,
                    "Descríbelo con tus palabras para seguir filtrando en catálogo.",
                    "Entendido.",
                ),
                "descubrimiento",
                ctx,
            )

        respuesta_valor = valor if tipo_resp in {"opcion", "texto_libre"} else str(mensaje or "").strip()
        campo = pregunta_actual.get("campo")
        mapa = pregunta_actual.get("mapa_valores") or {}
        respuesta_filtro = mapa.get(respuesta_valor, respuesta_valor)

    codigos = list(necesidad_ctx.get("candidatos_codigos") or [])
    productos = await cargar_productos_por_codigos(codigos)

    if not productos:
        dominio = necesidad_ctx.get("dominio") or "general"
        texto_base = necesidad_ctx.get("texto_original") or ""
        nuevos = await buscar_pool_por_dominio(
            dominio,
            f"{texto_base} {respuesta_valor}",
        )
        codigos = nuevos
        productos = await cargar_productos_por_codigos(codigos)
    else:
        productos = filtrar_productos_por_respuesta(
            productos, respuesta_filtro, campo=campo
        )
        codigos = [p.get("codigo") for p in productos if p.get("codigo")]

    dominio = necesidad_ctx.get("dominio") or "general"
    extractos = necesidad_ctx.get("extractos_libros") or []

    respuestas = list(necesidad_ctx.get("respuestas_hibridas") or [])
    respuestas.append(
        {
            "campo": campo,
            "valor": respuesta_valor,
            "clave": respuesta_filtro,
            "pregunta": pregunta_actual.get("texto"),
        }
    )

    productos = filtrar_productos_por_dominio(productos, dominio)
    productos = filtrar_productos_por_tecnologia_material(
        productos,
        respuestas,
        extractos_libros=extractos,
        texto_extra=respuesta_valor,
    )
    codigos = [p.get("codigo") for p in productos if p.get("codigo")]

    if campo == "nivel_1":
        nivel_afinado = await afinar_nivel_1_para_contexto(
            str(respuesta_filtro or respuesta_valor or "").strip(),
            respuestas,
            extractos_libros=extractos,
        )
        return await _transferir_hibrida_a_nivel_1(
            necesidad_ctx=necesidad_ctx,
            nivel_1=nivel_afinado or str(respuesta_filtro or respuesta_valor or "").strip(),
            respuestas_hibridas=respuestas,
            cliente=cliente,
            productos_acumulados=productos_acumulados,
        )

    campos_usados = set(necesidad_ctx.get("campos_usados") or [])
    if campo:
        campos_usados.add(campo)

    preguntas_realizadas = int(necesidad_ctx.get("preguntas_realizadas") or 0) + 1
    query = construir_query_hibrido(necesidad_ctx.get("texto_original") or "", respuestas)

    if len(productos) == 1:
        return await _finalizar_hibrida_con_producto(
            productos[0], necesidad_ctx, cliente, productos_acumulados
        )

    ok, producto = seleccionar_producto_final(
        productos,
        query,
        dominio=dominio,
        respuestas_previas=respuestas,
        extractos_libros=extractos,
    )
    if ok and producto and preguntas_realizadas >= 3:
        return await _finalizar_hibrida_con_producto(
            producto, necesidad_ctx, cliente, productos_acumulados
        )

    if preguntas_realizadas >= MAX_PREGUNTAS_HIBRIDAS:
        if producto:
            return await _finalizar_hibrida_con_producto(
                producto, necesidad_ctx, cliente, productos_acumulados
            )
        mejor = productos[0] if productos else None
        if mejor:
            res = {
                "estado": "relacionado",
                "producto": mejor,
                "razon": "Tras 5 preguntas, este es el candidato más cercano en catálogo.",
                "pregunta_sugerida": "¿Este equipo cumple con lo que necesitas?",
            }
            return construir_respuesta_desde_resultado(
                res=res,
                cliente=cliente,
                productos_acumulados=productos_acumulados,
                desde="hibrida_libros",
                necesidad_ctx_base={**necesidad_ctx, "opciones_actuales": []},
            )

    codigos_filtrados = [p.get("codigo") for p in productos if p.get("codigo")]
    dominio = necesidad_ctx.get("dominio") or "general"

    if campo == "fluido" and dominio:
        query_ctx = construir_query_hibrido(
            necesidad_ctx.get("texto_original") or "",
            respuestas,
        )
        nuevos = await buscar_pool_por_dominio(dominio, query_ctx)
        if nuevos:
            productos_ref = await cargar_productos_por_codigos(nuevos)
            productos_ref = filtrar_productos_por_dominio(productos_ref, dominio)
            if productos_ref:
                productos = productos_ref
                codigos_filtrados = [p.get("codigo") for p in productos if p.get("codigo")]

    siguiente = await generar_siguiente_pregunta_hibrida(
        dominio=dominio,
        productos=productos,
        respuestas_previas=respuestas,
        campos_usados=campos_usados,
        extractos_libros=extractos,
    )

    if not siguiente:
        if producto:
            return await _finalizar_hibrida_con_producto(
                producto, necesidad_ctx, cliente, productos_acumulados
            )
        return (
            _respuesta_pregunta_unica(
                cliente,
                "¿Puedes indicar rango, conexión o una especificación técnica clave?",
                None,
            ),
            "descubrimiento",
            {
                **necesidad_ctx,
                "candidatos_codigos": codigos_filtrados,
                "respuestas_hibridas": respuestas,
                "campos_usados": list(campos_usados),
                "preguntas_realizadas": preguntas_realizadas,
                "fase_descubrimiento": "esperando_otro_hibrida",
                "opciones_actuales": [],
            },
        )

    ctx = {
        **necesidad_ctx,
        "candidatos_codigos": codigos_filtrados,
        "respuestas_hibridas": respuestas,
        "campos_usados": list(campos_usados),
        "preguntas_realizadas": preguntas_realizadas,
        "pregunta_actual": siguiente,
        "fase_descubrimiento": "preguntas_hibridas",
        "opciones_actuales": siguiente.get("opciones") or [],
    }

    return (
        _respuesta_pregunta_unica(
            cliente,
            siguiente["texto"],
            None,
        ),
        "descubrimiento",
        ctx,
    )


async def _try_resolver_turno_hibrida(
    mensaje: str,
    necesidad_ctx: dict,
    cliente: dict,
    productos_acumulados: list,
) -> Optional[tuple[str, str, dict]]:
    if not (mensaje or "").strip():
        return None
    if not _en_flujo_hibrida(necesidad_ctx):
        return None

    logger.info(
        "Flujo híbrida activo: fase=%s preguntas=%s mensaje='%s'",
        necesidad_ctx.get("fase_descubrimiento"),
        necesidad_ctx.get("preguntas_realizadas"),
        mensaje[:80],
    )

    return await _continuar_hibrida_guiada(
        necesidad_ctx=necesidad_ctx,
        mensaje=mensaje,
        cliente=cliente,
        productos_acumulados=productos_acumulados,
    )


async def _buscar_y_responder_hibrido(
    mensaje: str,
    cliente: dict,
    productos_acumulados: list,
) -> tuple[str, str, dict]:
    """
    Modo 3 — búsqueda técnica/híbrida.

    Extrae campos técnicos del mensaje, infiere NIVEL_1 si hay instrumento
    y busca en catálogo combinando texto + especificaciones.
    """
    campos = extraer_campos_tecnicos(mensaje)
    palabra = extraer_palabra_clave(mensaje) or _extraer_keyword_instrumento(mensaje)
    nivel_1 = None

    if palabra:
        tipos = await obtener_tipos_nivel_1_por_texto(palabra, mensaje, top=1)
        if tipos:
            nivel_1 = tipos[0]["nivel_1"]

    respuestas_tecnicas = [
        str(valor).strip()
        for valor in campos.values()
        if str(valor).strip()
    ]

    logger.info(
        "Búsqueda híbrida: palabra=%s nivel_1=%s campos=%s mensaje='%s'",
        palabra,
        nivel_1,
        campos,
        mensaje[:100],
    )

    res = await buscar_con_descubrimiento_producto(
        palabra_clave=palabra or "",
        nivel_1=nivel_1,
        respuestas_tecnicas=respuestas_tecnicas,
        texto_original=mensaje,
    )

    ctx = {
        "texto_original": mensaje,
        "modo_busqueda": "hibrida",
        "palabra_clave": palabra,
        "campos_tecnicos": campos,
        "nivel_1_inferido": nivel_1,
        "respuestas_tecnicas": respuestas_tecnicas,
    }

    if res.get("estado") == "sin_resultado":
        ctx["busqueda_sin_resultado"] = True

    return construir_respuesta_desde_resultado(
        res=res,
        cliente=cliente,
        productos_acumulados=productos_acumulados,
        desde="busqueda_hibrida",
        necesidad_ctx_base=ctx,
    )


async def _buscar_y_responder_descubrimiento(
    query_e: str,
    necesidad_ctx: dict,
    cliente: dict,
    productos_acumulados: list,
    desde: str,
) -> tuple[str, str, dict]:
    """
    Ejecuta búsqueda en catálogo y conserva contexto si no hay match.
    """
    res = await buscar_en_catalogo(query_e)

    if debe_intentar_enriquecimiento(res):
        res = await enriquecer_y_buscar(query_e, necesidad_ctx=necesidad_ctx)

    ctx_base = _ctx_descubrimiento_base(necesidad_ctx, query_e)

    if res.get("estado") == "sin_resultado":
        ctx_base["busqueda_sin_resultado"] = True

    return construir_respuesta_desde_resultado(
        res=res,
        cliente=cliente,
        productos_acumulados=productos_acumulados,
        desde=desde,
        necesidad_ctx_base=ctx_base,
    )


def construir_respuesta_desde_resultado(
    res: dict,
    cliente: dict,
    productos_acumulados: list,
    desde: str,
    necesidad_ctx_base: Optional[dict] = None,
) -> tuple[str, str, dict]:
    """
    Convierte un resultado de catálogo en:
    - contexto_extra
    - nueva_etapa
    - necesidad_ctx actualizado

    Reglas:
    - encontrado: se agrega al carrito.
    - relacionado: no se agrega al carrito; se pide confirmación.
    - sin_resultado/pendiente: se mantiene descubrimiento.
    """
    necesidad_ctx_base = necesidad_ctx_base or {}

    estado = res.get("estado")

    if estado == "encontrado" and res.get("producto"):
        producto = res["producto"]

        productos_acumulados.append({
            "producto": producto,
            "cantidad": None,
            "desde": desde,
            "ts": datetime.utcnow().isoformat(),
            "contexto_aprendizaje": construir_contexto_aprendizaje_desde_necesidad(
                necesidad_ctx_base
            ),
            "feedback": None,
        })

        nota = necesidad_ctx_base.get("nota_tecnica") or ""

        return (
            _marcar_respuesta_segura(
                respuesta_producto_encontrado(
                    producto,
                    cliente,
                    nota_tecnica=nota,
                )
            ),
            "producto_encontrado",
            _ctx_confirmacion_producto(),
        )

    if estado == "necesita_marca":
        referencia = res.get("referencia_buscada") or ""
        candidatos = list(res.get("candidatos") or [])
        match_campo = res.get("match_campo") or "REFERENCIA"
        opciones = _opciones_marcas_referencia(candidatos)

        return (
            _marcar_respuesta_segura(
                _respuesta_pedir_marca_referencia(referencia, candidatos, match_campo)
            ),
            "esperando_marca_referencia",
            {
                **necesidad_ctx_base,
                "referencia_pendiente": referencia,
                "candidatos_referencia": candidatos,
                "match_campo_referencia": match_campo,
                "opciones_actuales": opciones,
            },
        )

    if estado == "multiples_candidatos" and res.get("candidatos"):
        candidatos = list(res.get("candidatos") or [])
        opciones = _opciones_candidatos_producto(candidatos)
        return (
            _marcar_respuesta_segura(
                _respuesta_multiples_candidatos(candidatos, cliente)
            ),
            "descubrimiento",
            {
                **necesidad_ctx_base,
                "fase_descubrimiento": "seleccion_producto_descubrimiento",
                "flujo_descubrimiento": necesidad_ctx_base.get(
                    "flujo_descubrimiento", "corta_larga"
                ),
                "candidatos_producto": candidatos,
                "opciones_actuales": opciones,
                "query_evaluada": res.get("query_catalogo")
                or necesidad_ctx_base.get("query_evaluada"),
            },
        )

    if estado == "relacionado" and res.get("producto"):
        producto = res["producto"]

        # Esta función es síncrona, por eso aquí NO usamos await.
        # Si existen preguntas técnicas, deben venir preparadas desde
        # buscar_en_catalogo(), que sí es async.
        preguntas_tecnicas = res.get("preguntas_tecnicas") or []

        respuesta_base = respuesta_producto_relacionado(
            producto=producto,
            razon=res.get("razon"),
            pregunta_sugerida=res.get("pregunta_sugerida"),
            cliente=cliente,
        )

        preguntas_limpias = _normalizar_preguntas(preguntas_tecnicas)

        if preguntas_limpias:
            respuesta = (
                f"{respuesta_base}\n\n"
                "Para validar mejor la solución, necesito confirmar:\n"
                f"{preguntas_limpias[0]}"
            )
        else:
            respuesta = respuesta_base

        necesidad_ctx = {
            **necesidad_ctx_base,
            "producto_relacionado": producto,
            "pregunta_sugerida": res.get("pregunta_sugerida"),
            "razon": res.get("razon"),
            "preguntas_tecnicas": preguntas_limpias,
            "preguntas_pendientes": preguntas_limpias,
            "pregunta_indice": 0,
            "respuestas_tecnicas": [],
        }

        return (
            _marcar_respuesta_segura(respuesta),
            "validando_relacionado",
            necesidad_ctx,
        )

    if estado == "pendiente":
        preguntas = res.get("preguntas", []) or []
        preguntas = [PREGUNTA_INICIAL_NECESIDAD] + [
            p for p in preguntas if _texto_pregunta(p) != PREGUNTA_INICIAL_NECESIDAD
        ][:2]
        return _iniciar_secuencia_preguntas(
            {
                **necesidad_ctx_base,
                "dominio": res.get("dominio", necesidad_ctx_base.get("dominio", "general")),
                "texto_original": (
                    necesidad_ctx_base.get("texto_original")
                    or necesidad_ctx_base.get("query_evaluada")
                ),
            },
            preguntas,
            cliente,
            "",
        )

    return (
        _marcar_respuesta_segura(
            respuesta_sin_resultado(
                pregunta_sugerida=res.get("pregunta_sugerida"),
                cliente=cliente,
            )
        ),
        "descubrimiento",
        {
            **necesidad_ctx_base,
            "busqueda_sin_resultado": True,
            "preguntas_pendientes": [],
            "pregunta_indice": 0,
        },
    )


def _extraer_respuesta_segura(contexto_extra: str) -> Optional[str]:
    """
    Extrae respuesta segura marcada.
    """
    if contexto_extra.startswith("[RESPUESTA_SEGURA]"):
        return contexto_extra.replace("[RESPUESTA_SEGURA]\n", "", 1).strip()

    return None

# ─────────────────────────────────────────────────────────────
# Estado comercial prioritario
# ─────────────────────────────────────────────────────────────

def _normalizar_intencion(texto: str) -> str:
    """
    Normaliza texto corto para interpretar confirmaciones,
    cierres y respuestas comerciales simples.

    No se usa para catálogo. Solo para control de flujo.
    """
    t = (texto or "").lower().strip()
    reemplazos = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ñ": "n",
    }

    for origen, destino in reemplazos.items():
        t = t.replace(origen, destino)

    t = re.sub(r"\s+", " ", t)
    return t.strip(" .,!¡¿?")


def _es_confirmacion_afirmativa(texto: str) -> bool:
    """
    Detecta respuestas afirmativas del cliente.

    Aplica para confirmar producto sugerido, no para buscar catálogo.
    """
    t = _normalizar_intencion(texto)

    afirmaciones_exactas = {
        "si",
        "correcto",
        "ese",
        "esa",
        "ese me sirve",
        "esa me sirve",
        "me sirve",
        "sirve",
        "ok",
        "dale",
        "perfecto",
        "confirmo",
        "confirmado",
    }

    return t in afirmaciones_exactas


def _es_confirmacion_negativa(texto: str) -> bool:
    """
    Detecta rechazo del producto sugerido.
    """
    t = _normalizar_intencion(texto)

    negativas_exactas = {
        "no",
        "no me sirve",
        "no es",
        "no corresponde",
        "otro",
        "otra",
        "diferente",
    }

    return t in negativas_exactas


def _extraer_cantidad_solicitada(texto: str) -> Optional[int]:
    """
    Extrae una cantidad comercial cuando NIA está esperando cantidad.

    Regla:
    - Acepta cantidades razonables de 1 a 5 dígitos.
    - Acepta sufijos: und, unds, uds, unidades, piezas, u.
    - Acepta puntuación final: "2,", "2 und.", "2 unidades,".
    - No interpreta números largos como cantidad para evitar confundir NIT,
      teléfonos o códigos.
    """
    if not texto:
        return None

    t = texto.lower().strip()
    t = re.sub(r"[.,;:\s]+$", "", t)

    m_solo = CANTIDAD_MENSAJE_SOLO_RE.match(t)
    if m_solo:
        try:
            cantidad = int(m_solo.group(1))
        except ValueError:
            return None
        return cantidad if cantidad > 0 else None

    m = re.search(
        r"(?<!\d)(\d{1,5})\s*(?:und|unds|uds?|unidad(?:es)?|pieza(?:s)?|u\.?)?(?!\d)",
        t,
        re.IGNORECASE,
    )

    if not m:
        return None

    try:
        cantidad = int(m.group(1))
    except ValueError:
        return None

    if cantidad <= 0:
        return None

    return cantidad


def _asignar_cantidad_ultimo_producto(productos_acumulados: list, cantidad: int) -> None:
    """
    Asigna la cantidad al último producto acumulado que aún no tenga cantidad.
    Si todos tienen cantidad, actualiza el último producto como decisión comercial.
    """
    if not productos_acumulados:
        return

    for item in reversed(productos_acumulados):
        if not item.get("cantidad"):
            item["cantidad"] = cantidad
            return

    productos_acumulados[-1]["cantidad"] = cantidad


_ACCIONES_UI_NO_NOMBRE = {
    "cotiza",
    "cotizar",
    "cotizar con esto",
    "con esto",
    "agregar",
    "agregar otro producto",
    "agregar_otro",
    "otro producto",
}


def _es_nombre_control_invalido(valor: Optional[str]) -> bool:
    """
    Rechaza valores que representan acciones de interfaz o comandos
    conversacionales, aunque un parser intermedio cambie mayúsculas,
    elimine conectores o reemplace guiones bajos.

    La regla se basa en intención y en la raíz del primer verbo,
    no en una única frase literal.
    """
    normalizado = _normalizar_intencion(
        str(valor or "")
    ).replace("_", " ").strip()

    if not normalizado:
        return False

    if normalizado in _ACCIONES_UI_NO_NOMBRE:
        return True

    tokens = re.findall(r"[a-z0-9]+", normalizado)

    if not tokens:
        return False

    primer_token = tokens[0]

    prefijos_accion = (
        "agreg",
        "anad",
        "cotiz",
        "compr",
        "busc",
        "consult",
        "continu",
        "confirm",
        "cancel",
        "finaliz",
        "termin",
        "volv",
        "envi",
        "seleccion",
        "escog",
    )

    return any(
        primer_token.startswith(prefijo)
        for prefijo in prefijos_accion
    )


def _sanitizar_cliente_control(cliente: dict) -> dict:
    """Elimina nombres contaminados por acciones de interfaz."""
    limpio = dict(cliente or {})

    if _es_nombre_control_invalido(limpio.get("nombre")):
        limpio.pop("nombre", None)

    return limpio

def _parece_nombre_simple(texto: str) -> Optional[str]:
    """
    Captura nombres escritos de forma directa, por ejemplo:
    - Luis
    - Luis Díaz
    - Juan Carlos Pérez

    No captura correos, números, NIT, frases de cierre ni solicitudes de producto.
    """
    if not texto:
        return None

    limpio = texto.strip(" ,.;:-")

    if not limpio or EMAIL_RE.search(limpio) or NIT_RE.search(limpio):
        return None

    t = _normalizar_intencion(limpio)

    if _es_nombre_control_invalido(limpio):
        return None

    bloqueados = PALABRAS_SALUDO | PALABRAS_FIN | {
        "si",
        "no",
        "ok",
        "dale",
        "correcto",
        "solo",
        "eso",
        "solo eso",
        "cotiza",
    }

    if t in bloqueados:
        return None

    if _parece_solicitud_de_producto(limpio):
        return None

    partes = limpio.split()

    if not (1 <= len(partes) <= 4):
        return None

    patron_nombre = re.compile(r"^[A-Za-zÁÉÍÓÚáéíóúÑñüÜ'-]{2,}$")

    if not all(patron_nombre.match(p) for p in partes):
        return None

    return " ".join(p.capitalize() for p in partes)


def _parece_empresa_simple(texto: str) -> Optional[str]:
    """
    Captura razón social escrita de forma directa cuando NIA está en etapa proforma.

    Ejemplos:
    - ViaIndustrial SAS
    - Equipos Industriales Fenix S.A.S
    - Industrias ABC
    """
    if not texto:
        return None

    limpio = texto.strip(" ,.;:-")

    if not limpio or EMAIL_RE.search(limpio) or NIT_RE.search(limpio):
        return None

    if _es_confirmacion_afirmativa(limpio) or _es_confirmacion_negativa(limpio):
        return None

    if _parece_solicitud_de_producto(limpio):
        return None

    if len(limpio) < 3 or len(limpio) > 80:
        return None

    if not re.search(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]", limpio):
        return None

    return limpio

def _extraer_datos_contacto_desde_mensaje(mensaje: str) -> dict:

    datos = {}
    if not mensaje:
        return datos

    texto_original = str(mensaje).strip()

    # ------------------------------------------------------------
    # 1. Extraer email si existe
    # ------------------------------------------------------------
    match_email = re.search(
        r"[\w\.-]+@[\w\.-]+\.\w+",
        texto_original,
        flags=re.IGNORECASE,
    )

    if match_email:
        datos["email"] = match_email.group(0).strip().lower()

    # ------------------------------------------------------------
    # 2. Construir candidato de nombre quitando email y frases comunes
    # ------------------------------------------------------------
    texto_nombre = texto_original

    if match_email:
        texto_nombre = texto_nombre.replace(match_email.group(0), " ")

    patrones_limpieza = [
        r"\bmi\s+nombre\s+es\b",
        r"\bnombre\s+es\b",
        r"\bme\s+llamo\b",
        r"\bsoy\b",
        r"\bmi\s+correo\s+es\b",
        r"\bcorreo\s+es\b",
        r"\bcorreo\s+electr[oó]nico\s+es\b",
        r"\bcorreo\s+electr[oó]nico\b",
        r"\bcorreo\b",
        r"\bemail\s+es\b",
        r"\bemail\b",
        r"\be-mail\s+es\b",
        r"\be-mail\b",
    ]

    for patron in patrones_limpieza:
        texto_nombre = re.sub(
            patron,
            " ",
            texto_nombre,
            flags=re.IGNORECASE,
        )

    # Quitar conectores sueltos que suelen quedar después de remover el correo.
    texto_nombre = re.sub(
        r"\b(y|con|para|al|a)\b",
        " ",
        texto_nombre,
        flags=re.IGNORECASE,
    )

    # Limpiar puntuación y espacios.
    texto_nombre = texto_nombre.strip(" ,.;:-")
    texto_nombre = re.sub(r"\s+", " ", texto_nombre).strip()

    # ------------------------------------------------------------
    # 3. Validar si lo restante parece nombre
    # ------------------------------------------------------------
    nombre = _parece_nombre_simple(texto_nombre)

    if nombre:
        datos["nombre"] = nombre

    return datos

def _extraer_rut_desde_mensaje(mensaje: str) -> Optional[str]:
    """
    Detecta si el cliente está compartiendo el RUT.

    El RUT es documento/soporte tributario, no debe bloquear el flujo.
    Si viene número, lo guardamos. Si solo dice que lo comparte, guardamos 'recibido'.
    """
    texto = (mensaje or "").strip()
    if not texto:
        return None

    texto_norm = _normalizar_intencion(texto)

    if "rut" not in texto_norm:
        return None

    nit_match = NIT_RE.search(texto)
    if nit_match:
        return nit_match.group(1).strip()

    if any(
        frase in texto_norm
        for frase in {
            "rut",
            "te comparto el rut",
            "envio el rut",
            "envie el rut",
            "adjunto el rut",
            "rut adjunto",
            "ya comparti el rut",
            "ya envie el rut",
        }
    ):
        return "recibido"

    return None

def _capturar_dato_comercial_por_etapa(mensaje: str, cliente: dict, etapa: str) -> dict:
    """
    Completa datos comerciales según el estado actual de la conversación.

    Esta función evita que datos como nombre, empresa o NIT sean tratados
    como búsqueda de producto cuando NIA está cerrando una cotización.

    Regla Cotización IV:
    Si NIA pide nombre y correo en un mismo turno, el backend debe poder
    capturar ambos datos correctamente desde una respuesta natural.
    """
    cliente = dict(cliente or {})
    mensaje = (mensaje or "").strip()

    if etapa in {"cotizacion", "calificacion", "confirmando_cierre"}:
        datos_contacto = _extraer_datos_contacto_desde_mensaje(mensaje)

        if not cliente.get("email") and datos_contacto.get("email"):
            cliente["email"] = datos_contacto["email"]
            logger.debug("Email capturado por parser de contacto: %s", cliente["email"])

        nombre_contacto = datos_contacto.get("nombre")

        if (
            not cliente.get("nombre")
            and nombre_contacto
            and not _es_nombre_control_invalido(nombre_contacto)
        ):
            cliente["nombre"] = nombre_contacto
            logger.debug("Nombre capturado por parser de contacto: %s", cliente["nombre"])

        # Fallback para el caso simple:

        if not cliente.get("nombre"):
            nombre = _parece_nombre_simple(mensaje)
            if nombre:
                cliente["nombre"] = nombre
                logger.debug("Nombre simple capturado por etapa: %s", nombre)

    if etapa in {"proforma", "proforma_lista"}:
        if not cliente.get("empresa"):
            empresa = _parece_empresa_simple(mensaje)
            if empresa:
                cliente["empresa"] = empresa
                logger.debug("Empresa simple capturada por etapa: %s", empresa)

        if not cliente.get("nit"):
            nit_match = NIT_RE.search(mensaje)
            if nit_match:
                cliente["nit"] = nit_match.group(1).strip()
                logger.debug("NIT capturado por etapa proforma: %s", cliente["nit"])

        rut = _extraer_rut_desde_mensaje(mensaje)
        if rut and cliente.get("rut") in {None, "", "pendiente", "pendiente_solicitado"}:
            cliente["rut"] = rut
            logger.debug("RUT capturado/actualizado por parser de RUT: %s", rut)
        elif not cliente.get("rut") or cliente.get("rut") in {
            "pendiente",
            "pendiente_solicitado",
        }:
            # Si solo adjunta/menciona documento fiscal sin la palabra RUT.
            texto_norm = _normalizar_intencion(mensaje)
            if any(
                f in texto_norm
                for f in {
                    "documento fiscal",
                    "rut adjunto",
                    "adjunto el rut",
                    "te envio el rut",
                    "envio el rut",
                    "archivo rut",
                }
            ):
                cliente["rut"] = "recibido"

    # Última barrera: elimina acciones de interfaz que hayan entrado por cualquier parser.
    return _sanitizar_cliente_control(cliente)

def _respuesta_siguiente_dato_comercial(
    cliente: dict,
    etapa_objetivo: str = "cotizacion",
) -> tuple[str, str]:
    """
    Decide el siguiente dato comercial que NIA debe pedir.

    Regla comercial actualizada:
    - Para cotización SOLO se pide nombre y correo.
    - Después del correo, NIA deja la solicitud lista para asesor/vendedor.
    - NIA NO pide razón social, NIT ni RUT en cotización.
    - Razón social, NIT y RUT solo se piden si la etapa objetivo es proforma.

    Esto implementa la barrera solicitada:
    cotización enviada/aprobada primero, proforma después.
    """

    nombre = (cliente.get("nombre") or "").strip()

    # ============================================================
    # ETAPA COTIZACIÓN
    # ============================================================
    if etapa_objetivo in {"cotizacion", "calificacion", "confirmando_cierre"}:
        if not cliente.get("nombre"):
            return "¿A nombre de quién va la cotización?", "cotizacion"

        if not cliente.get("email"):
            return (
                f"Gracias, {nombre}. ¿Cuál es el correo electrónico para enviar la cotización?",
                "cotizacion",
            )

        # Punto final de la etapa de cotización.
        # No pedimos empresa, NIT ni RUT aquí.
        return (
            "Registré tu solicitud con los productos y cantidades "
            "indicados. La cotización llegará lo más pronto posible.",
            "cotizacion_lista",
        )

    # ============================================================
    # ETAPA PROFORMA
    # ============================================================
    # Esta etapa solo debe activarse cuando exista una señal futura:
    # - vendedor confirmó que envió la cotización;
    # - cliente confirmó que cumple técnicamente.
    #==============================================================

    if etapa_objetivo == "proforma":
        if not cliente.get("empresa"):
            return (
                f"Perfecto, {nombre or 'cliente'}. Para preparar la proforma, "
                "¿cuál es la razón social de tu empresa?",
                "proforma",
            )

        if not cliente.get("nit"):
            return (
                "Gracias. ¿Cuál es el NIT o documento fiscal de la empresa? "
                "También envíame el RUT, por favor.",
                "proforma",
            )

        # ------------------------------------------------------------
        # RUT NO BLOQUEANTE
        # ------------------------------------------------------------
        # Ya se pidió junto con el NIT. Si no llegó, queda pendiente
        # sin frenar el avance a proforma_lista.
        if not cliente.get("rut"):
            cliente["rut"] = "pendiente"

        return (
            f"Perfecto, {nombre or 'cliente'}, ya tengo todos los datos. En breve recibirás la proforma.",
            "proforma_lista",
        )


    # Fallback seguro: si llega una etapa desconocida, no avanzar a proforma.
    return (
        "Perfecto, ya dejé la solicitud lista para que un asesor revise disponibilidad, precio y condiciones.",
        "cotizacion_lista",
    )


def _es_nueva_solicitud_durante_cierre(mensaje: str) -> bool:
    """
    Permite salir del flujo de cierre si el cliente realmente pide otro producto.

    Ejemplo:
    - también necesito una válvula
    - agrega otro sensor
    - necesito otro equipo
    """
    t = _normalizar_intencion(mensaje)

    if any(p in t for p in {"tambien necesito", "tambien quiero", "agrega", "agregar", "otro producto", "otra referencia"}):
        return True

    return _parece_solicitud_de_producto(mensaje)

# ─────────────────────────────────────────────────────────────
# Controlador determinístico de estado comercial
# ─────────────────────────────────────────────────────────────

ESTADOS_COMERCIALES = {
    "producto_encontrado",
    "esperando_cantidad",
    "confirmando_cierre",
    "cotizacion",
    "calificacion",
    "cotizacion_lista",
    "cotizacion_enviada",
    "proforma",
    "proforma_lista",
    "proforma_enviada",
    "pago",
    "pago_confirmado",
}

_CLAVES_CTX_DESCUBRIMIENTO = (
    "fase_descubrimiento",
    "flujo_descubrimiento",
    "preguntas_pendientes",
    "pregunta_actual",
    "alternativas_otro",
    "dominio",
    "esperando_otro_tecnico",
    "producto_candidato",
    "indice_pregunta",
    "texto_original",
    "palabra_clave",
    "nivel_actual",
    "opciones_actuales",
)


def _limpiar_ctx_para_cierre_comercial(necesidad_ctx: dict) -> dict:
    """
    Quita estado de descubrimiento/catálogo y deja la solicitud lista para asesor.
    """
    ctx = dict(necesidad_ctx or {})
    for clave in _CLAVES_CTX_DESCUBRIMIENTO:
        ctx.pop(clave, None)
    ctx["comercial_listo_asesor"] = True
    ctx["opciones_actuales"] = []
    return ctx

def _ultimo_turno_pide_datos_contacto(historial: list) -> bool:
    """
    Detecta si el último mensaje de NIA pidió datos básicos de contacto.

    Regla de negocio:
    Si NIA acaba de pedir nombre/correo, el siguiente mensaje del cliente
    debe ser tratado como dato comercial de cotización, no como búsqueda
    de catálogo.

    Esto protege el flujo cuando la etapa persistida queda inconsistente.
    """
    if not historial:
        return False

    # Buscar el último mensaje del asistente.
    ultimo_assistant = None

    for turno in reversed(historial):
        if turno.get("role") == "assistant":
            ultimo_assistant = turno.get("content", "")
            break

    if not ultimo_assistant:
        return False

    texto = _normalizar_intencion(ultimo_assistant)

    indicadores_contacto = [
        "nombre y correo",
        "nombre y correo electronico",
        "nombre y e mail",
        "a nombre de quien",
        "correo electronico",
        "correo para enviar la cotizacion",
        "dejar la solicitud lista",
        "datos basicos",
    ]

    return any(indicador in texto for indicador in indicadores_contacto)


def _ultimo_producto_pendiente_confirmacion(productos_acumulados: list) -> tuple[Optional[int], Optional[dict]]:
    """
    Último producto agregado al carrito que aún no tiene cantidad ni feedback.
    """
    for indice in range(len(productos_acumulados) - 1, -1, -1):
        item = productos_acumulados[indice]
        if not isinstance(item, dict):
            continue

        if item.get("feedback") in {"si", "no"}:
            continue

        if item.get("cantidad") is not None:
            continue

        return indice, item

    return None, None


def _preparar_feedback_aprendizaje(
    *,
    tipo: str,
    categoria: str,
    mensaje: str,
    productos_acumulados: list,
    historial: Optional[list],
    necesidad_ctx: Optional[dict] = None,
    producto_directo: Optional[dict] = None,
    contexto_extra: Optional[dict] = None,
    clave_aprendizaje: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Optional[dict]:
    """
    Arma evento de aprendizaje listo para persistir tras sí/no del cliente.
    """
    if not clave_aprendizaje:
        return None

    producto = producto_directo
    contexto = dict(contexto_extra or {})

    if not producto:
        _, item = _ultimo_producto_pendiente_confirmacion(productos_acumulados)
        if item:
            producto = item.get("producto")
            contexto = {
                **construir_contexto_aprendizaje_desde_necesidad(item),
                **contexto,
            }

    if necesidad_ctx and not contexto:
        contexto = construir_contexto_aprendizaje_desde_necesidad(necesidad_ctx)

    return construir_evento_feedback(
        clave_aprendizaje=clave_aprendizaje,
        tipo=tipo,
        categoria=categoria,
        mensaje_usuario=mensaje,
        session_id=session_id,
        producto=producto,
        contexto=contexto,
        historial=historial,
    )


def _contexto_cotizacion_aprendizaje(productos_acumulados: list) -> dict:
    """
    Resumen de productos en carrito para feedback de cotización.
    """
    items = []
    for item in productos_acumulados or []:
        if not isinstance(item, dict):
            continue

        producto = item.get("producto") or {}
        codigo = producto.get("codigo")
        if not codigo:
            continue

        items.append(
            {
                "codigo": codigo,
                "cantidad": item.get("cantidad"),
                "desde": item.get("desde"),
            }
        )

    return {"productos_carrito": items}


async def _persistir_aprendizaje_si_corresponde(evento: Optional[dict]):
    """
    Guarda feedback y refresca memoria activa del turno.
    """
    if not evento:
        return

    memoria = await registrar_feedback(evento)
    if memoria:
        activar_memoria_aprendizaje(memoria)


def _manejar_estado_comercial_prioritario(
    etapa: str,
    mensaje: str,
    cliente: dict,
    productos_acumulados: list,
    necesidad_ctx: dict,
    clasificacion: Optional[dict] = None,
    historial: Optional[list] = None,
    clave_aprendizaje: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Optional[dict]:
    """
    Controla estados comerciales de forma determinística.

    Regla central:
    Si esta función resuelve el turno, procesar_turno debe retornar
    inmediatamente sin buscar catálogo y sin llamar al LLM.

    Esto evita que:
    - una cantidad sea interpretada como código;
    - un NIT sea interpretado como producto;
    - un correo dispare búsqueda de catálogo;
    - el LLM cambie una etapa comercial ya decidida.
    """
    etapa = etapa or "inicio"
    mensaje = (mensaje or "").strip()
    cliente = dict(cliente or {})
    necesidad_ctx = dict(necesidad_ctx or {})
    productos_acumulados = productos_acumulados or []

    clasificacion = clasificacion or {}
    tipo_mensaje = clasificacion.get("tipo")

    if not mensaje or etapa not in ESTADOS_COMERCIALES:
        return None

    # ============================================================
    # Cotización recibida por el cliente
    # ============================================================
    if tipo_mensaje in {"cotizacion_recibida", "link_documento"} and etapa in {
        "cotizacion_lista",
        "cotizacion",
        "confirmando_cierre",
    }:
        necesidad_ctx["cotizacion_recibida"] = True

        if tipo_mensaje == "link_documento":
            necesidad_ctx["archivo_cotizacion"] = mensaje

        return {
            "handled": True,
            "respuesta": (
                "Perfecto, tomo esto como cotización recibida. "
                f"{PREGUNTA_COTIZACION_TECNICA}"
            ),
            "etapa": "cotizacion_enviada",
            "cliente": cliente,
            "necesidad_ctx": {
                **necesidad_ctx,
                **_ctx_confirmacion_cotizacion_tecnica(),
            },
            "productos_acumulados": productos_acumulados,
        }

    # ============================================================
    # Cliente valida cotización enviada
    # ============================================================
    if etapa == "cotizacion_enviada":
        if _es_confirmacion_afirmativa(mensaje):
            necesidad_ctx["cotizacion_aprobada_cliente"] = True
            # Limpia botones Sí/No; proforma pide datos, no confirmación.
            necesidad_ctx["opciones_actuales"] = []

            respuesta_dato, etapa_dato = _respuesta_siguiente_dato_comercial(
                cliente,
                etapa_objetivo="proforma",
            )

            return {
                "handled": True,
                "respuesta": respuesta_dato,
                "etapa": etapa_dato,
                "cliente": cliente,
                "necesidad_ctx": {
                    **necesidad_ctx,
                    "opciones_actuales": [],
                },
                "productos_acumulados": productos_acumulados,
                "aprendizaje": _preparar_feedback_aprendizaje(
                    tipo="si",
                    categoria="cotizacion",
                    mensaje=mensaje,
                    productos_acumulados=productos_acumulados,
                    historial=historial,
                    contexto_extra=_contexto_cotizacion_aprendizaje(productos_acumulados),
                    clave_aprendizaje=clave_aprendizaje,
                    session_id=session_id,
                ),
            }

        if _es_confirmacion_negativa(mensaje):
            necesidad_ctx["cotizacion_aprobada_cliente"] = False

            return {
                "handled": True,
                "respuesta": (
                    "Entiendo. Cuéntame qué ajuste necesitas en la cotización "
                    "o qué característica técnica no cumple."
                ),
                "etapa": "descubrimiento",
                "cliente": cliente,
                "necesidad_ctx": {
                    **necesidad_ctx,
                    "opciones_actuales": [],
                },
                "productos_acumulados": productos_acumulados,
                "aprendizaje": _preparar_feedback_aprendizaje(
                    tipo="no",
                    categoria="cotizacion",
                    mensaje=mensaje,
                    productos_acumulados=productos_acumulados,
                    historial=historial,
                    contexto_extra=_contexto_cotizacion_aprendizaje(productos_acumulados),
                    clave_aprendizaje=clave_aprendizaje,
                    session_id=session_id,
                ),
            }

        return {
            "handled": True,
            "respuesta": (
                "Para avanzar correctamente, necesito confirmar: "
                f"{PREGUNTA_COTIZACION_TECNICA}"
            ),
            "etapa": "cotizacion_enviada",
            "cliente": cliente,
            "necesidad_ctx": {
                **necesidad_ctx,
                **_ctx_confirmacion_cotizacion_tecnica(),
            },
            "productos_acumulados": productos_acumulados,
        }

    # ============================================================
    # Proforma recibida por el cliente
    # ============================================================
    if tipo_mensaje in {"proforma_recibida", "link_documento"} and etapa in {
        "proforma",
        "proforma_lista",
        "cotizacion_enviada",
    }:
        necesidad_ctx["proforma_recibida"] = True

        if tipo_mensaje == "link_documento":
            necesidad_ctx["archivo_proforma"] = mensaje

        return {
            "handled": True,
            "respuesta": (
                "Perfecto, tomo esto como proforma recibida. "
                "¿Deseas proceder con el pago?"
            ),
            "etapa": "proforma_enviada",
            "cliente": cliente,
            "necesidad_ctx": necesidad_ctx,
            "productos_acumulados": productos_acumulados,
        }

    # ============================================================
    # Cliente valida proforma para pago
    # ============================================================
    if etapa == "proforma_enviada":
        if _es_confirmacion_afirmativa(mensaje):
            return {
                "handled": True,
                "respuesta": (
                    "Perfecto. Puedes continuar con el pago por transferencia, PSE o tarjeta. "
                    "Un asesor confirmará el pago y coordinará el envío."
                ),
                "etapa": "pago",
                "cliente": cliente,
                "necesidad_ctx": _limpiar_ctx_para_cierre_comercial(necesidad_ctx),
                "productos_acumulados": productos_acumulados,
                "aprendizaje": _preparar_feedback_aprendizaje(
                    tipo="si",
                    categoria="proforma",
                    mensaje=mensaje,
                    productos_acumulados=productos_acumulados,
                    historial=historial,
                    contexto_extra=_contexto_cotizacion_aprendizaje(productos_acumulados),
                    clave_aprendizaje=clave_aprendizaje,
                    session_id=session_id,
                ),
            }

        if _es_confirmacion_negativa(mensaje):
            return {
                "handled": True,
                "respuesta": (
                    "Entiendo. Indícame qué ajuste necesitas en la proforma "
                    "para que el asesor pueda revisarlo."
                ),
                "etapa": "proforma",
                "cliente": cliente,
                "necesidad_ctx": necesidad_ctx,
                "productos_acumulados": productos_acumulados,
                "aprendizaje": _preparar_feedback_aprendizaje(
                    tipo="no",
                    categoria="proforma",
                    mensaje=mensaje,
                    productos_acumulados=productos_acumulados,
                    historial=historial,
                    contexto_extra=_contexto_cotizacion_aprendizaje(productos_acumulados),
                    clave_aprendizaje=clave_aprendizaje,
                    session_id=session_id,
                ),
            }

        return {
            "handled": True,
            "respuesta": "¿Deseas proceder con el pago?",
            "etapa": "proforma_enviada",
            "cliente": cliente,
            "necesidad_ctx": {
                **necesidad_ctx,
                **_ctx_confirmacion_producto(),
            },
            "productos_acumulados": productos_acumulados,
        }

    # ============================================================
    # Proforma lista, esperando confirmación externa
    # ============================================================
    if etapa == "proforma_lista":
        cliente = _capturar_dato_comercial_por_etapa(
            mensaje=mensaje,
            cliente=cliente,
            etapa="proforma_lista",
        )

        rut_valor = _extraer_rut_desde_mensaje(mensaje)
        if rut_valor and cliente.get("rut") in {None, "", "pendiente"}:
            cliente["rut"] = rut_valor
            logger.debug("RUT actualizado en proforma_lista: %s", rut_valor)

        if rut_valor:
            return {
                "handled": True,
                "respuesta": (
                        "Gracias, recibí el RUT y lo dejé asociado a tu solicitud. "
                        "En breve recibirás la proforma."
                ),
                "etapa": "proforma_lista",
                "cliente": cliente,
                "necesidad_ctx": necesidad_ctx,
                "productos_acumulados": productos_acumulados,
            }

        return {
            "handled": True,
            "respuesta": (
                "Tu proforma está en proceso. Cuando la recibas, me confirmas si deseas proceder con el pago."
            ),
            "etapa": "proforma_lista",
            "cliente": cliente,
            "necesidad_ctx": necesidad_ctx,
            "productos_acumulados": productos_acumulados,
        }

    # ============================================================
    # Pago: cierre comercial — no volver a catálogo ni preguntas técnicas
    # ============================================================
    if etapa in {"pago", "pago_confirmado"}:
        ctx_limpio = _limpiar_ctx_para_cierre_comercial(necesidad_ctx)

        if _es_confirmacion_afirmativa(mensaje):
            return {
                "handled": True,
                "respuesta": (
                    "Perfecto. Dejé tu solicitud lista para el equipo comercial. "
                    "Un asesor se comunicará contigo para confirmar el pago "
                    "(transferencia, PSE o tarjeta) y coordinar el envío."
                ),
                "etapa": "pago_confirmado",
                "cliente": cliente,
                "necesidad_ctx": ctx_limpio,
                "productos_acumulados": productos_acumulados,
            }

        if _es_confirmacion_negativa(mensaje):
            return {
                "handled": True,
                "respuesta": (
                    "Entiendo. Cuéntame qué necesitas ajustar sobre el pago o el envío "
                    "y lo dejo anotado para el asesor."
                ),
                "etapa": "pago",
                "cliente": cliente,
                "necesidad_ctx": ctx_limpio,
                "productos_acumulados": productos_acumulados,
            }

        return {
            "handled": True,
            "respuesta": (
                "Tu pedido ya está en manos del equipo comercial. "
                "Un asesor confirmará el pago y gestionará el envío. "
                "Si necesitas dejar algún comentario para el vendedor, escríbelo aquí."
            ),
            "etapa": "pago_confirmado" if etapa == "pago_confirmado" else "pago",
            "cliente": cliente,
            "necesidad_ctx": ctx_limpio,
            "productos_acumulados": productos_acumulados,
        }

    # 1) Producto encontrado: NIA espera confirmación explícita.
    if etapa == "producto_encontrado":
        # Saludo puro: no atrapar en el sí/no; deja que el flujo de saludo responda.
        if es_solo_saludo(mensaje):
            return {"handled": False}

        if _es_confirmacion_afirmativa(mensaje):
            indice, item = _ultimo_producto_pendiente_confirmacion(productos_acumulados)
            if indice is not None and item:
                productos_acumulados[indice]["feedback"] = "si"

            return {
                "handled": True,
                "respuesta": "Perfecto. ¿Cuál es la cantidad que necesitas?",
                "etapa": "esperando_cantidad",
                "cliente": cliente,
                "necesidad_ctx": {"esperando": "cantidad"},
                "productos_acumulados": productos_acumulados,
                "aprendizaje": _preparar_feedback_aprendizaje(
                    tipo="si",
                    categoria="producto",
                    mensaje=mensaje,
                    productos_acumulados=productos_acumulados,
                    historial=historial,
                    producto_directo=(item or {}).get("producto") if item else None,
                    contexto_extra=construir_contexto_aprendizaje_desde_necesidad(item or {}),
                    clave_aprendizaje=clave_aprendizaje,
                    session_id=session_id,
                ),
            }

        if _es_confirmacion_negativa(mensaje):
            indice, item = _ultimo_producto_pendiente_confirmacion(productos_acumulados)
            feedback = _preparar_feedback_aprendizaje(
                tipo="no",
                categoria="producto",
                mensaje=mensaje,
                productos_acumulados=productos_acumulados,
                historial=historial,
                producto_directo=(item or {}).get("producto") if item else None,
                contexto_extra=construir_contexto_aprendizaje_desde_necesidad(item or {}),
                clave_aprendizaje=clave_aprendizaje,
                session_id=session_id,
            )

            if indice is not None:
                productos_acumulados.pop(indice)

            return {
                "handled": True,
                "respuesta": (
                    "Entendido. Para buscar una mejor opción, ¿puedes indicarme "
                    "tipo de producto, aplicación, marca, referencia o especificación técnica requerida?"
                ),
                "etapa": "descubrimiento",
                "cliente": cliente,
                "necesidad_ctx": {},
                "productos_acumulados": productos_acumulados,
                "aprendizaje": feedback,
            }

        # Sale del sí/no para pedir otra referencia o otro producto.
        if _es_salida_confirmacion_producto(mensaje):
            indice, _item = _ultimo_producto_pendiente_confirmacion(productos_acumulados)
            if indice is not None:
                productos_acumulados.pop(indice)

            if _es_pedido_de_identificador_sin_valor(mensaje):
                if _es_pedido_solo_referencia(mensaje):
                    respuesta = _respuesta_pedir_referencia()
                else:
                    respuesta = _respuesta_pedir_identificador()
                return {
                    "handled": True,
                    "respuesta": respuesta,
                    "etapa": "esperando_codigo",
                    "cliente": cliente,
                    "necesidad_ctx": {
                        "esperando_codigo_producto": True,
                        "texto_original": mensaje,
                        "opciones_actuales": [],
                    },
                    "productos_acumulados": productos_acumulados,
                }

            return {
                "handled": True,
                "respuesta": (
                    "Entendido. ¿Qué otro producto o referencia necesitas? "
                    "Si tienes la marca, dámela también."
                ),
                "etapa": "inicio",
                "cliente": cliente,
                "necesidad_ctx": {},
                "productos_acumulados": productos_acumulados,
            }

        return {
            "handled": True,
            "respuesta": "¿Este producto cubre lo que necesitas? Puedes responder sí o no.",
            "etapa": "producto_encontrado",
            "cliente": cliente,
            "necesidad_ctx": _ctx_confirmacion_producto(),
            "productos_acumulados": productos_acumulados,
        }

    # 2) Esperando cantidad: un número corto es cantidad, no código.
    if etapa == "esperando_cantidad":
        cantidad = _extraer_cantidad_solicitada(mensaje)

        if not cantidad:
            return {
                "handled": True,
                "respuesta": "Para avanzar con la cotización necesito la cantidad en unidades. ¿Cuántas unidades necesitas?",
                "etapa": "esperando_cantidad",
                "cliente": cliente,
                "necesidad_ctx": {"esperando": "cantidad"},
                "productos_acumulados": productos_acumulados,
            }

        _asignar_cantidad_ultimo_producto(productos_acumulados, cantidad)

        resumen = _mensaje_resumen_carrito(productos_acumulados)
        intro = f"Listo, dejé la cantidad en {cantidad}."
        if resumen:
            intro = f"{intro} {resumen}"

        return {
            "handled": True,
            "respuesta": (
                f"{intro} ¿Qué prefieres: agregar otro producto o cotizar con lo que llevas?"
            ),
            "etapa": "confirmando_cierre",
            "cliente": cliente,
            "necesidad_ctx": {"opciones_actuales": _opciones_cierre_carrito()},
            "productos_acumulados": productos_acumulados,
        }

    # 3) Confirmando cierre: agregar al carrito o pasar a cotización.
    if etapa == "confirmando_cierre":
        if _es_pedido_cotizar(mensaje):
            # Nueva solicitud de cotización: pedir nombre y correo siempre,
            # sin reutilizar automáticamente datos de memoria permanente.
            cliente_cotizacion = {
                k: v
                for k, v in dict(cliente or {}).items()
                if k not in {"nombre", "email"}
            }
            return {
                "handled": True,
                "respuesta": "¿A nombre de quién va la cotización?",
                "etapa": "cotizacion",
                "cliente": cliente_cotizacion,
                "necesidad_ctx": {
                    "forzar_contacto_cotizacion": True,
                    "opciones_actuales": [],
                },
                "productos_acumulados": productos_acumulados,
            }

        if _es_pedido_agregar_producto(mensaje):
            if len(productos_acumulados) >= MAX_PRODUCTOS_CARRITO:
                respuesta_dato, etapa_dato = _respuesta_siguiente_dato_comercial(
                    cliente,
                    etapa_objetivo="cotizacion",
                )
                return {
                    "handled": True,
                    "respuesta": (
                        f"Ya tienes {MAX_PRODUCTOS_CARRITO} productos en tu solicitud "
                        f"(límite máximo). {respuesta_dato}"
                    ),
                    "etapa": etapa_dato,
                    "cliente": cliente,
                    "necesidad_ctx": {},
                    "productos_acumulados": productos_acumulados,
                }

            return {
                "handled": True,
                "respuesta": (
                    f"Perfecto. {_mensaje_resumen_carrito(productos_acumulados)} "
                    "¿Qué producto necesitas agregar?"
                ),
                "etapa": "inicio",
                "cliente": cliente,
                "necesidad_ctx": {},
                "productos_acumulados": productos_acumulados,
            }

        if _es_nueva_solicitud_durante_cierre(mensaje):
            if len(productos_acumulados) >= MAX_PRODUCTOS_CARRITO:
                respuesta_dato, etapa_dato = _respuesta_siguiente_dato_comercial(
                    cliente,
                    etapa_objetivo="cotizacion",
                )
                return {
                    "handled": True,
                    "respuesta": (
                        f"Ya tienes {MAX_PRODUCTOS_CARRITO} productos en tu solicitud "
                        f"(límite máximo). {respuesta_dato}"
                    ),
                    "etapa": etapa_dato,
                    "cliente": cliente,
                    "necesidad_ctx": {},
                    "productos_acumulados": productos_acumulados,
                }

            return {
                "handled": True,
                "respuesta": (
                    f"Perfecto. {_mensaje_resumen_carrito(productos_acumulados)} "
                    "¿Qué producto necesitas agregar?"
                ),
                "etapa": "inicio",
                "cliente": cliente,
                "necesidad_ctx": {},
                "productos_acumulados": productos_acumulados,
            }

        minutos_inactivo = _minutos_desde_ultimo_turno(historial or [])
        if (
            productos_acumulados
            and minutos_inactivo >= MINUTOS_INACTIVIDAD_CARRITO
            and not _es_pedido_agregar_producto(mensaje)
        ):
            respuesta_dato, etapa_dato = _respuesta_siguiente_dato_comercial(
                cliente,
                etapa_objetivo="cotizacion",
            )
            return {
                "handled": True,
                "respuesta": (
                    f"Llevas {int(minutos_inactivo)} minutos sin actividad. "
                    f"Voy a proceder a cotizar tus {len(productos_acumulados)} producto(s).\n\n"
                    f"{respuesta_dato}"
                ),
                "etapa": etapa_dato,
                "cliente": cliente,
                "necesidad_ctx": {},
                "productos_acumulados": productos_acumulados,
            }

        return {
            "handled": True,
            "respuesta": (
                f"{_mensaje_resumen_carrito(productos_acumulados)} "
                "¿Deseas agregar otro producto o cotizar con lo que llevas?"
            ).strip(),
            "etapa": "confirmando_cierre",
            "cliente": cliente,
            "necesidad_ctx": {"opciones_actuales": _opciones_cierre_carrito()},
            "productos_acumulados": productos_acumulados,
        }

    # 4) Cotización/proforma: capturar datos antes de cualquier búsqueda.
    if etapa in {"cotizacion", "calificacion", "proforma"}:
        if _es_nueva_solicitud_durante_cierre(mensaje):
            return None

        cliente = _capturar_dato_comercial_por_etapa(
            mensaje=mensaje,
            cliente=cliente,
            etapa=etapa,
        )

        etapa_objetivo = "proforma" if etapa == "proforma" else "cotizacion"

        respuesta_dato, etapa_dato = _respuesta_siguiente_dato_comercial(
            cliente,
            etapa_objetivo=etapa_objetivo,
        )

        ctx_out = {"opciones_actuales": []}
        if etapa_objetivo == "cotizacion" and etapa_dato == "cotizacion":
            # Sigue pidiendo nombre/correo en esta solicitud.
            ctx_out["forzar_contacto_cotizacion"] = True
        # Si ya cerró contacto (cotizacion_lista), no forzar más.

        return {
            "handled": True,
            "respuesta": respuesta_dato,
            "etapa": etapa_dato,
            "cliente": cliente,
            # Proforma pide texto/archivo (NIT, RUT), nunca Sí/No.
            "necesidad_ctx": ctx_out,
            "productos_acumulados": productos_acumulados,
        }

    # 5) Cotización lista: si el cliente indica que ya la recibió,
    #    pasa a validación técnica con botones Sí/No.
    if etapa == "cotizacion_lista":
        msg_norm = _normalizar_intencion(mensaje)
        indicios_recibida = any(
            frase in msg_norm
            for frase in {
                "ya tengo la cotizacion",
                "me llego la cotizacion",
                "ya recibi la cotizacion",
                "ya la recibi",
                "me la enviaron",
                "ya me cotizaron",
                "tengo la cotizacion",
                "ya esta la cotizacion",
                "ya esta lista la cotizacion",
            }
        ) or tipo_mensaje in {"cotizacion_recibida", "link_documento"}

        if indicios_recibida:
            necesidad_ctx["cotizacion_recibida"] = True
            if tipo_mensaje == "link_documento":
                necesidad_ctx["archivo_cotizacion"] = mensaje
            respuesta, etapa_dato, ctx_opts = _respuesta_confirmar_cotizacion_tecnica()
            return {
                "handled": True,
                "respuesta": respuesta,
                "etapa": etapa_dato,
                "cliente": cliente,
                "necesidad_ctx": {**necesidad_ctx, **ctx_opts},
                "productos_acumulados": productos_acumulados,
            }

        return {
            "handled": True,
            "respuesta": (
                "Tu cotización está en proceso. Cuando la recibas, me confirmas "
                "si cumple con lo que necesitas técnicamente."
            ),
            "etapa": "cotizacion_lista",
            "cliente": cliente,
            "necesidad_ctx": {},
            "productos_acumulados": productos_acumulados,
        }

    return None

async def _persistir_cliente_permanente(
    phone_id: Optional[str],
    cliente: Optional[dict],
) -> None:
    """
    Guarda datos comerciales del cliente en memoria permanente.

    Esta memoria NO reemplaza la sesión conversacional.
    Solo persiste datos reutilizables del cliente:
    - nombre
    - email
    - empresa
    - nit
    - rut
    - teléfono/phone_id

    Si Mongo falla, no rompemos la conversación del cliente.
    """
    if not phone_id or not cliente:
        return

    try:
        await upsert_cliente(phone_id, cliente)
    except Exception as e:
        logger.warning(
            "No fue posible persistir cliente permanente phone_id=%s error=%s",
            phone_id,
            e,
        )

async def _guardar_y_responder_turno(
    session_id: str,
    phone_id: Optional[str],
    historial: list,
    mensaje_usuario: str,
    respuesta: str,
    etapa: str,
    cliente: dict,
    productos_acumulados: list,
    necesidad_ctx: Optional[dict] = None,
    archivo_activo: Optional[dict] = None,
    items_resultado: Optional[list] = None,
    cotizacion_recibida: bool = False,
    archivo_cotizacion: Optional[str] = None,
    proforma_recibida: bool = False,
    archivo_proforma: Optional[str] = None,
):
    """
    Guarda sesión y retorna respuesta final sin pasar por LLM.

    Se usa cuando un estado comercial ya resolvió el turno.
    """
    turno_user = {
        "role": "user",
        "content": mensaje_usuario,
        "ts": datetime.utcnow().isoformat(),
    }

    turno_nia = {
        "role": "assistant",
        "content": respuesta,
        "ts": datetime.utcnow().isoformat(),
    }

    await _persistir_cliente_permanente(phone_id, cliente)

    # Garantiza cuadritos Sí/No en validación técnica de cotización.
    necesidad_ctx = dict(necesidad_ctx or {})
    resp_norm = (respuesta or "").lower()
    if (
        (not necesidad_ctx.get("opciones_actuales"))
        and "cumple con lo que necesitas t" in resp_norm
        and "cotizaci" in resp_norm
    ):
        necesidad_ctx.update(_ctx_confirmacion_cotizacion_tecnica())
        etapa = "cotizacion_enviada"

    await save_session(
        session_id=session_id,
        phone_id=phone_id,
        turnos=historial + [turno_user, turno_nia],
        etapa=etapa,
        archivo_activo=archivo_activo,
        necesidad_ctx=necesidad_ctx,
        cliente=cliente or {},
        productos_acumulados=productos_acumulados or [],
        cotizacion_recibida=cotizacion_recibida,
        archivo_cotizacion=archivo_cotizacion,
        proforma_recibida=proforma_recibida,
        archivo_proforma=archivo_proforma,
    )

    return {
        "respuesta": respuesta,
        "etapa": etapa,
        "opciones": necesidad_ctx.get("opciones_actuales") or None,
        "items_resultado": items_resultado or None,
        "cliente": cliente or None,
    }

# ─────────────────────────────────────────────────────────────
# Núcleo conversacional
# ─────────────────────────────────────────────────────────────

async def procesar_turno(
    session_id: str,
    mensaje: str,
    phone_id: Optional[str] = None,
    archivo_bytes: Optional[bytes] = None,
    archivo_nombre: Optional[str] = None,
) -> dict:
    logger.info(
        "Turno: session=%s etapa_msg='%s'",
        session_id,
        mensaje[:50] if mensaje else "[archivo]",
    )

    session = await get_session(session_id) or {}

    historial = session.get("turnos", [])
    etapa = session.get("etapa", "inicio")
        # ------------------------------------------------------------
    # Guardrail de estado comercial:
    # Si el último mensaje de NIA pidió datos de contacto, el
    # siguiente mensaje del cliente debe procesarse como cotización,
    # aunque la etapa guardada haya quedado inconsistente.
    # ------------------------------------------------------------
    if etapa in {"inicio", "descubrimiento"} and _ultimo_turno_pide_datos_contacto(historial):
            logger.info(
                "Corrigiendo etapa por último turno de contacto: session=%s etapa=%s -> cotizacion",
                session_id,
                etapa,
            )
            etapa = "cotizacion"
    archivo_activo = session.get("archivo_activo")
    necesidad_ctx = session.get("necesidad_ctx", {})

    # ------------------------------------------------------------
    # Cliente: sesión temporal + memoria permanente
    # ------------------------------------------------------------
    # La sesión tiene prioridad porque contiene lo más reciente
    # dentro de la conversación actual.
    cliente_sesion = session.get("cliente", {}) or {}

    cliente_permanente = {}

    if phone_id:
        try:
            cliente_permanente = await get_cliente(phone_id) or {}
        except Exception as e:
            logger.warning(
                "No fue posible cargar cliente permanente phone_id=%s error=%s",
                phone_id,
                e,
            )
            cliente_permanente = {}

    cliente = {
        **cliente_permanente,
        **cliente_sesion,
    }

    # Corrige datos históricos contaminados por botones de acción.
    cliente = _sanitizar_cliente_control(cliente)

    # Al cotizar, nombre/correo deben pedirse otra vez en esta solicitud,
    # aunque existan en memoria permanente de pruebas anteriores.
    if (necesidad_ctx or {}).get("forzar_contacto_cotizacion"):
        if "nombre" in cliente_sesion and str(cliente_sesion.get("nombre") or "").strip():
            cliente["nombre"] = cliente_sesion["nombre"]
        else:
            cliente.pop("nombre", None)

        if "email" in cliente_sesion and str(cliente_sesion.get("email") or "").strip():
            cliente["email"] = cliente_sesion["email"]
        else:
            cliente.pop("email", None)

    productos_acumulados = session.get("productos_acumulados", [])

    clave_aprendizaje = resolver_clave_aprendizaje(phone_id, cliente, session_id)
    memoria_aprendizaje = await obtener_memoria_aprendizaje(clave_aprendizaje)
    activar_memoria_aprendizaje(memoria_aprendizaje)

    cotizacion_recibida = bool(session.get("cotizacion_recibida", False))
    archivo_cotizacion = session.get("archivo_cotizacion")
    proforma_recibida = bool(session.get("proforma_recibida", False))
    archivo_proforma = session.get("archivo_proforma")

    en_etapa_comercial = etapa in ESTADOS_COMERCIALES

    # PRIORIDAD: saludo puro (incluso con preguntas pendientes en sesión)
    if (
        mensaje.strip()
        and not (archivo_bytes and archivo_nombre)
        and not en_etapa_comercial
    ):
        early_greeting = build_greeting_result(
            mensaje,
            seed_text=session_id or phone_id or mensaje,
            client_name=(cliente or {}).get("nombre"),
            excluir_respuesta=(necesidad_ctx or {}).get("ultimo_saludo"),
        )
        if early_greeting.get("matched") and early_greeting.get("should_respond_now"):
            ctx_limpio = {
                k: v
                for k, v in (necesidad_ctx or {}).items()
                if k not in _CLAVES_CTX_DESCUBRIMIENTO
            }
            ctx_limpio["preguntas_pendientes"] = []
            ctx_limpio["opciones_actuales"] = []
            ctx_limpio["ultimo_saludo"] = early_greeting.get("response")
            logger.info(
                "Saludo puro temprano: intent=%s session=%s (limpia descubrimiento)",
                early_greeting.get("intent"),
                session_id,
            )
            return await _guardar_y_responder_turno(
                session_id=session_id,
                phone_id=phone_id,
                historial=historial,
                mensaje_usuario=mensaje,
                respuesta=early_greeting["response"],
                etapa="saludo",
                cliente=cliente,
                productos_acumulados=productos_acumulados,
                necesidad_ctx=ctx_limpio,
                archivo_activo=archivo_activo,
                cotizacion_recibida=cotizacion_recibida,
                archivo_cotizacion=archivo_cotizacion,
                proforma_recibida=proforma_recibida,
                archivo_proforma=archivo_proforma,
            )

    # PRIORIDAD: intenciones comerciales con respuesta fija (escenarios 01-17)
    if mensaje.strip() and not (archivo_bytes and archivo_nombre):
        scripted = detect_scripted_intent(mensaje, historial=historial)
        if scripted.get("matched"):
            # En etapa comercial solo dejan pasar intenciones de servicio/cierre.
            intents_en_comercial = {
                "mensaje_automatico_ignorar",
                "asesor_humano",
                "datos_corporativos",
                "documentos_legales_bancarios",
                "pago_comprobante",
                "guia_envio",
                "factura_electronica",
                "flete_envio",
                "ficha_manual_catalogo",
                "agradecimiento",
                "adjunto_multimedia",
                "compra_proforma",
                "orden_directa",
            }
            if (not en_etapa_comercial) or scripted.get("intent") in intents_en_comercial:
                if scripted.get("silent"):
                    logger.info(
                        "Mensaje automático ignorado: session=%s",
                        session_id,
                    )
                    return {
                        "respuesta": "",
                        "etapa": "ignorado",
                        "opciones": None,
                        "items_resultado": None,
                        "cliente": cliente or None,
                    }

                logger.info(
                    "Intención fija: intent=%s session=%s",
                    scripted.get("intent"),
                    session_id,
                )
                return await _guardar_y_responder_turno(
                    session_id=session_id,
                    phone_id=phone_id,
                    historial=historial,
                    mensaje_usuario=mensaje,
                    respuesta=scripted["response"],
                    etapa=scripted.get("etapa") or "scripted",
                    cliente=cliente,
                    productos_acumulados=productos_acumulados,
                    necesidad_ctx=necesidad_ctx or {},
                    archivo_activo=archivo_activo,
                    cotizacion_recibida=cotizacion_recibida,
                    archivo_cotizacion=archivo_cotizacion,
                    proforma_recibida=proforma_recibida,
                    archivo_proforma=archivo_proforma,
                )

    # PRIORIDAD: flujo híbrida libros → catálogo
    if mensaje.strip() and not (archivo_bytes and archivo_nombre) and not en_etapa_comercial:
        turno_hibrida = await _try_resolver_turno_hibrida(
            mensaje=mensaje,
            necesidad_ctx=necesidad_ctx,
            cliente=cliente,
            productos_acumulados=productos_acumulados,
        )
        if turno_hibrida:
            contexto_extra, nueva_etapa, necesidad_ctx = turno_hibrida
            respuesta_segura = _extraer_respuesta_segura(contexto_extra)
            if respuesta_segura:
                return await _guardar_y_responder_turno(
                    session_id=session_id,
                    phone_id=phone_id,
                    historial=historial,
                    mensaje_usuario=mensaje,
                    respuesta=respuesta_segura,
                    etapa=nueva_etapa,
                    cliente=cliente,
                    productos_acumulados=productos_acumulados,
                    necesidad_ctx=necesidad_ctx,
                    archivo_activo=archivo_activo,
                    cotizacion_recibida=cotizacion_recibida,
                    archivo_cotizacion=archivo_cotizacion,
                    proforma_recibida=proforma_recibida,
                    archivo_proforma=archivo_proforma,
                )

    # PRIORIDAD: flujo descubrimiento NIVEL_1 (Otro → búsqueda textual)
    if mensaje.strip() and not (archivo_bytes and archivo_nombre) and not en_etapa_comercial:
        turno_corta_larga = await _try_resolver_turno_corta_larga(
            mensaje=mensaje,
            necesidad_ctx=necesidad_ctx,
            cliente=cliente,
            productos_acumulados=productos_acumulados,
        )
        if turno_corta_larga:
            contexto_extra, nueva_etapa, necesidad_ctx = turno_corta_larga
            respuesta_segura = _extraer_respuesta_segura(contexto_extra)
            if respuesta_segura:
                return await _guardar_y_responder_turno(
                    session_id=session_id,
                    phone_id=phone_id,
                    historial=historial,
                    mensaje_usuario=mensaje,
                    respuesta=respuesta_segura,
                    etapa=nueva_etapa,
                    cliente=cliente,
                    productos_acumulados=productos_acumulados,
                    necesidad_ctx=necesidad_ctx,
                    archivo_activo=archivo_activo,
                    cotizacion_recibida=cotizacion_recibida,
                    archivo_cotizacion=archivo_cotizacion,
                    proforma_recibida=proforma_recibida,
                    archivo_proforma=archivo_proforma,
                )

    contexto_extra = ""
    nueva_etapa = etapa
    items_resultado = []
    greeting_prefix = None
    mensaje_original = mensaje

    # ------------------------------------------------------------
    # Identificador explícito con prioridad global
    # ------------------------------------------------------------
    # Un código o referencia escrito por el cliente debe tener prioridad
    # sobre preguntas técnicas pendientes o estados de descubrimiento.

    tipo_identificador = None
    valor_identificador = None

    if mensaje.strip() and not (archivo_bytes and archivo_nombre):
        tipo_identificador, valor_identificador = detectar_identificador(mensaje)

    if mensaje.strip():
        cliente = extraer_datos_cliente(mensaje, cliente)

    # ------------------------------------------------------------
    # Clasificación de intención — Cotización V
    # ------------------------------------------------------------
    clasificacion = await clasificar_mensaje(mensaje, etapa)

    logger.info(
        "Clasificación mensaje: session=%s etapa=%s tipo=%s confianza=%s razon=%s",
        session_id,
        etapa,
        clasificacion.get("tipo"),
        clasificacion.get("confianza"),
        clasificacion.get("razon"),
    )

    # ══════════════════════════════════════════════════════
    # PRIORIDAD ABSOLUTA: ESTADO COMERCIAL
    # ══════════════════════════════════════════════════════
    # Si el turno pertenece a una etapa comercial, se resuelve aquí
    # y se retorna inmediatamente. No catálogo. No LLM. No reglas posteriores.
    # Excepciones: correo, cantidad ("2 und") o etapa esperando_cantidad
    # no deben saltarse el controlador comercial por parecer "referencia".
    tiene_email = bool(EMAIL_RE.search(mensaje or ""))
    es_cantidad_comercial = (
        etapa == "esperando_cantidad"
        or _parece_mensaje_solo_cantidad(mensaje)
    )
    bloquear_por_identificador = bool(tipo_identificador) and not (
        (tiene_email and etapa in ESTADOS_COMERCIALES)
        or es_cantidad_comercial
    )
    if (
        mensaje.strip()
        and not (archivo_bytes and archivo_nombre)
        and not bloquear_por_identificador
    ):
        comercial = _manejar_estado_comercial_prioritario(
            etapa=etapa,
            mensaje=mensaje,
            cliente=cliente,
            productos_acumulados=productos_acumulados,
            necesidad_ctx=necesidad_ctx,
            clasificacion=clasificacion,
            historial=historial,
            clave_aprendizaje=clave_aprendizaje,
            session_id=session_id,
        )

        if comercial and comercial.get("handled"):
            logger.info(
                "Turno resuelto por estado comercial: session=%s etapa=%s -> %s",
                session_id,
                etapa,
                comercial["etapa"],
            )

            await _persistir_aprendizaje_si_corresponde(comercial.get("aprendizaje"))

            necesidad_comercial = comercial.get("necesidad_ctx", {}) or {}

            return await _guardar_y_responder_turno(
                session_id=session_id,
                phone_id=phone_id,
                historial=historial,
                mensaje_usuario=mensaje,
                respuesta=comercial["respuesta"],
                etapa=comercial["etapa"],
                cliente=comercial["cliente"],
                productos_acumulados=comercial["productos_acumulados"],
                necesidad_ctx=necesidad_comercial,
                archivo_activo=archivo_activo,
                items_resultado=None,
                cotizacion_recibida=bool(
                    necesidad_comercial.get("cotizacion_recibida", cotizacion_recibida)
                ),
                archivo_cotizacion=necesidad_comercial.get(
                    "archivo_cotizacion", archivo_cotizacion
                ),
                proforma_recibida=bool(
                    necesidad_comercial.get("proforma_recibida", proforma_recibida)
                ),
                archivo_proforma=necesidad_comercial.get(
                    "archivo_proforma", archivo_proforma
                ),
            )

    # ══════════════════════════════════════════════════════
    # MODO ARCHIVO
    # ══════════════════════════════════════════════════════

    if archivo_bytes and archivo_nombre:
        logger.info("Procesando archivo: %s", archivo_nombre)

        # ------------------------------------------------------------
        # RUT (PDF DIAN): tomar NIT+DV y razón social, sin catálogo.
        # ------------------------------------------------------------
        nombre_archivo_l = str(archivo_nombre or "").lower()
        if nombre_archivo_l.endswith(".pdf"):
            datos_rut = extraer_datos_rut_pdf(archivo_bytes, archivo_nombre)
        else:
            datos_rut = None

        if datos_rut and datos_rut.get("es_rut"):
            if datos_rut.get("nit"):
                cliente["nit"] = datos_rut["nit"]
            if datos_rut.get("empresa"):
                cliente["empresa"] = datos_rut["empresa"]
            cliente["rut"] = "recibido"

            datos_leidos = []
            if cliente.get("empresa"):
                datos_leidos.append(f"razón social {cliente['empresa']}")
            if cliente.get("nit"):
                datos_leidos.append(f"NIT {cliente['nit']}")

            if datos_leidos:
                confirmacion_rut = (
                    "Recibí el RUT y tomé "
                    + " y ".join(datos_leidos)
                    + "."
                )
            else:
                confirmacion_rut = (
                    "Recibí el RUT, pero no pude leer con claridad el NIT "
                    "y la razón social. ¿Me los confirmas por texto?"
                )

            etapas_fiscales = {
                "proforma",
                "calificacion",
                "cotizacion_lista",
                "cotizacion_enviada",
            }

            if etapa in etapas_fiscales:
                respuesta_dato, etapa_dato = _respuesta_siguiente_dato_comercial(
                    cliente,
                    etapa_objetivo="proforma",
                )
                respuesta_rut = f"{confirmacion_rut} {respuesta_dato}".strip()
                etapa_rut = etapa_dato
                ctx_rut = {"opciones_actuales": []}
            else:
                respuesta_rut = confirmacion_rut
                etapa_rut = etapa or "inicio"
                ctx_rut = dict(necesidad_ctx or {})
                ctx_rut["opciones_actuales"] = []
                ctx_rut["rut_extraido"] = {
                    "nit": cliente.get("nit"),
                    "empresa": cliente.get("empresa"),
                }

            logger.info(
                "RUT PDF leído: session=%s nit=%s empresa=%s etapa=%s→%s",
                session_id,
                cliente.get("nit"),
                cliente.get("empresa"),
                etapa,
                etapa_rut,
            )

            return await _guardar_y_responder_turno(
                session_id=session_id,
                phone_id=phone_id,
                historial=historial,
                mensaje_usuario=mensaje or f"[Adjunto RUT: {archivo_nombre}]",
                respuesta=respuesta_rut,
                etapa=etapa_rut,
                cliente=cliente,
                productos_acumulados=productos_acumulados,
                necesidad_ctx=ctx_rut,
                archivo_activo={
                    "nombre": archivo_nombre,
                    "tipo": "rut",
                    "ts": datetime.utcnow().isoformat(),
                },
                items_resultado=[{
                    "estado": "rut_leido",
                    "nit": cliente.get("nit"),
                    "empresa": cliente.get("empresa"),
                    "texto_original": archivo_nombre,
                }],
                cotizacion_recibida=cotizacion_recibida,
                archivo_cotizacion=archivo_cotizacion,
                proforma_recibida=True,
                archivo_proforma=archivo_nombre,
            )

        items = await procesar_archivo(archivo_bytes, archivo_nombre)

        for item in items:
            tipo, valor = detectar_identificador(item["texto"])

            if tipo:
                res = await rama_codigo(valor, tipo)
            else:
                nec = await evaluar_necesidad(item["texto"])

                if nec["clara"]:
                    res = await buscar_en_catalogo(item["texto"])

                    if debe_intentar_enriquecimiento(res):
                        res = await enriquecer_y_buscar(item["texto"])

                else:
                    res = {
                        "estado": "pendiente",
                        "preguntas": nec["preguntas"],
                    }

            res["texto_original"] = item["texto"]
            res["fila"] = item.get("fila")
            res["cantidad"] = item.get("cantidad")
            items_resultado.append(res)

            if res["estado"] == "encontrado" and res.get("producto"):
                productos_acumulados.append({
                    "producto": res["producto"],
                    "cantidad": item.get("cantidad"),
                    "desde": "archivo",
                    "ts": datetime.utcnow().isoformat(),
                })

        encontrados = [i for i in items_resultado if i["estado"] == "encontrado"]
        pendientes = [
            i for i in items_resultado
            if i["estado"] in {"pendiente", "sin_resultado", "relacionado"}
        ]

        archivo_activo = {
            "nombre": archivo_nombre,
            "total_items": len(items),
            "items": items_resultado,
            "ts": datetime.utcnow().isoformat(),
        }

        # Un solo producto exacto → mismo flujo comercial (sí/no → cantidad).
        # Evita que el LLM reescriba la ficha y deje la etapa en procesando_archivo.
        if len(encontrados) == 1 and len(items_resultado) == 1:
            producto = encontrados[0]["producto"]
            cantidad_archivo = encontrados[0].get("cantidad")
            if productos_acumulados:
                productos_acumulados[-1]["cantidad"] = cantidad_archivo
                productos_acumulados[-1]["desde"] = "archivo"

            contexto_extra = _marcar_respuesta_segura(
                "Gracias por enviarlo, lo reviso ahora mismo.\n\n"
                + respuesta_producto_encontrado(producto, cliente)
            )
            nueva_etapa = "producto_encontrado"
            necesidad_ctx = _ctx_confirmacion_producto()
            logger.info(
                "Archivo con 1 producto exacto → flujo comercial producto_encontrado: %s",
                producto.get("codigo"),
            )
        else:
            nueva_etapa = "procesando_archivo"

            resumen = (
                f"[ARCHIVO: {archivo_nombre}]\n"
                f"Total: {len(items)} · Encontrados: {len(encontrados)} · "
                f"Pendientes/por validar: {len(pendientes)}\n"
            )

            for item_resultado in items_resultado:
                resumen += (
                    f"- {item_resultado['texto_original']}: "
                    f"{item_resultado['estado'].upper()}"
                )

                if item_resultado.get("producto"):
                    p = item_resultado["producto"]
                    resumen += f" → {p.get('codigo')} | {p.get('nombre')}"

                    if item_resultado["estado"] == "relacionado":
                        resumen += " [RELACIONADO — REQUIERE CONFIRMACIÓN]"
                    elif not item_resultado.get("exacto", True):
                        resumen += " [COINCIDENCIA CERCANA]"

                resumen += "\n"

            contexto_extra = resumen

    # ══════════════════════════════════════════════════════
    # MODO TEXTO
    # ══════════════════════════════════════════════════════

    elif mensaje.strip():
        greeting = build_greeting_result(
            mensaje_original,
            seed_text=session_id or phone_id or mensaje_original,
            client_name=(cliente or {}).get("nombre"),
            excluir_respuesta=(necesidad_ctx or {}).get("ultimo_saludo"),
        )

        # Saludo + solicitud comercial: el saludo es secundario.
        # Se responde con un prefijo breve y el resto del mensaje
        # continúa al router principal sin volver a preguntar qué necesita.
        if (
            greeting.get("matched")
            and greeting.get("continue_to_main_intent")
            and greeting.get("text_for_router")
        ):
            greeting_prefix = greeting.get("response")
            mensaje = greeting["text_for_router"]
            logger.info(
                "Saludo con solicitud: intent=%s secondary=%s router='%s'",
                greeting.get("intent"),
                greeting.get("secondary_intent"),
                mensaje[:80],
            )

        msg_lower = mensaje.lower().strip()
        estado_comercial_resuelto = False

        # El estado comercial se resuelve antes de entrar al modo texto.
        # Si llegó hasta aquí, este turno puede pasar a archivo/catálogo/LLM.

        # ============================================================
        # PRIORIDAD 0: saludo puro
        # ============================================================
        # Gana incluso si hay preguntas técnicas pendientes en la sesión.
        # Evita que "hola" se tome como respuesta a "¿qué fluido mide?".
        if (
            greeting.get("matched")
            and greeting.get("should_respond_now")
        ) or (
            not greeting.get("matched")
            and es_solo_saludo(mensaje_original)
        ):
            if greeting.get("matched") and greeting.get("response"):
                respuesta_saludo = greeting["response"]
            else:
                respuesta_saludo = select_response(
                    "saludo_general",
                    seed_text=session_id or phone_id or mensaje_original,
                    usar_hora_real=True,
                    excluir=(necesidad_ctx or {}).get("ultimo_saludo"),
                )
                respuesta_saludo = _personalizar(
                    respuesta_saludo, (cliente or {}).get("nombre")
                )

            contexto_extra = _marcar_respuesta_segura(respuesta_saludo)
            nueva_etapa = "saludo"
            # Limpia cola de descubrimiento para que el siguiente turno parta limpio.
            necesidad_ctx = {
                k: v
                for k, v in (necesidad_ctx or {}).items()
                if k not in _CLAVES_CTX_DESCUBRIMIENTO
            }
            necesidad_ctx["preguntas_pendientes"] = []
            necesidad_ctx["opciones_actuales"] = []
            necesidad_ctx["ultimo_saludo"] = respuesta_saludo
            logger.info(
                "Saludo puro prioritario: intent=%s session=%s",
                greeting.get("intent") or "saludo_general",
                session_id,
            )

        # ============================================================
        # PRIORIDAD 1: código o referencia explícita
        # ============================================================
        # Esta rama se ejecuta antes de:
        # - respuestas a preguntas técnicas;
        # - descubrimiento;
        # - búsquedas semánticas;
        # - generación de preguntas con OpenAI.

        elif tipo_identificador and valor_identificador:
            logger.info(
                "Procesando identificador explícito con prioridad: %s=%s",
                tipo_identificador,
                valor_identificador,
            )

            res = await rama_codigo(
                valor=valor_identificador,
                tipo=tipo_identificador,
                texto_original=mensaje,
            )

            contexto_extra, nueva_etapa, necesidad_ctx = (
                construir_respuesta_desde_resultado(
                    res=res,
                    cliente=cliente,
                    productos_acumulados=productos_acumulados,
                    desde="identificador_explicito",
                    necesidad_ctx_base={
                        "texto_original": mensaje,
                        "query_evaluada": valor_identificador,
                    },
                )
            )

        # Caso 1: respuesta a ítem pendiente de archivo
        elif archivo_activo:
            pendientes = [
                item for item in archivo_activo.get("items", [])
                if item["estado"] in {"pendiente", "sin_resultado", "relacionado"}
            ]

            if pendientes:
                item_pend = pendientes[0]
                query_e = f"{item_pend['texto_original']} {mensaje}".strip()

                res = await buscar_en_catalogo(query_e)

                if debe_intentar_enriquecimiento(res):
                    res = await enriquecer_y_buscar(
                        query_e,
                        necesidad_ctx={"texto_original": item_pend["texto_original"]},
                    )

                res["texto_original"] = item_pend["texto_original"]

                for idx, item in enumerate(archivo_activo["items"]):
                    if item["texto_original"] == item_pend["texto_original"]:
                        archivo_activo["items"][idx] = res
                        break

                contexto_extra, nueva_etapa, necesidad_ctx = construir_respuesta_desde_resultado(
                    res=res,
                    cliente=cliente,
                    productos_acumulados=productos_acumulados,
                    desde="archivo_pendiente",
                    necesidad_ctx_base={
                        "texto_original": item_pend["texto_original"],
                        "query_evaluada": query_e,
                    },
                )

        # Caso 2: validación de producto relacionado
        elif etapa == "validando_relacionado" and necesidad_ctx.get("producto_relacionado"):
            producto_relacionado = necesidad_ctx["producto_relacionado"]

            if any(palabra in msg_lower for palabra in {"sí", "si", "correcto", "ese", "me sirve", "sirve"}):
                contexto_relacionado = construir_contexto_aprendizaje_desde_necesidad(necesidad_ctx)
                productos_acumulados.append({
                    "producto": producto_relacionado,
                    "cantidad": None,
                    "desde": "confirmacion_relacionado",
                    "ts": datetime.utcnow().isoformat(),
                    "contexto_aprendizaje": contexto_relacionado,
                    "feedback": None,
                })

                await _persistir_aprendizaje_si_corresponde(
                    _preparar_feedback_aprendizaje(
                        tipo="si",
                        categoria="relacionado",
                        mensaje=mensaje,
                        productos_acumulados=productos_acumulados,
                        historial=historial,
                        producto_directo=producto_relacionado,
                        contexto_extra=contexto_relacionado,
                        clave_aprendizaje=clave_aprendizaje,
                        session_id=session_id,
                    )
                )

                contexto_extra = _marcar_respuesta_segura(
                    respuesta_producto_encontrado(producto_relacionado, cliente)
                )
                nueva_etapa = "producto_encontrado"
                necesidad_ctx = _ctx_confirmacion_producto()

            elif _es_confirmacion_negativa(mensaje):
                await _persistir_aprendizaje_si_corresponde(
                    _preparar_feedback_aprendizaje(
                        tipo="no",
                        categoria="relacionado",
                        mensaje=mensaje,
                        productos_acumulados=productos_acumulados,
                        historial=historial,
                        producto_directo=producto_relacionado,
                        contexto_extra=construir_contexto_aprendizaje_desde_necesidad(necesidad_ctx),
                        clave_aprendizaje=clave_aprendizaje,
                        session_id=session_id,
                    )
                )

                preguntas_pendientes = necesidad_ctx.get("preguntas_pendientes") or []

                if preguntas_pendientes:
                    respuesta_segura, etapa_resp, ctx_actualizado, accion = (
                        _continuar_secuencia_preguntas(
                            necesidad_ctx,
                            mensaje,
                            cliente,
                        )
                    )

                    if accion in {"continuar", "esperar_otro"}:
                        contexto_extra = respuesta_segura
                        nueva_etapa = "validando_relacionado"
                        necesidad_ctx = ctx_actualizado
                    else:
                        texto_original = (
                            necesidad_ctx.get("texto_original")
                            or necesidad_ctx.get("query_evaluada")
                            or mensaje
                        )
                        query_e = _construir_query_acumulado(ctx_actualizado)

                        res = await buscar_en_catalogo(query_e)

                        if debe_intentar_enriquecimiento(res):
                            res = await enriquecer_y_buscar(
                                query_e,
                                necesidad_ctx=ctx_actualizado,
                            )

                        contexto_extra, nueva_etapa, necesidad_ctx = (
                            construir_respuesta_desde_resultado(
                                res=res,
                                cliente=cliente,
                                productos_acumulados=productos_acumulados,
                                desde="validacion_relacionado",
                                necesidad_ctx_base={
                                    "texto_original": texto_original,
                                    "query_evaluada": query_e,
                                },
                            )
                        )
                else:
                    texto_original = (
                        necesidad_ctx.get("texto_original")
                        or necesidad_ctx.get("query_evaluada")
                        or mensaje
                    )
                    query_e = f"{texto_original} {mensaje}".strip()

                    res = await buscar_en_catalogo(query_e)

                    if debe_intentar_enriquecimiento(res):
                        res = await enriquecer_y_buscar(query_e, necesidad_ctx=ctx_actualizado)

                    contexto_extra, nueva_etapa, necesidad_ctx = construir_respuesta_desde_resultado(
                        res=res,
                        cliente=cliente,
                        productos_acumulados=productos_acumulados,
                        desde="validacion_relacionado",
                        necesidad_ctx_base={
                            "texto_original": texto_original,
                            "query_evaluada": query_e,
                        },
                    )

            else:
                preguntas_pendientes = necesidad_ctx.get("preguntas_pendientes") or []

                if preguntas_pendientes:
                    respuesta_segura, etapa_resp, ctx_actualizado, accion = (
                        _continuar_secuencia_preguntas(
                            necesidad_ctx,
                            mensaje,
                            cliente,
                        )
                    )

                    if accion in {"continuar", "esperar_otro"}:
                        contexto_extra = respuesta_segura
                        nueva_etapa = "validando_relacionado"
                        necesidad_ctx = ctx_actualizado
                    else:
                        texto_original = (
                            necesidad_ctx.get("texto_original")
                            or necesidad_ctx.get("query_evaluada")
                            or mensaje
                        )
                        query_e = _construir_query_acumulado(ctx_actualizado)

                        res = await buscar_en_catalogo(query_e)

                        if debe_intentar_enriquecimiento(res):
                            res = await enriquecer_y_buscar(
                                query_e,
                                necesidad_ctx=ctx_actualizado,
                            )

                        contexto_extra, nueva_etapa, necesidad_ctx = (
                            construir_respuesta_desde_resultado(
                                res=res,
                                cliente=cliente,
                                productos_acumulados=productos_acumulados,
                                desde="validacion_relacionado",
                                necesidad_ctx_base={
                                    "texto_original": texto_original,
                                    "query_evaluada": query_e,
                                },
                            )
                        )
                else:
                    texto_original = (
                        necesidad_ctx.get("texto_original")
                        or necesidad_ctx.get("query_evaluada")
                        or mensaje
                    )
                    query_e = f"{texto_original} {mensaje}".strip()

                    res = await buscar_en_catalogo(query_e)

                    if debe_intentar_enriquecimiento(res):
                        res = await enriquecer_y_buscar(query_e, necesidad_ctx=ctx_actualizado)

                    contexto_extra, nueva_etapa, necesidad_ctx = construir_respuesta_desde_resultado(
                        res=res,
                        cliente=cliente,
                        productos_acumulados=productos_acumulados,
                        desde="validacion_relacionado",
                        necesidad_ctx_base={
                            "texto_original": texto_original,
                            "query_evaluada": query_e,
                        },
                    )

        # Caso 2b: confirmación de marca para referencia ambigua
        elif etapa == "esperando_marca_referencia" and necesidad_ctx.get("referencia_pendiente"):
            referencia = str(necesidad_ctx.get("referencia_pendiente") or "").strip()
            match_campo = necesidad_ctx.get("match_campo_referencia") or "REFERENCIA"
            candidatos_ctx = list(necesidad_ctx.get("candidatos_referencia") or [])
            opciones_marca = _opciones_marcas_referencia(candidatos_ctx)

            # Cliente eligió "Otro": pedir marca en texto libre.
            if _normalizar_intencion(mensaje) in {"otro", "otra", "otra marca"}:
                contexto_extra = _marcar_respuesta_segura(
                    "Perfecto. Escríbeme la marca para confirmar la referencia."
                )
                nueva_etapa = "esperando_marca_referencia"
                necesidad_ctx = {
                    **necesidad_ctx,
                    "esperando_marca_libre": True,
                    "opciones_actuales": [],
                }
            else:
                marca = _extraer_marca_de_respuesta(mensaje)

                if not marca:
                    contexto_extra = _marcar_respuesta_segura(
                        "Para confirmar la referencia, indícame la marca."
                    )
                    nueva_etapa = "esperando_marca_referencia"
                    necesidad_ctx = {
                        **necesidad_ctx,
                        "opciones_actuales": opciones_marca,
                    }
                else:
                    res_ref = await buscar_por_referencia(referencia, marca=marca)

                    if res_ref.get("estado") == "encontrado" and res_ref.get("producto"):
                        contexto_extra, nueva_etapa, necesidad_ctx = (
                            construir_respuesta_desde_resultado(
                                res={
                                    "estado": "encontrado",
                                    "producto": res_ref["producto"],
                                },
                                cliente=cliente,
                                productos_acumulados=productos_acumulados,
                                desde="referencia_marca_confirmada",
                                necesidad_ctx_base={
                                    "texto_original": (
                                        necesidad_ctx.get("texto_original") or mensaje
                                    ),
                                    "query_evaluada": referencia,
                                    "marca_confirmada": marca,
                                    "match_campo_referencia": res_ref.get("match_campo"),
                                },
                            )
                        )
                    elif res_ref.get("estado") == "necesita_marca":
                        candidatos_filtrados = list(res_ref.get("candidatos") or [])
                        if marca and not candidatos_filtrados:
                            contexto_extra = _marcar_respuesta_segura(
                                f"No encontré la referencia {referencia} con marca {marca}. "
                                "¿Puedes verificar la marca o compartir otra?"
                            )
                            nueva_etapa = "esperando_marca_referencia"
                            necesidad_ctx = {
                                **necesidad_ctx,
                                "opciones_actuales": opciones_marca,
                            }
                        else:
                            contexto_extra, nueva_etapa, necesidad_ctx = (
                                construir_respuesta_desde_resultado(
                                    res={
                                        "estado": "necesita_marca",
                                        "referencia_buscada": referencia,
                                        "candidatos": candidatos_filtrados or candidatos_ctx,
                                        "match_campo": (
                                            res_ref.get("match_campo") or match_campo
                                        ),
                                    },
                                    cliente=cliente,
                                    productos_acumulados=productos_acumulados,
                                    desde="referencia_marca_pendiente",
                                    necesidad_ctx_base={
                                        "texto_original": (
                                            necesidad_ctx.get("texto_original") or mensaje
                                        ),
                                        "query_evaluada": referencia,
                                        "marca_intentada": marca,
                                    },
                                )
                            )
                    else:
                        contexto_extra = _marcar_respuesta_segura(
                            f"No encontré un producto con la referencia {referencia} "
                            f"y marca {marca} en el catálogo."
                        )
                        nueva_etapa = "esperando_marca_referencia"
                        necesidad_ctx = {
                            **necesidad_ctx,
                            "opciones_actuales": opciones_marca,
                        }

        # Caso 3b: respuesta a preguntas de descubrimiento (una por turno)
        elif etapa == "descubrimiento" and necesidad_ctx.get("preguntas_pendientes"):
            respuesta_segura, etapa_resp, ctx_actualizado, accion = (
                _continuar_secuencia_preguntas(
                    necesidad_ctx,
                    mensaje,
                    cliente,
                )
            )

            if accion in {"continuar", "esperar_otro"}:
                contexto_extra = respuesta_segura
                nueva_etapa = etapa_resp
                necesidad_ctx = ctx_actualizado
            else:
                query_e = _construir_query_acumulado(ctx_actualizado)
                texto_original = ctx_actualizado.get("texto_original") or query_e

                contexto_extra, nueva_etapa, necesidad_ctx = (
                    await _buscar_y_responder_descubrimiento(
                        query_e=query_e,
                        necesidad_ctx=ctx_actualizado,
                        cliente=cliente,
                        productos_acumulados=productos_acumulados,
                        desde="descubrimiento",
                    )
                )

        # Caso 4: respuesta a descubrimiento sin cola activa
        elif (
            etapa == "descubrimiento"
            and necesidad_ctx.get("texto_original")
            and not _en_flujo_corta_larga(necesidad_ctx)
        ):
            query_e = _construir_query_descubrimiento(necesidad_ctx, mensaje)
            ya_busco_sin_resultado = bool(necesidad_ctx.get("busqueda_sin_resultado"))
            tiene_contexto_tecnico = bool(necesidad_ctx.get("respuestas_tecnicas"))

            if ya_busco_sin_resultado or tiene_contexto_tecnico:
                contexto_extra, nueva_etapa, necesidad_ctx = (
                    await _buscar_y_responder_descubrimiento(
                        query_e=query_e,
                        necesidad_ctx=necesidad_ctx,
                        cliente=cliente,
                        productos_acumulados=productos_acumulados,
                        desde="descubrimiento_refino",
                    )
                )

                if (
                    nueva_etapa == "descubrimiento"
                    and necesidad_ctx.get("busqueda_sin_resultado")
                    and _es_respuesta_afirmativa_corta(mensaje)
                ):
                    pregunta = (
                        "¿Qué especificación podemos ajustar: rango de temperatura, "
                        "tamaño de dial, longitud de bulbo o tipo de conexión?"
                    )
                    contexto_extra = _respuesta_pregunta_unica(
                        cliente,
                        pregunta,
                        "Entiendo. Para afinar la búsqueda en el catálogo:",
                    )
                    necesidad_ctx = {
                        **necesidad_ctx,
                        "busqueda_sin_resultado": True,
                        "preguntas_pendientes": [],
                    }

            else:
                nec = await evaluar_necesidad(query_e)

                if nec["clara"]:
                    contexto_extra, nueva_etapa, necesidad_ctx = (
                        await _buscar_y_responder_descubrimiento(
                            query_e=query_e,
                            necesidad_ctx=necesidad_ctx,
                            cliente=cliente,
                            productos_acumulados=productos_acumulados,
                            desde="descubrimiento",
                        )
                    )
                else:
                    contexto_extra, nueva_etapa, necesidad_ctx = _iniciar_secuencia_preguntas(
                        {
                            "texto_original": necesidad_ctx.get("texto_original"),
                            "query_evaluada": query_e,
                            "dominio": nec.get("dominio", "general"),
                        },
                        nec["preguntas"],
                        cliente,
                        "Aún necesito un dato más:",
                    )

        # Caso 6: código, referencia o búsqueda por producto/instrumento
        else:
            if _en_flujo_corta_larga(necesidad_ctx):
                contexto_extra, nueva_etapa, necesidad_ctx = (
                    await _continuar_descubrimiento_corta_larga(
                        necesidad_ctx=necesidad_ctx,
                        mensaje=mensaje,
                        cliente=cliente,
                        productos_acumulados=productos_acumulados,
                    )
                )
            else:
                modo_busqueda = detectar_modo_busqueda(mensaje)
                logger.info(
                    "Modo de búsqueda detectado: %s mensaje='%s'",
                    modo_busqueda,
                    mensaje[:80],
                )

                if modo_busqueda == "codigo_exacto":
                    tipo, valor = detectar_identificador(mensaje)

                    res = await rama_codigo(valor, tipo, texto_original=mensaje)

                    contexto_extra, nueva_etapa, necesidad_ctx = construir_respuesta_desde_resultado(
                        res=res,
                        cliente=cliente,
                        productos_acumulados=productos_acumulados,
                        desde="codigo",
                        necesidad_ctx_base={
                            "texto_original": mensaje,
                            "query_evaluada": valor,
                            "modo_busqueda": modo_busqueda,
                        },
                    )

                elif modo_busqueda == "esperando_codigo":
                    if _es_pedido_solo_referencia(mensaje):
                        texto_pedido = _respuesta_pedir_referencia()
                    else:
                        texto_pedido = _respuesta_pedir_identificador()
                    contexto_extra = _marcar_respuesta_segura(texto_pedido)
                    nueva_etapa = "esperando_codigo"
                    necesidad_ctx = {
                        "esperando_codigo_producto": True,
                        "texto_original": mensaje,
                        "opciones_actuales": [],
                    }
                    logger.info(
                        "Cliente pide buscar por código/referencia sin valor aún: '%s'",
                        mensaje[:80],
                    )

                elif modo_busqueda == "producto_vago":
                    contexto_extra = _marcar_respuesta_segura(
                        _respuesta_pedir_detalle_producto()
                    )
                    nueva_etapa = "inicio"
                    necesidad_ctx = {
                        "texto_original": mensaje,
                        "opciones_actuales": [],
                    }
                    logger.info(
                        "Cliente pide otro producto sin detalle: '%s'",
                        mensaje[:80],
                    )

                elif modo_busqueda == "hibrida_guiada":
                    contexto_extra, nueva_etapa, necesidad_ctx = (
                        await _iniciar_hibrida_guiada(
                            mensaje=mensaje,
                            cliente=cliente,
                            productos_acumulados=productos_acumulados,
                        )
                    )

                elif modo_busqueda == "hibrida":
                    contexto_extra, nueva_etapa, necesidad_ctx = (
                        await _buscar_y_responder_hibrido(
                            mensaje=mensaje,
                            cliente=cliente,
                            productos_acumulados=productos_acumulados,
                        )
                    )

                elif modo_busqueda == "producto":
                    if _debe_preguntar_antes_de_buscar(mensaje):
                        contexto_extra, nueva_etapa, necesidad_ctx = (
                            await _iniciar_descubrimiento_producto_corta_larga(
                                mensaje=mensaje,
                                cliente=cliente,
                            )
                        )
                    else:
                        contexto_extra, nueva_etapa, necesidad_ctx = (
                            await _buscar_y_responder_descubrimiento(
                                query_e=mensaje,
                                necesidad_ctx={
                                    "texto_original": mensaje,
                                    "modo_busqueda": modo_busqueda,
                                    "categoria_detectada": detectar_categoria(mensaje),
                                },
                                cliente=cliente,
                                productos_acumulados=productos_acumulados,
                                desde="busqueda_producto",
                            )
                        )

                else:
                    nec = await evaluar_necesidad(mensaje)

                    if nec["clara"]:
                        contexto_extra, nueva_etapa, necesidad_ctx = (
                            await _buscar_y_responder_descubrimiento(
                                query_e=mensaje,
                                necesidad_ctx={
                                    "texto_original": mensaje,
                                    "modo_busqueda": "ambiguo",
                                },
                                cliente=cliente,
                                productos_acumulados=productos_acumulados,
                                desde="busqueda",
                            )
                        )
                    else:
                        preguntas = nec.get("preguntas") or []
                        preguntas = [PREGUNTA_INICIAL_NECESIDAD] + [
                            p
                            for p in preguntas
                            if _texto_pregunta(p) != PREGUNTA_INICIAL_NECESIDAD
                        ][:2]
                        contexto_extra, nueva_etapa, necesidad_ctx = _iniciar_secuencia_preguntas(
                            {
                                "texto_original": mensaje,
                                "dominio": nec.get("dominio", "general"),
                                "modo_busqueda": "ambiguo",
                            },
                            preguntas,
                            cliente,
                            "",
                        )


        # Intenciones comerciales transversales
        # Solo se aplican si el estado comercial prioritario no resolvió el turno.
        # Esto evita que datos como cantidad, nombre, empresa o NIT sean tratados
        # como nuevas búsquedas o cambien de etapa por accidente.
        if True:
            if any(w in msg_lower for w in PALABRAS_MAS):
                nueva_etapa = "acumulando"

            elif any(w in msg_lower for w in PALABRAS_FIN):
                nueva_etapa = "cotizacion"

            elif "presupuesto" in msg_lower or "fecha" in msg_lower:
                nueva_etapa = "calificacion"

            elif "rut" in msg_lower or "proforma" in msg_lower:
                nueva_etapa = "proforma"

            elif "pago" in msg_lower or "pse" in msg_lower or "transferencia" in msg_lower:
                nueva_etapa = "pago"
    # ─────────────────────────────────────────────────────────
    # Construcción de contexto para LLM o respuesta segura
    # ─────────────────────────────────────────────────────────

    ctx_cliente = ""
    if cliente:
        partes = []
        if cliente.get("nombre"):
            partes.append(f"Nombre: {cliente['nombre']}")
        if cliente.get("empresa"):
            partes.append(f"Empresa: {cliente['empresa']}")
        if cliente.get("nit"):
            partes.append(f"NIT: {cliente['nit']}")
        if cliente.get("email"):
            partes.append(f"Email: {cliente['email']}")

        if partes:
            ctx_cliente = "[DATOS DEL CLIENTE]\n" + "\n".join(partes) + "\n"

    ctx_carrito = ""
    if productos_acumulados:
        ctx_carrito = f"[CARRITO: {len(productos_acumulados)} producto(s)]\n"

        for i, item in enumerate(productos_acumulados[-5:], start=1):
            prod = item.get("producto", {})
            ctx_carrito += f"{i}. {prod.get('codigo', '—')} | {prod.get('nombre', '—')}"

            if item.get("cantidad"):
                ctx_carrito += f" | cant: {item['cantidad']}"

            ctx_carrito += "\n"

    ctx_faltantes = ""
    if nueva_etapa in {"cotizacion", "calificacion"}:
        faltantes = datos_faltantes(cliente, "cotizacion")
        if faltantes:
            ctx_faltantes = f"[DATO FALTANTE — pregunta solo este: {faltantes[0]}]\n"

    elif nueva_etapa == "proforma":
        faltantes = datos_faltantes(cliente, "proforma")
        if faltantes:
            ctx_faltantes = f"[DATO FALTANTE — pregunta solo este: {faltantes[0]}]\n"

    system = PROMPT_MAESTRO
    partes_ctx = [
        c for c in [ctx_cliente, ctx_carrito, ctx_faltantes, contexto_extra]
        if c
    ]

    if partes_ctx:
        system += "\n\n---\nCONTEXTO ACTUAL:\n" + "\n".join(partes_ctx)

    msg_llm = (
        mensaje_original
        if mensaje_original and mensaje_original.strip()
        else (
            mensaje
            if mensaje.strip()
            else f"[Cliente envió archivo: {archivo_nombre}]"
        )
    )

    respuesta_segura = _extraer_respuesta_segura(contexto_extra)

    if nueva_etapa == "cotizacion_lista" and not respuesta_segura:
        nombre_cliente = (cliente.get("nombre") or "").strip()

        if nombre_cliente:
            respuesta_segura = (
                f"{nombre_cliente}, registré tu solicitud con los "
                "productos y cantidades indicados. La cotización "
                "llegará lo más pronto posible."
            )
        else:
            respuesta_segura = (
                "Registré tu solicitud con los productos y "
                "cantidades indicados. La cotización llegará "
                "lo más pronto posible."
            )

    if respuesta_segura:
        respuesta = respuesta_segura

    else:
        respuesta = await call_nia(
            system=system,
            historial=historial[-20:],
            mensaje_usuario=msg_llm,
        )

        if contiene_placeholder(respuesta):
            logger.warning(
                "Respuesta del LLM contenía placeholders. Se reemplaza por respuesta segura sin resultado."
            )
            respuesta = respuesta_sin_resultado(cliente=cliente)
            nueva_etapa = "descubrimiento"

    # Prefijo de saludo cuando el mensaje venía con saludo + solicitud.
    if greeting_prefix and respuesta:
        resto = respuesta.strip()
        if resto:
            resto = resto[0].upper() + resto[1:]
        respuesta = f"{greeting_prefix} {resto}".strip()

    # Si se pregunta validación técnica de cotización, siempre van botones Sí/No.
    resp_norm = (respuesta or "").lower()
    if "cumple con lo que necesitas t" in resp_norm and "cotizaci" in resp_norm:
        nueva_etapa = "cotizacion_enviada"
        necesidad_ctx = {
            **(necesidad_ctx or {}),
            **_ctx_confirmacion_cotizacion_tecnica(),
        }

    logger.info("Respuesta generada: etapa=%s session=%s", nueva_etapa, session_id)

    turno_user = {
        "role": "user",
        "content": msg_llm,
        "ts": datetime.utcnow().isoformat(),
    }

    turno_nia = {
        "role": "assistant",
        "content": respuesta,
        "ts": datetime.utcnow().isoformat(),
    }

    await _persistir_cliente_permanente(phone_id, cliente)

    await save_session(
        session_id=session_id,
        phone_id=phone_id,
        turnos=historial + [turno_user, turno_nia],
        etapa=nueva_etapa,
        archivo_activo=archivo_activo,
        necesidad_ctx=necesidad_ctx or {},
        cliente=cliente or {},
        productos_acumulados=productos_acumulados or [],
        cotizacion_recibida=cotizacion_recibida,
        archivo_cotizacion=archivo_cotizacion,
        proforma_recibida=proforma_recibida,
        archivo_proforma=archivo_proforma,
    )


    return {
        "respuesta": respuesta,
        "etapa": nueva_etapa,
        "opciones": (necesidad_ctx or {}).get("opciones_actuales") or None,
        "items_resultado": items_resultado or None,
        "cliente": cliente or None,
    }


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.post("/nia/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def nia_chat_texto(request: Request, req: ChatRequest):
    return ChatResponse(**await procesar_turno(
        session_id=req.session_id,
        mensaje=req.mensaje,
        phone_id=req.phone_id,
    ))


@app.post("/nia/chat/archivo", response_model=ChatResponse)
@limiter.limit("10/minute")
async def nia_chat_archivo(
    request: Request,
    session_id: str = Form(...),
    mensaje: str = Form(default=""),
    phone_id: str = Form(default=None),
    archivo: UploadFile = File(...),
):
    contenido = await archivo.read()

    return ChatResponse(**await procesar_turno(
        session_id=session_id,
        mensaje=mensaje,
        phone_id=phone_id,
        archivo_bytes=contenido,
        archivo_nombre=archivo.filename,
    ))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "servicio": "NIA ViaIndustrial",
    }
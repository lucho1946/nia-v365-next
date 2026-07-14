"""
memory.py — Persistencia de sesión y clientes en MongoDB Atlas para NIA v365.

Responsabilidades:
- Cargar variables de entorno desde .env.
- Conectar con MongoDB Atlas.
- Crear índices de sesión temporal y clientes permanentes.
- Leer, guardar y eliminar sesiones conversacionales.
- Guardar datos permanentes del cliente por phone_id/email.

Base de datos:
- MONGO_DB=nia

Colecciones usadas:
- Sesiones temporales: nia_nueva_sessions
- Clientes permanentes: nia_nueva_clientes

TTL sesiones:
- 8 días desde updated_at.

Nota de arquitectura:
- Las sesiones tienen TTL y representan contexto conversacional.
- Los clientes NO tienen TTL; son memoria comercial reutilizable.
"""

import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient


# ============================================================
# CARGA SEGURA DEL .env
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

# override=True: el .env del proyecto manda sobre OPENAI_API_KEY
# u otras vars ya definidas en el entorno de Windows/usuario.
load_dotenv(dotenv_path=ENV_PATH, override=True)


# ============================================================
# CONFIGURACIÓN MONGODB
# ============================================================

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "nia")

# Colección propia para NIA v365.
SESSIONS_COLLECTION = os.getenv("MONGO_SESSIONS_COLLECTION", "nia_nueva_sessions")

# Nueva colección permanente de clientes.
CLIENTES_COLLECTION = os.getenv("MONGO_CLIENTES_COLLECTION", "nia_nueva_clientes")

TTL_SEGUNDOS = 60 * 60 * 24 * 8  # 8 días
MAX_TURNOS = 40
MAX_PRODUCTOS_ACUMULADOS = 20

_client: Optional[AsyncIOMotorClient] = None


def _validar_configuracion_mongo() -> None:
    """
    Valida que exista la configuración mínima de MongoDB.

    Sin esta validación, Motor/PyMongo podría intentar conectarse por defecto
    a localhost:27017, generando errores confusos.
    """
    if not MONGO_URI:
        raise RuntimeError(
            "Falta MONGO_URI en el archivo .env. "
            "Configura la cadena real de MongoDB Atlas."
        )

    if not MONGO_DB:
        raise RuntimeError(
            "Falta MONGO_DB en el archivo .env. "
            "Para este proyecto debe ser, por ahora: MONGO_DB=nia."
        )

    if not SESSIONS_COLLECTION:
        raise RuntimeError(
            "Falta MONGO_SESSIONS_COLLECTION o está vacío. "
            "Usa por defecto: MONGO_SESSIONS_COLLECTION=nia_nueva_sessions."
        )

    if not CLIENTES_COLLECTION:
        raise RuntimeError(
            "Falta MONGO_CLIENTES_COLLECTION o está vacío. "
            "Usa por defecto: MONGO_CLIENTES_COLLECTION=nia_nueva_clientes."
        )


def get_db():
    """
    Retorna la base de datos MongoDB configurada.
    La conexión se crea una sola vez y se reutiliza durante toda la ejecución.
    """
    global _client

    _validar_configuracion_mongo()

    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URI)

    return _client[MONGO_DB]


def get_sessions_collection():
    """
    Retorna la colección oficial de sesiones para NIA v365.
    """
    db = get_db()
    return db[SESSIONS_COLLECTION]


def get_clientes_collection():
    """
    Retorna la colección permanente de clientes para NIA v365.
    """
    db = get_db()
    return db[CLIENTES_COLLECTION]


# ============================================================
# ÍNDICES
# ============================================================

async def ensure_index_clientes():
    """
    Crea índices para memoria permanente de clientes.

    Índices:
    - phone_id único cuando exista.
    - email normalizado para búsquedas futuras.
    - updated_at para consultas administrativas.
    """
    collection = get_clientes_collection()

    await collection.create_index(
        [("phone_id", 1)],
        unique=True,
        background=True,
        name="idx_unique_phone_id",
        partialFilterExpression={"phone_id": {"$type": "string"}},
    )

    await collection.create_index(
        [("email", 1)],
        background=True,
        name="idx_email_cliente",
        partialFilterExpression={"email": {"$type": "string"}},
    )

    await collection.create_index(
        [("updated_at", -1)],
        background=True,
        name="idx_clientes_updated_at",
    )


async def ensure_index():
    """
    Crea los índices necesarios para sesiones de NIA v365 y clientes permanentes.

    Sesiones:
    - TTL por updated_at.
    - session_id único.

    Clientes:
    - phone_id único.
    - email consultable.
    """
    collection = get_sessions_collection()

    await collection.create_index(
        [("updated_at", 1)],
        expireAfterSeconds=TTL_SEGUNDOS,
        background=True,
        name="idx_ttl_updated_at",
    )

    await collection.create_index(
        [("session_id", 1)],
        unique=True,
        background=True,
        name="idx_unique_session_id",
        partialFilterExpression={"session_id": {"$type": "string"}},
    )

    await ensure_index_clientes()

    from learning_memory import ensure_index_aprendizaje

    await ensure_index_aprendizaje()


# ============================================================
# SESIONES TEMPORALES
# ============================================================

async def get_session(session_id: str) -> Optional[dict]:
    """
    Obtiene una sesión por session_id.

    Retorna:
    - dict si existe la sesión.
    - None si no existe.
    """
    collection = get_sessions_collection()

    return await collection.find_one(
        {"session_id": session_id},
        {"_id": 0},
    )


async def save_session(
    session_id: str,
    turnos: list,
    etapa: str,
    phone_id: Optional[str] = None,
    archivo_activo: Optional[dict] = None,
    necesidad_ctx: Optional[dict] = None,
    cliente: Optional[dict] = None,
    productos_acumulados: Optional[list] = None,
    cotizacion_recibida: bool = False,
    archivo_cotizacion: Optional[str] = None,
    proforma_recibida: bool = False,
    archivo_proforma: Optional[str] = None,
):
    """
    Guarda o actualiza una sesión conversacional.

    Reglas:
    - Conserva máximo los últimos 40 turnos.
    - Conserva máximo los últimos 20 productos acumulados.
    - Reinicia el TTL actualizando updated_at.
    - Guarda banderas futuras de cotización/proforma recibida.
    """
    if not session_id:
        raise ValueError("session_id es obligatorio para guardar una sesión.")

    collection = get_sessions_collection()

    payload = {
        "session_id": session_id,
        "phone_id": phone_id,
        "turnos": (turnos or [])[-MAX_TURNOS:],
        "etapa": etapa,
        "updated_at": datetime.now(timezone.utc),
        "necesidad_ctx": necesidad_ctx or {},
        "cliente": cliente or {},
        "productos_acumulados": (productos_acumulados or [])[-MAX_PRODUCTOS_ACUMULADOS:],
        "cotizacion_recibida": bool(cotizacion_recibida),
        "archivo_cotizacion": archivo_cotizacion,
        "proforma_recibida": bool(proforma_recibida),
        "archivo_proforma": archivo_proforma,
    }

    if archivo_activo is not None:
        payload["archivo_activo"] = archivo_activo

    await collection.update_one(
        {"session_id": session_id},
        {"$set": payload},
        upsert=True,
    )


async def delete_session(session_id: str):
    """
    Elimina una sesión manualmente si el cliente pide reiniciar.
    """
    if not session_id:
        raise ValueError("session_id es obligatorio para eliminar una sesión.")

    collection = get_sessions_collection()

    await collection.delete_one({"session_id": session_id})


# ============================================================
# CLIENTES PERMANENTES
# ============================================================

def _normalizar_email(email: Optional[str]) -> Optional[str]:
    """
    Normaliza email para almacenamiento/búsqueda.
    """
    if not email:
        return None

    email = str(email).strip().lower()

    return email or None


def _limpiar_valor_cliente(valor: Any) -> Any:
    """
    Limpia valores antes de guardarlos en clientes permanentes.

    Evita guardar strings vacíos y conserva estructuras útiles.
    """
    if isinstance(valor, str):
        valor = valor.strip()
        return valor or None

    return valor


def _filtrar_datos_cliente(datos: dict) -> dict:
    """
    Permite guardar solo campos comerciales seguros.

    No guardamos todo el contexto de sesión en clientes permanentes.
    """
    if not isinstance(datos, dict):
        return {}

    campos_permitidos = {
        "nombre",
        "email",
        "empresa",
        "nit",
        "rut",
        "telefono",
        "phone_id",
        "canal",
    }

    filtrado = {}

    for campo in campos_permitidos:
        valor = _limpiar_valor_cliente(datos.get(campo))

        if valor is not None:
            filtrado[campo] = valor

    if "email" in filtrado:
        filtrado["email"] = _normalizar_email(filtrado.get("email"))

    return filtrado


async def upsert_cliente(phone_id: str, datos: dict) -> Optional[dict]:
    """
    Crea o actualiza memoria permanente del cliente.

    Reglas:
    - Requiere phone_id.
    - No sobreescribe con campos vacíos.
    - Usa $set para datos nuevos y $setOnInsert para fecha de creación.
    """
    if not phone_id:
        return None

    collection = get_clientes_collection()

    datos_filtrados = _filtrar_datos_cliente(datos or {})
    datos_filtrados["phone_id"] = str(phone_id).strip()
    datos_filtrados["updated_at"] = datetime.now(timezone.utc)

    if "telefono" not in datos_filtrados:
        datos_filtrados["telefono"] = str(phone_id).strip()

    await collection.update_one(
        {"phone_id": str(phone_id).strip()},
        {
            "$set": datos_filtrados,
            "$setOnInsert": {
                "created_at": datetime.now(timezone.utc),
            },
        },
        upsert=True,
    )

    return await get_cliente(phone_id)


async def get_cliente(phone_id: str) -> Optional[dict]:
    """
    Obtiene datos permanentes del cliente por phone_id.
    """
    if not phone_id:
        return None

    collection = get_clientes_collection()

    return await collection.find_one(
        {"phone_id": str(phone_id).strip()},
        {"_id": 0},
    )


async def get_cliente_por_email(email: str) -> Optional[dict]:
    """
    Obtiene datos permanentes del cliente por email.
    """
    email_normalizado = _normalizar_email(email)

    if not email_normalizado:
        return None

    collection = get_clientes_collection()

    return await collection.find_one(
        {"email": email_normalizado},
        {"_id": 0},
    )
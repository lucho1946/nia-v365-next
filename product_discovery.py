"""
product_discovery.py — Descubrimiento guiado por catálogo real.

Flujo búsqueda por producto (sin código):
1. Buscar palabra clave en NIVEL_1.
2. Contar tipos y ofrecer los 3 más frecuentes + "Otro".
3. Tras elegir tipo, analizar DESCRIPCION_LARGA_PRE de ese grupo.
4. Generar 2 preguntas sobre los atributos con mayor peso discriminante.
5. Buscar el SKU con el contexto acumulado.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Optional

from memory import get_db
from product_fields import KEYWORDS_TO_CATEGORIA, get_campos
from knowledge import contexto_para_agente

logger = logging.getLogger("nia.product_discovery")

PRODUCTS_COLLECTION = "products_catalog"
MAX_PRODUCTOS_ANALISIS = 250

CAMPOS_CANONICOS = {
    "rango_temperatura": "rango",
    "resolucion": "resolucion",
    "conexion": "conexion",
    "recalibrable": "recalibrable",
}


def _campo_canonico(campo: str) -> str:
    return CAMPOS_CANONICOS.get(campo, campo)


ETIQUETAS_CAMPO = {
    "rango": "rango de temperatura",
    "rango_temperatura": "rango de temperatura",
    "conexion": "tipo de conexión o montaje",
    "longitud_vastago": "longitud de bulbo / vástago",
    "conexion_montaje": "tipo de conexión o montaje",
    "serie_modelo": "serie o modelo",
    "diametro_dial": "diámetro del dial",
    "senal_salida": "señal de salida",
    "rango_presion": "rango de presión",
    "material": "material",
    "proteccion": "grado de protección (IP)",
    "alimentacion": "alimentación eléctrica",
    "resolucion": "resolución",
    "recalibrable": "recalibración",
    "potencia": "potencia",
    "voltaje": "voltaje",
    "termostato": "opción de termostato",
}

EXTRACTORES_ATRIBUTOS: list[tuple[str, re.Pattern]] = [
    (
        "rango_temperatura",
        re.compile(
            r"(-?\d+)\s*[-–toa]+\s*(-?\d+)\s*([CFcf°])",
            re.IGNORECASE,
        ),
    ),
    (
        "rango_temperatura",
        re.compile(r"temperature\s*range[:\s,]*([^,■|]+)", re.IGNORECASE),
    ),
    (
        "longitud_vastago",
        re.compile(
            r"(\d+[\.,]?\d*)\s*(?:in\.?|inch(?:es)?)\.?\s*(?:stem|bulb|vástago|vastago|bulbo)",
            re.IGNORECASE,
        ),
    ),
    (
        "longitud_vastago",
        re.compile(r"stem\s*length[:\s,]*([^,■|]+)", re.IGNORECASE),
    ),
    (
        "conexion_montaje",
        re.compile(
            r"((?:1/2|3/8|1/4|1/4)\s*[-]?\s*(?:NPT|NF)[^,■|]*)",
            re.IGNORECASE,
        ),
    ),
    (
        "conexion_montaje",
        re.compile(r"(mounting\s*bushing[^,■|]*)", re.IGNORECASE),
    ),
    (
        "serie_modelo",
        re.compile(r"([HJ]\s*series[^,■|]*)", re.IGNORECASE),
    ),
    (
        "diametro_dial",
        re.compile(
            r"(\d+[\.,]?\d*)\s*(?:in\.?|inch(?:es)?)\s*(?:dial|carátula)",
            re.IGNORECASE,
        ),
    ),
    (
        "senal_salida",
        re.compile(r"(4\s*[-–]\s*20\s*mA|0\s*[-–]\s*10\s*V|hart|modbus)", re.IGNORECASE),
    ),
    (
        "rango_presion",
        re.compile(
            r"(\d+[\.,]?\d*)\s*[-–to]+\s*(\d+[\.,]?\d*)\s*(bar|psi|mbar|kpa)",
            re.IGNORECASE,
        ),
    ),
    (
        "material",
        re.compile(r"\b(acero\s+inoxidable|inox|bronce|latón|laton|pvc|ptfe)\b", re.IGNORECASE),
    ),
    (
        "proteccion",
        re.compile(r"\b(IP\s*\d{2})\b", re.IGNORECASE),
    ),
    (
        "alimentacion",
        re.compile(r"\b(24\s*VDC?|110\s*V|220\s*V|12\s*V)\b", re.IGNORECASE),
    ),
    (
        "potencia",
        re.compile(r"\b(\d+)\s*W\b", re.IGNORECASE),
    ),
    (
        "voltaje",
        re.compile(r"\b(\d+)\s*V(?:AC|DC)?\b", re.IGNORECASE),
    ),
    (
        "termostato",
        re.compile(r"\b(sin\s+termostato|con\s+termostato)\b", re.IGNORECASE),
    ),
]


def _normalizar_texto(valor: str) -> str:
    if not valor:
        return ""

    texto = str(valor).lower().strip()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(
        caracter
        for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )
    texto = re.sub(r"\s+", " ", texto)
    return texto


def extraer_palabra_clave(texto: str) -> Optional[str]:
    """
    Obtiene la palabra de instrumento/herramienta desde el mensaje.
    """
    texto_lower = (texto or "").lower()
    for kw in sorted(KEYWORDS_TO_CATEGORIA.keys(), key=len, reverse=True):
        if kw in texto_lower:
            return kw
    return None


def _limpiar_valor_extraido(campo: str, match: re.Match) -> str:
    if campo == "rango_temperatura" and match.lastindex and match.lastindex >= 3:
        return f"{match.group(1)}-{match.group(2)}{match.group(3).upper()}"

    if campo == "rango_presion" and match.lastindex and match.lastindex >= 3:
        return f"{match.group(1)}-{match.group(2)} {match.group(3).lower()}"

    if campo == "longitud_vastago" and match.lastindex and match.lastindex >= 1:
        if match.re.pattern.find("stem") >= 0 and match.lastindex == 1:
            return match.group(1).strip()
        return f"{match.group(1)} in"

    if campo == "potencia" and match.lastindex:
        return f"{match.group(1)} W"

    if campo == "voltaje" and match.lastindex:
        return f"{match.group(1)} V"

    if campo == "termostato" and match.lastindex:
        raw = _normalizar_texto(match.group(1))
        if "sin" in raw:
            return "Sin termostato"
        return "Con termostato"

    valor = match.group(1) if match.lastindex else match.group(0)
    return re.sub(r"\s+", " ", str(valor).strip())


def extraer_atributos_descripcion_larga(texto: str) -> dict[str, str]:
    """
    Extrae atributos técnicos desde DESCRIPCION_LARGA_PRE.
    Soporta texto libre (■), formato estructurado (¦ campo: valor)
    y bloques titulo/valor en CARACTERISTICAS.
    """
    if not texto:
        return {}

    atributos: dict[str, str] = {}
    texto_str = str(texto)

    for titulo, valor in re.findall(
        r"\{'titulo':\s*'([^']+)',\s*'valor':\s*'([^']*)'\}",
        texto_str,
        flags=re.IGNORECASE,
    ):
        campo_norm = _normalizar_texto(titulo).replace(" ", "_")
        valor_norm = str(valor).strip()
        if campo_norm and valor_norm:
            atributos[campo_norm] = valor_norm

    for separador in ("¦", "■", "|"):
        if separador in texto_str:
            partes = texto_str.split(separador)
            for parte in partes:
                parte = parte.strip()
                if ":" in parte:
                    campo, valor = parte.split(":", 1)
                    campo_norm = _normalizar_texto(campo).replace(" ", "_")
                    valor_norm = valor.strip()
                    if campo_norm and valor_norm:
                        atributos[campo_norm] = valor_norm
            break

    for campo, patron in EXTRACTORES_ATRIBUTOS:
        if campo in atributos:
            continue
        match = patron.search(texto_str)
        if match:
            atributos[campo] = _limpiar_valor_extraido(campo, match)

    # En fichas de producto: si hay potencia y no se menciona termostato,
    # contrastar con las variantes que sí dicen "sin termostato".
    # No aplicar a respuestas cortas del cliente (ej. solo "60 W").
    texto_l = texto_str.lower()
    es_ficha = len(texto_str) >= 40 or any(
        k in texto_l for k in ("calentador", "gabinete", "heater", "caja")
    )
    if (
        es_ficha
        and "potencia" in atributos
        and "termostato" not in atributos
        and "sin termostato" not in texto_l
        and "termostato" not in texto_l
    ):
        atributos["termostato"] = "Con termostato"

    return atributos


def hay_familia_ambigua_por_dl(
    candidatos: list[dict],
    tolerancia_score: float = 0.05,
) -> tuple[bool, list[dict], list[dict]]:
    """
    Detecta familia de variantes con score similar que se diferencian en DL.

    Retorna: (es_ambigua, grupo, preguntas_desde_dl)
    """
    if not candidatos or len(candidatos) < 2:
        return False, [], []

    ordenados = sorted(
        candidatos,
        key=lambda p: float(p.get("_score") or 0.0),
        reverse=True,
    )
    top_score = float(ordenados[0].get("_score") or 0.0)
    grupo = [
        p
        for p in ordenados
        if abs(float(p.get("_score") or 0.0) - top_score) <= tolerancia_score
    ]
    if len(grupo) < 2:
        return False, [], []

    nombres = {
        str(p.get("descripcion_corta") or p.get("nombre") or "").strip().lower()
        for p in grupo
    }
    mismas_corta = len([n for n in nombres if n]) == 1

    dls = [str(p.get("descripcion_larga") or "").strip() for p in grupo]
    dls = [d for d in dls if d]
    if len(dls) < 2:
        return False, [], []

    campos = analizar_campos_discriminantes(dls, top_n=3)
    if not campos:
        return False, [], []

    # Solo preguntar si comparten nombre corto o hay discriminantes claros en DL.
    if not mismas_corta and len(grupo) < 3:
        return False, [], []

    preguntas = generar_preguntas_desde_campos(campos)
    # Quitar preguntas genéricas o con una sola opción real (no discriminan).
    perguntas_utiles = []
    for preg in preguntas:
        opts = [
            o
            for o in (preg.get("opciones") or [])
            if str(o.get("valor") or "").lower() not in {"otro", ""}
        ]
        opts_dedup = _deduplicar_opciones_valores(
            [str(o.get("valor") or o.get("label") or "") for o in opts]
        )
        if len(opts_dedup) >= 2:
            perguntas_utiles.append(preg)
    if not perguntas_utiles:
        return False, [], []

    return True, grupo, perguntas_utiles[:2]


def _opciones_utiles_pregunta(pregunta: dict) -> list[dict]:
    return [
        o
        for o in ((pregunta or {}).get("opciones") or [])
        if str(o.get("valor") or "").lower() not in {"otro", ""}
    ]


def filtrar_preguntas_familia_no_respondidas(
    preguntas: list[dict],
    respuestas: list[str],
) -> list[dict]:
    """
    Descarta preguntas cuyo único valor útil ya fue respondido
    (evita bucles: rango -50 a 300 → otra vez el mismo rango).
    """
    claves_resp = {
        _clave_dedup_opcion(r)
        for r in (respuestas or [])
        if str(r or "").strip() and _clave_dedup_opcion(r)
    }
    claves_resp |= {
        _normalizar_texto(r)
        for r in (respuestas or [])
        if str(r or "").strip()
    }

    limpias = []
    for preg in preguntas or []:
        if not isinstance(preg, dict):
            continue
        opts = _opciones_utiles_pregunta(preg)
        if len(opts) < 2:
            continue
        claves_opts = {
            _clave_dedup_opcion(str(o.get("valor") or o.get("label") or ""))
            for o in opts
        }
        claves_opts.discard("")
        # Si todas las opciones útiles ya están respondidas, no aporta.
        if claves_opts and claves_opts.issubset(claves_resp):
            continue
        # Si tras quitar lo ya respondido queda < 2 opciones, tampoco.
        opts_nuevas = [
            o
            for o in opts
            if _clave_dedup_opcion(str(o.get("valor") or o.get("label") or ""))
            not in claves_resp
            and _normalizar_texto(str(o.get("valor") or "")) not in claves_resp
        ]
        if len(opts_nuevas) < 2 and len(opts) >= 2:
            # Puede haber opciones nuevas: conservar si hay ≥2 opciones originales
            # y no son exactamente la respuesta previa (caso 3 opciones con 1 ya dada).
            if len(opts_nuevas) == 0:
                continue
        if len(_deduplicar_opciones_valores(
            [str(o.get("valor") or "") for o in opts]
        )) < 2:
            continue
        limpias.append(preg)
    return limpias


def filtrar_candidatos_por_respuestas_dl(
    candidatos: list[dict],
    respuestas: list[str],
) -> list[dict]:
    """
    Reduce variantes de una familia usando respuestas técnicas (potencia, V, etc.).
    Coincide contra atributos extraídos de DESCRIPCION_LARGA o texto libre en DL.
    """
    if not candidatos:
        return []

    respuestas_utiles = [
        re.sub(r"\s+", " ", str(r or "").strip())
        for r in (respuestas or [])
        if str(r or "").strip()
    ]
    if not respuestas_utiles:
        return list(candidatos)

    attrs_cliente: dict[str, str] = {}
    for resp in respuestas_utiles:
        attrs_cliente.update(extraer_atributos_descripcion_larga(resp))

    filtrados = list(candidatos)
    for campo, valor_cliente in attrs_cliente.items():
        valor_norm = _normalizar_texto(valor_cliente)
        clave_cliente = _clave_dedup_opcion(valor_cliente)
        if not valor_norm and not clave_cliente:
            continue
        siguiente = []
        for prod in filtrados:
            dl = str(prod.get("descripcion_larga") or "")
            attrs_prod = extraer_atributos_descripcion_larga(dl)
            valor_prod = (
                attrs_prod.get(campo)
                or attrs_prod.get(_campo_canonico(campo))
                or attrs_prod.get("rango")
                or attrs_prod.get("rango_temperatura")
            )
            if valor_prod:
                if valor_norm and _normalizar_texto(valor_prod) == valor_norm:
                    siguiente.append(prod)
                    continue
                if clave_cliente and _clave_dedup_opcion(valor_prod) == clave_cliente:
                    siguiente.append(prod)
                    continue
            if valor_norm and valor_norm in _normalizar_texto(dl):
                siguiente.append(prod)
        if siguiente:
            filtrados = siguiente

    # Respuestas etiquetadas (chips): match por clave dedup contra attrs de DL.
    for resp in respuestas_utiles:
        clave = _clave_dedup_opcion(resp)
        if not clave:
            continue
        siguiente = []
        for prod in filtrados:
            dl = str(prod.get("descripcion_larga") or "")
            attrs_prod = extraer_atributos_descripcion_larga(dl)
            valores_attr = [str(v) for v in attrs_prod.values() if v]
            if any(_clave_dedup_opcion(v) == clave for v in valores_attr):
                siguiente.append(prod)
                continue
            resp_norm = _normalizar_texto(resp)
            if resp_norm and resp_norm in _normalizar_texto(dl):
                siguiente.append(prod)
        if siguiente:
            filtrados = siguiente

    return filtrados


def _formatear_nivel_1(valor: str) -> str:
    """
    Presenta NIVEL_1 de forma legible para el cliente.
    """
    texto = str(valor or "").strip().replace("-", " ")
    return re.sub(r"\s+", " ", texto)


async def obtener_tipos_nivel_1(
    palabra_clave: str,
    top: int = 3,
) -> list[dict]:
    """
    Busca la palabra en NIVEL_1 y devuelve los tipos más frecuentes.
    """
    palabra = _normalizar_texto(palabra_clave)
    if not palabra:
        return []

    db = get_db()
    collection = db[PRODUCTS_COLLECTION]

    pipeline = [
        {
            "$match": {
                "NIVEL_1": {"$regex": re.escape(palabra), "$options": "i"},
            }
        },
        {
            "$group": {
                "_id": "$NIVEL_1",
                "count": {"$sum": 1},
            }
        },
        {"$sort": {"count": -1}},
        {"$limit": max(top, 3)},
    ]

    resultados = await collection.aggregate(pipeline).to_list(max(top, 3))

    tipos = []
    for item in resultados:
        nivel = str(item.get("_id") or "").strip()
        if not nivel:
            continue
        tipos.append(
            {
                "nivel_1": nivel,
                "count": int(item.get("count") or 0),
            }
        )

    logger.info(
        "Tipos NIVEL_1 para '%s': %s",
        palabra_clave,
        [(t["nivel_1"], t["count"]) for t in tipos[:top]],
    )

    return tipos[:top]


SINONIMOS_PALABRA_NIVEL_1 = {
    "bota": ["bota", "botas", "calzado", "dielectric"],
    "botas": ["bota", "botas", "calzado", "dielectric"],
    "calzado": ["calzado", "dielectric", "bota"],
    "guante": ["guante", "guantes", "dielectric"],
    "guantes": ["guante", "guantes", "dielectric"],
    "electricidad": ["electric", "dielectric", "electrico"],
    "electrico": ["electric", "dielectric", "electrico"],
}


def _sinonimos_palabra_clave(palabra: str) -> list[str]:
    palabra = _normalizar_texto(palabra)
    if not palabra:
        return []
    base = SINONIMOS_PALABRA_NIVEL_1.get(palabra, [palabra])
    vistos = set()
    resultado = []
    for item in base:
        norm = _normalizar_texto(item)
        if norm and norm not in vistos:
            vistos.add(norm)
            resultado.append(norm)
    return resultado


def _filtrar_tipos_por_contexto_epi(tipos: list[dict], texto: str) -> list[dict]:
    from discovery_guards import es_producto_epi_seguridad

    if not tipos or not es_producto_epi_seguridad(texto):
        return tipos

    t = _normalizar_texto(texto)
    if any(k in t for k in ("bota", "calzado")):
        preferidos = [
            tipo
            for tipo in tipos
            if "calzado" in str(tipo.get("nivel_1") or "").lower()
        ]
        if preferidos:
            return preferidos

    if "guante" in t:
        preferidos = [
            tipo
            for tipo in tipos
            if "guante" in str(tipo.get("nivel_1") or "").lower()
        ]
        if preferidos:
            return preferidos

    return tipos


async def obtener_tipos_nivel_1_desde_campos_producto(
    texto_busqueda: str,
    top: int = 3,
) -> list[dict]:
    """
    Busca tipos NIVEL_1 a partir de texto en nombre/descripción del producto.
    Útil cuando el slug NIVEL_1 no contiene la palabra del cliente (ej. botas → calzado-dielectrico).
    """
    texto_norm = _normalizar_texto(texto_busqueda)
    tokens = _tokens_busqueda_cliente(texto_norm)
    if not tokens:
        return []

    db = get_db()
    collection = db[PRODUCTS_COLLECTION]

    condiciones = []
    for token in tokens[:6]:
        condiciones.extend(
            [
                {"NOMBRE_PRODUCTO": {"$regex": re.escape(token), "$options": "i"}},
                {"DESCRIPCION": {"$regex": re.escape(token), "$options": "i"}},
                {"DESCRIPCION_CORTA": {"$regex": re.escape(token), "$options": "i"}},
                {"NIVEL_1": {"$regex": re.escape(token), "$options": "i"}},
            ]
        )

    pipeline = [
        {"$match": {"$or": condiciones}},
        {"$group": {"_id": "$NIVEL_1", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": max(top, 6) * 3},
    ]

    resultados = await collection.aggregate(pipeline).to_list(max(top, 6) * 3)
    tipos = []
    for item in resultados:
        nivel = str(item.get("_id") or "").strip()
        if not nivel:
            continue
        tipos.append({"nivel_1": nivel, "count": int(item.get("count") or 0)})

    tipos = _filtrar_tipos_por_contexto_epi(tipos, texto_busqueda)
    logger.info(
        "Tipos NIVEL_1 desde campos producto '%s': %s",
        texto_busqueda[:80],
        [(t["nivel_1"], t["count"]) for t in tipos[:top]],
    )
    return tipos[:top]


async def resolver_tipos_catalogo_inicio(
    palabra_clave: str,
    mensaje: str,
    busqueda_textual: bool,
    top: int = 3,
) -> list[dict]:
    """
    Resuelve tipos NIVEL_1 para el inicio del flujo corta_larga.
    Combina búsqueda por slug, sinónimos y campos de producto.
    """
    texto_completo = str(mensaje or "").strip()
    if palabra_clave and palabra_clave not in _normalizar_texto(texto_completo):
        texto_completo = f"{palabra_clave} {texto_completo}".strip()

    if busqueda_textual:
        tipos = await obtener_tipos_nivel_1_por_texto(palabra_clave, mensaje, top=top)
        tipos = _filtrar_tipos_por_contexto_epi(tipos, texto_completo)
        if tipos:
            return tipos

    tipos = await obtener_tipos_nivel_1(palabra_clave, top=top)
    if tipos:
        return _filtrar_tipos_por_contexto_epi(tipos, texto_completo)

    for sinonimo in _sinonimos_palabra_clave(palabra_clave):
        tipos = await obtener_tipos_nivel_1(sinonimo, top=top)
        if tipos:
            tipos = _filtrar_tipos_por_contexto_epi(tipos, texto_completo)
            if tipos:
                logger.info(
                    "Tipos NIVEL_1 por sinónimo '%s' de '%s': %s",
                    sinonimo,
                    palabra_clave,
                    [t["nivel_1"] for t in tipos],
                )
                return tipos

    tipos = await obtener_tipos_nivel_1_desde_campos_producto(texto_completo, top=top)
    return tipos


# Alias de compatibilidad
obtener_tipos_descripcion_corta = obtener_tipos_nivel_1

PALABRAS_FUNCIONALES_BUSQUEDA = {
    "para",
    "con",
    "de",
    "del",
    "la",
    "el",
    "los",
    "las",
    "un",
    "una",
    "en",
    "y",
    "o",
    "necesito",
    "necesteto",
    "quiero",
    "busco",
    "requiero",
}


def _tokens_busqueda_cliente(texto: str) -> list[str]:
    texto_norm = _normalizar_texto(texto)
    return [
        token
        for token in texto_norm.split()
        if len(token) >= 3 and token not in PALABRAS_FUNCIONALES_BUSQUEDA
    ]


async def obtener_tipos_nivel_1_por_texto(
    palabra_clave: str,
    texto_busqueda: str,
    top: int = 3,
) -> list[dict]:
    """
    Busca tipos NIVEL_1 del instrumento más cercanos al texto del cliente.
    Se usa cuando el cliente escribe 2+ palabras (ej. termometro digital).
    """
    from difflib import SequenceMatcher

    palabra = _normalizar_texto(palabra_clave)
    texto_busqueda = str(texto_busqueda or "").strip()
    texto_norm = _normalizar_texto(texto_busqueda)
    if palabra and palabra not in texto_norm:
        texto_norm = _normalizar_texto(f"{palabra_clave} {texto_busqueda}".strip())
    tokens = _tokens_busqueda_cliente(texto_norm)
    tokens_extra = [t for t in tokens if t != palabra]

    if not palabra or not texto_norm:
        return []

    db = get_db()
    collection = db[PRODUCTS_COLLECTION]

    candidatos = await obtener_tipos_nivel_1(palabra_clave, top=40)
    if not candidatos:
        return []

    puntuados = []
    for tipo in candidatos:
        nivel = str(tipo.get("nivel_1") or "").strip()
        if not nivel:
            continue

        bloque = _normalizar_texto(f"{nivel} {_formatear_nivel_1(nivel)}")
        similitud = SequenceMatcher(None, texto_norm, bloque).ratio()

        for token in tokens:
            if token in bloque:
                similitud += 0.35

        for token in tokens_extra:
            if token not in bloque:
                similitud -= 0.2

        puntuados.append((similitud, tipo))

    puntuados.sort(key=lambda item: item[0], reverse=True)
    tipos = [tipo for sim, tipo in puntuados if sim > 0.1][:top]

    if not tipos:
        tipos = candidatos[:top]

    logger.info(
        "Tipos NIVEL_1 por texto '%s' instrumento='%s': %s",
        texto_busqueda,
        palabra_clave,
        [(t["nivel_1"], t["count"]) for t in tipos],
    )

    return tipos[:top]


def generar_pregunta_seleccion_otro(
    palabra_clave: str,
    tipos: list[dict],
    texto_busqueda: str,
) -> dict:
    """
    Segunda selección solo para flujo Otro: 3 alternativas textuales en botones.
    """
    instrumento = palabra_clave.replace("_", " ")
    opciones = []

    for idx, tipo in enumerate(tipos[:3], start=1):
        opciones.append(
            {
                "id": str(idx),
                "label": _formatear_nivel_1(tipo["nivel_1"]),
                "valor": str(idx),
            }
        )

    opciones.append(
        {
            "id": str(len(opciones) + 1),
            "label": "Otro",
            "valor": "otro",
        }
    )

    descripcion = texto_busqueda.strip()
    return {
        "texto": (
            f"Para \"{descripcion}\", estos son los tipos de {instrumento} "
            "más cercanos en catálogo. ¿Cuál se ajusta mejor?"
        ),
        "opciones": opciones,
    }


async def obtener_descripciones_largas_por_nivel_1(nivel_1: str) -> list[str]:
    """
    Recupera DESCRIPCION_LARGA_PRE de todos los productos de un NIVEL_1.
    """
    if not nivel_1:
        return []

    db = get_db()
    collection = db[PRODUCTS_COLLECTION]

    cursor = collection.find(
        {"NIVEL_1": nivel_1},
        {"DESCRIPCION_LARGA_PRE": 1, "CARACTERISTICAS": 1, "DIMENSION": 1, "_id": 0},
    ).limit(MAX_PRODUCTOS_ANALISIS)

    docs = await cursor.to_list(MAX_PRODUCTOS_ANALISIS)
    textos = []

    for doc in docs:
        bloques = [
            doc.get("DESCRIPCION_LARGA_PRE"),
            doc.get("CARACTERISTICAS"),
            doc.get("DIMENSION"),
        ]
        bloque = " ".join(str(b).strip() for b in bloques if b)
        if bloque:
            textos.append(bloque)

    return textos


# Alias de compatibilidad
async def obtener_descripciones_largas_por_tipo(nivel_1: str) -> list[str]:
    return await obtener_descripciones_largas_por_nivel_1(nivel_1)


def analizar_campos_discriminantes(
    descripciones_largas: list[str],
    top_n: int = 2,
) -> list[dict]:
    """
    Calcula los atributos con mayor peso discriminante dentro de un tipo.
    Peso = cobertura × valores_distintos.
    """
    if not descripciones_largas:
        return []

    total = len(descripciones_largas)
    valores_por_campo: dict[str, list[str]] = defaultdict(list)

    for descripcion in descripciones_largas:
        atributos = extraer_atributos_descripcion_larga(descripcion)
        for campo, valor in atributos.items():
            valor_limpio = re.sub(r"\s+", " ", str(valor).strip())
            if valor_limpio:
                valores_por_campo[_campo_canonico(campo)].append(valor_limpio)

    candidatos = []

    for campo, valores in valores_por_campo.items():
        if not valores:
            continue

        valores_utiles = [
            v
            for v in valores
            if v
            and len(v) <= 80
            and "[{'titulo'" not in v
            and "titulo':" not in v
        ]

        if not valores_utiles:
            continue

        # Deduplicar semánticamente (ej. "-50 a 300°C" vs "-50 a 300 ° C / °F")
        # para no tratar el mismo rango como varios discriminantes.
        valores_dedup = _deduplicar_opciones_valores(valores_utiles)
        if len(valores_dedup) <= 1:
            continue

        claves = [_clave_dedup_opcion(v) for v in valores_utiles if _clave_dedup_opcion(v)]
        conteo_claves = Counter(claves)
        distinct = len(conteo_claves)
        cobertura = len(valores_utiles) / total

        if distinct <= 1:
            continue

        peso = distinct * cobertura
        valores_frecuentes = valores_dedup[:5]

        candidatos.append(
            {
                "campo": campo,
                "peso": peso,
                "cobertura": cobertura,
                "distinct": distinct,
                "valores_frecuentes": valores_frecuentes,
            }
        )

    candidatos.sort(key=lambda x: x["peso"], reverse=True)

    seleccionados = []
    campos_vistos = set()
    for candidato in candidatos:
        canonico = _campo_canonico(candidato["campo"])
        if canonico in campos_vistos:
            continue
        campos_vistos.add(canonico)
        candidato["campo"] = canonico
        seleccionados.append(candidato)
        if len(seleccionados) >= top_n:
            break

    logger.info(
        "Campos discriminantes top=%s de %s productos: %s",
        top_n,
        total,
        [(c["campo"], round(c["peso"], 2)) for c in seleccionados],
    )

    return seleccionados


def _es_valor_temperatura(valor: str) -> bool:
    v = (valor or "").lower()
    if re.search(r"\b(psi|bar|mbar|kpa|mpa|pa|inhg|mmhg|kg/cm)\b", v):
        return False
    if re.search(r"[°º]|celsius|fahrenheit|kelvin|\b(c|f|k)\b", v):
        return True
    if re.search(r"-?\d+\s*[-–a]\s*-?\d+", v) and not re.search(
        r"\b(psi|bar|mbar|kpa)\b", v
    ):
        return True
    return False


def _es_valor_presion(valor: str) -> bool:
    v = (valor or "").lower()
    if re.search(r"\b(psi|bar|mbar|kpa|mpa|pa|inhg|mmhg|kg/cm)\b", v):
        return True
    return False


def _filtrar_valores_por_dominio(
    valores: list[str],
    campo: str,
    dominio: str,
) -> list[str]:
    if not valores or not dominio:
        return valores

    filtrados: list[str] = []
    for valor in valores:
        if dominio == "temperatura":
            if campo in ("rango", "rango_temperatura", "temperatura"):
                if _es_valor_temperatura(valor):
                    filtrados.append(valor)
            elif campo in CAMPOS_BLOQUEADOS_TEMPERATURA:
                continue
            else:
                filtrados.append(valor)
        elif dominio == "presion":
            if campo in ("rango", "rango_presion", "presion"):
                if _es_valor_presion(valor):
                    filtrados.append(valor)
            else:
                filtrados.append(valor)
        else:
            filtrados.append(valor)

    return filtrados


def _etiqueta_campo(campo: str) -> str:
    if campo in ETIQUETAS_CAMPO:
        return ETIQUETAS_CAMPO[campo]
    return campo.replace("_", " ")


def _etiqueta_campo_dominio(campo: str, dominio: str) -> str:
    if dominio == "temperatura" and campo in ("rango", "rango_temperatura"):
        return "rango de temperatura"
    if dominio == "presion" and campo in ("rango", "rango_presion"):
        return "rango de presión"
    return _etiqueta_campo(campo)


MAX_OPCIONES_PREGUNTA = 4

CAMPOS_BLOQUEADOS_NIVEL = frozenset({
    "rango",
    "rango_temperatura",
    "temperatura",
    "dimension",
    "precision",
    "diametro_dial",
    "resolucion",
    "recalibrable",
})

CAMPOS_PRIORITARIOS_NIVEL = (
    "rango_presion",
    "presion",
    "material",
    "senal_salida",
    "conexion",
    "conexion_montaje",
    "montaje",
    "alimentacion",
    "proteccion",
)

OPCIONES_ALTURA_TANQUE = [
    "Hasta 2 metros",
    "2 a 5 metros",
    "5 a 10 metros",
    "Más de 10 metros",
]

OPCIONES_PRESION_PROCESO = [
    "Hasta 10 bar",
    "10 a 40 bar",
    "Más de 40 bar",
]

OPCIONES_RANGO_TEMP_ALIMENTOS = [
    "0 a 60 °C",
    "0 a 100 °C",
    "0 a 200 °C",
    "-20 a 80 °C",
]

OPCIONES_RANGO_TEMP_INDUSTRIAL = [
    "-20 a 80 °C",
    "0 a 200 °C",
    "0 a 400 °C",
    "-50 a 500 °C",
]

CAMPOS_BLOQUEADOS_TEMPERATURA = frozenset({
    "rango_presion",
    "presion",
    "presion_maxima",
})

CAMPOS_PRIORITARIOS_TEMPERATURA = (
    "rango",
    "rango_temperatura",
    "resolucion",
    "resolucion_optica",
    "conexion",
    "conexion_montaje",
    "longitud_vastago",
    "senal_salida",
    "diametro_dial",
)

CAMPOS_BLOQUEADOS_PRESION = frozenset({
    "rango_temperatura",
    "temperatura",
    "temperatura_trabajo",
})

CAMPOS_PRIORITARIOS_PRESION = (
    "rango_presion",
    "rango",
    "conexion",
    "conexion_montaje",
    "diametro_dial",
    "material",
    "senal_salida",
)

OPCIONES_RANGO_PRESION = [
    "0-60 psi",
    "0-100 psi",
    "0-200 psi",
    "0-300 psi",
]

OPCIONES_CONEXION_PRESION = [
    "1/4 NPT",
    "1/2 NPT",
    "1/4 BSP",
    "1/2 BSP",
]

OPCIONES_RANGO_PUNTO_ROCIO = [
    "-30 a 100 °C",
    "-20 a 50 °C",
    "0 a 80 °C",
    "0 a 100 °C",
]

OPCIONES_RANGO_HUMEDAD_RELATIVA = [
    "0 a 100 % HR",
    "10 a 90 % HR",
    "0 a 95 % HR",
]

OPCIONES_RANGO_TEMP_AMBIENTE = [
    "-20 a 50 °C",
    "-30 a 100 °C",
    "0 a 60 °C",
    "0 a 80 °C",
]

CAMPOS_PRIORITARIOS_HUMEDAD = (
    "punto_de_rocio",
    "humedad",
    "rango_humedad",
    "temperatura",
    "rango_temperatura",
    "senal_salida",
    "precision",
)


_PREFIJO_MARCA_RANGO = re.compile(
    r"^(?:checktemp|modelo|serie|model|type)\s+",
    re.IGNORECASE,
)

_UNIDADES_NO_RANGO = re.compile(
    r"\b(ma|mv|v|hart|rs485|bar|psi|mbar|kpa|mpa|rh|npt|brida|panel)\b",
    re.IGNORECASE,
)


def _extraer_rango_numerico(valor: str) -> Optional[tuple[float, float]]:
    """Extrae par min/max de un texto con rango numérico."""
    v = str(valor or "").strip()
    if not v:
        return None

    limpio = _PREFIJO_MARCA_RANGO.sub("", v)
    limpio = limpio.replace(",", ".")
    norm = re.sub(r"[°º]", "", limpio.lower())
    norm = re.sub(r"\s+", " ", norm).strip()

    patrones = (
        r"(-?\d+(?:\.\d+)?)\s*(?:a|to)\s*(-?\d+(?:\.\d+)?)",
        r"(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)",
        r"(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)",
    )
    for patron in patrones:
        coincidencia = re.search(patron, norm, re.IGNORECASE)
        if not coincidencia:
            continue
        bajo = float(coincidencia.group(1))
        alto = float(coincidencia.group(2))
        if bajo > alto:
            bajo, alto = alto, bajo
        return bajo, alto
    return None


def _formatear_rango_temperatura_canonico(bajo: float, alto: float) -> str:
    def _fmt(numero: float) -> str:
        return str(int(numero)) if numero == int(numero) else f"{numero:g}"

    return f"{_fmt(bajo)} a {_fmt(alto)} °C"


def _puntuar_etiqueta_opcion(valor: str) -> float:
    """Mayor puntuación = etiqueta más legible para el chip."""
    v = str(valor or "").strip()
    norm = v.lower()
    puntaje = 0.0
    if re.search(r"\b(checktemp|modelo|serie|model)\b", norm):
        puntaje -= 20
    if re.search(r"\s+a\s+", v, re.IGNORECASE):
        puntaje += 5
    if re.search(r"[°º]\s*c|\bc\s*[°º]", v, re.IGNORECASE):
        puntaje += 3
    elif re.search(r"[°º]", v):
        puntaje += 1
    if re.search(r"-\d+[°º]?$", v):
        puntaje -= 2
    puntaje -= len(v) * 0.05
    return puntaje


def _clave_dedup_opcion(valor: str) -> str:
    """Clave estable para evitar opciones semánticamente duplicadas."""
    v = str(valor or "").strip()
    if not v:
        return ""

    if _UNIDADES_NO_RANGO.search(v):
        return f"txt:{_normalizar_texto(v)}"

    rango = _extraer_rango_numerico(v)
    if rango is not None:
        bajo, alto = rango
        if _es_valor_temperatura(v) or re.search(r"[°º]", v):
            return f"temp:{bajo:g}:{alto:g}"
        if "%" in v or "rh" in v.lower() or "roc" in v.lower():
            return f"hume:{bajo:g}:{alto:g}"

    return f"txt:{_normalizar_texto(v)}"


def _normalizar_etiqueta_opcion(valor: str) -> str:
    """Etiqueta unificada para chips de rango de temperatura."""
    v = str(valor or "").strip()
    if not v:
        return v
    rango = _extraer_rango_numerico(v)
    if rango is not None and (_es_valor_temperatura(v) or re.search(r"[°º]", v)):
        return _formatear_rango_temperatura_canonico(*rango)
    return v


def _deduplicar_opciones_valores(valores: list[str]) -> list[str]:
    """
    Elimina valores repetidos aunque cambien formato (-10 a 300°C vs -10-300°).
    Conserva la etiqueta más legible por cada rango numérico.
    """
    mejor_por_clave: dict[str, tuple[float, str]] = {}
    orden_claves: list[str] = []

    for valor in valores or []:
        v = str(valor).strip()
        if not v:
            continue
        clave = _clave_dedup_opcion(v)
        if not clave:
            continue
        puntaje = _puntuar_etiqueta_opcion(v)
        if clave not in mejor_por_clave:
            mejor_por_clave[clave] = (puntaje, v)
            orden_claves.append(clave)
        elif puntaje > mejor_por_clave[clave][0]:
            mejor_por_clave[clave] = (puntaje, v)

    salida: list[str] = []
    etiquetas_vistas: set[str] = set()
    for clave in orden_claves:
        _, raw = mejor_por_clave[clave]
        etiqueta = _normalizar_etiqueta_opcion(raw)
        etiqueta_key = _normalizar_texto(etiqueta)
        if etiqueta_key in etiquetas_vistas:
            continue
        etiquetas_vistas.add(etiqueta_key)
        salida.append(etiqueta)
    return salida


def _combinar_opciones_catalogo_fallback(
    catalogo: list[str],
    fallback: list[str],
) -> list[str]:
    combinados = list(catalogo or []) + list(fallback or [])
    return _deduplicar_opciones_valores(combinados)[:MAX_OPCIONES_PREGUNTA]


def _es_valor_rango_humedad_temperatura(valor: str) -> bool:
    v = str(valor or "").strip()
    if not v or len(v) > 45:
        return False
    norm = _normalizar_texto(v)
    if "+/-" in v or "precision" in norm or "rh (" in norm or "% rh" in norm:
        return False
    return bool(
        re.search(r"(\d+\s*[-–a]\s*\d+|\d+-\d+).*(°|c|f)\b", norm, re.IGNORECASE)
        or re.search(r"-?\d+\s*~\s*-?\d+", v)
    )


def _valores_unicos_atributos(
    descripciones: list[str],
    prefijos_campo: tuple[str, ...],
    *,
    solo_rangos: bool = False,
) -> list[str]:
    """
    Valores únicos del catálogo aunque todos los productos compartan el mismo rango.
    """
    claves_vistas: set[str] = set()
    valores: list[str] = []

    for descripcion in descripciones or []:
        atributos = extraer_atributos_descripcion_larga(descripcion)
        for campo, valor in atributos.items():
            campo_norm = _normalizar_texto(campo)
            if not any(pref in campo_norm for pref in prefijos_campo):
                continue
            valor_txt = re.sub(r"\s+", " ", str(valor).strip())
            if not valor_txt or len(valor_txt) > 80:
                continue
            if solo_rangos and not _es_valor_rango_humedad_temperatura(valor_txt):
                continue
            clave = _clave_dedup_opcion(valor_txt)
            if clave in claves_vistas:
                continue
            claves_vistas.add(clave)
            valores.append(valor_txt)

    return _deduplicar_opciones_valores(valores)


def _familia_campo_pregunta(campo: str) -> str:
    if campo in ("rango", "rango_presion", "rango_temperatura"):
        return "rango"
    if campo in ("conexion", "conexion_montaje", "montaje_conexion"):
        return "conexion"
    return campo


def _segunda_pregunta_dominio(
    dominio: str,
    campos: list[dict],
) -> dict:
    dom = (dominio or "").strip().lower()
    if dom == "presion":
        valores = _valores_catalogo_por_campos(
            campos,
            ("conexion", "conexion_montaje", "diametro_dial", "material"),
        )
        return {
            "texto": "¿Cuál es la conexión o el tamaño de carátula que necesitas?",
            "opciones": _construir_opciones(valores or OPCIONES_CONEXION_PRESION),
        }
    if dom == "temperatura":
        valores = _valores_catalogo_por_campos(
            campos,
            ("conexion", "conexion_montaje", "longitud_vastago"),
        )
        return {
            "texto": "¿Qué tipo de conexión o montaje necesitas?",
            "opciones": _construir_opciones(valores or ["Rosca NPT", "Brida", "Panel"]),
        }
    if dom == "humedad":
        valores = _valores_catalogo_por_campos(
            campos,
            ("temperatura", "rango_temperatura", "precision", "senal_salida"),
        )
        return {
            "texto": "¿Cuál es el rango de temperatura ambiente que necesitas medir?",
            "opciones": _construir_opciones(
                valores or OPCIONES_RANGO_TEMP_AMBIENTE
            ),
        }
    return {
        "texto": "¿Qué conexión, montaje o especificación adicional requieres?",
        "opciones": [{"id": "1", "label": "Otro", "valor": "otro"}],
    }


def _construir_opciones(valores: list[str]) -> list[dict]:
    """
    Construye opciones clicables: hasta 4 valores del catálogo + Otro.
    """
    valores = _deduplicar_opciones_valores(valores)
    opciones = []
    for idx, valor in enumerate(valores[:MAX_OPCIONES_PREGUNTA], start=1):
        valor_txt = str(valor).strip()
        if not valor_txt:
            continue
        opciones.append(
            {
                "id": str(idx),
                "label": valor_txt,
                "valor": valor_txt,
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


def _texto_pregunta(pregunta) -> str:
    if isinstance(pregunta, dict):
        return str(pregunta.get("texto") or "").strip()
    return str(pregunta or "").strip()


def _opciones_pregunta(pregunta) -> list[dict]:
    if isinstance(pregunta, dict):
        return list(pregunta.get("opciones") or [])
    return []


def _articulo_campo(campo: str) -> str:
    if campo in {
        "longitud_vastago",
        "potencia",
        "proteccion",
        "resolucion",
        "serie_modelo",
        "senal_salida",
        "termostato",
    }:
        return "la"
    return "el"


def generar_pregunta_seleccion_tipo(
    palabra_clave: str,
    tipos: list[dict],
) -> dict:
    """
    Primera pregunta: 3 tipos más frecuentes + Otro.
    """
    instrumento = palabra_clave.replace("_", " ")
    opciones = []

    for idx, tipo in enumerate(tipos, start=1):
        opciones.append(
            {
                "id": str(idx),
                "label": _formatear_nivel_1(tipo["nivel_1"]),
                "valor": str(idx),
            }
        )

    opciones.append(
        {
            "id": str(len(opciones) + 1),
            "label": "Otro",
            "valor": "otro",
        }
    )

    return {
        "texto": (
            f"En el catálogo encontré varios tipos de {instrumento}. "
            "¿Cuál necesitas?"
        ),
        "opciones": opciones,
    }


def generar_preguntas_desde_campos(
    campos: list[dict],
    dominio: Optional[str] = None,
) -> list[dict]:
    """
    Construye hasta 2 preguntas técnicas desde los campos discriminantes.
    Filtra valores incoherentes con el dominio y evita repetir la misma pregunta.
    """
    preguntas = []
    dom = (dominio or "").strip().lower()
    familias_vistas: set[str] = set()
    textos_vistos: set[str] = set()

    for campo_info in campos:
        if len(preguntas) >= 2:
            break

        campo = campo_info["campo"]
        ejemplos = campo_info.get("valores_frecuentes") or []
        ejemplos = _deduplicar_opciones_valores(
            _filtrar_valores_por_dominio(ejemplos, campo, dom)
        )

        if dom == "temperatura" and campo in CAMPOS_BLOQUEADOS_TEMPERATURA:
            continue
        if dom == "presion" and campo in CAMPOS_BLOQUEADOS_PRESION:
            continue
        if not ejemplos:
            continue

        familia = _familia_campo_pregunta(campo)
        if familia in familias_vistas:
            continue

        etiqueta = _etiqueta_campo_dominio(campo, dom)
        articulo = _articulo_campo(campo)
        texto = f"¿Cuál es {articulo} {etiqueta} que necesitas?"
        texto_key = _normalizar_texto(texto)
        if texto_key in textos_vistos:
            continue

        familias_vistas.add(familia)
        textos_vistos.add(texto_key)
        preguntas.append(
            {
                "texto": texto,
                "opciones": _construir_opciones(ejemplos),
            }
        )

    while len(preguntas) < 2:
        if len(preguntas) == 0:
            if dom == "temperatura":
                preguntas.append(
                    {
                        "texto": "¿Cuál es el rango de temperatura que necesitas?",
                        "opciones": _construir_opciones(OPCIONES_RANGO_TEMP_INDUSTRIAL),
                    }
                )
            elif dom == "presion":
                preguntas.append(
                    {
                        "texto": "¿Cuál es el rango de presión que necesitas?",
                        "opciones": _construir_opciones(OPCIONES_RANGO_PRESION),
                    }
                )
            elif dom == "humedad":
                preguntas.append(
                    {
                        "texto": "¿Cuál es el rango de humedad o punto de rocío que necesitas?",
                        "opciones": _construir_opciones(OPCIONES_RANGO_HUMEDAD_RELATIVA),
                    }
                )
            else:
                preguntas.append(
                    {
                        "texto": "¿Cuál es el rango de operación que necesitas?",
                        "opciones": [{"id": "1", "label": "Otro", "valor": "otro"}],
                    }
                )
        else:
            segunda = _segunda_pregunta_dominio(dom, campos)
            texto_key = _normalizar_texto(segunda["texto"])
            if texto_key not in textos_vistos:
                preguntas.append(segunda)

    return preguntas[:2]


def resolver_seleccion_tipo(
    mensaje: str,
    tipos: list[dict],
) -> tuple[str, Optional[str]]:
    """
    Interpreta la respuesta del cliente a la pregunta de tipo.

    Retorna:
    - ("tipo", nivel_1)
    - ("otro", None)
    - ("texto_libre", texto) cuando describe un tipo personalizado
    """
    texto = (mensaje or "").strip()
    texto_lower = _normalizar_texto(texto)

    if not texto_lower:
        return "otro", None

    if texto_lower in {"otro", "otra", "ninguno", "ninguna", "diferente"}:
        return "otro", None

    if re.fullmatch(r"\d+", texto_lower):
        idx = int(texto_lower)
        if 1 <= idx <= len(tipos):
            return "tipo", tipos[idx - 1]["nivel_1"]
        if idx == len(tipos) + 1:
            return "otro", None

    for tipo in tipos:
        nombre = tipo["nivel_1"]
        nombre_legible = _formatear_nivel_1(nombre)
        nombre_lower = _normalizar_texto(nombre)
        nombre_legible_lower = _normalizar_texto(nombre_legible)

        if (
            nombre_lower in texto_lower
            or texto_lower in nombre_lower
            or nombre_legible_lower in texto_lower
            or texto_lower in nombre_legible_lower
        ):
            return "tipo", nombre

        tokens_tipo = [
            t
            for t in nombre_legible_lower.split()
            if len(t) >= 5
        ]
        if tokens_tipo and sum(1 for t in tokens_tipo if t in texto_lower) >= 2:
            return "tipo", nombre

    return "texto_libre", texto


def _filtrar_campos_nivel(campos: list[dict]) -> list[dict]:
    return [c for c in campos if c.get("campo") not in CAMPOS_BLOQUEADOS_NIVEL]


def _valores_catalogo_por_campos(
    campos: list[dict],
    nombres: tuple[str, ...],
    limit: int = MAX_OPCIONES_PREGUNTA,
) -> list[str]:
    valores: list[str] = []

    for nombre in nombres:
        for campo_info in campos:
            canon = str(campo_info.get("campo") or "")
            if canon != nombre and nombre not in canon:
                continue
            for valor in campo_info.get("valores_frecuentes") or []:
                valor_txt = str(valor).strip()
                if valor_txt:
                    valores.append(valor_txt)

    return _deduplicar_opciones_valores(valores)[:limit]


def _pregunta_nivel_q2(nivel_1: str, campos_ok: list[dict], contexto_texto: str) -> tuple[str, list[str]]:
    """Segunda pregunta coherente con medición de nivel (libros + catálogo)."""
    nivel_slug = (nivel_1 or "").lower()
    campos_cat = get_campos(nivel_1)

    valores_presion = _valores_catalogo_por_campos(campos_ok, ("rango_presion", "presion"))
    valores_salida = _valores_catalogo_por_campos(campos_ok, ("senal_salida", "alimentacion"))
    valores_montaje = _valores_catalogo_por_campos(
        campos_ok, ("conexion_montaje", "conexion", "montaje", "material")
    )

    if "interruptor" in nivel_slug:
        texto = "¿Cuál es la presión máxima del proceso?"
        opciones = valores_presion or OPCIONES_PRESION_PROCESO
    elif "transmisor" in nivel_slug or "sensor" in nivel_slug:
        texto = "¿Qué señal de salida necesitas y cuál es la presión del proceso?"
        opciones = valores_salida or valores_presion or OPCIONES_PRESION_PROCESO
    elif "medidor" in nivel_slug or "radar" in nivel_slug or "ultrason" in nivel_slug:
        texto = "¿Cuál es el rango de nivel a medir y la presión del tanque?"
        opciones = valores_presion or OPCIONES_PRESION_PROCESO
    else:
        texto = campos_cat.get("q2_pregunta") or "¿Cuál es la presión máxima del proceso?"
        if "fluido" in (contexto_texto or "").lower():
            texto = "¿Cuál es la presión máxima del proceso y el tipo de montaje?"
        opciones = (
            valores_presion
            or valores_montaje
            or _valores_catalogo_por_campos(campos_ok, CAMPOS_PRIORITARIOS_NIVEL)
            or OPCIONES_PRESION_PROCESO
        )

    return texto, opciones


def generar_preguntas_nivel_coherentes(
    nivel_1: str,
    descripciones: list[str],
    contexto_texto: str = "",
) -> list[dict]:
    """
    Preguntas técnicas para dominio nivel: altura/rango primero, no temperatura.
    Usa libros Creus/Kuphaldt + campos del catálogo filtrados por dominio.
    """
    ctx_libros = contexto_para_agente(contexto_texto or "medición de nivel en tanques")
    terminos = ctx_libros.get("terminos") or []
    logger.info(
        "Preguntas nivel coherentes nivel_1=%s dominio_libros=%s terminos=%s",
        nivel_1,
        ctx_libros.get("dominio"),
        terminos[:4],
    )

    todos = analizar_campos_discriminantes(descripciones, top_n=10)
    campos_ok = _filtrar_campos_nivel(todos)

    texto_q2, opciones_q2 = _pregunta_nivel_q2(nivel_1, campos_ok, contexto_texto)

    return [
        {
            "texto": "¿Cuál es la altura del tanque o el rango de nivel que necesitas medir?",
            "opciones": _construir_opciones(OPCIONES_ALTURA_TANQUE),
        },
        {
            "texto": texto_q2,
            "opciones": _construir_opciones(opciones_q2),
        },
    ]


def _filtrar_campos_temperatura(campos: list[dict]) -> list[dict]:
    filtrados = []
    for campo_info in campos:
        campo = str(campo_info.get("campo") or "")
        if campo in CAMPOS_BLOQUEADOS_TEMPERATURA:
            continue
        valores = _filtrar_valores_por_dominio(
            campo_info.get("valores_frecuentes") or [],
            campo,
            "temperatura",
        )
        if campo in ("rango", "rango_temperatura") and not valores:
            continue
        if valores:
            filtrados.append({**campo_info, "valores_frecuentes": valores})
    return filtrados


def _filtrar_campos_presion(campos: list[dict]) -> list[dict]:
    filtrados = []
    for campo_info in campos:
        campo = str(campo_info.get("campo") or "")
        if campo in CAMPOS_BLOQUEADOS_PRESION:
            continue
        valores = _filtrar_valores_por_dominio(
            campo_info.get("valores_frecuentes") or [],
            campo,
            "presion",
        )
        if campo in ("rango", "rango_presion") and not valores:
            continue
        if valores:
            filtrados.append({**campo_info, "valores_frecuentes": valores})
    return filtrados


def generar_preguntas_presion_coherentes(
    nivel_1: str,
    descripciones: list[str],
    contexto_texto: str = "",
) -> list[dict]:
    """Preguntas técnicas de presión: rango psi + conexión, sin repetir."""
    ctx_libros = contexto_para_agente(contexto_texto or "medición de presión")
    logger.info(
        "Preguntas presión coherentes nivel_1=%s terminos=%s",
        nivel_1,
        (ctx_libros.get("terminos") or [])[:4],
    )

    todos = analizar_campos_discriminantes(descripciones, top_n=10)
    campos_ok = _filtrar_campos_presion(todos)

    if campos_ok:
        campos_ok.sort(
            key=lambda c: (
                0
                if c.get("campo") in ("rango_presion", "rango")
                else 1
                if c.get("campo") in CAMPOS_PRIORITARIOS_PRESION
                else 2
            )
        )
        return generar_preguntas_desde_campos(campos_ok, dominio="presion")

    return [
        {
            "texto": "¿Cuál es el rango de presión que necesitas?",
            "opciones": _construir_opciones(OPCIONES_RANGO_PRESION),
        },
        {
            "texto": "¿Cuál es la conexión o el tamaño de carátula que necesitas?",
            "opciones": _construir_opciones(OPCIONES_CONEXION_PRESION),
        },
    ]


def generar_preguntas_temperatura_coherentes(
    nivel_1: str,
    descripciones: list[str],
    contexto_texto: str = "",
) -> list[dict]:
    """Preguntas técnicas de temperatura sin mezclar unidades de presión."""
    ctx_libros = contexto_para_agente(contexto_texto or "medición de temperatura")
    logger.info(
        "Preguntas temperatura coherentes nivel_1=%s terminos=%s",
        nivel_1,
        (ctx_libros.get("terminos") or [])[:4],
    )

    todos = analizar_campos_discriminantes(descripciones, top_n=10)
    campos_ok = _filtrar_campos_temperatura(todos)

    if campos_ok:
        campos_ok.sort(
            key=lambda c: (
                0
                if c.get("campo") in ("rango", "rango_temperatura")
                else 1
                if c.get("campo") in CAMPOS_PRIORITARIOS_TEMPERATURA
                else 2
            )
        )
        return generar_preguntas_desde_campos(campos_ok, dominio="temperatura")

    ctx = (contexto_texto or "").lower()
    rangos = (
        OPCIONES_RANGO_TEMP_ALIMENTOS
        if "alimento" in ctx or "cocina" in ctx
        else OPCIONES_RANGO_TEMP_INDUSTRIAL
    )
    return [
        {
            "texto": "¿Cuál es el rango de temperatura que necesitas?",
            "opciones": _construir_opciones(rangos),
        },
        {
            "texto": "¿Qué tipo de conexión o montaje necesitas?",
            "opciones": _construir_opciones(["Rosca NPT", "Brida", "Panel"]),
        },
    ]


def generar_preguntas_humedad_coherentes(
    nivel_1: str,
    descripciones: list[str],
    contexto_texto: str = "",
) -> list[dict]:
    """
    Preguntas técnicas de humedad / punto de rocío.
    Usa rangos del catálogo aunque todos los psicrómetros compartan el mismo valor.
    """
    ctx = _normalizar_texto(contexto_texto or "")
    nivel_slug = (nivel_1 or "").lower()
    ctx_libros = contexto_para_agente(contexto_texto or "medición humedad punto de rocio")
    logger.info(
        "Preguntas humedad coherentes nivel_1=%s terminos=%s",
        nivel_1,
        (ctx_libros.get("terminos") or [])[:4],
    )

    rangos_rocio = _valores_unicos_atributos(
        descripciones, ("rocio",), solo_rangos=True
    )
    rangos_humedad = _valores_unicos_atributos(descripciones, ("humedad",))
    rangos_temp = _valores_unicos_atributos(
        descripciones,
        ("temperatura", "rango_temperatura"),
        solo_rangos=True,
    )

    pide_rocio = any(
        t in ctx or t in nivel_slug
        for t in ("rocio", "rocío", "punto de rocio", "punto de rocío", "dew")
    ) or "psicrom" in nivel_slug

    if pide_rocio:
        opciones_q1 = _combinar_opciones_catalogo_fallback(
            rangos_rocio, OPCIONES_RANGO_PUNTO_ROCIO
        )
        texto_q1 = "¿Cuál es el rango de punto de rocío que necesitas?"
    else:
        opciones_q1 = _combinar_opciones_catalogo_fallback(
            rangos_humedad, OPCIONES_RANGO_HUMEDAD_RELATIVA
        )
        texto_q1 = "¿Cuál es el rango de humedad relativa que necesitas?"

    opciones_q2 = _combinar_opciones_catalogo_fallback(
        rangos_temp, OPCIONES_RANGO_TEMP_AMBIENTE
    )

    if "transmisor" in nivel_slug:
        texto_q2 = "¿Qué señal de salida necesitas?"
        opciones_q2 = ["4-20 mA", "0-10 V", "HART", "RS485"]
    elif "registrador" in nivel_slug or "datalogger" in nivel_slug:
        texto_q2 = "¿Necesitas registro de datos o conectividad?"
        opciones_q2 = ["Registro local", "USB / PC", "Inalámbrico", "Pantalla"]
    else:
        texto_q2 = "¿Cuál es el rango de temperatura ambiente que necesitas medir?"

    return [
        {
            "texto": texto_q1,
            "opciones": _construir_opciones(opciones_q1),
        },
        {
            "texto": texto_q2,
            "opciones": _construir_opciones(opciones_q2),
        },
    ]


async def generar_preguntas_tecnicas_por_nivel_1(
    nivel_1: str,
    dominio: Optional[str] = None,
    contexto_texto: Optional[str] = None,
) -> list[dict]:
    """
    Analiza DESCRIPCION_LARGA_PRE del NIVEL_1 elegido y genera 2 preguntas.
    Si dominio=nivel, usa preguntas coherentes con medición de nivel.
    """
    descripciones = await obtener_descripciones_largas_por_nivel_1(nivel_1)

    dom = (dominio or "").strip().lower()
    if not dom and nivel_1 and "nivel" in nivel_1.lower():
        dom = "nivel"
    if not dom and nivel_1 and "temperatura" in nivel_1.lower():
        dom = "temperatura"
    if not dom and nivel_1 and "presion" in nivel_1.lower():
        dom = "presion"
    if not dom and nivel_1 and "manometro" in nivel_1.lower():
        dom = "presion"
    if not dom and contexto_texto and "nivel" in contexto_texto.lower():
        dom = "nivel"
    if not dom and contexto_texto and "temperatura" in contexto_texto.lower():
        dom = "temperatura"
    if not dom and contexto_texto and "presion" in contexto_texto.lower():
        dom = "presion"
    if not dom and nivel_1 and any(
        t in nivel_1.lower() for t in ("humedad", "higro", "psicrom", "termohigro")
    ):
        dom = "humedad"
    if not dom and contexto_texto and any(
        t in contexto_texto.lower()
        for t in ("humedad", "rocio", "rocío", "punto de rocio", "punto de rocío")
    ):
        dom = "humedad"

    if dom == "nivel":
        return generar_preguntas_nivel_coherentes(
            nivel_1=nivel_1,
            descripciones=descripciones,
            contexto_texto=contexto_texto or "",
        )

    if dom == "temperatura":
        return generar_preguntas_temperatura_coherentes(
            nivel_1=nivel_1,
            descripciones=descripciones,
            contexto_texto=contexto_texto or "",
        )

    if dom == "presion":
        return generar_preguntas_presion_coherentes(
            nivel_1=nivel_1,
            descripciones=descripciones,
            contexto_texto=contexto_texto or "",
        )

    if dom == "humedad":
        return generar_preguntas_humedad_coherentes(
            nivel_1=nivel_1,
            descripciones=descripciones,
            contexto_texto=contexto_texto or "",
        )

    campos = analizar_campos_discriminantes(descripciones, top_n=4)
    return generar_preguntas_desde_campos(campos, dominio=dom or None)


# Alias de compatibilidad
async def generar_preguntas_tecnicas_por_tipo(nivel_1: str) -> list[str]:
    return await generar_preguntas_tecnicas_por_nivel_1(nivel_1)


def construir_query_busqueda_final(
    palabra_clave: str,
    nivel_1: Optional[str],
    respuestas_tecnicas: list[str],
    texto_original: Optional[str] = None,
) -> str:
    """
    Arma la consulta final para catálogo con todo el contexto acumulado.
    """
    from discovery_guards import respuestas_utiles

    partes: list[str] = []

    original = str(texto_original or "").strip()
    if original:
        partes.append(original)

    palabra = str(palabra_clave or "").strip()
    if palabra:
        partes.append(palabra)

    if nivel_1:
        partes.append(_formatear_nivel_1(nivel_1))

    for respuesta in respuestas_utiles(respuestas_tecnicas):
        partes.append(respuesta)

    return " ".join(partes).strip()

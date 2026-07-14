"""
catalog.py — Consultas al catálogo real de ViaIndustrial en MongoDB Atlas.

Fuente oficial:
- Base de datos: MONGO_DB=nia
- Colección: products_catalog

Campos reales detectados:
- CODIGO
- REFERENCIA
- REF_ALTERNATIVA
- MARCA_LET
- DESCRIPCION_CORTA_PRE
- DESCRIPCION_LARGA_PRE
- NIVEL_0..NIVEL_4
- PRECIO_VENTA
- PV_FECHA
- STOCK_BOG
- STOCK_CALI
- STOCK_TOTAL
- VISIBLE_EN_LINEA
- EXISTENCIA
- texto_busqueda
- score_nia

Regla principal:
NIA no debe inventar productos. Si no hay coincidencia confiable,
debe pedir más información o indicar que no encontró una coincidencia suficiente.
"""

import logging
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Optional, Tuple

from memory import get_db


logger = logging.getLogger("nia.catalog")


# ============================================================
# CONFIGURACIÓN
# ============================================================

PRODUCTS_COLLECTION = "products_catalog"

DEFAULT_LIMIT = 30

# Umbrales internos.
# No son porcentajes de similitud textual pura, sino score compuesto.
UMBRAL_BASE = 0.55
UMBRAL_CON_CONTEXTO_TECNICO = 0.45


# ============================================================
# UTILIDADES DE TEXTO
# ============================================================

def _clean_text(value: Any) -> str:
    """
    Convierte un valor a texto limpio.
    """
    if value is None:
        return ""
    return str(value).strip()


def _normalize_text(value: Any) -> str:
    """
    Normaliza texto para comparación:
    - minúsculas;
    - sin tildes;
    - espacios compactos.
    """
    text = _clean_text(value).lower()

    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    text = re.sub(r"[^a-z0-9ñ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _tokens(text: str) -> list[str]:
    """
    Extrae tokens útiles para búsqueda sin aplicar una lista rígida de stopwords.

    Regla profesional:
    - No eliminamos palabras por criterio semántico fijo.
    - Solo normalizamos texto.
    - Solo descartamos tokens demasiado cortos porque normalmente no aportan
      valor de búsqueda y generan mucho ruido técnico.
    
    Esto evita que NIA pierda contexto importante en búsquedas industriales.
    """
    normalized = _normalize_text(text)
    raw_tokens = re.findall(r"[a-z0-9ñ]+", normalized)

    return [
        token
        for token in raw_tokens
        if len(token) >= 3
    ]


def _first_value(prod: dict, keys: list[str], default: Any = "") -> Any:
    """
    Retorna el primer valor no vacío de una lista de campos posibles.
    """
    for key in keys:
        value = prod.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return default


def _sim(a: str, b: str) -> float:
    """
    Similitud textual básica.
    """
    a_norm = _normalize_text(a)
    b_norm = _normalize_text(b)

    if not a_norm or not b_norm:
        return 0.0

    return SequenceMatcher(None, a_norm, b_norm).ratio()


def _token_coverage(query_tokens: list[str], target_text: str) -> float:
    """
    Mide cuántos tokens importantes de la consulta aparecen en el texto destino.

    Ejemplo:
    query_tokens = ["bomba", "agua"]
    target_text = "interruptor de presión de bomba de agua"
    coverage = 1.0
    """
    if not query_tokens:
        return 0.0

    normalized_target = _normalize_text(target_text)

    if not normalized_target:
        return 0.0

    hits = sum(1 for token in query_tokens if token in normalized_target)

    return hits / len(query_tokens)


def _build_search_text(prod: dict) -> str:
    """
    Construye un texto compuesto del producto normalizado para scoring.
    """
    fields = [
        prod.get("codigo"),
        prod.get("referencia"),
        prod.get("ref_alternativa"),
        prod.get("marca"),
        prod.get("nombre"),
        prod.get("descripcion_corta"),
        prod.get("descripcion_larga"),
        prod.get("categoria"),
        prod.get("nivel_0"),
        prod.get("nivel_1"),
        prod.get("nivel_2"),
        prod.get("nivel_3"),
        prod.get("nivel_4"),
        prod.get("texto_busqueda"),
    ]

    return " ".join(_clean_text(f) for f in fields if _clean_text(f))


# ============================================================
# NORMALIZACIÓN
# ============================================================

def normalizar_producto(prod: dict) -> dict:
    """
    Normaliza un documento real de products_catalog al contrato interno de NIA.

    Contrato interno:
    - codigo
    - referencia
    - ref_alternativa
    - nombre
    - marca
    - descripcion_corta
    - descripcion_larga
    - categoria
    - precio
    - moneda
    - stock_total
    - existencia
    - visible_en_linea
    - texto_busqueda
    - score_nia
    - raw
    """
    codigo = _clean_text(_first_value(prod, ["CODIGO", "codigo"]))
    referencia = _clean_text(_first_value(prod, ["REFERENCIA", "referencia"]))
    ref_alternativa = _clean_text(_first_value(prod, ["REF_ALTERNATIVA", "ref_alternativa"]))

    marca = _clean_text(_first_value(prod, ["MARCA_LET", "marca", "MARCA"]))

    descripcion_corta = _clean_text(
        _first_value(
            prod,
            [
                "DESCRIPCION_CORTA_PRE",
                "descripcion_corta_pre",
                "DESCRIPCION_CORTA",
                "descripcion_corta",
            ],
        )
    )

    descripcion_larga = _clean_text(
        _first_value(
            prod,
            [
                "DESCRIPCION_LARGA_PRE",
                "descripcion_larga_pre",
                "DESCRIPCION_LARGA",
                "descripcion_larga",
            ],
        )
    )

    nivel_0 = _clean_text(_first_value(prod, ["NIVEL_0", "nivel_0"]))
    nivel_1 = _clean_text(_first_value(prod, ["NIVEL_1", "nivel_1"]))
    nivel_2 = _clean_text(_first_value(prod, ["NIVEL_2", "nivel_2"]))
    nivel_3 = _clean_text(_first_value(prod, ["NIVEL_3", "nivel_3"]))
    nivel_4 = _clean_text(_first_value(prod, ["NIVEL_4", "nivel_4"]))

    # En este catálogo no siempre hay NOMBRE_PRODUCTO.
    # Para NIA usamos como nombre comercial preferido:
    # NIVEL_4 > DESCRIPCION_CORTA_PRE > NIVEL_3 > REFERENCIA.
    nombre = _clean_text(
        _first_value(
            prod,
            [
                "NOMBRE_PRODUCTO",
                "nombre_producto",
                "NOMBRE",
                "nombre",
            ],
        )
    )

    if not nombre:
        nombre = nivel_4 or descripcion_corta or nivel_3 or referencia or codigo

    categoria = nivel_4 or nivel_3 or nivel_2 or nivel_1 or nivel_0

    precio = _first_value(prod, ["PRECIO_VENTA", "precio_venta", "PRECIO", "precio"], None)
    stock_total = _first_value(prod, ["STOCK_TOTAL", "stock_total", "STOCK", "stock"], None)
    stock_bog = _first_value(prod, ["STOCK_BOG", "stock_bog"], None)
    stock_cali = _first_value(prod, ["STOCK_CALI", "stock_cali"], None)

    visible = prod.get("VISIBLE_EN_LINEA")
    if visible is None:
        visible = prod.get("visible_en_linea", True)

    texto_busqueda = _clean_text(_first_value(prod, ["texto_busqueda", "TEXTO_BUSQUEDA"]))
    existencia = _clean_text(_first_value(prod, ["EXISTENCIA", "existencia"]))
    pv_fecha = _clean_text(_first_value(prod, ["PV_FECHA", "pv_fecha"]))
    score_nia = _first_value(prod, ["score_nia"], None)

    return {
        "codigo": codigo,
        "referencia": referencia,
        "ref_alternativa": ref_alternativa,
        "nombre": nombre,
        "marca": marca,
        "descripcion_corta": descripcion_corta,
        "descripcion_larga": descripcion_larga,
        "descripcion": descripcion_corta or descripcion_larga,
        "categoria": categoria,
        "nivel_0": nivel_0,
        "nivel_1": nivel_1,
        "nivel_2": nivel_2,
        "nivel_3": nivel_3,
        "nivel_4": nivel_4,
        "precio": precio,
        "moneda": "COP",
        "stock_total": stock_total,
        "stock_bog": stock_bog,
        "stock_cali": stock_cali,
        "visible_en_linea": bool(visible),
        "existencia": existencia,
        "pv_fecha": pv_fecha,
        "texto_busqueda": texto_busqueda,
        "score_nia": score_nia,
        "_raw": prod,
    }


# ============================================================
# BÚSQUEDA POR CÓDIGO / REFERENCIA
# ============================================================

def normalizar_referencia(valor: str) -> str:
    """
    Forma canónica para comparar referencias:
    quita guiones, barras, espacios y puntos; mayúsculas.
    Ej: "EFS-40/S1" y "efs 40 s1" → "EFS40S1"
    """
    texto = _clean_text(valor).upper()
    return re.sub(r"[\s\-_./]+", "", texto)


def variantes_referencia_exacta(valor: str) -> list[str]:
    """
    Variantes mecánicas exactas del valor de entrada para $or / $in.

    Solo transforma separadores ya presentes (guion, barra, espacio, punto):
    no reconstruye tokens ni inventa separadores nuevos.
    """
    crudo = _clean_text(valor)
    if not crudo:
        return []

    upper = crudo.upper()
    compacta = normalizar_referencia(crudo)

    variantes = {crudo, upper, compacta}

    # Misma secuencia, cambiando solo el tipo de separador.
    guiones = re.sub(r"[\s_/.]+", "-", upper)
    guiones = re.sub(r"-{2,}", "-", guiones).strip("-")
    if guiones:
        variantes.add(guiones)

    barras = re.sub(r"[\s\-_.]+", "/", upper)
    barras = re.sub(r"/{2,}", "/", barras).strip("/")
    if barras:
        variantes.add(barras)

    espacios = re.sub(r"[\-_./]+", " ", upper)
    espacios = re.sub(r"\s+", " ", espacios).strip()
    if espacios:
        variantes.add(espacios)

    puntos = re.sub(r"[\s\-_./]+", ".", upper)
    puntos = re.sub(r"\.{2,}", ".", puntos).strip(".")
    if puntos:
        variantes.add(puntos)

    return list(variantes)


async def buscar_por_codigo(codigo: str) -> Optional[dict]:
    """
    Busca por CODIGO, REFERENCIA o REF_ALTERNATIVA exactos.
    Incluye variantes mecánicas de separadores (sin fuzzy ni regex).
    """
    valor = _clean_text(codigo)

    if not valor:
        return None

    db = get_db()
    collection = db[PRODUCTS_COLLECTION]

    variantes = variantes_referencia_exacta(valor)
    or_clauses = []
    for v in variantes:
        or_clauses.extend(
            [
                {"CODIGO": v},
                {"REFERENCIA": v},
                {"REF_ALTERNATIVA": v},
            ]
        )

    query = {"$or": or_clauses}

    prod = await collection.find_one(query, {"_id": 0})

    if not prod:
        logger.info("Producto no encontrado por código/referencia: %s", valor)
        return None

    normalizado = normalizar_producto(prod)
    normalizado["_match_type"] = "exacto_codigo_referencia"
    normalizado["_score"] = 1.0

    return normalizado


def _marca_coincide(marca_producto: str, marca_cliente: str) -> bool:
    """Compara marca del catálogo con la indicada por el cliente."""
    mp = _clean_text(marca_producto).lower()
    mc = _clean_text(marca_cliente).lower()
    if not mp or not mc:
        return False
    return mp == mc or mc in mp or mp in mc


def _variantes_referencia(valor: str) -> list[str]:
    """Alias: mismas variantes mecánicas exactas que buscar_por_codigo."""
    return variantes_referencia_exacta(valor)


async def _buscar_campo_exacto(collection, campo: str, valor: str) -> list:
    """Busca coincidencia exacta en REFERENCIA o REF_ALTERNATIVA."""
    variantes = _variantes_referencia(valor)
    if not variantes:
        return []

    query = {campo: {"$in": variantes}}
    cursor = collection.find(query, {"_id": 0}).limit(50)
    docs = await cursor.to_list(50)
    return [normalizar_producto(doc) for doc in docs]


async def buscar_por_referencia(valor: str, marca: Optional[str] = None) -> dict:
    """
    Búsqueda por referencia con prioridad comercial:

    1. REFERENCIA (exacta)
    2. REF_ALTERNATIVA (exacta)

    Reglas:
    - 1 match en REFERENCIA → entrega directa (confianza alta).
    - Varios matches o solo REF_ALTERNATIVA → pide MARCA_LET.
    - Con marca del cliente → filtra y entrega si queda 1.
    """
    valor = _clean_text(valor)
    if not valor:
        return {"estado": "sin_resultado", "candidatos": [], "match_campo": None}

    db = get_db()
    collection = db[PRODUCTS_COLLECTION]

    refs = await _buscar_campo_exacto(collection, "REFERENCIA", valor)
    if marca:
        refs = [p for p in refs if _marca_coincide(p.get("marca", ""), marca)]

    if len(refs) == 1:
        prod = refs[0]
        prod["_match_type"] = "exacto_referencia"
        prod["_score"] = 1.0
        return {
            "estado": "encontrado",
            "producto": prod,
            "candidatos": refs,
            "match_campo": "REFERENCIA",
            "confianza": "alta",
        }

    if len(refs) > 1:
        return {
            "estado": "necesita_marca",
            "producto": None,
            "candidatos": refs,
            "match_campo": "REFERENCIA",
            "confianza": "baja",
        }

    alts = await _buscar_campo_exacto(collection, "REF_ALTERNATIVA", valor)
    if marca:
        alts = [p for p in alts if _marca_coincide(p.get("marca", ""), marca)]

    if len(alts) == 1 and marca:
        prod = alts[0]
        prod["_match_type"] = "exacto_ref_alternativa"
        prod["_score"] = 0.95
        return {
            "estado": "encontrado",
            "producto": prod,
            "candidatos": alts,
            "match_campo": "REF_ALTERNATIVA",
            "confianza": "media",
        }

    if len(alts) >= 1:
        return {
            "estado": "necesita_marca",
            "producto": None,
            "candidatos": alts,
            "match_campo": "REF_ALTERNATIVA",
            "confianza": "baja",
        }

    return {"estado": "sin_resultado", "candidatos": [], "match_campo": None}


# ============================================================
# BÚSQUEDA POR TEXTO
# ============================================================

def _build_mongo_text_query(query: str) -> dict:
    """
    Construye una consulta MongoDB robusta usando texto_busqueda.

    Regla:
    - Con 1 token útil: busca ese token.
    - Con 2 o más tokens útiles: exige que al menos los tokens importantes
      aparezcan en texto_busqueda.

    Esto evita que "bomba de agua" traiga cualquier producto que solo diga "agua".
    """
    tokens = _tokens(query)

    if not tokens:
        return {}

    # Limitamos tokens para evitar consultas demasiado pesadas.
    tokens = tokens[:6]

    and_filters = []

    for token in tokens:
        safe = re.escape(token)
        and_filters.append(
            {
                "$or": [
                    {"texto_busqueda": {"$regex": safe, "$options": "i"}},
                    {"DESCRIPCION_CORTA_PRE": {"$regex": safe, "$options": "i"}},
                    {"DESCRIPCION_LARGA_PRE": {"$regex": safe, "$options": "i"}},
                    {"NIVEL_4": {"$regex": safe, "$options": "i"}},
                    {"NIVEL_3": {"$regex": safe, "$options": "i"}},
                    {"REFERENCIA": {"$regex": safe, "$options": "i"}},
                    {"MARCA_LET": {"$regex": safe, "$options": "i"}},
                ]
            }
        )

    base_filter = {
        "$and": and_filters
    }

    # Preferimos productos visibles, pero no bloqueamos al 100% por ahora
    # porque algunos productos internos pueden no estar visibles en línea.
    return base_filter


async def buscar_por_texto(query: str) -> Optional[list]:
    """
    Busca productos por texto en MongoDB usando products_catalog.

    Retorna lista de productos normalizados.
    """
    query = _clean_text(query)

    if not query:
        return None

    mongo_query = _build_mongo_text_query(query)

    if not mongo_query:
        return None

    db = get_db()
    collection = db[PRODUCTS_COLLECTION]

    cursor = collection.find(mongo_query, {"_id": 0}).limit(DEFAULT_LIMIT)
    docs = await cursor.to_list(length=DEFAULT_LIMIT)

    if not docs:
        logger.info("Sin resultados de catálogo para query: %s", query)
        return None

    normalizados = [normalizar_producto(doc) for doc in docs]

    logger.info(
        "Búsqueda catálogo query='%s' resultados=%s",
        query,
        len(normalizados),
    )

    return normalizados

async def buscar_con_campos(texto: str) -> Tuple[Optional[list], dict]:
    """
    Búsqueda técnica sobre MongoDB.

    Objetivo:
    - Extraer campos técnicos del mensaje del cliente.
    - Generar consultas limpias para el catálogo.
    - Mantener MongoDB como fuente oficial.
    - No depender de API externa.
    - No usar listas cerradas de productos o marcas.

    Importante:
    Esta función NO decide compatibilidad final.
    Solo recupera candidatos y entrega campos_query para scoring.
    """
    texto_original = _clean_text(texto)

    if not texto_original:
        return None, {}

    campos_query = _extraer_campos_query(texto_original)

    def _normalizar_query_catalogo(valor: str) -> str:
        """
        Limpia una frase conversacional y deja términos útiles de búsqueda.

        Esto no es una lista de productos. Son palabras funcionales genéricas
        del lenguaje que no aportan al catálogo.
        """
        valor_norm = _normalize_text(valor)

        # Quitamos valores técnicos ya detectados para que no dominen
        # la búsqueda textual.
        for campo_valor in campos_query.values():
            campo_valor_norm = _normalize_text(str(campo_valor))
            if campo_valor_norm:
                valor_norm = valor_norm.replace(campo_valor_norm, " ")

        tokens = [
            token
            for token in valor_norm.split()
            if len(token) > 2
            and token not in {
                "necesito",
                "necesitamos",
                "quiero",
                "requiere",
                "requiero",
                "buscar",
                "busco",
                "para",
                "con",
                "una",
                "uno",
                "del",
                "los",
                "las",
                "que",
                "por",
                "favor",
                "equipo",
                "producto",
                "sistema",
                "proceso",
            }
        ]

        return " ".join(tokens).strip()

    queries = []

    # 1. Consulta limpia principal.
    query_limpia = _normalizar_query_catalogo(texto_original)

    if query_limpia:
        queries.append(query_limpia)

    # 2. Consulta original como fallback.
    if texto_original and texto_original not in queries:
        queries.append(texto_original)

    # 3. Consulta corta con los primeros términos útiles.
    if query_limpia:
        tokens_base = query_limpia.split()[:4]
        query_base = " ".join(tokens_base).strip()

        if query_base and query_base not in queries:
            queries.append(query_base)

    # 4. Consulta técnica como último fallback.
    if campos_query and query_limpia:
        valores_tecnicos = [
            str(v).strip()
            for v in campos_query.values()
            if isinstance(v, str) and str(v).strip()
        ]

        if valores_tecnicos:
            query_tecnica = f"{query_limpia} {' '.join(valores_tecnicos[:2])}".strip()

            if query_tecnica and query_tecnica not in queries:
                queries.append(query_tecnica)

    logger.info(
        "buscar_con_campos texto='%s' campos=%s queries=%s",
        texto_original,
        campos_query,
        queries,
    )

    for query in queries:
        resultados = await buscar_por_texto(query)

        if resultados:
            logger.info(
                "buscar_con_campos encontró %s resultados con query='%s'",
                len(resultados),
                query,
            )
            return resultados, campos_query

    logger.info(
        "buscar_con_campos sin resultados texto='%s' queries=%s",
        texto_original,
        queries,
    )

    return None, campos_query


async def buscar_productos_por_nivel_1(
    nivel_1: str,
    palabra_clave: Optional[str] = None,
    limite: int = DEFAULT_LIMIT,
) -> Optional[list]:
    """
    Recupera productos de un tipo exacto en NIVEL_1.
    """
    nivel_1 = _clean_text(nivel_1)
    if not nivel_1:
        return None

    db = get_db()
    collection = db[PRODUCTS_COLLECTION]

    cursor = collection.find(
        {"NIVEL_1": nivel_1},
        {"_id": 0},
    ).limit(limite)

    docs = await cursor.to_list(limite)

    if not docs:
        logger.info("Sin productos para NIVEL_1='%s'", nivel_1)
        return None

    normalizados = [normalizar_producto(doc) for doc in docs]
    logger.info(
        "Productos por NIVEL_1 '%s': %s",
        nivel_1,
        len(normalizados),
    )
    return normalizados


# Alias de compatibilidad
buscar_productos_por_tipo_corta = buscar_productos_por_nivel_1


async def buscar_con_descubrimiento_producto(
    palabra_clave: str,
    nivel_1: Optional[str] = None,
    descripcion_corta: Optional[str] = None,
    respuestas_tecnicas: Optional[list] = None,
    respuestas_hibridas_previas: Optional[list] = None,
    extractos_libros: Optional[list] = None,
    texto_original: Optional[str] = None,
) -> dict:
    """
    Búsqueda final tras el flujo NIVEL_1 → DESCRIPCION_LARGA_PRE.
    Prioriza el NIVEL_1 elegido y puntúa con DESCRIPCION_LARGA_PRE.
    """
    from discovery_guards import respuestas_utiles

    nivel_1 = nivel_1 or descripcion_corta
    respuestas_tecnicas = respuestas_utiles(respuestas_tecnicas or [])
    partes_query = []
    original = str(texto_original or "").strip()
    if original:
        partes_query.append(original)
    if palabra_clave:
        partes_query.append(str(palabra_clave).strip())
    if nivel_1:
        partes_query.append(str(nivel_1).strip())
    partes_query.extend(str(r).strip() for r in respuestas_tecnicas if str(r).strip())
    query = " ".join(partes_query).strip()

    if respuestas_hibridas_previas:
        from hybrid_discovery import _contexto_material_nivel

        ctx = _contexto_material_nivel(respuestas_hibridas_previas)
        if ctx.get("es_solido_polvo"):
            query = f"{query} radar tdr onda guiada".strip()

    resultados = None
    campos_query = {}

    if nivel_1:
        resultados = await buscar_productos_por_nivel_1(
            nivel_1=nivel_1,
            palabra_clave=palabra_clave,
        )

    if respuestas_hibridas_previas and resultados:
        from hybrid_discovery import (
            filtrar_productos_por_tecnologia_material,
            producto_inadecuado_para_contexto,
        )

        filtrados = filtrar_productos_por_tecnologia_material(
            resultados,
            respuestas_hibridas_previas,
            extractos_libros=extractos_libros,
        )
        sin_ultrason = [
            p
            for p in filtrados
            if not producto_inadecuado_para_contexto(p, respuestas_hibridas_previas)
        ]
        if sin_ultrason:
            resultados = sin_ultrason
            logger.info(
                "Filtro material sólido/polvo: %s candidatos aptos para NIVEL_1='%s'",
                len(resultados),
                nivel_1,
            )

    if not resultados:
        resultados, campos_query = await buscar_con_campos(query)

    if not resultados:
        return {
            "estado": "sin_resultado",
            "razon": "No se encontraron productos para el tipo y especificaciones indicadas.",
            "pregunta_sugerida": "¿Puedes ajustar el tipo o alguna especificación técnica?",
            "candidatos_encontrados": False,
        }

    from learning_memory import filtrar_productos_por_aprendizaje

    resultados = filtrar_productos_por_aprendizaje(resultados)

    if not resultados:
        return {
            "estado": "sin_resultado",
            "razon": "Los productos de este tipo ya fueron descartados previamente por ti.",
            "pregunta_sugerida": "¿Puedes indicar otra especificación o tipo de producto?",
            "candidatos_encontrados": False,
        }

    ok, producto = evaluar_coincidencia(
        resultados,
        query,
        campos=max(2, len(respuestas_tecnicas)),
        campos_query=campos_query,
    )

    if ok and producto:
        producto["_compatibilidad"] = {
            "estado": "exact_match",
            "razon": "Coincidencia por tipo de catálogo y especificaciones técnicas.",
            "query_catalogo": query,
        }
        return {
            "estado": "encontrado",
            "producto": producto,
            "tipo": "descubrimiento_producto",
            "exacto": True,
            "razon": "Producto identificado por tipo y atributos técnicos.",
            "query_catalogo": query,
            "candidatos_encontrados": True,
        }

    destacados = obtener_candidatos_destacados(
        resultados,
        query,
        limit=4,
        campos_query=campos_query,
    )
    if len(destacados) >= 2:
        return {
            "estado": "multiples_candidatos",
            "candidatos": destacados,
            "tipo": "descubrimiento_producto_opciones",
            "exacto": False,
            "razon": "Hay varios productos del tipo elegido que podrían servir.",
            "query_catalogo": query,
            "candidatos_encontrados": True,
        }

    mejor = producto or (resultados[0] if resultados else None)
    if mejor:
        return {
            "estado": "relacionado",
            "producto": mejor,
            "tipo": "descubrimiento_producto_relacionado",
            "exacto": False,
            "razon": "Hay candidatos del tipo elegido, pero conviene confirmar la especificación.",
            "query_catalogo": query,
            "candidatos_encontrados": True,
        }

    return {
        "estado": "sin_resultado",
        "razon": "No hubo coincidencia confiable dentro del tipo seleccionado.",
        "pregunta_sugerida": "¿Puedes precisar rango, conexión o longitud de bulbo?",
        "candidatos_encontrados": True,
    }


# ============================================================
# SCORING
# ============================================================
def _parsear_campos(desc_larga: str) -> dict:
    """
    Extrae campos estructurados desde descripcion_larga.

    Formato esperado en catálogo:
    ¦ campo: valor ¦ campo: valor

    Ejemplo:
    ¦ rango: 0 a 10 bar ¦ salida: 4-20 mA ¦ conexion: 1/2 NPT
    """
    campos = {}

    if not desc_larga or "¦" not in desc_larga:
        return campos

    for parte in str(desc_larga).split("¦"):
        parte = parte.strip()

        if ":" not in parte:
            continue

        campo, valor = parte.split(":", 1)
        campo = _normalize_text(campo)
        valor = str(valor).strip()

        if 2 < len(campo) < 50 and valor:
            campos[campo] = valor

    return campos


def _score_campos_estructurados(
    campos_producto: dict,
    campos_query: dict,
) -> float:
    """
    Compara campos técnicos detectados en la consulta del cliente contra
    campos estructurados presentes en la descripción larga del producto.

    Retorna un promedio entre 0.0 y 1.0.
    """
    if not campos_producto or not campos_query:
        return 0.0

    scores = []

    for campo_q, valor_q in campos_query.items():
        campo_q_norm = _normalize_text(campo_q)
        valor_q_norm = _normalize_text(valor_q)

        if not campo_q_norm or not valor_q_norm:
            continue

        for campo_p, valor_p in campos_producto.items():
            campo_p_norm = _normalize_text(campo_p)
            valor_p_norm = _normalize_text(valor_p)

            # Coincidencia flexible de nombre de campo.
            # Ejemplo: "salida" puede coincidir con "señal de salida".
            if campo_q_norm in campo_p_norm or campo_p_norm in campo_q_norm:
                scores.append(_sim(valor_q_norm, valor_p_norm))
                break

    if not scores:
        return 0.0

    return sum(scores) / len(scores)

def _extraer_campos_query(texto: str) -> dict:
    """
    Extrae valores técnicos desde el mensaje del cliente usando patrones generales.

    Regla de diseño:
    - No depende de listas cerradas de materiales, marcas o protocolos.
    - Detecta estructuras comunes: rangos, unidades, conexión, salida/señal,
      material declarado, protección IP y voltaje.
    - No reemplaza al product_matcher; solo aporta contexto técnico para mejorar scoring.
    """
    campos = {}
    t_original = _clean_text(texto)
    t = t_original.lower()

    # ------------------------------------------------------------
    # 1. Rangos numéricos con unidades técnicas
    # Ejemplos:
    # - 0 a 10 bar
    # - 0-100 psi
    # - 10 a 80 °C
    # - 50 l/min
    # ------------------------------------------------------------
    rango = re.search(
        r"(-?\d+[\.,]?\d*)\s*(?:a|-|hasta)\s*(-?\d+[\.,]?\d*)\s*"
        r"([a-zA-Z°/%\.]+(?:/[a-zA-Z]+)?)",
        t,
        re.IGNORECASE,
    )

    if rango:
        inicio, fin, unidad = rango.groups()
        campos["rango"] = f"{inicio} a {fin} {unidad}"

    # Valor único con unidad cuando no hay rango.
    # Ejemplo: 80 C, 24 VDC, 50 l/min, 10 bar.
    valor_unico = re.search(
        r"\b(\d+[\.,]?\d*)\s*"
        r"(bar|psi|mbar|kpa|mpa|pa|°c|°f|c|f|v|vac|vdc|ma|a|gpm|lpm|l/min|m3/h|hz)\b",
        t,
        re.IGNORECASE,
    )

    if valor_unico and "rango" not in campos:
        valor, unidad = valor_unico.groups()
        unidad_norm = unidad.upper() if unidad in ["c", "f"] else unidad
        campos["valor_tecnico"] = f"{valor} {unidad_norm}"

    # ------------------------------------------------------------
    # 2. Señal / salida / comunicación / protocolo
    # Extrae lo que venga después de palabras técnicas.
    # Ejemplos:
    # - salida 4-20 mA
    # - señal 0-10 V
    # - protocolo Modbus RTU
    # - comunicación IO-Link
    # ------------------------------------------------------------
    salida = re.search(
        r"\b(?:salida|señal|senal|protocolo|comunicacion|comunicación)\s+"
        r"([a-zA-Z0-9\-\s/\.]+)",
        t_original,
        re.IGNORECASE,
    )

    if salida:
        valor = salida.group(1).strip()
        valor = re.split(r"[,.;\n]| con | para | y ", valor, maxsplit=1)[0].strip()

        if valor:
            campos["salida"] = valor

    # Si el cliente no escribió la palabra salida/señal, pero sí un patrón eléctrico claro.
    salida_patron = re.search(
        r"\b(\d+[\.,]?\d*)\s*-\s*(\d+[\.,]?\d*)\s*(ma|v)\b",
        t,
        re.IGNORECASE,
    )

    if salida_patron and "salida" not in campos:
        inicio, fin, unidad = salida_patron.groups()
        campos["salida"] = f"{inicio}-{fin} {unidad}"

    # ------------------------------------------------------------
    # 3. Conexión mecánica
    # Ejemplos:
    # - 1/2 NPT
    # - 3/4 BSP
    # - conexión roscada
    # - conexión bridada
    # ------------------------------------------------------------
    conexion_medida = re.search(
        r"\b(\d+/\d+|\d+\.\d+|\d+)\s*(?:\"|''|pulg|pulgada|pulgadas)?\s*"
        r"([a-zA-Z]{2,10})\b",
        t,
        re.IGNORECASE,
    )

    if conexion_medida:
        medida, tipo = conexion_medida.groups()

        # Solo medidas tipo 1/2, 3/4, etc. — evita falsos positivos con pulgadas.
        if "/" in medida and tipo.lower() not in {
            "v", "vac", "vdc", "ma", "a", "hz", "bar", "psi",
            "pulg", "pulgada", "pulgadas", "inch", "inches", "in",
        }:
            campos["conexion"] = f"{medida} {tipo}"

    conexion_texto = re.search(
        r"\bconexion\s+([a-zA-Z0-9\-\s/\.]+)",
        t_original,
        re.IGNORECASE,
    )

    if conexion_texto and "conexion" not in campos:
        valor = conexion_texto.group(1).strip()
        valor = re.split(r"[,.;\n]| con | para | y ", valor, maxsplit=1)[0].strip()

        if valor:
            campos["conexion"] = valor

    # ------------------------------------------------------------
    # 4. Material declarado explícitamente
    # No usamos lista cerrada. Solo extraemos si el usuario lo declara.
    # Ejemplos:
    # - material acero inoxidable
    # - cuerpo en bronce
    # - fabricado en PVC
    # ------------------------------------------------------------
    material = re.search(
        r"\b(?:material|cuerpo en|fabricado en|construido en|en material)\s+"
        r"([a-zA-Z0-9áéíóúñÁÉÍÓÚÑ\-\s/\.]+)",
        t_original,
        re.IGNORECASE,
    )

    if material:
        valor = material.group(1).strip()
        valor = re.split(r"[,.;\n]| con | para | y ", valor, maxsplit=1)[0].strip()

        if valor:
            campos["material"] = valor

    # ------------------------------------------------------------
    # 5. Protección IP
    # Ejemplos:
    # - IP65
    # - ip 67
    # ------------------------------------------------------------
    ip = re.search(r"\bip\s*(\d{2})\b", t, re.IGNORECASE)

    if ip:
        campos["proteccion"] = f"IP{ip.group(1)}"

    # ------------------------------------------------------------
    # 6. Voltaje / alimentación
    # Ejemplos:
    # - 24 VDC
    # - 110 VAC
    # - alimentación 220V
    # ------------------------------------------------------------
    voltaje = re.search(
        r"\b(\d+[\.,]?\d*)\s*(v|vac|vdc)\b",
        t,
        re.IGNORECASE,
    )

    if voltaje:
        valor, unidad = voltaje.groups()
        campos["voltaje"] = f"{valor} {unidad.upper()}"

    alimentacion = re.search(
        r"\b(?:alimentacion|alimentación)\s+([a-zA-Z0-9\-\s/\.]+)",
        t_original,
        re.IGNORECASE,
    )

    if alimentacion and "voltaje" not in campos:
        valor = alimentacion.group(1).strip()
        valor = re.split(r"[,.;\n]| con | para | y ", valor, maxsplit=1)[0].strip()

        if valor:
            campos["voltaje"] = valor

    # ------------------------------------------------------------
    # 7. Longitud bulbo / vástago
    # Ejemplos:
    # - bulbo 6 pulgadas
    # - vástago de 4 in
    # - 6 inch stem
    # ------------------------------------------------------------
    vastago = re.search(
        r"(?:bulbo|vástago|vastago|stem)\s*(?:de\s+)?"
        r"(\d+[\.,]?\d*)\s*(?:in\.?|inch(?:es)?|pulg(?:adas)?|\")",
        t_original,
        re.IGNORECASE,
    )

    if vastago:
        campos["longitud_vastago"] = f"{vastago.group(1)} in"

    vastago_inv = re.search(
        r"(\d+[\.,]?\d*)\s*(?:in\.?|inch(?:es)?|pulg(?:adas)?)\s*"
        r"(?:de\s+)?(?:bulbo|vástago|vastago|stem)",
        t_original,
        re.IGNORECASE,
    )

    if vastago_inv and "longitud_vastago" not in campos:
        campos["longitud_vastago"] = f"{vastago_inv.group(1)} in"

    return campos


def extraer_campos_tecnicos(texto: str) -> dict:
    """
    API pública: extrae campos técnicos del mensaje del cliente.
    Usado por el modo de búsqueda híbrida (texto + especificaciones).
    """
    return _extraer_campos_query(texto)

def _contiene_indicador_accesorio(texto: str) -> bool:
    """
    Detecta si un producto parece ser accesorio, control, repuesto,
    kit, switch, interruptor o componente relacionado con otro equipo.

    Regla general:
    No se usa para bloquear siempre. Se usa para penalizar cuando el cliente
    pidió el equipo principal y el resultado parece ser solo un accesorio.
    """
    texto_norm = _normalize_text(texto)

    indicadores = [
        "control de",
        "control para",
        "switch",
        "interruptor",
        "kit para",
        "kit de",
        "repuesto",
        "accesorio",
        "modulo para",
        "tarjeta para",
        "soporte para",
        "base para",
        "cable para",
        "sensor para",
        "protector para",
        "arrancador para",
        "contactores para",
        "valvula para",
    ]

    return any(indicador in texto_norm for indicador in indicadores)


def _consulta_pide_accesorio(query: str) -> bool:
    """
    Determina si el usuario explícitamente pidió un accesorio/componente.

    Si el usuario pide "control de presión para bomba", sí podemos devolver
    controles. Si pide solo "bomba de agua", no deberíamos devolver controles
    como primera opción.
    """
    query_norm = _normalize_text(query)

    indicadores = [
        "control",
        "switch",
        "interruptor",
        "kit",
        "repuesto",
        "accesorio",
        "modulo",
        "tarjeta",
        "soporte",
        "base",
        "cable",
        "sensor",
        "protector",
        "arrancador",
        "contactor",
        "valvula",
    ]

    return any(indicador in query_norm for indicador in indicadores)


def _penalizacion_por_incompatibilidad(prod: dict, query: str) -> float:
    """
    Calcula penalización si el producto parece accesorio pero la consulta
    no pidió explícitamente un accesorio.

    Retorna:
    - 1.0: sin penalización.
    - 0.65: penalización moderada por posible accesorio.
    """
    if _consulta_pide_accesorio(query):
        return 1.0

    texto_producto = " ".join(
        [
            _clean_text(prod.get("nombre")),
            _clean_text(prod.get("descripcion_corta")),
            _clean_text(prod.get("descripcion_larga")),
            _clean_text(prod.get("categoria")),
        ]
    )

    if _contiene_indicador_accesorio(texto_producto):
        return 0.65

    return 1.0

def _score_producto(
    prod: dict,
    query: str,
    campos_query: Optional[dict] = None,
) -> float:
    """
    Score compuesto para evaluar relevancia real.

    Mantiene el scoring estable actual basado en cobertura textual y agrega,
    de forma opcional, scoring por campos técnicos estructurados.

    Si no hay campos_query, el comportamiento sigue siendo el mismo.
    """
    query_tokens = _tokens(query)

    if not query_tokens:
        return 0.0

    texto_total = _build_search_text(prod)
    nombre_categoria = " ".join(
        [
            _clean_text(prod.get("nombre")),
            _clean_text(prod.get("categoria")),
            _clean_text(prod.get("nivel_4")),
            _clean_text(prod.get("nivel_3")),
        ]
    )

    coverage_total = _token_coverage(query_tokens, texto_total)
    coverage_nombre = _token_coverage(query_tokens, nombre_categoria)

    sim_nombre = _sim(query, prod.get("nombre", ""))
    sim_desc = _sim(query, prod.get("descripcion_corta", ""))

    raw_score_nia = prod.get("score_nia")
    try:
        score_nia_norm = min(float(raw_score_nia or 0) / 100, 1.0)
    except (TypeError, ValueError):
        score_nia_norm = 0.0

    score_textual = (
        coverage_total * 0.45
        + coverage_nombre * 0.25
        + sim_nombre * 0.15
        + sim_desc * 0.10
        + score_nia_norm * 0.05
    )

    score_campos = 0.0

    if campos_query:
        descripcion_larga = _clean_text(prod.get("descripcion_larga"))
        campos_producto = _parsear_campos(descripcion_larga)
        score_campos = _score_campos_estructurados(campos_producto, campos_query)

    if campos_query and score_campos > 0:
        # El texto sigue pesando más porque no todos los productos tienen
        # descripción larga estructurada con separador ¦.
        score = (score_textual * 0.70) + (score_campos * 0.30)
    else:
        score = score_textual

    if prod.get("visible_en_linea") is False:
        score *= 0.90

    return round(score, 4)


def evaluar_coincidencia(
    resultados: Optional[list],
    query: str,
    campos: int = 1,
    marca_presente: bool = False,
    campos_query: Optional[dict] = None,
) -> Tuple[bool, Optional[dict]]:
    """
    Evalúa resultados y retorna UNO solo: el mejor producto confiable.

    Reglas:
    - Si no hay resultados, retorna False.
    - Si hay más contexto técnico, permite un umbral menor.
    - Si hay campos técnicos detectados, se usa scoring estructurado adicional.
    - No acepta productos sin nombre/categoría/descripción.
    """
    if not resultados:
        return False, None

    if campos >= 3:
        umbral = UMBRAL_CON_CONTEXTO_TECNICO
    elif campos == 2 or marca_presente:
        umbral = 0.50
    else:
        umbral = UMBRAL_BASE

    if campos_query:
        cantidad_campos_query = len(campos_query)

        if cantidad_campos_query >= 3:
            umbral = min(umbral, 0.45)
        elif cantidad_campos_query == 2:
            umbral = min(umbral, 0.50)

    candidatos = []

    for producto in resultados:
        if not producto.get("nombre") and not producto.get("descripcion_corta"):
            continue

        score = _score_producto(
            producto,
            query,
            campos_query=campos_query,
        )

        candidatos.append((score, producto))

    if not candidatos:
        return False, None

    candidatos.sort(key=lambda x: x[0], reverse=True)

    mejor_score, mejor_prod = candidatos[0]

    logger.debug(
        "Mejor score catálogo: %.3f | umbral: %.2f | codigo: %s | nombre: %s | campos_query: %s",
        mejor_score,
        umbral,
        mejor_prod.get("codigo"),
        mejor_prod.get("nombre"),
        list(campos_query.keys()) if campos_query else [],
    )

    if mejor_score >= umbral:
        mejor_prod["_score"] = mejor_score
        mejor_prod["_campos_match"] = list(campos_query.keys()) if campos_query else []
        return True, mejor_prod

    return False, None


def obtener_candidatos_destacados(
    resultados: Optional[list],
    query: str,
    limit: int = 4,
    campos_query: Optional[dict] = None,
    umbral_minimo: float = 0.32,
) -> list[dict]:
    """
    Devuelve los mejores candidatos cuando no hay match único confiable.
    """
    if not resultados:
        return []

    puntuados: list[tuple[float, dict]] = []
    for producto in resultados[:40]:
        if not producto.get("nombre") and not producto.get("descripcion_corta"):
            continue
        score = _score_producto(producto, query, campos_query=campos_query)
        if score >= umbral_minimo:
            copia = dict(producto)
            copia["_score"] = score
            puntuados.append((score, copia))

    if not puntuados:
        return []

    puntuados.sort(key=lambda x: x[0], reverse=True)
    mejor_score = puntuados[0][0]
    umbral = max(umbral_minimo, mejor_score * 0.82)

    vistos: set[str] = set()
    salida: list[dict] = []
    for score, producto in puntuados:
        codigo = str(producto.get("codigo") or "").strip()
        if codigo and codigo in vistos:
            continue
        if score < umbral and len(salida) >= 2:
            break
        if codigo:
            vistos.add(codigo)
        salida.append(producto)
        if len(salida) >= limit:
            break

    return salida


# ============================================================
# FORMATO DE RESPUESTA
# ============================================================

def formatear_producto(p: dict) -> str:
    """
    Formatea un producto para mostrarlo al cliente.

    Nunca usa placeholders tipo [código] o [marca].
    Si falta un dato real, muestra 'No disponible'.
    """
    codigo = p.get("codigo") or "No disponible"
    referencia = p.get("referencia") or "No disponible"
    nombre = p.get("nombre") or "No disponible"
    marca = p.get("marca") or "No disponible"
    desc = p.get("descripcion_corta") or p.get("descripcion") or "No disponible"
    existencia = p.get("existencia") or "No disponible"
    stock_total = p.get("stock_total")

    stock_texto = "No disponible"
    if stock_total is not None:
        stock_texto = str(stock_total)

    return (
        f"Código: {codigo}\n"
        f"Referencia: {referencia}\n"
        f"Nombre: {nombre}\n"
        f"Marca: {marca}\n"
        f"Descripción: {desc}\n"
        f"Existencia: {existencia}\n"
        f"Stock total: {stock_texto}"
    )
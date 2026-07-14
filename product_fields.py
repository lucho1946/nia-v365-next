"""
product_fields.py — Campos técnicos reales por categoría
Extraído del catálogo real de ViaIndustrial (289.017 productos)
Estructura de desc_larga: ¦ campo: valor ¦ campo: valor

Cada categoría tiene:
  - campos_q2: campos para la pregunta 2 (rango + condición)
  - campos_q3: campos para la pregunta 3 (interfaz + material)
  - keywords:  palabras clave que el cliente usa para llegar a esta categoría
  - dominio:   dominio técnico de los libros (Creus/Kuphaldt)
"""

# ─── Mapeo palabras cliente → categoría del catálogo ─────────────────────────
KEYWORDS_TO_CATEGORIA = {
    # Presión
    "manometro": "manometros-con-glicerina",
    "manómetro": "manometros-con-glicerina",
    "presion": "transmisores-de-presion",
    "presión": "transmisores-de-presion",
    "transmisor presion": "transmisores-de-presion",
    "transmisor presión": "transmisores-de-presion",
    "presion diferencial": "transmisores-de-presion-diferencial",
    "dp": "transmisores-de-presion-diferencial",
    # Temperatura
    "termometro": "termometros-bimetalicos",
    "termómetro": "termometros-bimetalicos",
    "temperatura": "controles-de-temperatura",
    "termopar": "controles-de-temperatura",
    "rtd": "controles-de-temperatura",
    "pt100": "controles-de-temperatura",
    "infrarrojo": "termometros-infrarrojos-portatiles",
    "pirometro": "termometros-infrarrojos-portatiles",
    # Caudal / flujo
    "rotametro": "rotametros-flujometros",
    "rotámetro": "rotametros-flujometros",
    "flujometro": "rotametros-flujometros",
    "caudal": "rotametros-flujometros",
    "flujo": "rotametros-flujometros",
    "medidor agua": "medidores-para-agua",
    # Nivel
    "nivel": "interruptores-de-nivel",
    "interruptor nivel": "interruptores-de-nivel",
    "switch nivel": "interruptores-de-nivel",
    # Control
    "pid": "controles-de-procesos-pid",
    "controlador": "controles-de-procesos-pid",
    "control proceso": "controles-de-procesos-pid",
    "control temperatura": "controles-de-temperatura",
    # Válvulas
    "valvula solenoide": "valvulas-solenoides",
    "válvula solenoide": "valvulas-solenoides",
    "solenoide": "valvulas-solenoides",
    "valvula bola": "valvulas-de-bola",
    "válvula bola": "valvulas-de-bola",
    "valvula reguladora": "valvulas-reductoras-reguladoras-de-presion",
    "valvula reductora": "valvulas-reductoras-reguladoras-de-presion",
    # Analítica
    "ph": "ph-metros",
    "phmetro": "ph-metros",
    "ph metro": "ph-metros",
    "conductividad": "ph-metros",
    # Sensores
    "sensor proximidad": "sensores-de-proximidad",
    "proximidad": "sensores-de-proximidad",
    # Pesaje
    "celda carga": "celdas-de-carga",
    "celda de carga": "celdas-de-carga",
    "balanza": "balanzas-digitales",
    "bascula": "basculas-digitales",
    # Neumática
    "cilindro": "cilindros-neumaticos",
    "valvula neumatica": "valvulas-solenoides",
    "racor": "racores-rapidos-para-aire-conectores-instantaneos",
    # Medición dimensional
    "calibrador": "calibradores-pie-de-rey",
    "pie de rey": "calibradores-pie-de-rey",
    "micrometro": "micrometros-para-exteriores",
    # EPI / seguridad eléctrica
    "botas": "calzado-dielectrico",
    "bota": "calzado-dielectrico",
    "calzado": "calzado-dielectrico",
    "guantes": "guantes-dielectricos",
    "guante": "guantes-dielectricos",
    "dielectrico": "calzado-dielectrico",
    "dieléctrico": "calzado-dielectrico",
    "dielectrica": "calzado-dielectrico",
    "dieléctrica": "calzado-dielectrico",
    # Calentamiento / gabinetes
    "calentador": "calentadores-flexibles",
    "calentadores": "calentadores-flexibles",
    "heater": "calentadores-flexibles",
}

# ─── Campos técnicos por categoría ────────────────────────────────────────────
# Extraídos del análisis real de desc_larga del catálogo ViaIndustrial
# campos_q2: los más discriminantes (rango/condición)
# campos_q3: los finales (interfaz/material/certificación)

CAMPOS_POR_CATEGORIA = {

    "transmisores-de-presion": {
        "campos_q2": ["rango de presión", "rango", "temperatura fluido", "exactitud"],
        "campos_q3": ["salida", "conexion", "conexion electrica", "voltaje"],
        "q2_pregunta": "¿Cuál es el rango de presión y el tipo de fluido?",
        "q3_pregunta": "¿Qué señal de salida necesitas y cuál es la conexión al proceso?",
        "dominio": "presion",
    },

    "transmisores-de-presion-diferencial": {
        "campos_q2": ["límites de presión", "rango", "rango de temperatura", "exactitud"],
        "campos_q3": ["salida", "tiempo de respuesta", "estabilidad"],
        "q2_pregunta": "¿Cuál es el rango de presión diferencial y la temperatura del proceso?",
        "q3_pregunta": "¿Qué señal de salida y qué exactitud necesitas?",
        "dominio": "presion",
    },

    "manometros-con-glicerina": {
        "campos_q2": ["rango", "temperatura trabajo", "caratula"],
        "campos_q3": ["conexion", "ventana", "material"],
        "q2_pregunta": "¿Cuál es el rango de presión y el tamaño de carátu­la?",
        "q3_pregunta": "¿Cuál es la conexión (tamaño y tipo NPT/BSP) y el material?",
        "dominio": "presion",
    },

    "rotametros-flujometros": {
        "campos_q2": ["rango", "temperatura", "presión", "presión máxima"],
        "campos_q3": ["cuerpo", "flotador", "conexion", "sellos"],
        "q2_pregunta": "¿Cuál es el rango de caudal, el fluido y la temperatura del proceso?",
        "q3_pregunta": "¿Qué material de cuerpo necesitas y cuál es la conexión?",
        "dominio": "caudal",
    },

    "medidores-para-agua": {
        "campos_q2": ["flujo normal", "flujo maximo", "rango lectura", "flujo minimo"],
        "campos_q3": ["conexion", "conexion conectores", "norma"],
        "q2_pregunta": "¿Cuál es el caudal normal y máximo de operación?",
        "q3_pregunta": "¿Cuál es la conexión y necesita norma específica?",
        "dominio": "caudal",
    },

    "interruptores-de-nivel": {
        "campos_q2": ["presion maxima", "temperatura operación", "material"],
        "campos_q3": ["salida", "alimentacion", "montaje conexion", "montaje"],
        "q2_pregunta": "¿Cuál es la presión máxima del proceso y el tipo de fluido?",
        "q3_pregunta": "¿Qué tipo de salida necesitas y cómo va montado?",
        "dominio": "nivel",
    },

    "controles-de-temperatura": {
        "campos_q2": ["entrada", "modos de control", "dimensiones"],
        "campos_q3": ["salida(s) de control", "alarma(s)", "proteccion", "alimentacion"],
        "q2_pregunta": "¿Qué tipo de sensor usas (termopar/RTD) y qué modo de control necesitas?",
        "q3_pregunta": "¿Qué tipo de salida de control necesitas y cuál es la alimentación?",
        "dominio": "temperatura",
    },

    "termometros-bimetalicos": {
        "campos_q2": ["rango", "dial", "longitud bulbo"],
        "campos_q3": ["conexion", "resolucion", "recalibrable"],
        "q2_pregunta": "¿Cuál es el rango de temperatura y el tamaño del dial?",
        "q3_pregunta": "¿Cuál es la longitud del bulbo y el tipo de conexión?",
        "dominio": "temperatura",
    },

    "termometros-infrarrojos-portatiles": {
        "campos_q2": ["rango", "resolucion optica", "exactitud"],
        "campos_q3": ["emisibilidad", "mira laser", "resolucion"],
        "q2_pregunta": "¿Cuál es el rango de temperatura que necesitas medir?",
        "q3_pregunta": "¿Necesitas mira láser y cuál es la distancia de medición?",
        "dominio": "temperatura",
    },

    "controles-de-procesos-pid": {
        "campos_q2": ["entrada", "entradas universales", "dimensiones"],
        "campos_q3": ["salida(s) de control", "alarma(s)", "proteccion", "alimentacion"],
        "q2_pregunta": "¿Qué tipo de entrada de señal necesitas (4-20mA, TC, RTD)?",
        "q3_pregunta": "¿Cuántas salidas de control y alarmas necesitas?",
        "dominio": "control_pid",
    },

    "valvulas-solenoides": {
        "campos_q2": ["temperatura fluido", "presion", "presion de operacion"],
        "campos_q3": ["conexion / orificio", "cuerpo", "voltaje", "operación"],
        "q2_pregunta": "¿Cuál es el fluido, la temperatura y la presión de operación?",
        "q3_pregunta": "¿Cuál es la conexión, el material del cuerpo y el voltaje de la bobina?",
        "dominio": "valvulas_control",
    },

    "valvulas-de-bola": {
        "campos_q2": ["presion", "diámetro", "material del cuerpo"],
        "campos_q3": ["conexion", "caracteristica", "empaque"],
        "q2_pregunta": "¿Cuál es el diámetro, la presión de trabajo y el fluido?",
        "q3_pregunta": "¿Qué tipo de conexión y material de cuerpo necesitas?",
        "dominio": "valvulas_control",
    },

    "valvulas-reductoras-reguladoras-de-presion": {
        "campos_q2": ["diámetro", "tipo", "aplicaciones"],
        "campos_q3": ["conexión", "tipo de conexión", "diámetro de conexión"],
        "q2_pregunta": "¿Cuál es el diámetro y la presión de entrada/salida?",
        "q3_pregunta": "¿Qué tipo de conexión necesitas?",
        "dominio": "presion",
    },

    "sensores-de-proximidad": {
        "campos_q2": ["alcance", "deteccion", "distancia de deteccion"],
        "campos_q3": ["alimentacion", "salida", "proteccion", "respuesta en frecuencia"],
        "q2_pregunta": "¿Qué distancia de detección necesitas y qué material detecta?",
        "q3_pregunta": "¿Qué tipo de salida y cuál es la alimentación?",
        "dominio": "transmisores",
    },

    "ph-metros": {
        "campos_q2": ["rango", "exactitud", "resolucion"],
        "campos_q3": ["electrodo", "alimentacion", "calibracion"],
        "q2_pregunta": "¿Cuál es el rango de pH y la exactitud requerida?",
        "q3_pregunta": "¿Es portátil o de panel, y qué tipo de electrodo necesitas?",
        "dominio": "analitica_proceso",
    },

    "celdas-de-carga": {
        "campos_q2": ["capacidad", "temperatura de operacion", "limite de sobrecarga"],
        "campos_q3": ["material", "proteccion", "sensibilidad (cn)"],
        "q2_pregunta": "¿Cuál es la capacidad de carga máxima y la temperatura de operación?",
        "q3_pregunta": "¿Qué material y protección IP necesitas?",
        "dominio": "transmisores",
    },

    "cilindros-neumaticos": {
        "campos_q2": ["diametro", "carrera en milimetros", "carrera"],
        "campos_q3": ["presión de prueba", "tipo", "montaje"],
        "q2_pregunta": "¿Cuál es el diámetro del émbolo y la carrera requerida?",
        "q3_pregunta": "¿Qué tipo de cilindro (simple/doble efecto) y cómo va montado?",
        "dominio": "plc_automatizacion",
    },

    "reles-de-estado-solido": {
        "campos_q2": ["carga", "rango de voltaje carga", "voltaje de control"],
        "campos_q3": ["temperatura de funcionamiento", "rango de frecuencia"],
        "q2_pregunta": "¿Cuál es la carga y el voltaje de la carga a controlar?",
        "q3_pregunta": "¿Cuál es el voltaje de control y la temperatura de operación?",
        "dominio": "plc_automatizacion",
    },

    "default": {
        "campos_q2": ["rango", "temperatura", "presion"],
        "campos_q3": ["salida", "conexion", "material", "proteccion"],
        "q2_pregunta": "¿Cuál es el rango de operación y las condiciones del proceso?",
        "q3_pregunta": "¿Qué señal de salida, conexión o material necesitas?",
        "dominio": "general",
    },
}

def get_campos(categoria: str) -> dict:
    """Retorna los campos técnicos para una categoría. Usa default si no existe."""
    return CAMPOS_POR_CATEGORIA.get(categoria, CAMPOS_POR_CATEGORIA["default"])

def detectar_categoria(texto: str) -> str:
    """Detecta la categoría del catálogo a partir del texto del cliente."""
    texto_lower = texto.lower()
    # Buscar match exacto primero
    for kw, cat in sorted(KEYWORDS_TO_CATEGORIA.items(), key=lambda x: -len(x[0])):
        if kw in texto_lower:
            return cat
    return "default"

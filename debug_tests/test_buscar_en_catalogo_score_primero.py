"""
Score determinístico decide primero; LLM solo como respaldo.
MAX_CANDIDATOS=1: el LLM recibe solo el candidato más puntuado.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from catalog import extraer_campos_tecnicos, score_producto
import main as appmain
from product_matcher import MAX_CANDIDATOS


def test_dimensiones_normalizadas_orden_ascendente():
    campos = extraer_campos_tecnicos("Caja gabinete de 77 X 99 X 24")
    assert campos["dimensiones"] == "24x77x99", campos


def test_max_candidatos_es_uno():
    assert MAX_CANDIDATOS == 1


def _prod(codigo: str, dims_en_larga: str) -> dict:
    return {
        "codigo": codigo,
        "nombre": f"Caja gabinete {dims_en_larga}",
        "descripcion_corta": f"Caja gabinete {dims_en_larga}",
        "descripcion_larga": f"Gabinete metálico dimensiones {dims_en_larga}",
        "nivel_1": "cajas-gabinetes",
        "nivel_2": "cajas gabinetes",
        "visible_en_linea": True,
        "score_nia": 50,
    }


async def test_score_claro_no_llama_llm():
    query = "Caja gabinete de 77 X 99 X 24"
    bueno = _prod("GOOD77", "77x99x24")
    malo = _prod("BAD60", "60x80x20")
    # Intencionalmente en orden "malo" primero para probar el sort.
    candidatos = [malo, bueno]
    campos = extraer_campos_tecnicos(query)
    llm = AsyncMock(return_value={"estado": "no_match", "producto": None})

    with (
        patch.object(
            appmain, "generar_queries_catalogo", AsyncMock(return_value=[query])
        ),
        patch.object(
            appmain,
            "buscar_con_campos",
            AsyncMock(return_value=(candidatos, campos)),
        ),
        patch.object(
            appmain,
            "filtrar_productos_por_aprendizaje",
            side_effect=lambda xs: xs,
        ),
        patch.object(appmain, "validar_compatibilidad_producto", llm),
    ):
        res = await appmain.buscar_en_catalogo(query)

    assert res["estado"] == "encontrado", res
    assert res["producto"]["codigo"] == "GOOD77", res["producto"]
    assert res.get("exacto") is True
    llm.assert_not_awaited()


async def test_score_bajo_si_llama_llm_con_candidato_top1():
    query = "zzz producto inexistente xyz"
    a = {
        "codigo": "A",
        "nombre": "algo leve",
        "descripcion_corta": "algo leve",
        "descripcion_larga": "",
        "visible_en_linea": True,
        "score_nia": 1,
    }
    b = {
        "codigo": "B",
        "nombre": "otro leve",
        "descripcion_corta": "otro leve",
        "descripcion_larga": "",
        "visible_en_linea": True,
        "score_nia": 2,
    }
    candidatos = [a, b]
    campos: dict = {}
    called_with: dict = {}

    async def _llm(necesidad_cliente, candidatos, contexto_tecnico=None):
        called_with["candidatos"] = list(candidatos)
        scores = [
            score_producto(p, necesidad_cliente, campos_query=campos)
            for p in candidatos
        ]
        assert scores == sorted(scores, reverse=True)
        assert candidatos[0]["_score"] == max(p["_score"] for p in candidatos)
        # Con MAX_CANDIDATOS=1 el matcher solo mira el primero;
        # buscar_en_catalogo ya entrega la lista ordenada completa,
        # y product_matcher la corta a 1.
        return {
            "estado": "no_match",
            "producto": None,
            "razon": "sin match",
            "pregunta_sugerida": "?",
        }

    llm = AsyncMock(side_effect=_llm)

    with (
        patch.object(
            appmain, "generar_queries_catalogo", AsyncMock(return_value=[query])
        ),
        patch.object(
            appmain,
            "buscar_con_campos",
            AsyncMock(return_value=(candidatos, campos)),
        ),
        patch.object(
            appmain,
            "filtrar_productos_por_aprendizaje",
            side_effect=lambda xs: xs,
        ),
        patch.object(appmain, "validar_compatibilidad_producto", llm),
    ):
        res = await appmain.buscar_en_catalogo(query)

    assert res["estado"] == "sin_resultado"
    llm.assert_awaited()
    tops = called_with["candidatos"]
    assert tops[0]["_score"] >= tops[-1]["_score"]


def run() -> None:
    test_dimensiones_normalizadas_orden_ascendente()
    test_max_candidatos_es_uno()
    asyncio.run(test_score_claro_no_llama_llm())
    asyncio.run(test_score_bajo_si_llama_llm_con_candidato_top1())
    print("OK: score primero, LLM respaldo, MAX_CANDIDATOS=1, dimensiones OK")


if __name__ == "__main__":
    run()

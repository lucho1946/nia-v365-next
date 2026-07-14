"""
Variantes mecánicas exactas de referencia (sin fuzzy ni reconstrucción de tokens).
"""
from __future__ import annotations

import asyncio

from catalog import (
    buscar_por_codigo,
    normalizar_referencia,
    variantes_referencia_exacta,
)

CANONICA = "EFS-40/S1"
ENTRADAS = ["EFS-40/S1", "EFS40S1", "EFS-40S1", "efs 40 s1"]


def test_normalizar_referencia_colapsa_separadores():
    esperada = "EFS40S1"
    for entrada in ENTRADAS:
        assert normalizar_referencia(entrada) == esperada


def test_variantes_mecanicas_desde_forma_catalogo():
    vars_ = set(variantes_referencia_exacta("EFS-40/S1"))
    assert "EFS-40/S1" in vars_
    assert "EFS40S1" in vars_
    assert "EFS-40-S1" in vars_ or "EFS 40 S1" in vars_


def test_variantes_mecanicas_no_reconstruyen_desde_compacta():
    """EFS40S1 no inventa EFS-40/S1 (solo mecánicas)."""
    vars_ = set(variantes_referencia_exacta("EFS40S1"))
    assert "EFS40S1" in vars_
    assert "EFS-40/S1" not in vars_


def test_variantes_desde_espacios_no_mezclan_separadores():
    """efs 40 s1 → EFS-40-S1 o EFS/40/S1, no EFS-40/S1."""
    vars_ = set(variantes_referencia_exacta("efs 40 s1"))
    assert "EFS40S1" in vars_
    assert "EFS 40 S1" in vars_
    assert "EFS-40-S1" in vars_
    assert "EFS/40/S1" in vars_
    assert "EFS-40/S1" not in vars_


async def _buscar_variantes_si_existe():
    base = await buscar_por_codigo(CANONICA)
    if not base:
        print(f"SKIP: no hay producto con referencia {CANONICA} en Mongo")
        return

    codigo_esperado = base.get("codigo")
    for entrada in ENTRADAS:
        prod = await buscar_por_codigo(entrada)
        print(f"entrada={entrada!r} ->", None if not prod else prod.get("codigo"))
        if entrada == "EFS-40/S1":
            assert prod is not None
            assert prod.get("codigo") == codigo_esperado
        elif prod is not None:
            # Coincide solo si el catálogo también tiene una variante mecánica
            # (p.ej. REF_ALTERNATIVA compacta EFS40S1).
            assert prod.get("codigo") == codigo_esperado


def test_buscar_por_codigo_variantes_mismo_producto_si_existe():
    asyncio.run(_buscar_variantes_si_existe())


if __name__ == "__main__":
    test_normalizar_referencia_colapsa_separadores()
    test_variantes_mecanicas_desde_forma_catalogo()
    test_variantes_mecanicas_no_reconstruyen_desde_compacta()
    test_variantes_desde_espacios_no_mezclan_separadores()
    test_buscar_por_codigo_variantes_mismo_producto_si_existe()
    print("OK")

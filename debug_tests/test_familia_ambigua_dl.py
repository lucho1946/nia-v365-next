from product_discovery import (
    extraer_atributos_descripcion_larga,
    filtrar_candidatos_por_respuestas_dl,
    hay_familia_ambigua_por_dl,
)


def _cands():
    dls = [
        "■ Calentador de Caja 60 W 120 V sin termostato",
        "■ Calentador de Caja 120 W 120 V sin termostato",
        "■ Calentador de Caja 60 W 120 V con termostato",
        "■ Calentador de Caja 200 W 240 V",
    ]
    return [
        {
            "codigo": f"P{i}",
            "nombre": "Calentadores de gabinetes o cajas de caucho de silicio",
            "descripcion_larga": dl,
            "_score": 0.67,
        }
        for i, dl in enumerate(dls)
    ]


def test_extrae_potencia_voltaje_termostato():
    attrs = extraer_atributos_descripcion_larga(
        "■ Calentador de Caja 60 W 120 V sin termostato"
    )
    assert attrs["potencia"] == "60 W"
    assert attrs["voltaje"] == "120 V"
    assert attrs["termostato"] == "Sin termostato"


def test_familia_ambigua_pregunta_potencia():
    ambigua, grupo, preguntas = hay_familia_ambigua_por_dl(_cands())
    assert ambigua is True
    assert len(grupo) == 4
    assert preguntas
    texto = (preguntas[0].get("texto") or "").lower()
    assert "potencia" in texto
    assert "la potencia" in texto
    labels = {o.get("label") for o in preguntas[0].get("opciones") or []}
    assert "60 W" in labels
    assert "120 W" in labels


def test_filtrar_por_60w_deja_dos_y_pide_termostato():
    filtrados = filtrar_candidatos_por_respuestas_dl(_cands(), ["60 W"])
    assert len(filtrados) == 2
    ambigua, grupo, preguntas = hay_familia_ambigua_por_dl(filtrados)
    assert ambigua is True
    assert len(grupo) == 2
    texto = " ".join((p.get("texto") or "").lower() for p in preguntas)
    assert "termostato" in texto or "voltaje" in texto


def test_filtrar_60w_sin_termostato_unico():
    filtrados = filtrar_candidatos_por_respuestas_dl(
        _cands(),
        ["60 W", "Sin termostato"],
    )
    assert len(filtrados) == 1
    assert filtrados[0]["codigo"] == "P0"

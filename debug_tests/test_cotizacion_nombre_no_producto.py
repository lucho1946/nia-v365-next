from main import (
    _es_nueva_solicitud_durante_cierre,
    _manejar_estado_comercial_prioritario,
    _parece_nombre_simple,
)


def test_nombre_persona_no_es_nueva_solicitud():
    assert _parece_nombre_simple("andres valencia") == "Andres Valencia"
    assert _es_nueva_solicitud_durante_cierre("andres valencia") is False


def test_escape_explicito_sigue_funcionando():
    assert _es_nueva_solicitud_durante_cierre("tambien necesito una valvula") is True
    assert _es_nueva_solicitud_durante_cierre("agregar otro producto") is True


def test_cotizacion_captura_nombre_y_pide_correo():
    result = _manejar_estado_comercial_prioritario(
        mensaje="andres valencia",
        etapa="cotizacion",
        cliente={},
        productos_acumulados=[
            {"producto": {"codigo": "P1", "nombre": "Demo"}, "cantidad": 1}
        ],
        historial=[],
        necesidad_ctx={"forzar_contacto_cotizacion": True},
    )
    assert result is not None
    assert result.get("handled") is True
    assert (result.get("cliente") or {}).get("nombre") == "Andres Valencia"
    assert "correo" in (result.get("respuesta") or "").lower()
    assert "tipos claros" not in (result.get("respuesta") or "").lower()

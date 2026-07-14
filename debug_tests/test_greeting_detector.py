import pytest

from greeting_detector import (
    _periodo_actual,
    _personalizar,
    build_greeting_result,
    detect_greeting,
    select_response,
)


def test_pure_greeting():
    result = detect_greeting("Buenos días")
    assert result["intent"] == "saludo_horario"
    assert result["pure_greeting"] is True


def test_greeting_with_request():
    result = detect_greeting("Hola, necesito un manómetro de 0 a 100 psi")
    assert result["intent"] == "saludo_con_solicitud"
    assert result["secondary_intent"] == "saludo"
    assert result["continue_to_main_intent"] is True
    assert result["remainder"] == "necesito un manometro de 0 a 100 psi"


def test_no_greeting():
    result = detect_greeting("Necesito un termómetro")
    assert result["matched"] is False


def test_every_intent_has_response():
    for intent in (
        "saludo_general",
        "saludo_horario",
        "saludo_formal",
        "saludo_informal",
        "saludo_con_solicitud",
    ):
        assert isinstance(select_response(intent, seed_text="prueba"), str)


def test_stable_response_with_seed():
    first = select_response("saludo_general", seed_text="Hola")
    second = select_response("saludo_general", seed_text="Hola")
    assert first == second


def test_build_result_for_pure_greeting():
    result = build_greeting_result("Hola", seed_text="sesion-1")
    assert result["response"]
    assert result["should_respond_now"] is True
    assert result["text_for_router"] == ""


def test_build_result_routes_request():
    result = build_greeting_result("Hola, busco el código P251802", seed_text="sesion-2")
    assert result["response"]
    assert result["should_respond_now"] is False
    assert result["text_for_router"] == "busco el codigo p251802"


def test_unknown_intent_fails_fast():
    with pytest.raises(ValueError):
        select_response("saludo_inexistente")


def test_typo_hole_como_estas_is_pure_greeting():
    result = detect_greeting("hole como estas?")
    assert result["matched"] is True
    assert result["pure_greeting"] is True
    built = build_greeting_result("hole como estas?", seed_text="s1")
    assert built["should_respond_now"] is True
    assert built["text_for_router"] == ""


def test_hola_como_estas_is_pure_greeting():
    result = detect_greeting("hola como estas?")
    assert result["matched"] is True
    assert result["pure_greeting"] is True
    assert result["continue_to_main_intent"] is False
    built = build_greeting_result("hola como estas?", seed_text="s2")
    assert built["should_respond_now"] is True
    assert built["text_for_router"] == ""


def test_hola_buneas_tardes_is_pure_greeting():
    result = detect_greeting("hola .. buneas tardes")
    assert result["matched"] is True
    assert result["pure_greeting"] is True
    built = build_greeting_result("hola .. buneas tardes", seed_text="s3")
    assert built["should_respond_now"] is True
    assert built["text_for_router"] == ""


def test_hola_buenas_tardes_is_pure_greeting():
    result = detect_greeting("hola buenas tardes")
    assert result["matched"] is True
    assert result["pure_greeting"] is True
    assert result["continue_to_main_intent"] is False


# ─────────────────────────────────────────────────────────────
# Humanización: no repetir, nombre del cliente, hora real
# ─────────────────────────────────────────────────────────────

def test_excluir_evita_repetir_ultima_respuesta():
    primera = select_response("saludo_general", seed_text="misma-sesion")
    segunda = select_response(
        "saludo_general", seed_text="misma-sesion", excluir=primera
    )
    assert segunda != primera


def test_excluir_no_revienta_si_solo_hay_una_opcion():
    # saludo_formal tiene varias respuestas; igual no debe fallar si se
    # excluye una y solo queda esa disponible tras un pool artificialmente chico.
    respuesta = select_response("saludo_formal", seed_text="x", excluir="no existe")
    assert isinstance(respuesta, str)


def test_personalizar_inserta_nombre_tras_signo_exclamacion():
    resultado = _personalizar("¡Hola! Soy NIA. ¿Qué necesitas?", "Andrés")
    assert resultado == "¡Hola, Andrés! Soy NIA. ¿Qué necesitas?"


def test_personalizar_inserta_nombre_tras_punto():
    resultado = _personalizar("Cordial saludo. ¿En qué te ayudo?", "Andrés")
    assert resultado == "Cordial saludo, Andrés. ¿En qué te ayudo?"


def test_personalizar_sin_nombre_no_cambia_nada():
    original = "¡Hola! Soy NIA. ¿Qué necesitas?"
    assert _personalizar(original, None) == original
    assert _personalizar(original, "") == original


def test_build_result_personaliza_con_nombre_cliente():
    result = build_greeting_result("Hola", seed_text="sesion-nombre", client_name="Andrés")
    assert "Andrés" in result["response"]


def test_build_result_excluye_ultima_respuesta_de_la_sesion():
    primera = build_greeting_result("Hola", seed_text="sesion-norepeat")
    segunda = build_greeting_result(
        "Hola",
        seed_text="sesion-norepeat",
        excluir_respuesta=primera["response"],
    )
    assert segunda["response"] != primera["response"]


def test_saludo_general_usa_hora_real_del_servidor():
    # "hola" sin franja horaria explícita → debe responder con una variante
    # de saludo_horario coherente con la hora real, no con saludo_general.
    periodo = _periodo_actual()
    keywords = {"manana": ("día", "dia"), "tarde": ("tarde",), "noche": ("noche",)}[periodo]
    respuesta = select_response("saludo_general", seed_text="hola", usar_hora_real=True)
    assert any(k in respuesta.lower() for k in keywords)


def test_saludo_general_sin_hora_real_mantiene_comportamiento_anterior():
    respuesta = select_response("saludo_general", seed_text="hola", usar_hora_real=False)
    assert respuesta in select_response.__globals__["INTENT_CONFIG"]["saludo_general"]["responses"]

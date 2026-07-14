"""
Guardrail comercial de cotización.

Confirma el cierre de solicitud y comunica que la cotización
llegará lo más pronto posible.
"""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
MAIN_FILE = BASE_DIR / "main.py"


def test_mensaje_cierre_solicitud() -> None:
    source = MAIN_FILE.read_text(encoding="utf-8-sig")

    expected = (
        "La cotización llegará lo más pronto posible."
    )

    assert expected in source, (
        "La respuesta de cierre debe indicar que la "
        "cotización llegará lo más pronto posible."
    )


def test_sin_mensaje_antiguo_de_asesor() -> None:
    source = MAIN_FILE.read_text(encoding="utf-8-sig")

    forbidden = (
        "Un asesor debe validarla y continuar el proceso."
    )

    assert forbidden not in source, (
        "Ya no debe usarse el mensaje de validación "
        "por asesor en el cierre de cotización."
    )


def run() -> None:
    test_mensaje_cierre_solicitud()
    test_sin_mensaje_antiguo_de_asesor()

    print(
        "OK: mensaje de cierre de cotización actualizado."
    )


if __name__ == "__main__":
    run()

from fastapi.testclient import TestClient

from main import UPLOAD_TMP_DIR, _parece_empresa_simple, app


def test_proforma_acepta_nombre_persona_con_prefijo():
    assert _parece_empresa_simple("a nombre de david valencia") == "David Valencia"
    assert _parece_empresa_simple("A nombre de David Valencia") == "David Valencia"
    assert _parece_empresa_simple("david valencia") == "David Valencia"
    assert _parece_empresa_simple("valencia") == "Valencia"


def test_proforma_acepta_razon_social_y_empresa():
    assert _parece_empresa_simple("ViaIndustrial SAS") is not None
    assert "via" in (_parece_empresa_simple("ViaIndustrial SAS") or "").lower()
    assert _parece_empresa_simple("empresa es Via Industrial") == "Via Industrial"
    assert _parece_empresa_simple("razon social Equipos Industriales Fenix") is not None


def test_proforma_rechaza_pedido_de_producto():
    assert _parece_empresa_simple("necesito una valvula") is None
    assert _parece_empresa_simple("quiero cotizar un termometro") is None


def test_upload_archivo_legacy_guarda_y_responde_metadatos():
    client = TestClient(app)
    files = {"archivo": ("rut.pdf", b"%PDF-1.4 demo", "application/pdf")}
    r = client.post("/upload-archivo", files=files)
    assert r.status_code == 200
    data = r.json()
    assert data["archivo_nombre"] == "rut.pdf"
    assert data["archivo_tipo"] == "pdf"
    assert data["archivo_ruta"]
    assert data["archivo_ruta"].startswith(str(UPLOAD_TMP_DIR))
    from pathlib import Path
    assert Path(data["archivo_ruta"]).is_file()

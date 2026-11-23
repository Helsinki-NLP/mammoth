"""Mammoth server, step 3: HTTP API (OPUS-MT compatible endpoints + Mammoth extensions)."""

import threading

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from mammoth.serving import ComponentPool, GenerationSettings, TranslationEngine, open_source  # noqa: E402
from mammoth.serving.app import create_app  # noqa: E402
from mammoth.tests.test_hf_modular import checkpoint  # noqa: E402,F401  (fixture)


@pytest.fixture
def engine(checkpoint):
    ckpt, _, _ = checkpoint
    pool = ComponentPool(open_source(str(ckpt)))
    return TranslationEngine(pool, GenerationSettings(num_beams=2, max_new_tokens=6))


@pytest.fixture
def client(engine):
    engine.pool.add_task("eng-spa")
    engine.pool.add_task("deu-fra")  # zero-shot
    return TestClient(create_app(engine))


def test_ready(engine):
    client = TestClient(create_app(engine))
    assert client.get("/api/ready").status_code == 503
    engine.pool.add_task("eng-spa")
    r = client.get("/api/ready")
    assert r.status_code == 204 and r.content == b""


def test_languages(client):
    assert client.get("/api/languages").json() == {"languages": {"deu": ["fra"], "eng": ["spa"]}}


def test_translate_opus_mt_format(client, engine):
    r = client.post("/api/translate", json={"from": "eng", "to": "spa", "source": "eng5 eng6"})
    assert r.status_code == 200
    assert r.json() == {"translation": engine.translate("eng-spa", "eng5 eng6")}
    assert r.json()["translation"].split()[0].startswith("spa")


def test_translate_list_and_overrides(client, engine):
    texts = ["deu5 deu6 deu7", "", "deu8"]
    r = client.post("/api/translate", json={"from": "deu", "to": "fra", "source": texts,
                                            "num_beams": 1, "max_new_tokens": 3})
    assert r.status_code == 200
    assert r.json()["translation"] == engine.translate("deu-fra", texts, num_beams=1, max_new_tokens=3)
    assert r.json()["translation"][1] == ""


def test_translate_errors(client):
    def post(**body):
        return client.post("/api/translate", json={"from": "eng", "to": "spa", "source": "eng5", **body})

    r = post(**{"from": "spa", "to": "eng"})
    assert r.status_code == 404 and r.json() == {"error": "Language pair spa-eng not supported"}
    r = post(**{"from": "eng", "to": "fra"})  # servable, but not loaded
    assert r.status_code == 404 and "POST /api/tasks" in r.json()["error"]
    assert post(num_beams=0).status_code == 422
    assert post(num_beams=100).status_code == 422
    r = post(max_new_tokens=7)  # above the server's max_new_tokens=6
    assert r.status_code == 422 and "at most 6" in r.json()["error"]
    assert client.post("/api/translate", json={"from": "eng", "source": "x"}).status_code == 422


def test_tasks_load_unload(client, engine):
    tasks = client.get("/api/tasks").json()
    assert set(tasks) == {"eng-spa", "eng-fra", "deu-spa", "deu-fra"}
    assert tasks["deu-fra"] == dict(src="deu", tgt="fra", encoder=["deu", "all"],
                                    decoder=["fra", "all"], trained=False, loaded=True)
    assert tasks["eng-fra"]["loaded"] is False and tasks["eng-fra"]["trained"] is True

    r = client.post("/api/tasks", json={"task": "eng-fra"})
    assert r.status_code == 201 and r.json()["eng-fra"]["loaded"] is True
    assert client.get("/api/languages").json()["languages"]["eng"] == ["fra", "spa"]
    assert client.post("/api/translate", json={"from": "eng", "to": "fra", "source": "eng5"}).status_code == 200

    assert client.delete("/api/tasks/eng-spa").status_code == 204
    assert client.get("/api/languages").json() == {"languages": {"deu": ["fra"], "eng": ["fra"]}}
    assert "dec__0__spa" not in engine.pool.loaded_components()
    r = client.delete("/api/tasks/eng-spa")
    assert r.status_code == 404 and "not loaded" in r.json()["error"]

    assert client.post("/api/tasks", json={"task": "spa-eng"}).status_code == 404
    assert client.post("/api/tasks", json={"task": "engspa"}).status_code == 422


def test_health(client):
    health = client.get("/api/health").json()
    assert health["status"] == "ok" and health["device"] == "cpu" and health["dtype"] == "float32"
    assert health["loaded_tasks"] == ["deu-fra", "eng-spa"]
    assert health["loaded_components"]["enc__1__all"] == ["deu-fra", "eng-spa"]


def test_reads_do_not_wait_for_a_load(engine):
    """While a direction loads (slow: disk or Hub), listing and translating keep working."""
    engine.pool.add_task("eng-spa")
    client = TestClient(create_app(engine))
    source = engine.pool.source
    started, release = threading.Event(), threading.Event()
    real_load = source.load_component

    def slow_load(name):
        started.set()
        assert release.wait(10)
        return real_load(name)

    source.load_component = slow_load
    loader = threading.Thread(target=lambda: client.post("/api/tasks", json={"task": "deu-fra"}))
    loader.start()
    try:
        assert started.wait(10)
        assert client.get("/api/languages").json() == {"languages": {"eng": ["spa"]}}
        r = client.post("/api/translate", json={"from": "eng", "to": "spa", "source": "eng5"})
        assert r.status_code == 200
    finally:
        release.set()
        loader.join()
    assert client.get("/api/languages").json() == {"languages": {"deu": ["fra"], "eng": ["spa"]}}


def test_web_page(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    # The page only uses API endpoints that exist.
    for path in ("/api/tasks", "/api/health", "/api/translate"):
        assert f'"{path}"' in r.text

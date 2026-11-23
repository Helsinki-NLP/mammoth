"""Mammoth server, step 4: the mammoth_serve command (without starting uvicorn)."""

import pytest

from mammoth.bin import serve
from mammoth.tests.test_hf_modular import checkpoint  # noqa: F401  (fixture)

AVAILABLE = {"eng-spa": {"trained": True}, "deu-fra": {"trained": False}, "deu-spa": {"trained": True}}


def test_startup_tasks():
    assert serve.startup_tasks("trained", AVAILABLE) == ["deu-spa", "eng-spa"]
    assert serve.startup_tasks("all", AVAILABLE) == ["deu-fra", "deu-spa", "eng-spa"]
    assert serve.startup_tasks("none", AVAILABLE) == []
    assert serve.startup_tasks(" deu-fra, eng-spa ", AVAILABLE) == ["deu-fra", "eng-spa"]
    with pytest.raises(ValueError, match="Cannot serve.*spa-eng"):
        serve.startup_tasks("eng-spa,spa-eng", AVAILABLE)


def test_build_engine(checkpoint):
    ckpt, _, _ = checkpoint
    args = serve.get_parser().parse_args(["--model", str(ckpt), "--tasks", "eng-spa,deu-fra",
                                          "--device", "cpu", "--dtype", "bf16", "--num-beams", "2"])
    engine = serve.build_engine(args)
    assert sorted(engine.pool.loaded_tasks()) == ["deu-fra", "eng-spa"]
    assert engine.settings.num_beams == 2
    model = engine.pool.loaded_tasks()["deu-fra"].model
    assert all(p.dtype.is_floating_point and str(p.dtype) == "torch.bfloat16" for p in model.parameters())
    assert engine.translate("deu-fra", "deu5 deu6", max_new_tokens=3).split()[0].startswith("fra")

    args = serve.get_parser().parse_args(["--model", str(ckpt), "--device", "cpu"])
    assert sorted(serve.build_engine(args).pool.loaded_tasks()) == ["deu-spa", "eng-fra", "eng-spa"]


def test_main_reports_bad_tasks(checkpoint, monkeypatch, capsys):
    ckpt, _, _ = checkpoint
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: pytest.fail("server must not start"))
    with pytest.raises(SystemExit):
        serve.main(["--model", str(ckpt), "--tasks", "spa-eng"])
    assert "Cannot serve" in capsys.readouterr().err


def test_main_starts_uvicorn(checkpoint, monkeypatch):
    ckpt, _, _ = checkpoint
    started = {}
    monkeypatch.setattr("uvicorn.run", lambda app, host, port: started.update(app=app, host=host, port=port))
    serve.main(["--model", str(ckpt), "--tasks", "none", "--port", "9999"])
    assert started["host"] == "127.0.0.1" and started["port"] == 9999
    assert {r.path for r in started["app"].routes} >= {"/api/languages", "/api/translate", "/api/tasks"}

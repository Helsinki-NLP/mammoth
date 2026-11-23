"""HTTP API of the Mammoth server (FastAPI; install with `pip install mammoth-nmt[server]`).

OPUS-MT compatible (github.com/Helsinki-NLP/Opus-MT), so its clients work unchanged:

    GET  /api/languages              {"languages": {src: [tgt, ...]}}   loaded directions
    POST /api/translate              {"from", "to", "source"} → {"translation"}
    GET  /api/ready                  204 once a direction is loaded, else 503

Mammoth extensions:

    POST /api/translate              "source" may also be a list (→ list of translations);
                                     optional "num_beams", "max_new_tokens"
    GET  /api/tasks                  every direction the model can serve, trained or
                                     zero-shot: {task: {src, tgt, encoder, decoder, trained, loaded}}
    POST /api/tasks {"task"}         load a direction at runtime (only missing components are read)
    DELETE /api/tasks/{task}         unload it; components no other direction uses are freed
    GET  /api/health                 device, dtype, loaded tasks and components

Language codes are the model's own (e.g. "eng", not OPUS-MT's "en"). Errors come back as
{"error": message}, as in OPUS-MT.

Endpoints are plain functions, which FastAPI runs in a thread pool: a long load does not
block translation of loaded directions, and the engine runs one generation at a time.
"""

from typing import Optional, Union

try:
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, ConfigDict, Field
except ImportError as e:  # pragma: no cover
    raise ImportError("The Mammoth server needs FastAPI: pip install 'mammoth-nmt[server]'") from e

from mammoth.serving.engine import TranslationEngine
from mammoth.serving.pool import split_task

# Per-request caps, so one request cannot exhaust a laptop's memory.
MAX_NUM_BEAMS = 16


class TranslateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    src: str = Field(alias="from")
    tgt: str = Field(alias="to")
    source: Union[str, list[str]]
    num_beams: Optional[int] = Field(default=None, ge=1, le=MAX_NUM_BEAMS)
    max_new_tokens: Optional[int] = Field(default=None, ge=1)


class LoadRequest(BaseModel):
    task: str


def create_app(engine: TranslationEngine) -> FastAPI:
    pool = engine.pool
    app = FastAPI(title="Mammoth translation server")

    @app.exception_handler(HTTPException)
    def _error(request: Request, exc: HTTPException):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    # -- OPUS-MT compatible ------------------------------------------------------

    @app.get("/api/ready", status_code=204)
    def ready():
        if not pool.loaded_tasks():
            raise HTTPException(503, "No translation direction is loaded")
        return Response(status_code=204)

    @app.get("/api/languages")
    def languages():
        out = {}
        for loaded in pool.loaded_tasks().values():
            out.setdefault(loaded.spec["src"], []).append(loaded.spec["tgt"])
        return {"languages": {src: sorted(tgts) for src, tgts in sorted(out.items())}}

    @app.post("/api/translate")
    def translate(req: TranslateRequest):
        task = f"{req.src}-{req.tgt}"
        if task not in pool.loaded_tasks():
            if task in pool.available_tasks():
                raise HTTPException(404, f"Language pair {task} is not loaded; "
                                         f"load it with POST /api/tasks {{\"task\": \"{task}\"}}")
            raise HTTPException(404, f"Language pair {task} not supported")
        if req.max_new_tokens is not None and req.max_new_tokens > engine.settings.max_new_tokens:
            raise HTTPException(422, f"max_new_tokens may be at most {engine.settings.max_new_tokens}")
        try:
            translation = engine.translate(task, req.source, num_beams=req.num_beams,
                                           max_new_tokens=req.max_new_tokens)
        except KeyError:  # unloaded between the check and the call
            raise HTTPException(404, f"Language pair {task} is not loaded")
        return {"translation": translation}

    # -- Mammoth extensions -----------------------------------------------------------

    @app.get("/api/tasks")
    def tasks():
        loaded = pool.loaded_tasks()
        return {name: {**info, "loaded": name in loaded}
                for name, info in pool.available_tasks().items()}

    @app.post("/api/tasks", status_code=201)
    def load_task(req: LoadRequest):
        try:
            split_task(req.task)
        except ValueError as e:
            raise HTTPException(422, str(e))
        info = pool.available_tasks().get(req.task)
        if info is None:
            raise HTTPException(404, f"Language pair {req.task} not supported")
        pool.add_task(req.task)
        return {req.task: {**info, "loaded": True}}

    @app.delete("/api/tasks/{task}", status_code=204)
    def unload_task(task: str):
        try:
            pool.remove_task(task)
        except KeyError:
            raise HTTPException(404, f"Task {task} is not loaded")
        return Response(status_code=204)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "device": str(pool.device), "dtype": str(pool.dtype).removeprefix("torch."),
                "loaded_tasks": sorted(pool.loaded_tasks()), "loaded_components": pool.loaded_components()}

    return app

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from webapp.app.api.routes import router as api_router
from webapp.app.config import CACHE_PATH, DEFAULT_MODEL_MODE, MODEL_PATHS
from webapp.app.services.analyzer_service import UrlAnalyzerService
from webapp.app.services.cache_store import JsonCacheStore
from webapp.app.services.model_service import ModelService


def create_app() -> FastAPI:
    app = FastAPI(title="Phishing URL Analyzer", version="1.0.0")
    ui_root = Path(__file__).resolve().parent / "ui"

    app.mount("/static", StaticFiles(directory=str(ui_root / "static")), name="static")

    cache_store = JsonCacheStore(CACHE_PATH)

    analyzers = {}
    for mode, model_path in MODEL_PATHS.items():
        if model_path.exists():
            analyzers[mode] = UrlAnalyzerService(
                model_service=ModelService(model_path),
                cache_store=cache_store,
            )

    if not analyzers:
        configured = ", ".join(f"{name}={path}" for name, path in MODEL_PATHS.items())
        raise RuntimeError(f"No model artifacts found. Checked: {configured}")

    default_mode = DEFAULT_MODEL_MODE if DEFAULT_MODEL_MODE in analyzers else next(iter(analyzers.keys()))
    app.state.url_analyzers = analyzers
    app.state.default_model_mode = default_mode

    templates = Jinja2Templates(directory=str(ui_root / "templates"))

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "available_modes": sorted(analyzers.keys()),
                "default_mode": default_mode,
            },
        )

    app.include_router(api_router)
    return app


app = create_app()

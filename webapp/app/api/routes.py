from fastapi import APIRouter, HTTPException, Request

from webapp.app.models import AnalyzeRequest, AnalyzeResponse

router = APIRouter(prefix="/api", tags=["analysis"])


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_url(payload: AnalyzeRequest, request: Request):
    analyzers = request.app.state.url_analyzers
    default_mode = request.app.state.default_model_mode
    requested_mode = (payload.mode or default_mode).strip().lower()

    analyzer = analyzers.get(requested_mode)
    if analyzer is None:
        available_modes = ", ".join(sorted(analyzers.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported mode '{requested_mode}'. Available modes: {available_modes}",
        )

    url = payload.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
    try:
        result = analyzer.analyze(url, recheck=payload.recheck)
        result["mode"] = requested_mode
        result["model_type"] = analyzer.model_service.model_type
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to analyze URL: {exc}") from exc

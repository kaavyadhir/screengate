"""FastAPI surface.

Serves the gateway API and the demo UI from one process, so the whole
thing is one container and one deployment.

The `layers` query parameter is what makes the demo honest: the UI's
toggle switches map directly onto real policy, so turning a layer off in
the browser turns it off in the gateway. Nothing in the demo is replayed
from a script - type a new attack and it runs for real.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import ingest
from .audit import JsonlAuditLog
from .config import LayerToggles, Policy, Settings
from .models import Decision, Finding, GatewayRequest, ScreeningResult, Segment, Trust
from .pipeline import Gateway
from .providers.base import DEFAULT_JOB_DESCRIPTION

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

app = FastAPI(
    title="ScreenGate",
    description=(
        "A security gateway for LLM applications. Inspects untrusted input "
        "and model output against policy before either crosses a trust "
        "boundary. Demonstrated on a resume-screening assistant."
    ),
    version="1.0.0",
)

_settings = Settings.from_env()
_gateway = Gateway(settings=_settings, audit_sink=JsonlAuditLog(_settings.log_path))


class DocumentIn(BaseModel):
    content: str = Field(max_length=100_000)
    label: str | None = None


class AnalyzeRequest(BaseModel):
    user_input: str = Field(max_length=10_000)
    documents: list[DocumentIn] = Field(default_factory=list, max_length=10)
    session_id: str | None = None


class AnalyzeResponse(BaseModel):
    request_id: str
    decision: Decision
    output: str | None
    result: ScreeningResult | None
    findings: list[Finding]
    layers_run: list[str]
    latency_ms: float


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "provider": _gateway.provider.name,
        "semantic_reportable": _gateway.semantic_reportable,
    }


@app.get("/policy")
def policy() -> dict[str, object]:
    p = _gateway.policy
    return {
        "block_at": p.block_at.value,
        "layers_enabled": p.layers.enabled_names,
        "allowed_link_hosts": sorted(p.allowed_link_hosts),
        "max_document_chars": p.max_document_chars,
        "semantic_threshold": p.semantic_threshold,
        "score_anomaly_floor": p.score_anomaly_floor,
        "job_description": DEFAULT_JOB_DESCRIPTION,
    }


@app.post("/v1/analyze", response_model=AnalyzeResponse)
def analyze(
    body: AnalyzeRequest,
    layers: str | None = Query(
        default=None,
        description=(
            "Comma-separated layers to enable: inbound_patterns, spotlighting, "
            "outbound_policy, semantic. Omit for all; pass an empty string for none."
        ),
    ),
) -> AnalyzeResponse:
    try:
        toggles = LayerToggles.from_csv(layers)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    request = GatewayRequest(
        user_input=body.user_input,
        documents=[
            Segment(trust=Trust.DOCUMENT, content=d.content, label=d.label)
            for d in body.documents
        ],
        session_id=body.session_id,
    )
    res = _gateway.handle(request, policy=Policy().with_layers(toggles))
    return AnalyzeResponse(
        request_id=res.request_id,
        decision=res.decision,
        output=res.output,
        result=res.result,
        findings=res.findings,
        layers_run=res.layers_run,
        latency_ms=round(res.latency_ms, 2),
    )


class ConcealedOut(BaseModel):
    reason: str
    detail: str
    page: int
    text: str


class ScreenPdfResponse(AnalyzeResponse):
    extracted_text: str = Field(description="What the model actually reads.")
    pages: int
    concealed: list[ConcealedOut] = Field(
        default_factory=list,
        description="Text a human reader would not have seen.",
    )
    truncated: bool = False


@app.post("/v1/screen-pdf", response_model=ScreenPdfResponse)
async def screen_pdf(
    file: UploadFile = File(...),
    user_input: str = Form("Screen this candidate against the role."),
    layers: str | None = Query(default=None),
) -> ScreenPdfResponse:
    """Screen an uploaded resume PDF.

    Two detections happen here that cannot happen on pasted text: the
    file's content stream is scanned for text hidden from the reader, and
    those findings are handed to the gateway, which fails closed on them
    before any layer runs.

    The size cap is enforced while reading rather than after, so an
    oversized upload is abandoned mid-stream instead of being buffered in
    full. This endpoint takes files from strangers; it is attack surface
    in its own right.
    """
    try:
        toggles = LayerToggles.from_csv(layers)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    data = bytearray()
    while chunk := await file.read(256 * 1024):
        data.extend(chunk)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
            )

    try:
        extraction = ingest.extract(bytes(data), max_bytes=MAX_UPLOAD_BYTES)
    except ingest.PdfTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ingest.PdfUnreadable as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    label = (file.filename or "resume.pdf").split("/")[-1][:80]
    prior = ingest.to_findings(extraction, label=label)

    request = GatewayRequest(
        user_input=user_input[:10_000],
        documents=[
            Segment(trust=Trust.DOCUMENT, content=extraction.text, label=label)
        ],
    )
    res = _gateway.handle(
        request, policy=Policy().with_layers(toggles), prior_findings=prior
    )

    return ScreenPdfResponse(
        request_id=res.request_id,
        decision=res.decision,
        output=res.output,
        result=res.result,
        findings=res.findings,
        layers_run=res.layers_run,
        latency_ms=round(res.latency_ms, 2),
        extracted_text=extraction.text,
        pages=extraction.pages,
        concealed=[
            ConcealedOut(reason=c.reason, detail=c.detail, page=c.page, text=c.short())
            for c in extraction.concealed
        ],
        truncated=extraction.truncated,
    )


@app.get("/cases")
def cases() -> dict[str, object]:
    """Preset demo cases, read from the real benchmark corpus.

    The UI does not carry its own copies: what you click in the browser is
    a case from the corpus the numbers were measured on.
    """
    import yaml

    root = Path(__file__).resolve().parent.parent / "bench" / "corpus"
    out: dict[str, list[dict[str, object]]] = {"attacks": [], "benign": []}
    for name, key in (("attacks.yaml", "attacks"), ("benign.yaml", "benign")):
        path = root / name
        if not path.exists():
            continue
        for raw in yaml.safe_load(path.read_text(encoding="utf-8")) or []:
            docs = [
                d for d in (raw.get("documents") or []) if "content_repeat" not in d
            ]
            out[key].append(
                {
                    "id": raw["id"],
                    "category": raw.get("category") or raw.get("tripwire"),
                    "user_input": raw["user_input"].strip(),
                    "documents": [
                        {"content": d.get("content", "").strip(), "label": d.get("label")}
                        for d in docs
                    ],
                }
            )
    return out


class NoCacheStatic(StaticFiles):
    """Serve UI assets with caching disabled.

    Browsers aggressively cache same-URL JS, so an edited app.js keeps
    running the previous version until a hard refresh. During a live demo
    that looks like the gateway is broken - stale JS rendering against
    fresh HTML produces dead controls. The UI is a handful of KB served
    from the same process, so there is nothing to gain from caching it.
    """

    def is_not_modified(self, *args, **kwargs) -> bool:
        return False

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        return response


if UI_DIR.exists():
    app.mount("/static", NoCacheStatic(directory=str(UI_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        """Serve the page with a cache-busted script URL.

        No-store headers stop a browser caching the script in future, but
        they do not evict a copy it already holds - so an edited app.js can
        keep running stale against fresh HTML, which looks exactly like the
        gateway being broken. Stamping the script URL with the file's
        mtime changes the URL whenever the file changes, and a URL the
        browser has never seen cannot be served from cache.
        """
        html = (UI_DIR / "index.html").read_text(encoding="utf-8")
        try:
            version = str(int((UI_DIR / "app.js").stat().st_mtime))
        except OSError:
            version = "0"
        return HTMLResponse(
            html.replace("__ASSET_V__", version),
            headers={"Cache-Control": "no-store, must-revalidate"},
        )

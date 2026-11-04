"""Policy and settings.

Every defense layer is independently toggleable. That is not a convenience
feature - it is what makes the ablation possible. A gateway whose layers
cannot be switched off one at a time is a gateway whose individual
contribution you cannot prove, and an unprovable contribution is
indistinguishable from decoration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from .models import Decision, Severity, SEVERITY_ORDER


def _load_dotenv(path: Path | None = None) -> None:
    """Read .env into the environment if it is there.

    Every entry point - the API, the benchmark, the held-out runner -
    reaches settings through this module, so loading here means a key in
    .env works everywhere without each script remembering to do it. A
    plain parser rather than a dependency: the file is a handful of
    KEY=value lines, and real values already in the environment win, so
    an explicit `set GEMINI_API_KEY=...` still overrides the file.
    """
    path = path or Path(__file__).resolve().parent.parent / ".env"
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return

    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

LAYER_NAMES = ("inbound_patterns", "spotlighting", "outbound_policy", "semantic")


@dataclass(frozen=True)
class LayerToggles:
    inbound_patterns: bool = True
    spotlighting: bool = True
    outbound_policy: bool = True
    semantic: bool = True

    @property
    def enabled_names(self) -> list[str]:
        return [n for n in LAYER_NAMES if getattr(self, n)]

    @classmethod
    def none(cls) -> "LayerToggles":
        return cls(False, False, False, False)

    @classmethod
    def only(cls, *names: str) -> "LayerToggles":
        base = cls.none()
        for n in names:
            if n not in LAYER_NAMES:
                raise ValueError(f"unknown layer: {n}")
        return replace(base, **{n: True for n in names})

    @classmethod
    def from_csv(cls, csv: str | None) -> "LayerToggles":
        """Parse a 'layers=inbound_patterns,semantic' query parameter.

        An absent parameter means all layers on; an empty string means all
        off, which is what the UI sends when every switch is flipped.
        """
        if csv is None:
            return cls()
        wanted = [p.strip() for p in csv.split(",") if p.strip()]
        unknown = [w for w in wanted if w not in LAYER_NAMES]
        if unknown:
            raise ValueError(f"unknown layer(s): {', '.join(unknown)}")
        return cls.only(*wanted) if wanted else cls.none()


@dataclass(frozen=True)
class Policy:
    """What the gateway refuses to let through."""

    block_at: Severity = Severity.HIGH

    # Embedded in the system prompt. If it ever appears in output, the
    # operator instructions have leaked.
    system_prompt_canaries: tuple[str, ...] = (
        "SG-CANARY-4D81B6",
        "internal scoring rubric",
    )

    # The model may only link to these hosts. Anything else in output is a
    # candidate exfiltration channel.
    allowed_link_hosts: frozenset[str] = frozenset({"careers.example-corp.com"})

    max_document_chars: int = 25_000

    # Cosine similarity at or above which the semantic layer fires.
    #
    # Measured, not guessed: `python -m bench.tune_semantic` embeds the
    # whole corpus and sweeps thresholds. On gemini-embedding-2 the two
    # populations overlap (lowest attack 0.555, highest benign 0.692), so
    # no threshold separates them perfectly.
    #
    # The sweep's own pick was 0.63, by Youden's J. Rejected: J weights a
    # missed attack and a false positive equally, and here they are not
    # equal. A false positive blocks a real candidate's application; a
    # missed attack falls through to layer 3, which catches 86% alone. A
    # layer in a stack should not be tuned as though it were the last one.
    #
    #   0.72  24/33 attacks, 0 false positives
    #   0.68  28/33 attacks, 1 false positive
    #   0.63  30/33 attacks, 2 false positives (one of them new)
    #
    # 0.68 it is: the single false positive is the resume that *quotes* an
    # instruction, already documented as failing on L1, so L4 agreeing
    # costs nothing new. 0.63 would newly block an innocent case. The
    # paraphrase attacks this layer exists for go from 2/4 to 3/4.
    semantic_threshold: float = 0.68

    # A score at or above this from a resume whose retrieved text contains
    # no corroborating evidence is treated as anomalous. Catches hijacks
    # that produce contract-valid but implausible output.
    score_anomaly_floor: int = 90

    layers: LayerToggles = field(default_factory=LayerToggles)

    def decision_for(self, severity: Severity | None) -> Decision:
        if severity is None:
            return Decision.ALLOW
        if SEVERITY_ORDER[severity] >= SEVERITY_ORDER[self.block_at]:
            return Decision.BLOCK
        return Decision.FLAG

    def with_layers(self, layers: LayerToggles) -> "Policy":
        return replace(self, layers=layers)


@dataclass(frozen=True)
class Settings:
    provider: str = "mock"
    embedder: str = "auto"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-flash-latest"
    embedding_model: str = "gemini-embedding-001"
    vector_store: str = "auto"
    chroma_path: str = ".chroma"
    log_path: str = "audit.log"

    # Public-demo rate limit, applied per client to /v1/ only. Generous
    # enough that nobody clicking through the UI will ever see it, tight
    # enough that a script cannot drain a free-tier quota.
    rate_limit_events: int = 30
    rate_limit_window_s: float = 60.0

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            provider=os.getenv("SCREENGATE_PROVIDER", "mock"),
            embedder=os.getenv("SCREENGATE_EMBEDDER", "auto"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            gemini_model=os.getenv("SCREENGATE_GEMINI_MODEL", "gemini-flash-latest"),
            embedding_model=os.getenv(
                "SCREENGATE_EMBEDDING_MODEL", "gemini-embedding-001"
            ),
            vector_store=os.getenv("SCREENGATE_VECTOR_STORE", "auto"),
            chroma_path=os.getenv("SCREENGATE_CHROMA_PATH", ".chroma"),
            log_path=os.getenv("SCREENGATE_LOG_PATH", "audit.log"),
            rate_limit_events=int(os.getenv("SCREENGATE_RATE_LIMIT_EVENTS", "30")),
            rate_limit_window_s=float(
                os.getenv("SCREENGATE_RATE_LIMIT_WINDOW_S", "60")
            ),
        )

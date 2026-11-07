"""Safe, auditable LLM-assisted factor discovery."""

from .adapter import DiscoveryAdapter, DiscoveryDataset
from .evaluator import ExpressionEvaluator
from .evidence import EvidenceResult, RollingEvidenceEngine
from .factor import ExpressionFactor
from .llm import LLMClient, OpenAIResponsesClient, ReplayLLMClient
from .workflow import (
    FactorDiscoveryRunner,
    create_discovery_plan,
    load_discovery_config,
    load_discovery_plan,
    run_discovery,
    write_discovery_report,
)
from .models import (
    CandidateBatch,
    CandidateSpec,
    DiscoveryPlan,
    DiscoveryRunResult,
    EvidenceRecipe,
    ExpressionSpec,
    FactorDiscoveryConfig,
    LLMSettings,
    Operator,
)

__all__ = [
    "CandidateBatch",
    "CandidateSpec",
    "DiscoveryPlan",
    "DiscoveryAdapter",
    "DiscoveryDataset",
    "DiscoveryRunResult",
    "EvidenceRecipe",
    "EvidenceResult",
    "ExpressionEvaluator",
    "ExpressionFactor",
    "ExpressionSpec",
    "FactorDiscoveryRunner",
    "FactorDiscoveryConfig",
    "LLMSettings",
    "LLMClient",
    "OpenAIResponsesClient",
    "Operator",
    "ReplayLLMClient",
    "RollingEvidenceEngine",
    "create_discovery_plan",
    "load_discovery_config",
    "load_discovery_plan",
    "run_discovery",
    "write_discovery_report",
]

"""Optional RD-Agent v1.0.0 bridge for ZyQuant.

The bridge is deliberately outside the core :mod:`zyquant` package.  Importing
it never imports RD-Agent; the dependency is resolved only when an RD-Agent
loop or feedback object is explicitly requested.
"""

from .bridge import (
    RDAGENT_COMMIT,
    RDAGENT_VERSION,
    BridgeExperiment,
    BridgeFeedback,
    RDAgentCompatibilityError,
    RDAgentProcessClient,
    RDAgentUnavailableError,
    ZyQuantExperimentSummarizer,
    ZyQuantFactorModelRunner,
    ZyQuantHypothesis2Experiment,
    ZyQuantRDLoop,
    ZyQuantScenario,
    assert_no_qlib_modules_loaded,
    create_rdagent_loop,
    verify_rdagent_install,
)
from .handler import (
    OpenAIResponsesBackend,
    RDAgentAPIBackendAdapter,
    StructuredCompletion,
    StructuredRDAgentHandler,
)

__all__ = [
    "RDAGENT_COMMIT",
    "RDAGENT_VERSION",
    "BridgeExperiment",
    "BridgeFeedback",
    "OpenAIResponsesBackend",
    "RDAgentAPIBackendAdapter",
    "RDAgentCompatibilityError",
    "RDAgentProcessClient",
    "RDAgentUnavailableError",
    "StructuredRDAgentHandler",
    "StructuredCompletion",
    "ZyQuantExperimentSummarizer",
    "ZyQuantFactorModelRunner",
    "ZyQuantHypothesis2Experiment",
    "ZyQuantRDLoop",
    "ZyQuantScenario",
    "assert_no_qlib_modules_loaded",
    "create_rdagent_loop",
    "verify_rdagent_install",
]

"""Strict, code-free contracts for RD-Agent driven ZyQuant research."""

from .models import (
    AgentCampaignPlan,
    AgentCampaignResult,
    AgentResearchConfig,
    AgentResearchWindows,
    AgentProviderSettings,
    CandidateFeedback,
    CostModel,
    DateWindow,
    DatasetAdapterSpec,
    ExecutorSpec,
    ExperimentFeedback,
    FactorCandidateSpec,
    JointExperimentSpec,
    LabelContract,
    ModelRecipeSpec,
    ResourceGuards,
    ResearchHypothesis,
    StageBudgets,
    StrategyContract,
)
from .executor import ZyQuantExperimentExecutor
from .replay import ReplayAgentClient

__all__ = [
    "AgentCampaignPlan",
    "AgentCampaignResult",
    "AgentResearchConfig",
    "AgentResearchWindows",
    "AgentProviderSettings",
    "CandidateFeedback",
    "CostModel",
    "DateWindow",
    "DatasetAdapterSpec",
    "ExecutorSpec",
    "ExperimentFeedback",
    "FactorCandidateSpec",
    "JointExperimentSpec",
    "LabelContract",
    "ModelRecipeSpec",
    "ResourceGuards",
    "ResearchHypothesis",
    "ReplayAgentClient",
    "StageBudgets",
    "StrategyContract",
    "ZyQuantExperimentExecutor",
]

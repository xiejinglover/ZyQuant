class ZyQuantError(RuntimeError):
    """Base exception for expected framework failures."""


class DataContractError(ZyQuantError):
    """Raised when a canonical dataset violates its contract."""


class FutureDataError(DataContractError):
    """Raised when a point-in-time view is asked for future data."""


class FactorError(ZyQuantError):
    """Raised by the factor graph or cache."""


class FactorCacheMiss(FactorError):
    """Raised when a read-only factor engine cannot resolve a cache entry."""


class StrategyError(ZyQuantError):
    """Raised when a strategy emits an invalid decision."""


class ConstraintError(StrategyError):
    """Raised when a target portfolio cannot satisfy hard constraints."""


class ConstraintProtocolError(ConstraintError):
    """A candidate contains illegal values; fallback must not hide this."""


class BacktestError(ZyQuantError):
    """Raised when execution or accounting invariants are violated."""


class ExperimentError(ZyQuantError):
    """Raised by experiment or search persistence."""


class SchemaVersionError(ZyQuantError):
    """A persisted protocol cannot be consumed by this framework version."""


class PluginError(ZyQuantError):
    """A plugin is missing metadata, incompatible, or violates its contract."""


class ResourceError(ZyQuantError):
    """A retryable worker, I/O, or operating-system resource failure."""


class AccountingError(BacktestError):
    """A master/sleeve or P&L conservation invariant was violated."""

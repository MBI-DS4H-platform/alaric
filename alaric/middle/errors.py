class MiddleError(RuntimeError):
    """Base error for middle-level command failures."""


class SchemaError(MiddleError):
    """Invalid alaric.yaml."""


class ResolveError(MiddleError):
    """An auto value could not be resolved."""


class GraphError(MiddleError):
    """Invalid action graph."""


class ResultError(MiddleError):
    """Invalid result lifecycle operation."""


class PoolGraphError(MiddleError):
    """Invalid pose-pool connectivity/provenance graph."""

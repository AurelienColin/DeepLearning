from typing import Any

def logger(*args: Any, **kwargs: Any) -> None:
    """
    This is a mock implementation of the logger.
    It accepts any arguments and does nothing.
    """
    pass

# If other specific items are imported from Rignak.logging_utils by the codebase,
# they would need mock implementations here too. For now, just 'logger' is known.

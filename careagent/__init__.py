"""CareAgent: an agentic customer self-service assistant.

A multi-step agent that resolves order, return and delivery questions end-to-end by
calling typed tools, grounding its answers in retrieved policy documents, verifying
every claim against its citations, and handing off to a human agent when uncertain.
"""

from careagent.config import Settings, get_settings

__version__ = "0.1.0"
__all__ = ["Settings", "get_settings", "__version__"]

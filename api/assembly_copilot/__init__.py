"""Robot-independent assembly copilot scaffold."""

from .manual import load_manual
from .state_tracker import AssemblyStateTracker

__all__ = ["AssemblyStateTracker", "load_manual"]

from __future__ import annotations

import warnings
from typing import Any, Mapping

try:
    from .command_models import CommandContext
    from .command_router import CommandRouter
except ImportError:
    from command_models import CommandContext
    from command_router import CommandRouter


class ExtractKeyword:
    """Structured parser with a disabled legacy tuple adapter.

    ``extract_keyword`` is retained only so old callers fail closed. Consumers must
    migrate to ``parse_command`` and preserve its grounding and snapshot metadata.
    """

    def __init__(self, router: CommandRouter | None = None) -> None:
        self.router = router or CommandRouter()

    def parse_command(
        self,
        output_message: str,
        context: CommandContext | Mapping[str, Any] | None = None,
    ):
        return self.router.parse_command(output_message, context)

    def extract_keyword(
        self,
        output_message: str,
        context: CommandContext | Mapping[str, Any] | None = None,
    ):
        warnings.warn(
            "ExtractKeyword.extract_keyword() is disabled and always returns None; "
            "use parse_command() so resolved object and snapshot metadata are preserved.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


if __name__ == "__main__":
    parser = ExtractKeyword()
    print(parser.parse_command("hammer를 pos1으로 가져와").model_dump_json(indent=2))

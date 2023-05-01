"""General utility functions."""

import warnings

import datasets.utils.logging as ds_logging
import transformers.utils.logging as hf_logging
from datasets.utils import disable_progress_bar


def block_terminal_output() -> None:
    """Blocks undesired terminal output."""

    # Ignore user warnings throughout the codebase
    warnings.filterwarnings("ignore", category=UserWarning)

    # Disable logging from the `datasets` library
    ds_logging.set_verbosity_error()

    # Disable the tokeniser progress bars from the `datasets` library
    disable_progress_bar()


class ignore_transformers_output:
    """Context manager to block terminal output."""

    def __enter__(self) -> None:
        hf_logging.set_verbosity_error()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        hf_logging.set_verbosity_info()

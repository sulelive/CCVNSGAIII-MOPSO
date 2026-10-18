"""
Runtime path helper for the frozen CCVNSGAIII-MOPSO implementation.

This file configures Python's module search path without modifying the
canonical scientific implementation.
"""

from pathlib import Path
import sys


def configure_runtime(repo_root=None):

    if repo_root is None:
        repo_root = Path(__file__).resolve().parent

    repo_root = Path(repo_root).resolve()

    cc_dir = (
        repo_root
        / "algorithm"
        / "ccvnsgaiii_mopso"
    )

    paper2_dir = (
        repo_root
        / "paper3_audit"
        / "PAPER2_Q1_STAGE2C"
    )

    for path in [
        str(paper2_dir),
        str(cc_dir)
    ]:
        if path not in sys.path:
            sys.path.insert(0, path)

    return cc_dir, paper2_dir

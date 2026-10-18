"""
Minimal import example for CCVNSGAIII-MOPSO.

This confirms that the canonical algorithm and its inheritance chain
can be loaded from the public repository.
"""

from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO))

from runtime import configure_runtime

configure_runtime(REPO)

from q1_cooperative_phase8 import CCVNSGAIII_MOPSO_CPU_P8


def main():

    print("CCVNSGAIII-MOPSO successfully imported.")
    print()
    print("Canonical class:")
    print(CCVNSGAIII_MOPSO_CPU_P8.__name__)

    print()
    print("Inheritance chain:")

    for cls in CCVNSGAIII_MOPSO_CPU_P8.__mro__:
        print(" -", cls.__name__)


if __name__ == "__main__":
    main()

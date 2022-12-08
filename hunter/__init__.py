from dataclasses import dataclass

import sys
from pathlib import Path

package_path = Path(__file__).parent.absolute()

sys.path.append(str(package_path))

__all__ = ["model", "preprocessing"]

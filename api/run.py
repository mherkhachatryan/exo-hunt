import sys
from pathlib import Path

package_path = Path(__file__).parent.parent.absolute()
sys.path.append(str(package_path))

from api import application

if __name__ == '__main__':
    application.run(debug=True)

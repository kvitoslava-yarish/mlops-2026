"""Print md5 and size of every file in a directory (what `make data` should deliver)."""

import hashlib
import sys
from pathlib import Path

for path in sorted(Path(sys.argv[1]).iterdir()):
    if path.is_file():
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        print(f"{md5}  {path.stat().st_size:>10}  {path}")

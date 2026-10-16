"""perception/input/eth80.py -- extracted from sensory_floor.py (Move 3a).

MOVE ONLY: whole top-level statements, verbatim. The re-export
shim in sensory_floor.py keeps `from sensory_floor import X`
working until the importer repoint retires it.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional, Union

_ETH80_VIEW = re.compile(r"^([a-z]+)-?(\d+)[-_](\d+)-(\d+)\.png$", re.I)

def parse_eth80_view(path: Union[str, os.PathLike]) -> Optional[Dict[str, Any]]:
    """Parse ETH-80 image filename into structured view metadata.
    Standard ETH-80 format: <class><id>-<colatitude>-<azimuth>.png.
    Returns the raw `colat` alongside true elevation `el = 90 - colat`, so
    _dOmega and all other geometry treat `el` as a true elevation.
    """
    m = _ETH80_VIEW.match(os.path.basename(str(path)))
    if not m:
        return None
    colat = float(m.group(3))
    az = float(m.group(4))
    return {
        "label": m.group(1).lower(),
        "instance": m.group(1).lower() + m.group(2),
        "colat": colat,
        "el": 90.0 - colat,
        "az": az,
    }


"""Compatibility shim: aggregate moved modules under legacy top-level name."""
import core.types as _m1
import core.hashing as _m2
import perception.observation as _m3
import perception.observation_2d as _m4
import perception.floor as _m5
import perception.contours as _m6
import perception.polar as _m7
for _m in (_m1, _m2, _m3, _m4, _m5, _m6, _m7):
    for _name in dir(_m):
        if not _name.startswith("__"):
            globals()[_name] = getattr(_m, _name)

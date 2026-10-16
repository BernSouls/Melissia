import learning.fertility as _m
for _name in dir(_m):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_m, _name)

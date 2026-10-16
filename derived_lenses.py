import perception.lenses.derived as _m
import perception.lenses.peak_lens as _peak

for _source in (_m, _peak):
    for _name in dir(_source):
        if not _name.startswith("__"):
            globals()[_name] = getattr(_source, _name)

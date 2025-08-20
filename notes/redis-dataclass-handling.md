Yes—**it works out of the box** with dataclasses as values, because you’re defaulting to `pickle`. A (top-level) `@dataclass` is just a normal Python class; `pickle.dumps/loads` will handle it as long as all its fields are picklable.

### Make sure these are true

* The dataclass is defined at **module top level** (not inside a function).
* Your Redis client is created with **`decode_responses=False`** (so `.get()` returns `bytes` for `pickle.loads`).
* You’re okay with the usual **pickle caveats**: unsafe for untrusted data, brittle across code refactors, and tied to Python.

### Quick smoke test

```python
from dataclasses import dataclass
import redis

@dataclass
class User:
    id: int
    name: str

# create OAL with your pool (ensure decode_responses=False)
oal.create("u:1", User(1, "Ada"))
assert oal.exists("u:1")
obj = oal.read("u:1")
assert isinstance(obj, User) and obj.name == "Ada"
```

### Optional: JSON (instead of pickle) for dataclasses

If you prefer a text format, pass a serializer/deserializer pair. For a **flat** dataclass (no nested dataclasses, custom types), this is enough:

```python
import json
from dataclasses import asdict

def dataclass_json_codec(cls):
    def ser(obj):
        if not isinstance(obj, cls):
            raise TypeError(f"expected {cls.__name__}")
        return json.dumps(asdict(obj), ensure_ascii=False).encode("utf-8")
    def de(b):
        return cls(**json.loads(b))
    return ser, de

# usage
ser, de = dataclass_json_codec(User)
oal = ObjectAccessLayer(pool, serializer=ser, deserializer=de)
```

> If you have **nested dataclasses**, `asdict()` will nest dicts, but `cls(**...)` won’t automatically rebuild nested types—you’d need a slightly richer deserializer (or use a library like `msgspec`, `pydantic`, etc.).

### Minor niceties (optional)

* Use `pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)` for compact/faster pickles.
* Consider a tiny **version field** in the object or key namespace to survive schema changes later.

If you want, I can sketch a minimal nested-dataclass JSON codec, but for most cases your current `pickle` default is perfectly fine.

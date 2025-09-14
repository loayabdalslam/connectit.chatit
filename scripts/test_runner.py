import importlib
import pkgutil
import sys
import traceback
import os


def main():
    # Ensure project root on path
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    success = True
    for mod in pkgutil.iter_modules(["tests"]):
        if not mod.ispkg and mod.name.startswith("test_"):
            name = f"tests.{mod.name}"
            try:
                m = importlib.import_module(name)
                if hasattr(m, "run"):
                    m.run()
                print(f"[OK] {name}")
            except Exception:
                success = False
                print(f"[FAIL] {name}")
                traceback.print_exc()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

import importlib
import pkgutil
import sys
import traceback
import os
import time


def main():
    # Ensure project root on path
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    # Prepare logging
    try:
        from connectit.utils import data_file
        log_path = data_file("test.log")
    except Exception:
        log_path = os.path.join(root, "test.log")
    logs = []
    start_all = time.time()
    success = True
    discovered = [m for m in pkgutil.iter_modules(["tests"]) if (not m.ispkg and m.name.startswith("test_"))]
    print(f"Discovered {len(discovered)} test modules")
    for mod in discovered:
        name = f"tests.{mod.name}"
        t0 = time.time()
        try:
            m = importlib.import_module(name)
            if hasattr(m, "run"):
                m.run()
            dur = time.time() - t0
            line = f"[OK] {name} ({dur:.3f}s)"
            print(line)
            logs.append(line)
        except Exception:
            success = False
            dur = time.time() - t0
            line = f"[FAIL] {name} ({dur:.3f}s)\n{traceback.format_exc()}"
            print(line)
            logs.append(line)
    total = time.time() - start_all
    summary = f"Summary: passed={success} total={len(discovered)} duration={total:.3f}s"
    print(summary)
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(logs + [summary]))
        print(f"Wrote log to {log_path}")
    except Exception:
        pass
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

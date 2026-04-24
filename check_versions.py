"""CI helper — syntax checks and import checks every snake version.
"""
import ast
import importlib.util
import os
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

VERSIONS = [
    "snake_v0.py",
    "snake_v1.py",
    "snake_v2.py",
    "snake_v3.py",
    "snake_v4.py",
    "snake_v5.py",
    "snake_v6.py",
]

failed = []

for filename in VERSIONS:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", filename)
    print(f"Checking {filename}...")

    # Syntax check
    try:
        with open(path) as f:
            ast.parse(f.read())
        print("  syntax OK")
    except SyntaxError as e:
        print(f"  SYNTAX ERROR: {e}")
        failed.append(filename)
        continue

    # Import check
    try:
        spec = importlib.util.spec_from_file_location("mod", path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print("  imports OK")
    except Exception as e:
        print(f"  IMPORT ERROR: {e}")
        failed.append(filename)

if failed:
    print(f"\nFailed: {failed}")
    sys.exit(1)

print("\nAll versions OK")
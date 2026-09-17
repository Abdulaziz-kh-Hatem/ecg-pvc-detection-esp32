#!/usr/bin/env python3
"""
Automated Test Runner for the PVC detection suite.
Discovers and executes all test functions across test_*.py files.
Auto-delegates to project .venv if executed from an external python environment.
"""

import os
import sys
import time
import traceback
import importlib.util
import subprocess

# Auto-delegate to local virtualenv if present and not already active
_repo_root = os.path.abspath(os.path.dirname(__file__))
_venv_python = os.path.join(_repo_root, ".venv", "Scripts", "python.exe")
if os.path.exists(_venv_python) and os.path.abspath(sys.executable).lower() != os.path.abspath(_venv_python).lower():
    _res = subprocess.run([_venv_python, os.path.abspath(__file__)] + sys.argv[1:])
    sys.exit(_res.returncode)

# Stabilize OpenBLAS thread allocation on Windows
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

# Add repository root to python search path
sys.path.insert(0, _repo_root)


def main():
    tests_dir = os.path.abspath(os.path.join(_repo_root, "tests"))
    
    test_entries = []
    for root, _, files in os.walk(tests_dir):
        for f in sorted(files):
            if f.startswith("test_") and f.endswith(".py"):
                filepath = os.path.join(root, f)
                rel_path = os.path.relpath(filepath, tests_dir)
                mod_name = rel_path.replace(os.sep, ".").replace("/", ".")[:-3]
                test_entries.append((mod_name, filepath))

    test_entries.sort(key=lambda x: x[0])

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    print("\n" + "=" * 80)
    print(" RUNNING SCIENTIFIC TEST SUITE")
    print("=" * 80)

    start_time = time.time()

    for module_name, filepath in test_entries:
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Find all functions starting with test_
        test_funcs = [getattr(mod, name) for name in dir(mod) if name.startswith("test_") and callable(getattr(mod, name))]

        for func in test_funcs:
            total_tests += 1
            test_desc = f"{module_name}.{func.__name__}"
            print(f"Running {test_desc:<55} ... ", end="", flush=True)
            try:
                func()
                print("PASSED [OK]")
                passed_tests += 1
            except Exception as e:
                print("FAILED [X]")
                failed_tests.append((test_desc, str(e), traceback.format_exc()))

    elapsed = time.time() - start_time
    print("-" * 80)
    print(f"Results: {passed_tests}/{total_tests} tests passed in {elapsed:.2f} seconds.")
    print("=" * 80)

    if failed_tests:
        print("\nFailures:")
        for name, err, tb in failed_tests:
            print(f"\n--- {name} ---")
            print(tb)
        sys.exit(1)
    else:
        print("\nALL SCIENTIFIC TESTS PASSED SUCCESSFULLY! 100% INVARIANTS VERIFIED.")
        sys.exit(0)


if __name__ == "__main__":
    main()

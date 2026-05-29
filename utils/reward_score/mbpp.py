"""Reward scoring utilities for MBPP."""

import multiprocessing
import os
import re
import signal
import threading
from contextlib import contextmanager
from typing import Any, Dict, List

# Set multiprocessing start method to 'fork' for better compatibility
# This needs to be done before any multiprocessing operations
try:
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    # Already set, ignore
    pass


class TimeoutError(Exception):
    pass


def _is_main_thread() -> bool:
    """Return True when signal-based timeout can be used safely."""
    return threading.current_thread() is threading.main_thread()


@contextmanager
def timeout(seconds: int = 5):
    """Context manager for execution timeout.

    signal.SIGALRM only works in the main thread; in worker threads we skip
    signal-based timeout to keep RandOpt evaluation stable.
    """
    if not _is_main_thread():
        yield
        return

    def handler(signum, frame):
        raise TimeoutError("Code execution timed out")

    try:
        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    except ValueError:
        # Defensive fallback when environment rejects signal registration.
        yield


def _execute_in_subprocess(code: str, test_list: List[str], test_setup_code: str, result_queue):
    """Execute code and tests in a subprocess."""
    try:
        namespace = {}
        
        # Execute main code
        exec(code, namespace)
        
        # Run all tests
        for test in test_list:
            exec(test, namespace)
        
        # All tests passed
        result_queue.put(True)
    except Exception:
        result_queue.put(False)


def execute_code_with_tests(
    code: str,
    test_list: List[str],
    test_setup_code: Any = "",
    timeout_sec: int = 5,
) -> bool:
    """Execute candidate code and run all tests with multiprocessing-based timeout."""
    if not code.strip():
        return False

    if hasattr(test_list, "tolist"):
        test_list = test_list.tolist()
    test_list = list(test_list) if test_list is not None else []

    if test_setup_code is None:
        test_setup_code = ""
    elif hasattr(test_setup_code, "tolist"):
        items = test_setup_code.tolist()
        test_setup_code = "\n".join([s for s in items if s]) if items else ""
    elif isinstance(test_setup_code, list):
        test_setup_code = "\n".join([s for s in test_setup_code if s])
    elif not isinstance(test_setup_code, str):
        test_setup_code = ""

    full_code = ""
    if isinstance(test_setup_code, str) and test_setup_code.strip():
        full_code += test_setup_code + "\n"
    full_code += code + "\n"

    # Try multiprocessing-based timeout first (more robust)
    try:
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_execute_in_subprocess,
            args=(full_code, test_list, test_setup_code, result_queue)
        )
        
        process.start()
        process.join(timeout=timeout_sec)
        
        if process.is_alive():
            # Timeout occurred
            process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join()
            return False
        
        # Check if we got a result
        if not result_queue.empty():
            return result_queue.get()
        else:
            # Process exited but no result (likely crashed)
            return False
            
    except Exception as e:
        # Multiprocessing failed, fall back to signal-based timeout (if in main thread)
        # or no timeout (if in worker thread)
        if process and hasattr(process, 'is_alive') and process.is_alive():
            try:
                process.terminate()
                process.join(timeout=1)
                if process.is_alive():
                    process.kill()
                    process.join()
            except Exception:
                pass
        
        # Fallback to original implementation
        namespace = {}
        try:
            with timeout(timeout_sec):
                exec(full_code, namespace)
        except Exception:
            return False

        for test in test_list:
            try:
                with timeout(timeout_sec):
                    exec(test, namespace)
            except (AssertionError, TimeoutError, Exception):
                return False

        return True


def extract_answer(response: str) -> str:
    """Extract code from `<answer>...</answer>` tags or code fences."""
    matches = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if matches:
        return matches[-1].strip()

    code_matches = re.findall(r"```python\n?(.*?)```", response, re.DOTALL)
    if code_matches:
        return code_matches[-1].strip()

    code_matches = re.findall(r"```\n?(.*?)```", response, re.DOTALL)
    if code_matches:
        return code_matches[-1].strip()

    return ""


def compute_score(code: str, ground_truth: Dict, score: float = 1.0) -> float:
    """Return `score` when generated code passes all tests, else 0.0."""
    if not code:
        return 0.0

    if isinstance(ground_truth, dict):
        test_list = ground_truth.get("test_list", [])
        test_setup_code = ground_truth.get("test_setup_code", "")
    else:
        try:
            if hasattr(ground_truth, "item"):
                ground_truth = dict(ground_truth)
            test_list = ground_truth.get("test_list", [])
            test_setup_code = ground_truth.get("test_setup_code", "")
        except Exception:
            return 0.0

    passed = execute_code_with_tests(
        code,
        test_list,
        test_setup_code,
    )
    return score if passed else 0.0

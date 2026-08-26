"""
record_environment.py
---------------------
Freeze the execution environment block reported in the paper.

Measurements are only comparable across a campaign if the platform is the
same, and "same GPU model" is not the same thing as "same host": a driver
revision alone can move a timing. Recording the environment mechanically, once
per campaign, means the claim can be checked instead of remembered.

Usage:
    python tools/record_environment.py
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     ".."))



def _redact(path):
    """Replace a home directory with ~ so no username is published."""
    home = os.path.expanduser("~")
    return path.replace(home, "~") if home and path.startswith(home) else path

def smi(query):
    try:
        return subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}",
             "--format=csv,noheader,nounits"],
            text=True, timeout=15).strip().splitlines()[0].strip()
    except Exception:                                           # noqa: BLE001
        return None


def main():
    info = {
        "os": platform.platform(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        # Redacted: the interpreter path adds nothing over the version and
        # the environment name, and the full path carries a user directory
        # into a published artifact.
        "python_executable": _redact(sys.executable),
    }

    for mod in ("numpy", "scipy", "cupy", "matplotlib", "pandas", "pyvista",
                "skimage", "torch"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "?")
        except Exception:                                       # noqa: BLE001
            info[f"{mod}_version"] = None

    try:
        import cupy as cp
        info["cuda_runtime_version"] = cp.cuda.runtime.runtimeGetVersion()
        info["cuda_driver_version"] = cp.cuda.runtime.driverGetVersion()
        props = cp.cuda.runtime.getDeviceProperties(0)
        info["gpu_name"] = props["name"].decode()
        info["gpu_compute_capability"] = f"{props['major']}.{props['minor']}"
        info["gpu_sm_count"] = props["multiProcessorCount"]
        free, total = cp.cuda.runtime.memGetInfo()
        info["gpu_total_bytes"] = int(total)
        info["gpu_total_GiB"] = total / 2**30
        info["gpu_free_at_start_GiB"] = free / 2**30
        info["gpu_used_by_other_processes_GiB"] = (total - free) / 2**30
    except Exception as ex:                                     # noqa: BLE001
        info["cupy_error"] = repr(ex)

    info["nvidia_driver"] = smi("driver_version")
    info["gpu_power_limit_W"] = smi("power.limit")
    info["gpu_memory_total_MiB"] = smi("memory.total")
    info["CUDA_PATH"] = os.environ.get("CUDA_PATH")
    info["GPU_FEM_ENV_BOOTSTRAPPED"] = os.environ.get(
        "GPU_FEM_ENV_BOOTSTRAPPED")
    info["gpu_clocks_locked"] = False
    info["gpu_drives_display"] = True
    info["note"] = (
        "The GPU also drives the desktop, and its clocks are not locked "
        "(locking them is a system-wide setting). Short-kernel measurements "
        "therefore carry occasional contention outliers; the microbenchmark "
        "reports the median of independent timing blocks with the "
        "interquartile range and full range, rather than a single run.")

    out = os.path.join(_ROOT, "results", "G0", "environment.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)

    txt = os.path.join(_ROOT, "results", "G0", "repro_environment_v2.txt")
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write("Execution environment\n")
        fh.write("=" * 62 + "\n")
        for k, v in info.items():
            fh.write(f"{k:34s} {v}\n")

    for k, v in info.items():
        print(f"{k:34s} {v}")
    print(f"\nwritten: {out}\n         {txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

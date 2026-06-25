"""Disposable Docker sandbox for executing untrusted shell/Python.

The container carries NO host secrets (no ``.env``, no ``~/.ssh``) and only the
data dir (read-only) plus a shared work dir are mounted. Because the container —
not a command denylist — is the security boundary, a full shell is allowed
inside it. Secret-bearing tools (search, renewables, tariffs, ...) run on the
host and never enter the sandbox.
"""

import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "energyevals-sandbox:latest")
# Default "bridge" network gives the container unrestricted internet and LAN
# access. Set SANDBOX_NETWORK to a firewalled Docker network to restrict it.
SANDBOX_NETWORK = os.getenv("SANDBOX_NETWORK", "bridge")
SANDBOX_DNS = os.getenv("SANDBOX_DNS", "1.1.1.1")  # public DNS so LAN can be firewalled
SANDBOX_MEMORY = os.getenv("SANDBOX_MEMORY", "2g")
SANDBOX_CPUS = os.getenv("SANDBOX_CPUS", "2")
SANDBOX_PIDS = os.getenv("SANDBOX_PIDS", "256")
SANDBOX_TMPFS_SIZE = os.getenv("SANDBOX_TMPFS_SIZE", "512m")
SANDBOX_DOCKER_OVERHEAD_S = 15  # extra wall-clock beyond the command timeout

# Per-process run id so two benchmarks running concurrently (separate repos /
# processes against the same Docker daemon) only ever reap THEIR OWN containers.
# Without it, one run's end-of-run cleanup -- docker rm -f by the shared "ee-sbx-"
# prefix -- would force-kill the other run's live sandbox containers. Override via
# SANDBOX_RUN_ID to pin it.
_RUN_ID = os.getenv("SANDBOX_RUN_ID") or uuid.uuid4().hex[:8]
_CONTAINER_PREFIX = f"ee-sbx-{_RUN_ID}-"

_REPO_ROOT = Path(__file__).resolve().parents[2]
# data = read-only inputs; work = shared scratch (also where host tools drop CSVs).
SANDBOX_DATA_DIR = _REPO_ROOT / "data"
SANDBOX_WORK_DIR = _REPO_ROOT / "run_outputs"
# Path the shared work dir is mounted at INSIDE the container. Host tools that
# drop files into SANDBOX_WORK_DIR should report this path so the model can read
# them with run_python_code/run_shell_command (which only run in the sandbox).
SANDBOX_WORK_MOUNT = "/work"
SANDBOX_DATA_MOUNT = "/data"
SANDBOX_WORK_KEEP = {"tool_output_logs"}


def sandbox_path(path: str | Path) -> str:
    """Host path under the shared work dir -> the path the sandbox sees (/work/...).

    Host tools that write files for the model to read with run_python_code should
    report this. Paths not under the work dir are returned unchanged.
    """
    try:
        rel = Path(path).resolve().relative_to(SANDBOX_WORK_DIR.resolve())
        return f"{SANDBOX_WORK_MOUNT}/{rel.as_posix()}"
    except ValueError:
        return str(path)


def host_path(path: str | Path) -> str:
    """Sandbox-visible path (/work, /data) -> the real host path.

    Host tools that receive a path the model obtained inside the sandbox (e.g. a
    CSV at ``/work/x.csv``) must map it back to the host directory. ``/tmp`` is
    per-call ephemeral and unreadable from the host, so it is left unchanged.
    """
    p = str(path)
    for mount, host_dir in ((SANDBOX_WORK_MOUNT, SANDBOX_WORK_DIR),
                            (SANDBOX_DATA_MOUNT, SANDBOX_DATA_DIR)):
        if p == mount or p.startswith(mount + "/"):
            rel = p[len(mount):].lstrip("/")
            return str(host_dir / rel) if rel else str(host_dir)
    return p


def sandbox_available() -> bool:
    """True if Docker is reachable and the sandbox image is built."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", SANDBOX_IMAGE],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception as exc:  # docker missing / daemon down
        logger.debug(f"sandbox_available check failed: {exc}")
        return False


def _docker_run_argv(name: str) -> list[str]:
    """Build the hardened ``docker run`` argument list (no command yet)."""
    argv = [
        "docker", "run", "--rm", "-i",
        "--name", name,
        "--network", SANDBOX_NETWORK,
        "--dns", SANDBOX_DNS,
        "--read-only",                                   # host/root fs immutable
        "--tmpfs", f"/tmp:rw,size={SANDBOX_TMPFS_SIZE}",
        "--workdir", SANDBOX_WORK_MOUNT,
        "--user", "1000:1000",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", SANDBOX_PIDS,                    # fork-bomb guard
        "--memory", SANDBOX_MEMORY,
        "--cpus", SANDBOX_CPUS,
        "--env", f"HOME={SANDBOX_WORK_MOUNT}",           # NO secrets in env
        "-v", f"{SANDBOX_WORK_DIR}:{SANDBOX_WORK_MOUNT}:rw",
    ]
    if SANDBOX_DATA_DIR.exists():
        argv += ["-v", f"{SANDBOX_DATA_DIR}:{SANDBOX_DATA_MOUNT}:ro"]
    return argv


def _execute(
    container_argv: list[str],
    name: str,
    timeout: int,
    stdin: str | None,
    nonzero_is_error: bool,
) -> dict[str, Any]:
    """Run the container, enforcing ``timeout`` and guaranteeing teardown.

    ``nonzero_is_error`` controls how a nonzero exit is reported: True for code
    execution (a failing script is an error), False for shell commands (a
    nonzero exit -- e.g. ``grep`` finding nothing -- is a normal result, not a
    tool failure). Either way ``returncode`` is included.
    """
    try:
        result = subprocess.run(
            container_argv,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout + SANDBOX_DOCKER_OVERHEAD_S,
        )
    except subprocess.TimeoutExpired:
        # The docker CLI was killed; make sure the container is gone too.
        subprocess.run(["docker", "kill", name], capture_output=True, timeout=10)
        return {
            "status": "error",
            "error": f"Execution timed out after {timeout}s",
            "stdout": "",
            "stderr": "",
        }
    except FileNotFoundError:
        return {"status": "error", "error": "docker not available", "stdout": "", "stderr": ""}

    if result.returncode == 0 or not nonzero_is_error:
        return {
            "status": "success",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    stderr = (result.stderr or "").strip()
    err_line = stderr.splitlines()[-1] if stderr else f"Exited with code {result.returncode}"
    return {
        "status": "error",
        "returncode": result.returncode,
        "error": err_line,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def run_shell(command: str, timeout: int) -> dict[str, Any]:
    """Run a full shell command (pipes/heredocs/etc.) inside the sandbox."""
    SANDBOX_WORK_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{_CONTAINER_PREFIX}{uuid.uuid4().hex[:12]}"
    inner = ["timeout", "--signal=KILL", f"{timeout}s", "bash", "-lc", command]
    return _execute(
        _docker_run_argv(name) + [SANDBOX_IMAGE, *inner], name, timeout, None,
        nonzero_is_error=False,
    )


def run_python(code: str, timeout: int) -> dict[str, Any]:
    """Run Python code (fed via stdin) inside the sandbox."""
    SANDBOX_WORK_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{_CONTAINER_PREFIX}{uuid.uuid4().hex[:12]}"
    inner = ["timeout", "--signal=KILL", f"{timeout}s", "python3", "-"]
    return _execute(
        _docker_run_argv(name) + [SANDBOX_IMAGE, *inner], name, timeout, code,
        nonzero_is_error=True,
    )


def reset_work_dir() -> None:
    """Clear the bind-mounted scratch dir (``/work``), keeping ``SANDBOX_WORK_KEEP``.

    Called at the start of each benchmark question so files generated for one
    question (CSV offloads, plots, model-written scratch) can't leak into a later
    question's ``list_files``/``grep_files``. Diagnostic log subdirs are preserved.
    Safe to call when the work dir doesn't exist yet.
    """
    if not SANDBOX_WORK_DIR.exists():
        return
    for child in SANDBOX_WORK_DIR.iterdir():
        if child.name in SANDBOX_WORK_KEEP:
            continue
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError as exc:
            logger.warning(f"sandbox cleanup: could not remove {child}: {exc}")


def cleanup_run() -> None:
    """Tear down sandbox state at the end of a benchmark run.

    Per-call containers are already ``--rm``; this clears the persistent
    bind-mounted scratch dir (``/work``, keeping ``SANDBOX_WORK_KEEP`` subdirs
    like logs) and force-removes any orphaned sandbox containers left by
    timed-out calls. Safe to call even when the docker backend wasn't used.
    """
    reset_work_dir()

    try:
        listed = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"name={_CONTAINER_PREFIX}"],
            capture_output=True, text=True, timeout=10,
        )
        ids = listed.stdout.split()
        if ids:
            subprocess.run(["docker", "rm", "-f", *ids], capture_output=True, timeout=30)
            logger.info(f"sandbox cleanup: removed {len(ids)} stray container(s)")
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        logger.debug(f"sandbox cleanup: container reap skipped ({exc})")

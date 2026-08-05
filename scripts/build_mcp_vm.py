#!/usr/bin/env python3
"""
One-time setup: create Ubuntu-MCP.qcow2 with MCP server pre-installed.

Usage:
    python OSWorld-MCP/setup_mcp_vm.py

What it does:
  1. Boots the Ubuntu VM in Docker (QEMU, CPU-only is fine but slow ~5-15 min)
  2. pip-installs fastmcp and injects MCP server files into the VM
  3. Copies the QCOW2 overlay out of the container before shutdown
  4. Merges overlay + base → Ubuntu-MCP.qcow2  (standalone, no backing file)

After success, do ONE of:
  Option A (replace):
    cp docker_vm_data/Ubuntu.qcow2  docker_vm_data/Ubuntu-orig.qcow2
    cp docker_vm_data/Ubuntu-MCP.qcow2  docker_vm_data/Ubuntu.qcow2

  Option B (edit manager.py to point at Ubuntu-MCP.qcow2 by default)
"""

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import docker
import requests

# ── paths (override via env vars) ─────────────────────────────────────────────
MCP_REPO = Path(os.environ.get("MCP_SRC_ROOT", "mcp_tools"))   # vendored MCP server/tools (contains mcp/)
MCP_DIR  = MCP_REPO / "mcp"
VM_DIR   = Path(os.environ.get("VM_DIR", "OSWorld-main/docker_vm_data"))  # dir holding the stock Ubuntu.qcow2

UBUNTU_QCOW2   = VM_DIR / "Ubuntu.qcow2"
OUTPUT_QCOW2   = VM_DIR / "Ubuntu-MCP.qcow2"
OVERLAY_OUTDIR = Path("/tmp/osworld_mcp_setup")  # host dir → mounted at /setup-output/

SERVER_PORT = 5099   # fixed host port for this session
VNC_PORT    = 8099

OVERLAY_OUTDIR.mkdir(parents=True, exist_ok=True)

# Pre-downloaded fastmcp wheels (Python 3.10 linux x86_64).
# Populate with: pip download fastmcp --dest WHEELS_DIR \
#   --index-url https://pypi.org/simple/ \
#   --python-version 3.10 --platform manylinux_2_17_x86_64 --abi cp310 \
#   --only-binary :all:
WHEELS_DIR = Path("/tmp/fastmcp_wheels_310")

# Session that bypasses any HTTP proxy (critical for localhost health checks)
_session = requests.Session()
_session.trust_env = False


# ── Docker helpers ─────────────────────────────────────────────────────────────

def start_container() -> docker.models.containers.Container:
    client = docker.from_env()

    devices = []
    env = {"DISK_SIZE": "32G", "RAM_SIZE": "8G", "CPU_CORES": "4"}
    if os.path.exists("/dev/kvm"):
        devices.append("/dev/kvm")
        print("  KVM found – hardware acceleration enabled")
    else:
        env["KVM"] = "N"
        print("  No KVM – software emulation (VM boot will take ~5-15 min, be patient)")
    if os.path.exists("/dev/net/tun"):
        devices.append("/dev/net/tun")

    volumes = {
        str(UBUNTU_QCOW2.resolve()): {"bind": "/System.qcow2", "mode": "ro"},
        str(OVERLAY_OUTDIR.resolve()): {"bind": "/setup-output", "mode": "rw"},
    }
    # Mount pre-downloaded wheels so we can serve them to the VM without internet
    if WHEELS_DIR.exists() and any(WHEELS_DIR.iterdir()):
        volumes[str(WHEELS_DIR.resolve())] = {"bind": "/wheels", "mode": "ro"}
        print(f"  Wheels dir : {WHEELS_DIR}  (will serve to VM via HTTP)")

    container = client.containers.run(
        "happysixd/osworld-docker",
        environment=env,
        cap_add=["NET_ADMIN"],
        privileged=True,
        devices=devices,
        volumes=volumes,
        ports={5000: SERVER_PORT, 8006: VNC_PORT},
        detach=True,
    )
    print(f"  Container : {container.short_id}")
    print(f"  VNC debug : http://localhost:{VNC_PORT}  (open in browser to watch)")
    return container


def wait_for_vm(container, timeout: int = 900) -> bool:
    print(f"  Polling VM python-server on :{SERVER_PORT} (timeout {timeout}s) …", flush=True)
    start = time.time()
    tick = 0
    while time.time() - start < timeout:
        # Check if container exited early
        try:
            container.reload()
        except Exception:
            print("\n  Container no longer exists – was it removed externally? ✗")
            return False
        if container.status in ("exited", "dead"):
            print(f"\n  Container exited early (status={container.status}) ✗")
            try:
                logs = container.logs(tail=60).decode("utf-8", errors="replace")
                print("  Container logs (last 60 lines):\n")
                print(logs)
            except Exception as e:
                print(f"  (Could not read logs: {e})")
            return False

        try:
            r = _session.get(f"http://127.0.0.1:{SERVER_PORT}/screenshot", timeout=5)
            if r.status_code == 200:
                elapsed = int(time.time() - start)
                print(f"\n  VM ready after {elapsed}s ✓")
                return True
        except Exception:
            pass

        tick += 1
        if tick % 15 == 0:  # print container status every ~2 min
            elapsed = int(time.time() - start)
            print(f"\n  [{elapsed}s] container status={container.status}, still waiting…", flush=True)
        else:
            print(".", end="", flush=True)
        time.sleep(8)
    print("\n  TIMEOUT ✗")
    # Print last container logs to help diagnose
    try:
        logs = container.logs(tail=40).decode("utf-8", errors="replace")
        print("  Container logs (last 40 lines):\n")
        print(logs)
    except Exception:
        pass
    return False


# ── VM communication via /execute endpoint ─────────────────────────────────────
#
# All code is base64-encoded before sending so there are ZERO quoting issues,
# regardless of what the Python code or file content contains.

_EXEC_PREFIX = "import pyautogui; import time; pyautogui.FAILSAFE = False; "


def _vm_exec_b64(python_code: str, timeout: int = 120) -> dict:
    """
    Encode python_code as base64, send:
        python -c "...prefix...; exec(b64decode('...').decode())"
    Returns the response JSON dict (keys: output, error).
    """
    b64 = base64.b64encode(python_code.encode()).decode()
    one_liner = (
        _EXEC_PREFIX +
        f"exec(__import__('base64').b64decode('{b64}').decode())"
    )
    payload = json.dumps({"command": ["python", "-c", one_liner], "shell": False})
    try:
        r = _session.post(
            f"http://127.0.0.1:{SERVER_PORT}/execute",
            headers={"Content-Type": "application/json"},
            data=payload,
            timeout=timeout,
        )
        if r.status_code == 200:
            return r.json()
        return {"output": "", "error": f"HTTP {r.status_code}"}
    except requests.exceptions.Timeout:
        return {"output": "(timeout)", "error": "timeout"}
    except Exception as e:
        return {"output": "", "error": str(e)}


def vm_shell(cmd: str, timeout: int = 300) -> str:
    """Run a shell command in the Ubuntu VM; return combined stdout/stderr."""
    cmd_b64 = base64.b64encode(cmd.encode()).decode()
    script = f"""
import subprocess, base64
cmd = base64.b64decode('{cmd_b64}').decode()
r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout={max(timeout - 10, 10)})
out = r.stdout or ''
err = r.stderr or ''
print(out[-3000:] if len(out) > 3000 else out, end='')
if err.strip():
    print('[STDERR]', err[-500:])
print('[RC]', r.returncode)
""".strip()
    result = _vm_exec_b64(script, timeout=timeout)
    return (result.get("output") or "").strip()


_CHUNK_SIZE = 32 * 1024  # 32 KB per chunk – stays well under /execute body limit

def vm_write_file(remote_path: str, content_bytes: bytes) -> bool:
    """Write arbitrary bytes into the Ubuntu VM.

    Large files are split into 32 KB chunks to avoid HTTP 500 from the
    /execute endpoint's body-size limit.
    """
    # ── first chunk: create/truncate the file ─────────────────────────
    chunks = [content_bytes[i:i + _CHUNK_SIZE]
              for i in range(0, max(len(content_bytes), 1), _CHUNK_SIZE)]

    for idx, chunk in enumerate(chunks):
        chunk_b64 = base64.b64encode(chunk).decode()
        mode = "'wb'" if idx == 0 else "'ab'"
        inner_script = f"""
import base64, os
dest = r'{remote_path}'
os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
open(dest, {mode}).write(base64.b64decode('{chunk_b64}'))
print('chunk{idx}')
""".strip()
        result = _vm_exec_b64(inner_script, timeout=30)
        out = (result.get("output") or "")
        if f"chunk{idx}" not in out:
            err = result.get("error") or ""
            print(f"    ✗  {remote_path}  chunk={idx}  err={err!r}  out={out!r}")
            return False

    # ── verify final size ──────────────────────────────────────────────
    verify = _vm_exec_b64(
        f"import os; print('size', os.path.getsize(r'{remote_path}'))", timeout=10
    )
    out = verify.get("output") or ""
    if "size" in out:
        got = int(out.split("size")[1].strip())
        if got != len(content_bytes):
            print(f"    ✗  {remote_path}  size mismatch: got {got}, want {len(content_bytes)}")
            return False
    return True


# ── MCP installation inside the VM ────────────────────────────────────────────

def install_mcp_in_vm(container) -> bool:

    # ── 1. Directory structure ─────────────────────────────────────────────
    print("\n[1/5] Creating directories …")
    out = vm_shell(
        "mkdir -p /home/user/mcp_server/tools/package "
        "/home/user/mcp_server/tools/apis && echo MKDIR_OK"
    )
    print(f"  {out[:200]}")
    if "MKDIR_OK" not in out:
        print("  WARNING: mkdir may have failed")

    # Empty __init__.py so Python treats tools/ as packages
    for pkg in [
        "/home/user/mcp_server/__init__.py",
        "/home/user/mcp_server/tools/__init__.py",
        "/home/user/mcp_server/tools/package/__init__.py",
    ]:
        vm_write_file(pkg, b"")
    print("  __init__.py files created ✓")

    # ── 2. pip install fastmcp ─────────────────────────────────────────────
    print("\n[2/5] Installing fastmcp …")
    fastmcp_ok = False

    if WHEELS_DIR.exists():
        wheel_files = sorted(WHEELS_DIR.glob("*.whl"))
    else:
        wheel_files = []

    if wheel_files:
        # Extract all wheels on the HOST (wheels are zip files), bundle the
        # extracted Python packages as tar.gz, and inject directly into the VM's
        # user site-packages. No pip needed inside the VM – avoids all pip
        # resolver / network / glob issues.
        import io, tarfile as _tarfile, zipfile as _zipfile, shutil as _shutil

        stage = Path("/tmp/fastmcp_stage")
        if stage.exists():
            _shutil.rmtree(stage)
        stage.mkdir()

        print(f"  Extracting {len(wheel_files)} wheels on host …")
        for wf in wheel_files:
            with _zipfile.ZipFile(wf) as z:
                z.extractall(stage)

        buf = io.BytesIO()
        with _tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(str(stage), arcname=".")
        pkg_bytes = buf.getvalue()
        print(f"  Package bundle: {len(pkg_bytes) // 1024} KB")

        print("  Injecting package bundle into VM …")
        ok = vm_write_file("/tmp/fastmcp_pkg.tar.gz", pkg_bytes)
        if not ok:
            print("  ✗  Failed to inject package bundle")
        else:
            print("  ✓  Bundle injected – extracting to site-packages …")
            site = "/home/user/.local/lib/python3.10/site-packages"
            out = vm_shell(
                f"mkdir -p {site} && "
                f"tar xzf /tmp/fastmcp_pkg.tar.gz -C {site}/ && echo PKG_OK",
                timeout=300,
            )
            print(f"  {out[:300]}")
            if "PKG_OK" in out:
                verify = vm_shell(
                    "python3 -c \"import mcp.types; print('FASTMCP_OK')\" 2>&1",
                    timeout=60,
                )
                print(f"  import verify: {verify[:300]}")
                if "FASTMCP_OK" in verify:
                    print("  fastmcp installed (no-pip) ✓")
                    fastmcp_ok = True
                else:
                    print("  ✗  import check failed after extraction")

    if not fastmcp_ok:
        print("  Trying PyPI directly (fallback) …")
        out = vm_shell(
            "pip3 install --user fastmcp --timeout 180 --retries 3 "
            "--index-url https://pypi.org/simple/ 2>&1 | tail -15",
            timeout=600,
        )
        print(f"  {out[:800]}")
        verify = vm_shell(
            "python3 -c \"import mcp.types; print('FASTMCP_OK')\" 2>&1",
            timeout=15,
        )
        print(f"  import verify: {verify[:200]}")
        if "FASTMCP_OK" in verify:
            print("  fastmcp installed via PyPI ✓")
            fastmcp_ok = True
        else:
            print("  WARNING: fastmcp install may have failed – smoke test will confirm")

    # ── 2b. Install playwright + deps (needed by google_chrome MCP tool) ──
    # playwright is large (~46 MB wheel); inject separately so a failure here
    # does not block the rest.  server.py has try/except for the import.
    print("\n[2b/5] Installing playwright …")
    playwright_wheels = [wf for wf in sorted(WHEELS_DIR.glob("*.whl"))
                         if any(n in wf.name.lower() for n in ("playwright", "pyee", "greenlet"))]
    if not playwright_wheels:
        print("  No playwright wheels found in WHEELS_DIR – skipping.")
    else:
        import io as _io2, tarfile as _tf2, zipfile as _zf2, shutil as _sh2
        pw_stage = Path("/tmp/playwright_stage")
        if pw_stage.exists():
            _sh2.rmtree(pw_stage)
        pw_stage.mkdir()
        for wf in playwright_wheels:
            with _zf2.ZipFile(wf) as z:
                z.extractall(pw_stage)
        buf2 = _io2.BytesIO()
        with _tf2.open(fileobj=buf2, mode="w:gz") as tar:
            tar.add(str(pw_stage), arcname=".")
        pw_bytes = buf2.getvalue()
        print(f"  playwright bundle: {len(pw_bytes) // 1024 // 1024} MB  "
              f"({len(playwright_wheels)} wheels) – injecting …")
        ok2 = vm_write_file("/tmp/playwright_pkg.tar.gz", pw_bytes)
        if ok2:
            site = "/home/user/.local/lib/python3.10/site-packages"
            out2 = vm_shell(
                f"tar xzf /tmp/playwright_pkg.tar.gz -C {site}/ && echo PW_OK",
                timeout=300,
            )
            print(f"  {out2[:200]}")
            if "PW_OK" in out2:
                pw_verify = vm_shell(
                    "python3 -c \"from playwright.async_api import async_playwright; "
                    "print('PW_VERIFY_OK')\" 2>&1",
                    timeout=60,
                )
                print(f"  playwright verify: {pw_verify[:200]}")
                if "PW_VERIFY_OK" not in pw_verify:
                    print("  WARNING: playwright import failed – chrome tools unavailable")
            else:
                print("  ✗  playwright extraction failed")
        else:
            print("  ✗  Failed to inject playwright bundle")

    # ── 3. Inject MCP server files ─────────────────────────────────────────
    print("\n[3/5] Injecting MCP server files …")
    mcp_server = MCP_DIR / "mcp_server"

    files: list[tuple[Path, str]] = [
        (mcp_server / "server.py",        "/home/user/mcp_server/server.py"),
        (mcp_server / "launch_server.sh", "/home/user/mcp_server/launch_server.sh"),
    ]
    for f in (mcp_server / "tools" / "package").glob("*.py"):
        files.append((f, f"/home/user/mcp_server/tools/package/{f.name}"))
    for f in (mcp_server / "tools" / "apis").glob("*.json"):
        files.append((f, f"/home/user/mcp_server/tools/apis/{f.name}"))
    # Simplified MCP client (no Node.js)
    files.append((MCP_DIR / "osworld_mcp_client.py", "/home/user/osworld_mcp_client.py"))

    for local, remote in files:
        ok = vm_write_file(remote, local.read_bytes())
        status = "✓" if ok else "✗"
        print(f"  {status}  {local.name}")

    vm_shell("chmod +x /home/user/mcp_server/launch_server.sh")

    # ── 4. Smoke-test: start MCP server briefly ────────────────────────────
    print("\n[4/5] Smoke-testing MCP server (5 s) …")
    # Quick import check before starting server
    import_check = vm_shell("python3 -c \"import mcp; print('mcp_import_ok')\" 2>&1", timeout=15)
    print(f"  import check: {import_check[:200]}")

    out = vm_shell(
        "cd /home/user/mcp_server && "
        "nohup python3 server.py > /tmp/mcp.log 2>&1 & "
        "MCP_PID=$! && sleep 5 && "
        "curl -sf --max-time 3 http://localhost:9292/mcp | head -c 80 || echo '(no response)' && "
        "kill $MCP_PID 2>/dev/null && echo SERVER_TEST_DONE",
        timeout=30,
    )
    print(f"  {out[:400]}")
    if "SERVER_TEST_DONE" not in out:
        log = vm_shell("cat /tmp/mcp.log 2>/dev/null | tail -20")
        print(f"  server.log:\n{log[:600]}")
        print("  WARNING: smoke test inconclusive – will still save QCOW2")

    # ── 5. Copy overlay out of container while it's still running ─────────
    print("\n[5/5] Copying QCOW2 overlay from container …")
    exit_code, raw = container.exec_run("cp /boot.qcow2 /setup-output/boot-mcp.qcow2")
    if exit_code != 0:
        print(f"  ERROR: docker exec cp failed (rc={exit_code}): {raw.decode()[:300]}")
        return False
    size_mb = (OVERLAY_OUTDIR / "boot-mcp.qcow2").stat().st_size / 1024 / 1024
    print(f"  Overlay saved: {OVERLAY_OUTDIR}/boot-mcp.qcow2  ({size_mb:.0f} MB) ✓")
    return True


def shutdown_vm():
    print("\nShutting down Ubuntu VM …")
    try:
        vm_shell("sudo shutdown -h now 2>/dev/null || poweroff", timeout=15)
    except Exception:
        pass
    print("  Shutdown command sent.")


# ── QCOW2 post-processing on host ──────────────────────────────────────────────

def convert_qcow2() -> bool:
    overlay = OVERLAY_OUTDIR / "boot-mcp.qcow2"
    base    = UBUNTU_QCOW2.resolve()
    output  = OUTPUT_QCOW2.resolve()

    if not overlay.exists():
        print(f"  ERROR: overlay not found at {overlay}")
        return False

    print(f"\n[A] Rebasing overlay → backing file = {base}")
    r = subprocess.run(
        ["qemu-img", "rebase", "-u", "-f", "qcow2",
         "-b", str(base), "-F", "qcow2", str(overlay)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  rebase failed: {r.stderr}")
        return False
    print("  Rebase OK ✓")

    print(f"\n[B] Converting (merging overlay + base) → {output}")
    print("  This may take 5-20 min depending on disk speed …")
    r = subprocess.run(
        ["qemu-img", "convert",
         "-f", "qcow2", "-O", "qcow2",
         "-c",             # enable compression (keeps file size similar to original)
         str(overlay), str(output)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  convert failed: {r.stderr}")
        return False

    size_gb = output.stat().st_size / 1024 ** 3
    print(f"  Convert OK ✓  →  {output}  ({size_gb:.1f} GB)")
    return True


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  OSWorld-MCP – One-time VM Setup")
    print("=" * 62)

    # Pre-flight
    if not UBUNTU_QCOW2.exists():
        sys.exit(f"ERROR: Ubuntu.qcow2 not found at {UBUNTU_QCOW2}")
    r = subprocess.run(["which", "qemu-img"], capture_output=True)
    if r.returncode != 0:
        sys.exit("ERROR: qemu-img not found – install with: apt-get install qemu-utils")

    if OUTPUT_QCOW2.exists():
        ans = input(f"\n{OUTPUT_QCOW2.name} already exists. Overwrite? [y/N] ").strip().lower()
        if ans != "y":
            sys.exit("Aborted.")

    print(f"\n  Base   : {UBUNTU_QCOW2}  ({UBUNTU_QCOW2.stat().st_size / 1024**3:.1f} GB)")
    print(f"  Output : {OUTPUT_QCOW2}")
    print(f"  Overlay: {OVERLAY_OUTDIR}")

    # ── Phase 1: VM setup ──────────────────────────────────────────────────
    print("\n── Starting Docker container ────────────────────────────────")
    container = start_container()

    try:
        print("\n── Waiting for Ubuntu VM ────────────────────────────────────")
        if not wait_for_vm(container, timeout=1800):
            sys.exit("ERROR: VM failed to boot within 15 min")
        time.sleep(10)  # let desktop settle

        print("\n── Installing MCP ───────────────────────────────────────────")
        if not install_mcp_in_vm(container):
            sys.exit("ERROR: MCP installation failed – see messages above")

        shutdown_vm()

        print("\nWaiting for container to stop …", end="", flush=True)
        for _ in range(90):
            container.reload()
            if container.status in ("exited", "dead"):
                break
            print(".", end="", flush=True)
            time.sleep(5)
        print(f" [{container.status}]")

    finally:
        try:
            container.remove(force=True)
            print("Container removed.")
        except Exception:
            pass

    # ── Phase 2: Build standalone QCOW2 ───────────────────────────────────
    print("\n── Building Ubuntu-MCP.qcow2 ────────────────────────────────")
    if not convert_qcow2():
        print("\nConversion failed.  Raw overlay is preserved:")
        print(f"  {OVERLAY_OUTDIR}/boot-mcp.qcow2")
        print("\nRetry manually:")
        print(f"  qemu-img rebase -f qcow2 -b {UBUNTU_QCOW2} -F qcow2 \\")
        print(f"      {OVERLAY_OUTDIR}/boot-mcp.qcow2")
        print(f"  qemu-img convert -f qcow2 -O qcow2 -c \\")
        print(f"      {OVERLAY_OUTDIR}/boot-mcp.qcow2 {OUTPUT_QCOW2}")
        sys.exit(1)

    print("\n" + "=" * 62)
    print("  SUCCESS!  Ubuntu-MCP.qcow2 is ready.")
    print("=" * 62)
    print(f"\n  {OUTPUT_QCOW2}")
    print("""
Next steps
──────────
1. Replace the base QCOW2 (simplest):
     cd ./OSWorld
     cp docker_vm_data/Ubuntu.qcow2 docker_vm_data/Ubuntu-orig.qcow2
     cp docker_vm_data/Ubuntu-MCP.qcow2 docker_vm_data/Ubuntu.qcow2

2. Run experiments with  --action_space mcp
""")


if __name__ == "__main__":
    main()

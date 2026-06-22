"""Stream subprocess stdout line-by-line and surface parsed progress.

Both yt-dlp and Demucs draw in-place progress bars using `\r` carriage returns,
so a normal `for line in stdout` loop only sees the final flushed line. We read
byte-by-byte and split on either `\r` or `\n` to catch every update.
"""
from __future__ import annotations

import re
import subprocess
from typing import Callable, Optional


ProgressCallback = Optional[Callable[[Optional[int], str], None]]


_YTDLP_PCT = re.compile(r"\[download\]\s+(\d+(?:\.\d+)?)%")
_DEMUCS_PCT = re.compile(r"(\d+)%\|")


def parse_ytdlp(line: str) -> Optional[int]:
    m = _YTDLP_PCT.search(line)
    if not m:
        return None
    try:
        return max(0, min(100, int(float(m.group(1)))))
    except (TypeError, ValueError):
        return None


def parse_demucs(line: str) -> Optional[int]:
    m = _DEMUCS_PCT.search(line)
    if not m:
        return None
    try:
        return max(0, min(100, int(m.group(1))))
    except (TypeError, ValueError):
        return None


def stream_subprocess(
    cmd: list[str],
    on_line: Callable[[str], None],
    *,
    timeout: Optional[float] = None,
) -> subprocess.CompletedProcess:
    """Run `cmd` and call `on_line(text)` for each line OR CR-terminated chunk.

    Returns a CompletedProcess with stdout/stderr fields populated (combined into
    stdout because we redirect stderr→stdout to interleave progress bars in order).
    Raises subprocess.TimeoutExpired on timeout, FileNotFoundError if the binary
    is missing.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        text=False,  # read bytes so we can split on \r without TextIOWrapper buffering
    )

    captured: list[str] = []
    buf = bytearray()

    assert proc.stdout is not None
    try:
        while True:
            chunk = proc.stdout.read(1)
            if not chunk:
                break
            if chunk in (b"\r", b"\n"):
                if buf:
                    line = buf.decode("utf-8", errors="replace")
                    captured.append(line)
                    try:
                        on_line(line)
                    except Exception:  # noqa: BLE001
                        # never let a progress callback kill the subprocess loop
                        pass
                    buf.clear()
            else:
                buf.extend(chunk)
        if buf:
            line = buf.decode("utf-8", errors="replace")
            captured.append(line)
            try:
                on_line(line)
            except Exception:  # noqa: BLE001
                pass
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise

    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout="\n".join(captured),
        stderr="",
    )

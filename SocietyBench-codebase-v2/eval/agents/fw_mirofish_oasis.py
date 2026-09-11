"""MiroFish — social-simulation predictor based on open-source OASIS (route B).

camel-oasis needs Python 3.10/3.11 while the system is 3.12, so this framework runs
mirofish_sim.py as a **subprocess** inside a separate OASIS venv
(default ~/societybench_oasis_venv311) and retrieves the answer text.

Public API: answer(base_model, prompt, kind="brier") -> str (same format as bare models)

Environment variables (tunable):
  OASIS_VENV_PY    python of the OASIS venv (default ~/societybench_oasis_venv311/bin/python)
  MIROFISH_AGENTS  number of simulated netizens (default 8; smoke tests may use 4)
  MIROFISH_STEPS   number of simulation interaction steps (default 3; smoke tests may use 2)
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import queue
import re
import subprocess
import sys
import tempfile
import threading

# Multiple question batches for the same prediction point (same context) share one social
# simulation: cache the transcript so each batch doesn't rerun the simulation.
# (Only used by the legacy one-shot answer() path; for continuous rollout sessions see _MiroFishSession.)
_TRANSCRIPT_CACHE: dict = {}

_HERE = pathlib.Path(__file__).resolve().parent
_EVAL = _HERE.parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

import adapter  # noqa: E402  (process-level call counter STATS)

OASIS_PY = os.environ.get(
    "OASIS_VENV_PY",
    str(pathlib.Path.home() / "societybench_oasis_venv311" / "bin" / "python"),
)
_SIM = _HERE / "mirofish_sim.py"
_N_AGENTS = os.environ.get("MIROFISH_AGENTS", "8")
_N_STEPS = os.environ.get("MIROFISH_STEPS", "3")


def _extract_context(prompt: str) -> str:
    """Extract the background section from a 3B/3F prompt (used as the event brief for the
    social simulation). Recognizes both headers (Chinese '已知信息' / English 'Known information'),
    otherwise falls back to truncation."""
    head = r"(?:已知信息|Known information)"
    m = re.search(head + r"[^\n]*={2,}\n(.*?)\n={3,}", prompt, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(head + r"[^\n]*\n(.*?)\n={3,}", prompt, re.S)
    if m:
        return m.group(1).strip()
    return prompt[:4000]


def answer(
    base_model: str,
    prompt: str,
    kind: str = "brier",
    max_tokens: int = 32000,
    thinking_budget: int = 24000,
) -> str:
    if not _SIM.exists():
        raise SystemExit(f"找不到 mirofish_sim.py: {_SIM}")
    if not pathlib.Path(OASIS_PY).exists():
        raise SystemExit(
            f"OASIS venv python 不存在: {OASIS_PY}。请先 `uv venv --python 3.11 "
            "~/societybench_oasis_venv311 && uv pip install --python <它> camel-oasis`。"
        )
    ctx = _extract_context(prompt)
    key = (base_model, hashlib.sha1(ctx.encode("utf-8")).hexdigest())
    cached = _TRANSCRIPT_CACHE.get(key)

    job = {
        "base_model": base_model, "prompt": prompt, "kind": kind, "context": ctx,
        "n_agents": int(_N_AGENTS), "n_steps": int(_N_STEPS),
        "max_tokens": int(max_tokens), "thinking_budget": int(thinking_budget),
    }
    if cached is not None:
        job["transcript"] = cached  # reuse this point's already-run social simulation, only rerun the reader

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False)
        jobf = f.name
    try:
        r = subprocess.run([OASIS_PY, str(_SIM), jobf],
                           capture_output=True, text=True, timeout=1800, env=dict(os.environ))
    finally:
        try:
            os.unlink(jobf)
        except OSError:
            pass

    m = re.search(r"<<<TRANSCRIPT>>>\n(.*?)\n<<<ANSWER>>>\n(.*?)\n<<<END>>>", r.stdout, re.S)
    if m:
        if cached is None:
            try:
                _TRANSCRIPT_CACHE[key] = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        return m.group(2)
    raise RuntimeError(f"mirofish_sim 无 answer 输出。\nstderr 尾部:\n{r.stderr[-800:]}")


# ===========================================================================
# Continuous rollout session (M3 single-rollout refactor): resident OASIS subprocess,
# build once, step along the timeline, accumulate reads across spans.
# ===========================================================================
# The subprocess runs mirofish_sim.py --resident; advance/answer use a stdin/stdout JSON
# command protocol. Key point: the subprocess routes all library prints to stderr; stdout
# carries only <<<SBRESP>>>..<<<SBEND>>> protocol frames.
# Only 1 simulation is built per event (vs. legacy rebuilding per point -> MiroFish request
# volume drops by an order of magnitude).

_SB_RESP_BEGIN = "<<<SBRESP>>>"
_SB_RESP_END = "<<<SBEND>>>"


class _MiroFishSession:
    def __init__(self, base_model: str, *, max_tokens: int, thinking_budget: int, sim_params=None) -> None:
        if not _SIM.exists():
            raise SystemExit(f"找不到 mirofish_sim.py: {_SIM}")
        if not pathlib.Path(OASIS_PY).exists():
            raise SystemExit(
                f"OASIS venv python 不存在: {OASIS_PY}。请先 `uv venv --python 3.11 "
                "~/societybench_oasis_venv311 && uv pip install --python <它> camel-oasis`。"
            )
        self.base = base_model
        self.max_tokens = int(max_tokens)
        self.thinking_budget = int(thinking_budget)
        sp = sim_params or {}
        self.n_agents = int(sp.get("n_agents", _N_AGENTS))
        self.n_steps = int(sp.get("n_steps", _N_STEPS))
        self.cutoff = ""
        self.n_advance = self.n_answer = self.n_sim = self.n_llm = 0
        self._lock = threading.Lock()
        self._timeout = float(os.environ.get("MIROFISH_CMD_TIMEOUT", "1800"))
        # Route subprocess stderr to a temp file so OASIS logs can't fill the PIPE buffer and
        # block the subprocess's writes (classic deadlock); read the file tail for diagnostics
        # on failure. stdout carries protocol frames only.
        self._errf = tempfile.NamedTemporaryFile("w+", suffix=".mirofish.err",
                                                 delete=False, encoding="utf-8")
        self.proc = subprocess.Popen(
            [OASIS_PY, str(_SIM), "--resident"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._errf,
            text=True, bufsize=1, env=dict(os.environ),
        )
        # A dedicated reader thread parses stdout frames and puts them on a queue; _recv takes
        # from the queue (with timeout). Do not mix select+readline (mixing loses lines already
        # in Python's buffer: when a whole frame is written at once, readline takes the first
        # line, the rest go to the buffer, and select sees no new data on the fd and waits
        # until timeout — this was the root cause of the earlier hang).
        self._q: "queue.Queue" = queue.Queue()
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

    def _reader_loop(self) -> None:
        capturing, lines = False, []
        try:
            while True:
                line = self.proc.stdout.readline()  # blocking line read; returns as soon as each \n arrives, no readahead delay
                if line == "":  # EOF: subprocess exited
                    break
                s = line.rstrip("\n")
                if s == _SB_RESP_BEGIN:
                    capturing, lines = True, []
                elif s == _SB_RESP_END:
                    if capturing:
                        capturing = False
                        try:
                            self._q.put(("frame", json.loads("\n".join(lines))))
                        except Exception as e:  # noqa: BLE001
                            self._q.put(("error", f"bad frame json: {e}"))
                elif capturing:
                    lines.append(s)
                # Stray lines outside frames (shouldn't happen; stdout carries frames only) are ignored
        except Exception as e:  # noqa: BLE001
            self._q.put(("error", str(e)))
        finally:
            self._q.put(("eof", None))

    def _send(self, obj: dict) -> None:
        if self.proc.poll() is not None:
            raise RuntimeError("mirofish resident subprocess already exited")
        self.proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def _err_tail(self) -> str:
        try:
            self._errf.flush()
            return pathlib.Path(self._errf.name).read_text(encoding="utf-8")[-1000:]
        except Exception:  # noqa: BLE001
            return ""

    def _recv(self) -> dict:
        # Take one frame from the reader-thread queue; whole-command timeout (a single advance/answer may run multiple LLM calls with minutes of no stdout).
        try:
            kind, payload = self._q.get(timeout=self._timeout)
        except queue.Empty:
            raise RuntimeError(
                f"mirofish resident _recv 超时({self._timeout}s)。stderr 尾部:\n{self._err_tail()}")
        if kind == "frame":
            return payload
        if kind == "eof":
            raise RuntimeError(f"mirofish resident died.\nstderr 尾部:\n{self._err_tail()}")
        raise RuntimeError(
            f"mirofish resident reader error: {payload}\nstderr 尾部:\n{self._err_tail()}")

    def advance(self, delta_context: str, cutoff: str) -> None:
        self.cutoff = cutoff or self.cutoff
        delta = (delta_context or "").strip()
        if not delta:
            return
        self.n_advance += 1
        adapter.STATS.bump("advance")
        msg = {"cmd": "ADVANCE", "delta": delta[:8000], "cutoff": cutoff, "steps": self.n_steps,
               "base_model": self.base, "n_agents": self.n_agents, "n_steps": self.n_steps,
               "max_tokens": self.max_tokens, "thinking_budget": self.thinking_budget}
        with self._lock:
            self._send(msg)
            resp = self._recv()
        if resp.get("event") == "built_and_advanced":
            self.n_sim += 1  # the only build for the whole event (vs. legacy one build per point)
            adapter.STATS.bump("sim")
        if not resp.get("ok"):
            # v7.1 (2026-07-03): rollout errors are no longer swallowed as warnings — raise to
            # run_agents' persistent retry (otherwise failures like "empty-scene rollout" would
            # be silently waved through into answering).
            print(f"[mirofish] advance failed: {resp.get('error')}", file=sys.stderr)
            raise RuntimeError(f"mirofish advance failed: {resp.get('error')}")
        # Rollout/answer decoupling (2026-07-05): after each point's rollout succeeds, auto-save an opinion snapshot; the answering stage can read it offline and rerun with different params.
        _snap_dir = os.environ.get("MIROFISH_SNAPSHOT_DIR", "").strip()
        if _snap_dir and cutoff:
            try:
                os.makedirs(_snap_dir, exist_ok=True)
                tr = self.snapshot(limit=int(os.environ.get("MIROFISH_SNAPSHOT_LIMIT", "3000")))
                safe = cutoff.replace("/", "-").replace(" ", "_")
                with open(os.path.join(_snap_dir, f"{safe}.json"), "w", encoding="utf-8") as f:
                    json.dump({"cutoff": cutoff, "n": len(tr), "transcript": tr}, f, ensure_ascii=False)
                print(f"[mirofish] snapshot saved: {cutoff} ({len(tr)} items)", file=sys.stderr)
            except Exception as _se:  # noqa: BLE001  snapshot failure is non-fatal (does not affect the main rollout/answer flow)
                print(f"[mirofish] snapshot warn: {_se}", file=sys.stderr)

    def answer(self, point_prompt: str, kind: str = "brier") -> str:
        self.n_answer += 1
        adapter.STATS.bump("answer")
        with self._lock:
            self._send({"cmd": "ANSWER", "prompt": point_prompt, "kind": kind})
            resp = self._recv()
        self.n_llm += 1
        adapter.STATS.bump("llm")
        if resp.get("ok"):
            return resp.get("answer", "") or ""
        raise RuntimeError(f"mirofish answer failed: {resp.get('error')}")

    def snapshot(self, limit: int = 3000) -> list:
        """Dump the full opinion transcript up to the current rollout point (rollout/answer decoupling: phase 1 saves per point, phase 2 answers offline)."""
        with self._lock:
            self._send({"cmd": "SNAPSHOT", "limit": int(limit)})
            resp = self._recv()
        if resp.get("ok"):
            return resp.get("transcript", []) or []
        raise RuntimeError(f"mirofish snapshot failed: {resp.get('error')}")

    def close(self) -> None:
        try:
            with self._lock:
                if self.proc.poll() is None:
                    self._send({"cmd": "CLOSE"})
                    try:
                        self._recv()
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001
            pass
        for closer in (lambda: self.proc.stdin.close(),):
            try:
                closer()
            except Exception:  # noqa: BLE001
                pass
        try:
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            try:
                self.proc.kill()
            except Exception:  # noqa: BLE001
                pass
        for fclose in (lambda: self._errf.close(),
                       lambda: os.unlink(self._errf.name)):
            try:
                fclose()
            except Exception:  # noqa: BLE001
                pass

    def stats(self) -> dict:
        return {"advance": self.n_advance, "answer": self.n_answer, "llm": self.n_llm, "sim": self.n_sim}


def make_session(
    base_model: str,
    *,
    max_tokens: int = 32000,
    thinking_budget: int = 24000,
    sim_params=None,
) -> _MiroFishSession:
    return _MiroFishSession(base_model, max_tokens=max_tokens,
                            thinking_budget=thinking_budget, sim_params=sim_params)

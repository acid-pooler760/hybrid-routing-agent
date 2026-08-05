

import os
import re
import json
import hashlib
import argparse
from termcolor import cprint
from typing import Any, Dict, List, Tuple, Optional, Set
import math
from collections import OrderedDict, defaultdict

# Token length calculator (using tiktoken for OpenAI-style tokenizers, fallback to char-based estimate)
_TOKENIZER = None
def get_token_length(text: str) -> int:
    """Calculate token length for response text. Falls back to char estimation if tokenizer unavailable."""
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            import tiktoken
            _TOKENIZER = tiktoken.get_encoding("cl100k_base")  # OpenAI's encoding, rough approximation for Qwen
        except Exception:
            _TOKENIZER = "char"  # Fallback: 1 char ≈ 0.33 tokens

    if _TOKENIZER == "char":
        # Rough estimate: 1 token ≈ 3-4 chars for English, Qwen may be similar
        return max(1, len(text) // 3)
    else:
        try:
            return len(_TOKENIZER.encode(text))
        except Exception:
            return max(1, len(text) // 3)


def is_dir(p: str) -> bool:
    return os.path.isdir(p)

def list_subdirs(p: str) -> List[str]:
    try:
        return [d for d in sorted(os.listdir(p)) if is_dir(os.path.join(p, d))]
    except FileNotFoundError:
        return []

def list_numeric_dirs(p: str) -> List[str]:

    try:
        subs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
    except FileNotFoundError:
        return []
    if subs and all(s.isdigit() for s in subs):
        return sorted(subs, key=lambda x: int(x))
    return sorted(subs)



def _response_to_str(resp: Any) -> str:
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        for k in ("text", "response", "output", "content"):
            v = resp.get(k)
            if isinstance(v, str):
                return v
        chs = resp.get("choices")
        if isinstance(chs, list) and chs:
            ch = chs[0]
            if isinstance(ch, dict):
                msg = ch.get("message")
                if isinstance(msg, dict):
                    c = msg.get("content")
                    if isinstance(c, str):
                        return c
                txt = ch.get("text")
                if isinstance(txt, str):
                    return txt
    if isinstance(resp, list):
        texts = []
        for it in resp:
            if isinstance(it, str):
                texts.append(it)
            elif isinstance(it, dict):
                t = it.get("text") or it.get("content")
                if isinstance(t, str):
                    texts.append(t)
        if texts:
            return "\n".join(texts)
    try:
        return json.dumps(resp, ensure_ascii=False)
    except Exception:
        return str(resp)



def _normalize_one_messages(ms: Any) -> List[Dict[str, Any]]:
    if isinstance(ms, list) and all(isinstance(x, dict) and ("role" in x) for x in ms):
        return ms
    if isinstance(ms, dict) and "role" in ms:
        return [ms]
    return []

def normalize_message_and_response_from_trajectory(traj: Any) -> Tuple[List[List[Dict[str, Any]]], List[str]]:
    messages_list: List[List[Dict[str, Any]]] = []
    responses: List[str] = []

    def _push(ms_any: Any, resp_any: Any):
        ms = _normalize_one_messages(ms_any)
        if not ms:
            return
        messages_list.append(ms)
        responses.append(_response_to_str(resp_any))

    if isinstance(traj, list):
        for step in traj:
            if not isinstance(step, dict):
                continue
            ms_any = step.get("messages") or step.get("message") or step.get("input") or step.get("prompt")
            resp_any = step.get("response") or step.get("output") or step.get("completion")
            _push(ms_any, resp_any)
    elif isinstance(traj, dict):
        ms_any = traj.get("messages") or traj.get("message") or traj.get("input") or traj.get("prompt")
        resp_any = traj.get("response") or traj.get("output") or traj.get("completion")
        _push(ms_any, resp_any)

    return messages_list, responses

def sort_traj_list_inplace(traj_list: List[Dict[str, Any]]):

    def _key(x):
        v = x.get("run_id")
        if isinstance(v, int):
            return (0, v)
        try:
            return (0, int(v))
        except Exception:
            return (1, str(v))
    traj_list.sort(key=_key)



def collect_messages_with_responses(root_dir: str, output_json: str) -> Tuple[int, int, int]:
    domains = list_subdirs(root_dir)
    tasks: List[Tuple[str, str, str, str]] = []  # (domain, example, run, traj_json)

    for domain in domains:
        domain_dir = os.path.join(root_dir, domain)
        examples = list_subdirs(domain_dir)
        for ex in examples:
            ex_dir = os.path.join(domain_dir, ex)
            runs = list_numeric_dirs(ex_dir)
            for run in runs:
                traj_json = os.path.join(ex_dir, run, "trajectory.json")
                if os.path.isfile(traj_json):
                    tasks.append((domain, ex, run, traj_json))

    out_index: Dict[Tuple[str, str], int] = {}
    result: List[Dict[str, Any]] = []
    for domain, ex, _run, _ in tasks:
        key = (domain, ex)
        if key not in out_index:
            out_index[key] = len(result)
            result.append({"domain": domain, "example": ex, "trajctory_list": []})

    for domain, ex, run, traj_json in tasks:
        try:
            with open(traj_json, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] failed to open, skip: {traj_json} | reason: {e}")
            continue

        meta_result = (data.get("meta") or {}).get("result")
        traj = data.get("trajectory", [])
        messages_list, responses = normalize_message_and_response_from_trajectory(traj)

        idx = out_index[(domain, ex)]
        run_id = int(run) if str(run).isdigit() else run
        action_types = data.get("action_types", [])
        exec_oks = data.get("exec_oks", [])
        tools_available = data.get("tools_available", [])

        # Empty trajectory still represents one task attempt; emit stub so
        # downstream acc aggregation counts it instead of silently dropping
        # collapsed-model tasks from both numerator and denominator.
        result[idx]["trajctory_list"].append({
            "run_id": run_id,
            "message": messages_list,
            "response": responses,
            "result": meta_result,
            "action_types": action_types,
            "exec_oks": exec_oks,
            "tools_available": tools_available,
        })

    for item in result:
        sort_traj_list_inplace(item["trajctory_list"])

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[DONE] messages+responses write: {output_json}")
    return len(domains), len(out_index), len(tasks)


# ----------------- aggregate reward slide -----------------
def _load_json_list(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return []
    if isinstance(obj, list):
        return obj
    elif isinstance(obj, dict):
        return [obj]
    else:
        return []

def merge_reward_shards(merge_dir: str, num_nodes: int, out_path: str) -> int:
    shards = []
    for i in range(num_nodes):
        p = os.path.join(merge_dir, f"reward_rollout_results_node_{i}.json")
        if not os.path.isfile(p):
            print(f"[INFO] don't exists, skip: {p}")
            continue
        arr = _load_json_list(p)
        print(f"[INFO] read {i}: {p}, items={len(arr)}")
        shards.extend(arr)

    if not shards:
        print("[WARN] dont find any reward shard to merge.")
        return 0

    merged = OrderedDict()
    for item in shards:
        try:
            d = item["domain"]; e = item["example"]
            tl = item.get("trajctory_list", [])
        except Exception:
            continue
        key = (d, e)
        if key not in merged:
            merged[key] = {"domain": d, "example": e, "trajctory_list": []}
        if isinstance(tl, list):
            merged[key]["trajctory_list"].extend(tl)

    for _, ex_item in merged.items():
        sort_traj_list_inplace(ex_item["trajctory_list"])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(list(merged.values()), f, ensure_ascii=False, indent=2)

    print(f"[DONE] reward merge completed {out_path} | items={len(merged)}")
    return len(merged)



def safe_outcome_reward(result_val: Any) -> int:

    try:
        return 1 if float(result_val) > 0 else -1
    except Exception:
        return -1


_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def parse_terminate(last_resp: str) -> Optional[str]:
    """Detect terminate tool_call and return its claimed status.

    Returns:
        None     — no terminate in last <tool_call> block (or no tool_call at all)
        "success" — terminate claimed success
        "failure" — terminate claimed failure (or any non-success status)
                    (model said "I give up" or similar)

    Distinguishes voluntary termination from cap-induced FAIL (which
    lib_run_single force-appends without a model tool_call). Key signal: the
    last <tool_call> block contains `"action": "terminate"`. The status is
    extracted from the same block via `"status": "..."`. Defaults to "failure"
    if terminate present but no parseable status (conservative; treats malformed
    as "fail" so no false-success-pen is wrongly applied)."""
    if not last_resp:
        return None
    blocks = _TOOL_CALL_RE.findall(last_resp)
    if not blocks:
        return None
    last_block = blocks[-1].lower()
    is_term = bool(re.search(r'"action"\s*:\s*"terminate"', last_block)) or \
              bool(re.search(r'\bterminate\s*\(', last_block))
    if not is_term:
        return None
    # Parse status: "success" wins, anything else (failure/empty/missing) → "failure"
    m = re.search(r'"status"\s*:\s*"([^"]+)"', last_block)
    if m and m.group(1).strip().lower() == "success":
        return "success"
    return "failure"


def is_voluntary_terminate(last_resp: str) -> bool:
    """Back-compat: True iff model emitted any terminate tool_call (success or failure)."""
    return parse_terminate(last_resp) is not None


def last_image_hash(msg) -> str:
    """Cheap hash of the last image part embedded in a prompt-messages payload.

    Returns "" when no image part is found (e.g. message is a plain string).
    The screenshot is base64 in the message dict, so hashing the string itself
    is sufficient and avoids decoding."""
    if not isinstance(msg, list):
        return ""
    last_img = None
    for m in msg:
        content = m.get("content") if isinstance(m, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image":
                img = part.get("image")
                if isinstance(img, str) and img:
                    last_img = img
            elif part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url")
                if isinstance(url, str) and url:
                    last_img = url
    if last_img is None:
        return ""
    return hashlib.sha1(last_img.encode("utf-8", errors="ignore")).hexdigest()


def action_key_from_response(resp: str) -> str:
    """Extract a stable action identity from a step's response for repeat detection.

    owl_parser executes the LAST <tool_call> block, so we key on that block's
    content (whitespace-normalized). This ignores the per-step reasoning text
    (which varies every step even when the executed action repeats) — the reason
    the old `response[k] == response[k-1]` check almost never fired. Returns ""
    when no tool_call is present (don't penalize)."""
    if not resp:
        return ""
    m = _TOOL_CALL_RE.findall(resp)
    if not m:
        return ""
    return " ".join(m[-1].split())


# ── Hybrid routing: action type bonus ────────────────────────────────────────

def compute_action_type_bonus(
    action_type: Optional[str],
    exec_ok: bool,
) -> float:
    """Return action type bonus: MCP success +1, GUI 0, fail -1."""
    if action_type is None or not exec_ok:
        return -1.0
    if action_type == "mcp":
        return 1.0
    return 0.0


# ── Format reward (A3) + safe MCP tool-use bonus (hybrid-action specific) ─────
# Read-only tool-name hints: substrings marking a tool as non-effectful (inspects
# state without changing it). Used to DENY the MCP tool-use bonus to read-only
# calls so it can't be farmed by spamming list/get/read. Over-flagging is safe
# (just fewer bonuses); under-flagging is the risk we guard against.
_READONLY_TOOL_HINTS = (
    "list", "get", "read", "query", "find", "search", "describe",
    "show", "view", "fetch", "info", "status", "count", "exists",
)
_TOOLS_BLOCK_RE = re.compile(r"<tools>(.*?)</tools>", re.DOTALL)
_JSON_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')


def _prompt_text(prompt_messages: Any) -> str:
    """Flatten a chat messages list into plain text (system+user text parts)."""
    out: List[str] = []
    try:
        for m in prompt_messages or []:
            c = m.get("content")
            if isinstance(c, str):
                out.append(c)
            elif isinstance(c, list):
                for blk in c:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        out.append(blk.get("text", ""))
    except Exception:
        return ""
    return "\n".join(out)


def injected_tool_names(prompt_messages: Any) -> Set[str]:
    """Tool names from the <tools> block of the step prompt. Empty set when the
    block can't be found → caller should SKIP the name-validity check (never
    penalize on missing data). computer_use is always present in <tools>, so
    legit GUI actions are always in this set."""
    mb = _TOOLS_BLOCK_RE.search(_prompt_text(prompt_messages))
    if not mb:
        return set()
    return set(_JSON_NAME_RE.findall(mb.group(1)))


def tool_name_from_response(resp: str) -> str:
    """Tool name from the LAST <tool_call> block; '' if absent/unparseable."""
    if not resp:
        return ""
    blocks = _TOOL_CALL_RE.findall(resp)
    if not blocks:
        return ""
    try:
        obj = json.loads(blocks[-1].strip())
        return str(obj.get("name", "")).strip()
    except Exception:
        return ""


def is_read_only_tool(name: str) -> bool:
    if not name:
        return False
    leaf = name.lower().rsplit(".", 1)[-1]   # strip namespace: app.get_x → get_x
    return any(h in leaf for h in _READONLY_TOOL_HINTS)


def format_violation(resp: str, valid_names: Set[str]) -> bool:
    """A3 format check (True = violation → penalize). Trips when the step output
    is NOT exactly one well-formed tool_call with valid {name, arguments} JSON,
    or names a tool not in valid_names (only when valid_names non-empty — this is
    what catches a hallucinated name like 'libreoffice_impress.left_click')."""
    blocks = _TOOL_CALL_RE.findall(resp or "")
    if len(blocks) != 1:
        return True
    try:
        obj = json.loads(blocks[0].strip())
    except Exception:
        return True
    if not (isinstance(obj, dict) and "name" in obj and "arguments" in obj):
        return True
    if valid_names:
        return str(obj.get("name", "")).strip() not in valid_names
    return False


def success01(result_val: Any) -> int:
    try:
        return 1 if float(result_val) > 0 else 0
    except Exception:
        return 0

def compute_acc_map_from_rollout(rollout_json: str) -> Dict[Tuple[str, str], float]:

    with open(rollout_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: Dict[Tuple[str, str], float] = {}
    for item in data:
        d = item.get("domain"); e = item.get("example")
        if d is None or e is None:
            continue
        tl = item.get("trajctory_list", []) or []
        flags = [success01(t.get("result")) for t in tl]
        out[(d, e)] = (sum(flags) / len(flags)) if flags else 0.0
    return out

def build_id2domain_from_rollout(rollout_json: str) -> Dict[str, str]:
    with open(rollout_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    m = {}
    for item in data:
        d = item.get("domain"); e = item.get("example")
        if isinstance(d, str) and isinstance(e, str):
            m[e] = d
    return m



def list_example_node_files(manifest_dir: str) -> List[str]:
    if not os.path.isdir(manifest_dir):
        return []
    out = []
    for fn in sorted(os.listdir(manifest_dir)):
        if re.fullmatch(r"example_node_\d+\.json", fn):
            out.append(os.path.join(manifest_dir, fn))
    return out

def normalize_manifest_obj(manifest_obj: Any, id2domain: Dict[str, str]) -> Dict[str, List[Dict[str, str]]]:

    def _norm_list(lst: Any) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        if not isinstance(lst, list):
            return out
        for x in lst:
            if isinstance(x, dict) and ("domain" in x) and ("example_id" in x):
                out.append({"domain": str(x["domain"]), "example_id": str(x["example_id"])})
            elif isinstance(x, str):
                d = id2domain.get(x)
                if d is not None:
                    out.append({"domain": d, "example_id": x})
        return out

    if isinstance(manifest_obj, list):
        return {"active": _norm_list(manifest_obj), "temp": []}

    if isinstance(manifest_obj, dict):
        return {
            "active": _norm_list(manifest_obj.get("active", [])),
            "temp": _norm_list(manifest_obj.get("temp", [])),
        }

    return {"active": [], "temp": []}

def aggregate_active_temp_sets(manifest_dir: str, id2domain: Dict[str, str]) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    node_files = list_example_node_files(manifest_dir)
    active_set: Set[Tuple[str, str]] = set()
    temp_set: Set[Tuple[str, str]] = set()

    for p in node_files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue
        norm = normalize_manifest_obj(obj, id2domain)
        for it in (norm.get("active") or []):
            active_set.add((it["domain"], it["example_id"]))
        for it in (norm.get("temp") or []):
            temp_set.add((it["domain"], it["example_id"]))

    return active_set, temp_set



def _stepwise_standardize_inplace(records: List[Dict[str, Any]], require_multi_traj: bool = True) -> None:
    """Original per-step-index z-score. Preserves per-step shaping variance but
    introduces a structural bias: late step-indices are populated only by
    never-terminate (cap-hitting) trajectories, so non-DONE actions at those
    indices receive systematically positive advantage → suppresses terminate."""
    by_step = defaultdict(list)
    for i, rec in enumerate(records):
        by_step[int(rec["step"])].append(i)

    for step_k, idxs in by_step.items():
        run_ids = {records[i]["run_id"] for i in idxs}
        vals = [float(records[i]["raw_reward"]) for i in idxs]

        if require_multi_traj and len(run_ids) <= 1:
            for i in idxs:
                records[i]["reward"] = float(records[i]["raw_reward"])
            continue

        if len(vals) == 1:
            i = idxs[0]
            records[i]["reward"] = float(records[i]["raw_reward"])
            continue

        mu = sum(vals) / len(vals)
        var = sum((x - mu) ** 2 for x in vals) / len(vals)
        std = math.sqrt(var)

        # Degenerate group (all rollouts agree at this step-index): fall back to
        # raw - mu instead of writing 0.0. Most members will still be ~0, but any
        # surviving cross-trajectory shaping signal (length_pen/cap_pen differing
        # between trajectories) is preserved instead of being wiped. The downstream
        # |r| < eps filter (not r == 0) then drops only the genuinely flat samples.
        if std < 1e-8:
            for i in idxs:
                records[i]["reward"] = float(records[i]["raw_reward"]) - mu
            continue

        for i in idxs:
            x = float(records[i]["raw_reward"])
            records[i]["reward"] = (x - mu) / std


def _groupwise_standardize_inplace(records: List[Dict[str, Any]], require_multi_traj: bool = True) -> None:
    """Standard GRPO: trajectory-level return (mean raw_reward across steps),
    per-group (same task) z-score, broadcast back to every step token.
    Eliminates the late-step bias of _stepwise_standardize_inplace because
    the advantage scalar is computed before any step-index split.

    Per-token normalization: the broadcasted advantage is divided by trajectory
    length T_i so that each trajectory contributes equal total gradient regardless
    of length. Without this, a 30-step trajectory contributes 30/7 ≈ 4× more
    gradient than a 7-step trajectory with the same advantage scalar, giving
    never-terminate (long) trajectories disproportionate influence."""
    # group by run_id within this task's records
    by_run: Dict[str, List[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        by_run[str(rec["run_id"])].append(i)

    if require_multi_traj and len(by_run) <= 1:
        for rec in records:
            rec["reward"] = float(rec["raw_reward"])
        return

    # trajectory-level return = mean raw_reward across steps
    run_ids = list(by_run.keys())
    returns = []
    for rid in run_ids:
        idxs = by_run[rid]
        ret = sum(float(records[i]["raw_reward"]) for i in idxs) / len(idxs)
        returns.append(ret)

    mu = sum(returns) / len(returns)
    var = sum((r - mu) ** 2 for r in returns) / len(returns)
    std = math.sqrt(var)

    for rid, ret in zip(run_ids, returns):
        if std < 1e-8:
            adv = ret - mu
        else:
            adv = (ret - mu) / std
        T_i = max(len(by_run[rid]), 1)
        for i in by_run[rid]:
            records[i]["reward"] = adv / T_i



# ----------------- generate RL training data -----------------
def generate_rl_datasets(
    rollout_json: str,
    reward_json: str,
    out_dir: str,
    acc_lower: float = 0.125,
    acc_upper: float = 0.875,
    normalize: bool = True,
    policy_out_name: str = "policy_optimization_data.json",
    allow_set: Optional[Set[Tuple[str, str]]] = None,
    lambda_coef: float = 0.3,
    eta_initial: float = 0.02,
    eta_decay_steps: int = 30,
    rl_step: int = 0,
    positive_advantage_only: bool = False,
    length_penalty_coef: float = 0.0,
    max_steps_ref: int = 30,
    step_fail_pen: float = 0.0,
    repeat_pen: float = 0.0,
    cap_penalty: float = 0.0,
    # 5-case terminate classifier — trajectory-level shaping. Combines model's
    # claimed terminate status (success/failure parsed from tool_call) with the
    # true outcome from evaluator, plus T_pol bucketing:
    #   A. terminate + outcome=1 + T <  cap*efficient_ratio  → +terminate_bonus_efficient
    #   B. terminate + outcome=1 + T >= cap*efficient_ratio  → +terminate_bonus_slow
    #   C. terminate(status=failure) + outcome=0 + T < cap*early_quit_ratio → -early_quit_pen (gave up too soon)
    #   D. terminate(status=failure) + outcome=0 + T >= cap*early_quit_ratio → +graceful_fail_bonus (legit give-up)
    #   E. terminate(status=success) + outcome=0  → -false_success_pen  (REWARD HACKING: lied about success)
    # Case E directly counters the e17 failure mode where the model emits
    # terminate(status=success) on step 3 with no real progress to grab the
    # efficient bonus. Status is parsed from the LAST <tool_call> block.
    # Cases C/D only fire when the model itself said status=failure (it knew it
    # gave up). Cap-induced FAIL → no terminate → no signal at all.
    # All 0 by default → opt-in shaping. Kept independent from cap_penalty (which fires
    # on hit_cap regardless of terminate). Legacy `terminate_bonus` arg is supported as
    # a shorthand for setting both efficient & slow to the same value.
    #
    # term_amortize_mode controls how the trajectory-level signal hits each step:
    #   "per_step"  (default): every step gets the FULL signal, so per-step-index
    #                          GRPO z-score sees a O(0.3) cross-traj delta on top of
    #                          O(1) outcome — keeps the signal alive after z-score.
    #   "amortized":           signal / T_pol per step (legacy behavior). Smaller
    #                          per-step delta — easily drowned by outcome std.
    terminate_bonus_efficient: float = 0.0,
    terminate_bonus_slow:      float = 0.0,
    early_quit_pen:            float = 0.0,
    graceful_fail_bonus:       float = 0.0,
    false_success_pen:         float = 0.0,
    efficient_ratio:           float = 0.7,
    early_quit_ratio:          float = 0.5,
    term_amortize_mode:        str   = "per_step",
    # Legacy: if set, overrides both terminate_bonus_efficient and terminate_bonus_slow.
    terminate_bonus:           float = 0.0,
    # Per-step penalty when the next-step screenshot equals the current screenshot
    # (cheap proxy for "I did something but the screen didn't change"). Hashes the
    # last image part of pol_msg_list[k+1] vs pol_msg_list[k]. Off by default.
    no_progress_pen: float = 0.0,
    # ── Format reward (A3) + safe MCP tool-use bonus — hybrid-action specific ──
    # All default 0/off → existing experiments byte-identical. See module helpers
    # format_violation / is_read_only_tool / injected_tool_names.
    #   format_pen: per-step penalty when output is not exactly one well-formed
    #     tool_call with valid {name, arguments} JSON, or names a tool absent from
    #     the injected <tools> list (hallucinated). Syntactic, outcome-independent.
    #   mcp_tool_bonus: "safe eta" — per-step bonus for a SUCCESSFUL + EFFECTFUL
    #     (non read-only) + NON-DUPLICATE MCP call; nudges the ~1% MCP-adoption
    #     bottleneck without the read-only/repeat farming the legacy eta_initial
    #     path allows. Outcome-gated by default (shapes "use tools toward a win").
    #     Decays over rl_step like eta_t.
    format_pen: float = 0.0,
    mcp_tool_bonus: float = 0.0,
    mcp_tool_bonus_decay_steps: int = 30,
    mcp_tool_bonus_outcome_gated: bool = True,
    # DENSE MCP bonus: when True, the MCP tool-use bonus is NOT folded into
    # raw_reward (which would be averaged into the trajectory return, z-scored, and
    # ÷T — annihilating it to ~1e-4, see the format-reward diagnostic). Instead it is added directly to
    # the post-normalization `reward` per-step, so a successful effectful MCP step
    # gets a genuine per-step advantage of ~mcp_tool_bonus (comparable to the ~0.06
    # per-step outcome advantage). mcp_tool_bonus_outcome_gated applies to the dense
    # path too (gated variant): gated → bonus only on MCP steps of winning trajectories,
    # coupling the per-step signal to outcome; ungated (ungated dense) → exec_ok only,
    # bootstraps adoption in failing trajectories but rewards "call tools" per se.
    # Read-only filter + first-occurrence-per-key dedup + decay still guard farming.
    mcp_tool_bonus_dense: bool = False,
    # False-success exception to positive_advantage_only (false-success exception). pos-adv-only diagnosis:
    # pos-only drops ALL negative-adv samples, which makes false-success
    # trajectories (claimed terminate(success) but outcome=0) INVISIBLE — the only
    # negative examples that teach "don't claim success when not done" vanish, and
    # cloning short winners generalizes premature terminate(success) unchecked
    # (held-out greedy false_success_rate 0.30→0.54 in 5 steps). Fix is surgical:
    # keep the negative advantage ONLY on the LAST step (the lying terminate
    # response) of false-success trajectories, with ÷T undone (otherwise the
    # punishment is diluted to ~-0.05) and capped at -2.0. All other negative
    # samples are still dropped (they are the noise pos-only exists to remove).
    # No effect unless positive_advantage_only is also true.
    false_success_keep_neg: bool = False,
    # Diagnostic: replace every per-step raw_reward with U(-1, 1) iid noise.
    # Used to verify that infra (GRPO z-score, gradient flow, optimizer) is alive
    # independently of whether the task reward signal is meaningful.
    random_reward: bool = False,
    # "groupwise" (default): standard GRPO trajectory-level return, per-group
    #   z-score, broadcast to all tokens. Eliminates late-step bias that
    #   suppresses terminate. This is the project default — all live configs use it.
    # "stepwise" (legacy): per-step-index z-score within task; kept only for the
    #   A1 ablation. (Footgun history: this default used to be "stepwise" while
    #   every yaml overrode it to groupwise — flipped so a yaml omission can't
    #   silently revert. See CLAUDE.md "RL ablation matrix / known issues".)
    normalize_mode: str = "groupwise",
):
    if not os.path.isfile(rollout_json):
        raise FileNotFoundError(f"rollout_json not exist: {rollout_json}")

    outcome_only = (lambda_coef == 0.0)

    with open(rollout_json, "r", encoding="utf-8") as f:
        ROLL = json.load(f)

    if outcome_only:
        REWA = []
    else:
        if not os.path.isfile(reward_json):
            raise FileNotFoundError(f"reward_json not exist: {reward_json}")
        with open(reward_json, "r", encoding="utf-8") as f:
            REWA = json.load(f)

    def _index_by_de(arr):
        m = {}
        for it in arr:
            d = it.get("domain"); e = it.get("example")
            if d is None or e is None:
                continue
            m[(d, e)] = it
        return m

    idx_roll = _index_by_de(ROLL)
    idx_rewa = _index_by_de(REWA)

    policy_dataset: List[Dict[str, Any]] = []

    # pre-standardization stats accumulators
    _raw_rewards: List[float] = []
    _prcs:        List[float] = []
    _outcomes:    List[float] = []
    _traj_lens:   List[int]   = []   # per-trajectory step count (never-terminate monitor)
    _traj_lens_success: List[int] = []  # per-trajectory step count, success outcomes only
    _traj_lens_fail:    List[int] = []  # per-trajectory step count, failed outcomes only
    _resp_lens:   List[int]   = []   # per-record response char length (train-side mirror of monitor_greedy/resp_len_mean)
    _length_pens: List[float] = []   # per-record length penalty magnitude
    _step_pens:   List[float] = []   # per-record cheap step penalty magnitude
    _cap_pens:    List[float] = []   # per-record hit-cap penalty magnitude
    _term_bonuses:  List[float] = [] # per-record terminate signal magnitude (signed; can be -)
    _noprog_pens:   List[float] = [] # per-record no-progress penalty magnitude
    # 5-case terminate counters (mutually exclusive, only fires when last-step terminate emitted)
    _n_term_efficient: int = 0   # A: outcome=1 + short
    _n_term_slow:      int = 0   # B: outcome=1 + long
    _n_term_early:     int = 0   # C: status=failure + outcome=0 + short  (gave up too early)
    _n_term_graceful:  int = 0   # D: status=failure + outcome=0 + long   (legit give-up)
    _n_term_false_success: int = 0  # E: status=success + outcome=0       (reward hacking)
    _n_noprog_steps:   int = 0   # count of steps that triggered the no-progress pen
    _n_format_viol:    int = 0   # A3: steps penalized for malformed/hallucinated tool_call
    _n_tool_bonus:     int = 0   # safe eta: steps that earned the MCP tool-use bonus
    _n_fs_neg_kept:    int = 0   # false-success exception: lie steps kept with negative adv
    _n_format_steps:   int = 0   # total steps eligible for A3 check (denominator for viol rate)

    # per-domain counters: {domain -> {mcp, gui, success, total_trajs}}
    _domain_stats: Dict[str, Dict[str, int]] = {}
    # tasks that passed the acc filter and produced records — gradient
    # concentration diagnostic (a step training on 1 task is a red flag)
    _n_tasks_kept: int = 0

    for key, item_roll in idx_roll.items():
        d, e = key

        if allow_set is not None and key not in allow_set:
            continue

        item_rewa = idx_rewa.get(key)
        if not item_rewa and not outcome_only:
            print(f"[WARN] miss reward data, skip example: {key}")
            continue

        traj_roll = item_roll.get("trajctory_list", []) or []
        traj_rewa = item_rewa.get("trajctory_list", []) if item_rewa else []
        sort_traj_list_inplace(traj_roll)
        sort_traj_list_inplace(traj_rewa)

        def _by_run_id(lst):
            m = {}
            for t in lst:
                m[t.get("run_id")] = t
            return m

        roll_by_run = _by_run_id(traj_roll)
        rewa_by_run = _by_run_id(traj_rewa)
        if outcome_only:
            common_run_ids = list(roll_by_run.keys())
        else:
            common_run_ids = [rid for rid in roll_by_run.keys() if rid in rewa_by_run]
        if not common_run_ids:
            print(f"[WARN] no mutual run_id, skip example: {key}")
            continue

        results_sign = []
        for rid in common_run_ids:
            res = roll_by_run[rid].get("result")
            results_sign.append(1 if safe_outcome_reward(res) > 0 else 0)
        acc = sum(results_sign) / len(results_sign) if results_sign else 0.0

        # accumulate per-domain stats (all trajs, regardless of keep_policy filter)
        if d not in _domain_stats:
            _domain_stats[d] = {"mcp": 0, "gui": 0, "success": 0, "total_trajs": 0}
        for rid in common_run_ids:
            rt = roll_by_run[rid]
            _domain_stats[d]["total_trajs"] += 1
            _domain_stats[d]["success"] += 1 if safe_outcome_reward(rt.get("result")) > 0 else 0
            for at in (rt.get("action_types") or []):
                if at == "mcp":
                    _domain_stats[d]["mcp"] += 1
                else:
                    _domain_stats[d]["gui"] += 1

        if acc_lower is not None and acc_upper is not None:
            if not (acc_lower <= acc <= acc_upper):
                continue

        policy_records: List[Dict[str, Any]] = []

        def _rid_key(v):
            if isinstance(v, int):
                return (0, v)
            if str(v).isdigit():
                return (0, int(v))
            return (1, str(v))

        for rid in sorted(common_run_ids, key=_rid_key):
            roll_t = roll_by_run[rid]
            rewa_t = rewa_by_run.get(rid, {})

            outcome = float(safe_outcome_reward(roll_t.get("result")))
            if outcome_only:
                pol_msg_list_len = len(roll_t.get("message", []) or [])
                proc_list = [0.0] * pol_msg_list_len
            else:
                proc_list = rewa_t.get("process_reward", []) or []

            pol_msg_list  = roll_t.get("message", []) or []
            pol_resp_list = roll_t.get("response", []) or []

            T_pol = min(len(proc_list), len(pol_msg_list), len(pol_resp_list))

            # --- policy ---
            action_types = roll_t.get("action_types", []) or []
            exec_oks     = roll_t.get("exec_oks",     []) or []

            eta_t = max(0.0, eta_initial * (1 - rl_step / eta_decay_steps))
            # Safe MCP tool-use bonus (decays like eta_t). Separate knob from the
            # legacy eta_initial/bonus path so old configs are unaffected.
            mcp_bonus_t = (
                max(0.0, mcp_tool_bonus * (1 - rl_step / max(1, mcp_tool_bonus_decay_steps)))
                if mcp_tool_bonus else 0.0
            )

            # Length penalty: proportional to total trajectory length T_pol. Subtracted
            # identically on every step of this trajectory, but because it scales with
            # T_pol it differs across trajectories at the same step index k, so it
            # survives the per-step-index GRPO standardization (see
            # _stepwise_standardize_inplace). Net effect: "solve faster is better",
            # directly counteracting the never-terminate failure mode.
            length_pen = length_penalty_coef * (T_pol / max(1, max_steps_ref))
            # Cap penalty: flat negative on every step of a trajectory that spun to
            # the step cap without terminating on its own (T_pol >= max_steps_ref).
            # Directly targets never-terminate — unlike the linear length penalty it
            # does NOT punish a genuinely long *successful* trajectory, only "ran out
            # of steps". Trajectories that voluntarily stopped (T_pol < cap) pay 0, so
            # there's cross-trajectory variance at each step-index → survives the GRPO
            # z-score. Reward-side soft signal, replaces the misfiring rollout loop-break.
            hit_cap = T_pol >= max_steps_ref
            cap_pen = cap_penalty if hit_cap else 0.0
            _traj_lens.append(T_pol)
            # Partition trajectory lengths by outcome — exposes the "success short /
            # fail spin-to-cap" split that the aggregate traj_len_mean hides. Mirrors
            # held-out monitor_greedy/avg_steps_{success,fail}.
            if outcome > 0:
                _traj_lens_success.append(T_pol)
            else:
                _traj_lens_fail.append(T_pol)

            # 5-case terminate classifier (trajectory-level). Parses model's claimed
            # terminate status from the LAST tool_call ("success"/"failure"/None).
            # Cap-induced FAIL → no terminate tool_call → claimed_status is None →
            # no signal applied (handled by outcome ± cap_pen alone).
            last_resp = pol_resp_list[T_pol - 1] if T_pol > 0 else ""
            claimed_status = parse_terminate(last_resp)   # "success" / "failure" / None
            voluntary_terminate = (claimed_status is not None) and not hit_cap

            # Legacy `terminate_bonus` overrides per-case bonuses (back-compat).
            _eff_bonus  = terminate_bonus if terminate_bonus else terminate_bonus_efficient
            _slow_bonus = terminate_bonus if terminate_bonus else terminate_bonus_slow

            _eff_cap  = max(1.0, max_steps_ref * efficient_ratio)   # success: short < this
            _early_cap = max(1.0, max_steps_ref * early_quit_ratio) # fail:    short < this

            term_signal_total = 0.0   # signed; positive = bonus, negative = penalty
            if voluntary_terminate:
                if outcome > 0:
                    # Outcome=1 doesn't care about claimed_status — the model got
                    # the job done. (Even if it said status=failure but evaluator
                    # judged success, that's a happy accident → still bonus.)
                    if T_pol < _eff_cap:
                        term_signal_total = +_eff_bonus
                        _n_term_efficient += 1
                    else:
                        term_signal_total = +_slow_bonus
                        _n_term_slow += 1
                else:
                    # Outcome=0. Now claimed_status matters:
                    #   status=success + outcome=0 = REWARD HACKING (model lied).
                    #     → -false_success_pen, directly punish the e17 failure mode.
                    #   status=failure + outcome=0 = model knew it failed.
                    #     short → early quit (penalty), long → graceful (bonus).
                    if claimed_status == "success":
                        term_signal_total = -false_success_pen
                        _n_term_false_success += 1
                    else:
                        if T_pol < _early_cap:
                            term_signal_total = -early_quit_pen
                            _n_term_early += 1
                        else:
                            # Graceful fail: model legitimately gave up after honest
                            # effort. Small positive to signal "this beats cap-spin".
                            term_signal_total = +graceful_fail_bonus
                            _n_term_graceful += 1
            # per_step (default): full signal at every step → larger cross-traj delta
            # at each step-index for GRPO z-score. amortized: signal/T_pol (legacy).
            if term_signal_total:
                if term_amortize_mode == "amortized":
                    term_bonus_per_step = term_signal_total / max(T_pol, 1)
                else:  # "per_step"
                    term_bonus_per_step = term_signal_total
            else:
                term_bonus_per_step = 0.0

            # Distinct effectful MCP action keys already rewarded in THIS trajectory
            # → the safe-eta bonus fires only on a tool call's first occurrence.
            _seen_tool_keys: Set[str] = set()

            for k in range(T_pol):
                prc = float(proc_list[k])

                a_type = action_types[k] if k < len(action_types) else "gui"
                e_ok   = bool(exec_oks[k]) if k < len(exec_oks) else True
                bonus = compute_action_type_bonus(a_type, e_ok)

                # Cheap heuristic per-step process signal (no reward API / no human
                # labels). Penalize failed exec and repeated actions. Like
                # length_pen these are negative and only bite cross-trajectory, so they
                # require positive_advantage_only=False.
                # Repeat detection keys on the executed <tool_call> (via
                # action_key_from_response), NOT the full response text — reasoning
                # text differs every step even when the action repeats, which is why
                # the old verbatim-response check never fired. This is the reward-side
                # soft replacement for the removed rollout loop-break.
                step_pen = 0.0
                if step_fail_pen and not e_ok:
                    step_pen += step_fail_pen
                if repeat_pen and k > 0:
                    _cur = action_key_from_response(pol_resp_list[k])
                    _prev = action_key_from_response(pol_resp_list[k - 1])
                    if _cur and _cur == _prev:
                        step_pen += repeat_pen
                # No-progress penalty: screenshot in the NEXT prompt (pol_msg_list[k+1])
                # equals the current prompt's screenshot (pol_msg_list[k]). This means
                # the action at step k produced no visible state change. Skipped on the
                # last step (no k+1) and when no_progress_pen is 0.
                noprog_pen_this = 0.0
                if no_progress_pen and (k + 1) < len(pol_msg_list):
                    h_now  = last_image_hash(pol_msg_list[k])
                    h_next = last_image_hash(pol_msg_list[k + 1])
                    if h_now and h_next and h_now == h_next:
                        noprog_pen_this = no_progress_pen
                        step_pen += noprog_pen_this
                        _n_noprog_steps += 1

                # Format reward (A3): syntactic well-formedness + tool-name validity.
                # injected_tool_names returns {} when the <tools> block isn't found
                # → name check skipped (never penalize on missing data).
                format_pen_this = 0.0
                if format_pen:
                    _n_format_steps += 1
                    _valid = injected_tool_names(pol_msg_list[k]) if k < len(pol_msg_list) else set()
                    if format_violation(pol_resp_list[k], _valid):
                        format_pen_this = format_pen
                        _n_format_viol += 1

                # Safe MCP tool-use bonus: successful + effectful (non read-only) +
                # non-duplicate. Folded path (default): outcome-gated, added to raw
                # (then z-scored — weak). Dense path: exec_ok-gated only, NOT
                # added to raw — accumulated here and added to `reward` AFTER
                # normalization so it survives as a real per-step advantage.
                tool_bonus_this = 0.0   # folded into raw (legacy)
                tool_bonus_dense = 0.0  # added post-normalization (dense mode)
                if mcp_bonus_t and a_type == "mcp" and e_ok:
                    # Gate applies to BOTH folded and dense paths (gated variant). Dense runs
                    # the ungated dense runs set mcp_tool_bonus_outcome_gated=false explicitly, so
                    # their behavior is unchanged by the dense path now honoring it.
                    _gate_ok = (not mcp_tool_bonus_outcome_gated) or outcome > 0
                    if _gate_ok:
                        _nm = tool_name_from_response(pol_resp_list[k])
                        if _nm and not is_read_only_tool(_nm):
                            _ak = action_key_from_response(pol_resp_list[k])
                            if _ak and _ak not in _seen_tool_keys:
                                if mcp_tool_bonus_dense:
                                    tool_bonus_dense = mcp_bonus_t
                                else:
                                    tool_bonus_this = mcp_bonus_t
                                _n_tool_bonus += 1
                            if _ak:
                                _seen_tool_keys.add(_ak)

                raw = outcome + lambda_coef * prc + eta_t * bonus \
                      - length_pen - step_pen - cap_pen + term_bonus_per_step \
                      + tool_bonus_this - format_pen_this
                if random_reward:
                    import random as _random
                    raw = float(_random.random() < 0.4)
                # false-success exception: tag the lying terminate step — the LAST step of a
                # voluntarily-terminated trajectory that claimed success while
                # outcome=0. Consumed by the pos-only filter exception below.
                _fs_last = (
                    voluntary_terminate
                    and claimed_status == "success"
                    and outcome <= 0
                    and k == T_pol - 1
                )
                policy_records.append({
                    "domain": d,
                    "example": e,
                    "run_id": rid,
                    "step": k,
                    "prompt_messages": pol_msg_list[k],
                    "response": pol_resp_list[k],
                    "action_type": a_type,
                    "action_type_bonus": bonus,
                    "eta_t": eta_t,
                    "length_pen": length_pen,
                    "step_pen": step_pen,
                    "cap_pen": cap_pen,
                    "term_bonus": term_bonus_per_step,
                    "no_progress_pen": noprog_pen_this,
                    "format_pen": format_pen_this,
                    "tool_bonus": tool_bonus_this,
                    "tool_bonus_dense": tool_bonus_dense,
                    "false_success_last": _fs_last,
                    "traj_T": T_pol,
                    "raw_reward": raw,
                })
                _raw_rewards.append(raw)
                _prcs.append(prc)
                _outcomes.append(outcome)
                _length_pens.append(length_pen)
                _step_pens.append(step_pen)
                _cap_pens.append(cap_pen)
                _term_bonuses.append(term_bonus_per_step)
                _noprog_pens.append(noprog_pen_this)
                _resp_lens.append(get_token_length(pol_resp_list[k]) if isinstance(pol_resp_list[k], str) else 0)  # Token-level length

        if policy_records:
            _n_tasks_kept += 1

        if normalize:
            if policy_records:
                if normalize_mode == "groupwise":
                    _groupwise_standardize_inplace(policy_records, require_multi_traj=True)
                else:
                    _stepwise_standardize_inplace(policy_records, require_multi_traj=True)
            else:
                policy_records = []
        else:
            for rec in policy_records:
                rec["reward"] = float(rec["raw_reward"])

        # DENSE MCP bonus: add the per-step tool-use bonus to `reward` AFTER
        # normalization so it bypasses trajectory-mean + z-score + ÷T (which crushed
        # the folded path to ~1e-4 in the format-reward diagnostic). This is a direct per-step advantage of
        # ~mcp_tool_bonus on each successful effectful MCP step — also lifts those
        # steps above the |r|<eps filter below so they always carry gradient.
        if mcp_tool_bonus_dense:
            for rec in policy_records:
                _tbd = float(rec.get("tool_bonus_dense", 0.0) or 0.0)
                if _tbd:
                    rec["reward"] = float(rec["reward"]) + _tbd

        # |r| < eps instead of == 0.0: catches the degenerate-group fallback (where
        # we wrote raw - mu, often ~0 but possibly tiny non-zero) but keeps any
        # sample with a real advantage. Was r == 0.0 — that exactly matched the
        # legacy std==0 sentinel and silently dropped every flat group.
        _ADV_EPS = 1e-6
        for rec in policy_records:
            r = float(rec["reward"])
            if positive_advantage_only and r < 0.0:
                # false-success exception: keep the negative advantage on the lying
                # terminate step of false-success trajectories — the ONLY class
                # of negative sample that carries "don't claim success when not
                # done". Undo the ÷T dilution (this is a single decisive step,
                # not an amortized trajectory signal) and cap at -2.0 so one
                # sample can't dominate a batch.
                if false_success_keep_neg and rec.get("false_success_last"):
                    out = dict(rec)
                    _T = max(int(rec.get("traj_T", 1)), 1)
                    out["reward"] = max(r * _T, -2.0)
                    out.pop("raw_reward", None)
                    policy_dataset.append(out)
                    _n_fs_neg_kept += 1
                continue
            if abs(r) < _ADV_EPS:
                continue
            out = dict(rec)
            out.pop("raw_reward", None)
            policy_dataset.append(out)

    os.makedirs(out_dir or ".", exist_ok=True)
    policy_out = os.path.join(out_dir, policy_out_name)

    with open(policy_out, "w", encoding="utf-8") as f:
        json.dump(policy_dataset, f, ensure_ascii=False, indent=2)

    print(f"[DONE] generate policy training data {policy_out} | {len(policy_dataset)} samples")

    # return pre-standardization rollout stats for external logging
    rollout_stats: Dict[str, Any] = {}
    if _raw_rewards:
        import statistics as _stat
        rollout_stats = {
            "raw_reward_mean": round(sum(_raw_rewards) / len(_raw_rewards), 6),
            "raw_reward_std":  round(_stat.stdev(_raw_rewards), 6) if len(_raw_rewards) > 1 else 0.0,
            "prc_mean":        round(sum(_prcs) / len(_prcs), 6),
            "prc_std":         round(_stat.stdev(_prcs), 6) if len(_prcs) > 1 else 0.0,
            "outcome_rate":    round(sum(1 for o in _outcomes if o > 0) / len(_outcomes), 6),
            # trajectory-level success rate over kept tasks (outcome_rate above is
            # per-record, i.e. step-weighted — longer trajectories count more)
            "outcome_rate_traj": round(len(_traj_lens_success) / len(_traj_lens), 6) if _traj_lens else 0.0,
            # gradient concentration: how many tasks / trajectories survived the
            # acc filter and actually feed this step's update
            "n_tasks_kept":    int(_n_tasks_kept),
            "n_trajs_kept":    int(len(_traj_lens)),
            "n_records":       len(_raw_rewards),
            # records actually fed to training (after z-score + r==0.0 / positive-only
            # filter). A big gap vs n_records = GRPO zeroed-out groups (the all-zero
            # gradient failure: same-outcome trajectories at a step-index → 0 variance
            # → reward 0 → dropped). frac_kept near 1.0 = healthy; near 0.1 = e8-style.
            "n_train_samples": len(policy_dataset),
            "frac_kept":       round(len(policy_dataset) / len(_raw_rewards), 4) if _raw_rewards else 0.0,
            "traj_len_mean":   round(sum(_traj_lens) / len(_traj_lens), 3) if _traj_lens else 0.0,
            # success/fail length split — danger signal when avg_steps_success ↑
            # and avg_steps_fail ↓ together (model spins on wins, gives up on losses).
            "avg_steps_success": round(sum(_traj_lens_success) / len(_traj_lens_success), 3) if _traj_lens_success else 0.0,
            "avg_steps_fail":    round(sum(_traj_lens_fail)    / len(_traj_lens_fail),    3) if _traj_lens_fail    else 0.0,
            # train-side mean response char length — pair with monitor_greedy/resp_len_mean
            # to compare temp=1 (train) vs temp=0 (eval) degeneration. Sustained rise =
            # response inflation precedes greedy-side collapse by 1-2 steps.
            "resp_len_mean":   round(sum(_resp_lens) / len(_resp_lens), 1) if _resp_lens else 0.0,
            # fraction of trajectories that spun to the step cap (never-terminate
            # signal on the temp=1.0 train side, mirrors held-out monitor_greedy)
            "frac_hit_cap":    round(sum(1 for L in _traj_lens if L >= max_steps_ref) / len(_traj_lens), 4) if _traj_lens else 0.0,
            # penalty magnitudes — confirm shaping is actually applied & sane
            "length_pen_mean": round(sum(_length_pens) / len(_length_pens), 6) if _length_pens else 0.0,
            "step_pen_mean":   round(sum(_step_pens) / len(_step_pens), 6) if _step_pens else 0.0,
            "cap_pen_mean":    round(sum(_cap_pens) / len(_cap_pens), 6) if _cap_pens else 0.0,
            # New shaping signals (e17+): 4-case terminate classifier + no-progress pen.
            # term_signal_mean / _std measure the per-step SIGNED magnitude across all
            # records. _std is the critical diagnostic: GRPO z-scores within each
            # step-index, so what matters is the cross-trajectory std of this signal
            # AT a single step k. If _std is much smaller than the cross-traj std of
            # the dominant signal (outcome ±1, std ~0.85 at outcome_rate ~0.3), the
            # shaping is being drowned by outcome — switch term_amortize_mode to
            # "per_step" (default) or scale up the bonuses.
            # The 4 counters give the trajectory-level category distribution.
            "term_signal_mean":   round(sum(_term_bonuses) / len(_term_bonuses), 6) if _term_bonuses else 0.0,
            "term_signal_std":    round(_stat.stdev(_term_bonuses), 6) if len(_term_bonuses) > 1 else 0.0,
            "term_signal_abs_mean": round(sum(abs(x) for x in _term_bonuses) / len(_term_bonuses), 6) if _term_bonuses else 0.0,
            "n_term_efficient":   int(_n_term_efficient),       # A
            "n_term_slow":        int(_n_term_slow),            # B
            "n_term_early":       int(_n_term_early),           # C: failure status + short
            "n_term_graceful":    int(_n_term_graceful),        # D: failure status + long
            "n_term_false_success": int(_n_term_false_success), # E: success status + outcome=0 (reward hacking!)
            "noprog_pen_mean":    round(sum(_noprog_pens) / len(_noprog_pens), 6) if _noprog_pens else 0.0,
            "n_noprog_steps":     int(_n_noprog_steps),
            "n_format_viol":      int(_n_format_viol),
            "format_viol_rate":   round(_n_format_viol / _n_format_steps, 4) if _n_format_steps else 0.0,
            "n_tool_bonus":       int(_n_tool_bonus),
            "n_fs_neg_kept":      int(_n_fs_neg_kept),
            "mcp_bonus_dense":    bool(mcp_tool_bonus_dense),
        }
        # per-domain breakdown
        domain_mcp_rates: Dict[str, float] = {}
        domain_outcome_rates: Dict[str, float] = {}
        for dom, ds in _domain_stats.items():
            total_acts = ds["mcp"] + ds["gui"]
            domain_mcp_rates[dom] = round(ds["mcp"] / total_acts, 4) if total_acts > 0 else 0.0
            domain_outcome_rates[dom] = round(ds["success"] / ds["total_trajs"], 4) if ds["total_trajs"] > 0 else 0.0
        rollout_stats["domain_mcp_rates"] = domain_mcp_rates
        rollout_stats["domain_outcome_rates"] = domain_outcome_rates
    return rollout_stats


# ----------------- log_policy_accuracy -----------------
def log_policy_accuracy_from_rollout(
    rollout_json: str,
    outputs_result_name: str,
    mode: str,
    allow_set: Optional[Set[Tuple[str, str]]] = None,
):
    if not os.path.isfile(rollout_json):
        print(f"[INFO] rollout_json not exists, acc summary{rollout_json}")
        return

    with open(rollout_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    domain_stat = {}
    total = 0
    correct = 0

    for item in data:
        domain = item.get("domain", "unknown")
        example = item.get("example")

        if allow_set is not None:
            if not isinstance(example, str) or (domain, example) not in allow_set:
                continue

        traj_list = item.get("trajctory_list", [])
        for t in traj_list:
            outcome = safe_outcome_reward(t.get("result"))
            total += 1
            if domain not in domain_stat:
                domain_stat[domain] = [0, 0]
            domain_stat[domain][0] += 1
            if outcome > 0:
                domain_stat[domain][1] += 1
                correct += 1

    if total == 0:
        print(f"[WARN] not effective result, can not summarize acc: {rollout_json}")
        return

    parts = []
    for d in sorted(domain_stat.keys()):
        tot, cor = domain_stat[d]
        if tot == 0:
            continue
        acc = cor / tot
        parts.append(f"{d}:{acc:.4f}")

    overall_acc = correct / total
    parts.append(f"ALL:{overall_acc:.4f}")

    line = f"[{mode}] " + " ".join(parts)

    os.makedirs(os.path.dirname(outputs_result_name), exist_ok=True)
    with open(outputs_result_name, "a", encoding="utf-8") as f:
        cprint("\n\n\n" + line, color="green")
        f.write(line + "\n")


# -----------------  entry point -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True, help="domain/example/run/trajectory.json root dir")

    parser.add_argument("--num-nodes", type=int, default=-1,
                        help=">=0 merge reward files: reward_rollout_results_node_{i}.json, i=0..num_nodes")
    parser.add_argument("--merge-dir", type=str,
                        default="./temp_data",
                        help="reward slide dir, (also contains example_node_*.json)")

    parser.add_argument("--acc-lower", type=float, default=0.0)
    parser.add_argument("--acc-upper", type=float, default=1.0)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--policy-out-name", type=str, default="policy_optimization_data.json")
    parser.add_argument("--type", type=str, default="train")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--lambda-coef", type=float, default=0.3)
    parser.add_argument("--eta-initial", type=float, default=0.02)
    parser.add_argument("--eta-decay-steps", type=int, default=30)
    parser.add_argument("--positive-advantage-only", action="store_true", default=False)
    parser.add_argument("--length-penalty-coef", type=float, default=0.0,
                        help="penalty proportional to trajectory length T/max_steps_ref; counters the never-terminate failure mode")
    parser.add_argument("--max-steps-ref", type=int, default=30,
                        help="reference horizon for length penalty normalization (should match rollout max_steps)")
    parser.add_argument("--step-fail-pen", type=float, default=0.0,
                        help="cheap per-step penalty for failed exec (0=off)")
    parser.add_argument("--repeat-pen", type=float, default=0.0,
                        help="cheap per-step penalty for verbatim-repeated action (0=off)")
    parser.add_argument("--cap-penalty", type=float, default=0.0,
                        help="flat penalty on every step of a trajectory that hit max_steps without terminating (0=off); directly targets never-terminate")
    # 4-case terminate classifier — see generate_rl_datasets docstring above the kwargs
    parser.add_argument("--terminate-bonus-efficient", type=float, default=0.0,
                        help="bonus when voluntary terminate + success + T < cap*efficient_ratio")
    parser.add_argument("--terminate-bonus-slow", type=float, default=0.0,
                        help="bonus when voluntary terminate + success + T >= cap*efficient_ratio (smaller than efficient)")
    parser.add_argument("--early-quit-pen", type=float, default=0.0,
                        help="penalty when terminate(status=failure) + outcome=0 + T < cap*early_quit_ratio (gave up too soon)")
    parser.add_argument("--graceful-fail-bonus", type=float, default=0.0,
                        help="small bonus when terminate(status=failure) + outcome=0 + T >= cap*early_quit_ratio (legit give-up; better than cap-spin)")
    parser.add_argument("--false-success-pen", type=float, default=0.0,
                        help="penalty when terminate(status=success) + outcome=0 — reward hacking: model lied about success. Should typically be > efficient_bonus to net-negative the hacked path.")
    parser.add_argument("--efficient-ratio", type=float, default=0.7,
                        help="success-trajectory boundary between 'efficient' (T < cap*ratio) and 'slow'")
    parser.add_argument("--early-quit-ratio", type=float, default=0.5,
                        help="fail-trajectory boundary between 'early-quit' (T < cap*ratio) and 'graceful'")
    parser.add_argument("--term-amortize-mode", type=str, default="per_step",
                        choices=["per_step", "amortized"],
                        help="'per_step' (default): every step gets the full signal (recommended; survives z-score). 'amortized': legacy, signal/T_pol per step.")
    parser.add_argument("--terminate-bonus", type=float, default=0.0,
                        help="LEGACY: single bonus value used for both efficient and slow when set. Overrides --terminate-bonus-{efficient,slow}.")
    parser.add_argument("--no-progress-pen", type=float, default=0.0,
                        help="per-step penalty when next-step screenshot equals current screenshot (cheap 'did nothing visible' proxy). (0=off)")
    parser.add_argument("--format-pen", type=float, default=0.0,
                        help="A3: per-step penalty when output is not exactly one well-formed tool_call with valid {name,arguments}, or names a tool not in the injected <tools> list (hallucinated). Syntactic, outcome-independent. (0=off)")
    parser.add_argument("--mcp-tool-bonus", type=float, default=0.0,
                        help="safe eta: per-step bonus for a successful + effectful (non read-only) + non-duplicate MCP call; nudges MCP adoption without read-only/repeat farming. (0=off)")
    parser.add_argument("--mcp-tool-bonus-decay-steps", type=int, default=30,
                        help="decay horizon for --mcp-tool-bonus (linear to 0 by this rl_step)")
    parser.add_argument("--mcp-tool-bonus-ungated", action="store_true", default=False,
                        help="DISABLE outcome-gating of the MCP tool-use bonus (grant it on failed trajectories too). Default: gated (safe — bonus only on successful trajectories).")
    parser.add_argument("--false-success-keep-neg", action="store_true", default=False,
                        help="false-success exception to --positive-advantage-only — keep the negative advantage on the LAST step (the lying terminate response) of false-success trajectories (claimed success, outcome=0), with the div-T dilution undone and capped at -2.0. Restores the only negative signal that teaches 'don't claim success when not done'. No effect without --positive-advantage-only.")
    parser.add_argument("--mcp-tool-bonus-dense", action="store_true", default=False,
                        help="DENSE mode: add the MCP tool-use bonus to reward AFTER normalization (per-step direct advantage ~mcp_tool_bonus), instead of folding into raw_reward where z-score+÷T crush it. Combine with/without --mcp-tool-bonus-ungated: gated dense = bonus only in winning trajectories; ungated dense = exec_ok only. (default off = legacy folded)")
    parser.add_argument("--random-reward", action="store_true", default=False,
                        help="replace every raw_reward with U(-1,1) iid noise — infra diagnostic only")
    parser.add_argument("--normalize-mode", type=str, default="groupwise",
                        choices=["stepwise", "groupwise"],
                        help="groupwise (default): standard GRPO trajectory-level return + per-group z-score; stepwise: legacy per-step-index z-score (A1 ablation only)")

    args = parser.parse_args()

    # collect message+response
    if args.type == "train":
        output_rollout_result = os.path.join(args.merge_dir, "rl_rollout_results.json")
    else:
        output_rollout_result = os.path.join(args.merge_dir, "eval_rollout_results.json")
    _n_domains, _n_tasks, _n_trajs = collect_messages_with_responses(args.root_dir, output_rollout_result)
    if args.type != "train":
        print(f"[EVAL COVERAGE] step={args.step} domains={_n_domains} unique_tasks={_n_tasks} total_trajs={_n_trajs}")

    # get id2domain,after_acc_map, then aggregate active/temp
    id2domain = build_id2domain_from_rollout(output_rollout_result)
    after_acc_map = compute_acc_map_from_rollout(output_rollout_result)

    active_set, _ = aggregate_active_temp_sets(args.merge_dir, id2domain)
    cprint(f"[INFO] active_set={len(active_set)} (from {args.merge_dir}/example_node_*.json)", "green")

    _merge_abs = os.path.abspath(args.merge_dir.rstrip("/"))
    project_name = os.path.basename(os.path.dirname(_merge_abs))
    base_dir = os.path.dirname(os.path.dirname(_merge_abs))

    allow_set: Optional[Set[Tuple[str, str]]] = None
    if args.type == "train":
        allow_set = set(active_set)
        cprint(f"[INFO] allow_set={len(allow_set)}", "green")

    # merge reward slides
    reward_merged_path = ""
    if args.type == "train":
        reward_merged_path = os.path.join(args.merge_dir, "rl_reward_rollout_results.json")
        merge_reward_shards(args.merge_dir, args.num_nodes, reward_merged_path)

    # generate RL training data
    if args.type == "train":
        rollout_json = output_rollout_result
        reward_json = reward_merged_path

        # Use strict > 0 / < 1 so passing the argparse defaults (0.0 / 1.0)
        # disables the filter — explicit 0.05/0.95 from osworld_rl.py turns it on.
        lower = args.acc_lower if args.acc_lower is not None and args.acc_lower > 0 else None
        upper = args.acc_upper if args.acc_upper is not None and args.acc_upper < 1 else None
        normalize = not args.no_normalize

        rollout_stats = generate_rl_datasets(
            rollout_json=rollout_json,
            reward_json=reward_json,
            out_dir=args.merge_dir,
            acc_lower=lower if (lower is not None and upper is not None) else None,
            acc_upper=upper if (lower is not None and upper is not None) else None,
            normalize=normalize,
            policy_out_name=args.policy_out_name,
            allow_set=allow_set,
            lambda_coef=args.lambda_coef,
            eta_initial=args.eta_initial,
            eta_decay_steps=args.eta_decay_steps,
            rl_step=args.step,
            positive_advantage_only=args.positive_advantage_only,
            length_penalty_coef=args.length_penalty_coef,
            max_steps_ref=args.max_steps_ref,
            step_fail_pen=args.step_fail_pen,
            repeat_pen=args.repeat_pen,
            cap_penalty=args.cap_penalty,
            terminate_bonus=args.terminate_bonus,
            terminate_bonus_efficient=args.terminate_bonus_efficient,
            terminate_bonus_slow=args.terminate_bonus_slow,
            early_quit_pen=args.early_quit_pen,
            graceful_fail_bonus=args.graceful_fail_bonus,
            false_success_pen=args.false_success_pen,
            efficient_ratio=args.efficient_ratio,
            early_quit_ratio=args.early_quit_ratio,
            term_amortize_mode=args.term_amortize_mode,
            no_progress_pen=args.no_progress_pen,
            format_pen=args.format_pen,
            mcp_tool_bonus=args.mcp_tool_bonus,
            mcp_tool_bonus_decay_steps=args.mcp_tool_bonus_decay_steps,
            mcp_tool_bonus_outcome_gated=(not args.mcp_tool_bonus_ungated),
            mcp_tool_bonus_dense=args.mcp_tool_bonus_dense,
            false_success_keep_neg=args.false_success_keep_neg,
            random_reward=args.random_reward,
            normalize_mode=args.normalize_mode,
        )

        # write pre-standardization rollout stats
        if rollout_stats:
            _stats_log = os.path.join(base_dir, project_name, "results", "rollout-stats.jsonl")
            os.makedirs(os.path.dirname(_stats_log), exist_ok=True)
            with open(_stats_log, "a") as _sf:
                _sf.write(json.dumps({"rl_step": args.step, **rollout_stats}) + "\n")
            print(f"[DONE] rollout stats → {_stats_log}")

    # acc
    if args.type == "train":
        outputs_result_name = os.path.join(base_dir, project_name, "results", "rl-results.txt")
        mode_str = "TRAIN " + str(args.step)
    else:
        outputs_result_name = os.path.join(base_dir, project_name, "results", "eval-results.txt")
        mode_str = "EVAL " + str(args.step)

    log_policy_accuracy_from_rollout(
        rollout_json=output_rollout_result,
        outputs_result_name=outputs_result_name,
        mode=mode_str,
        allow_set=active_set if args.type == "train" else None,
    )

    if args.type != "train":
        coverage_line = f"[EVAL COVERAGE] step={args.step} unique_tasks={_n_tasks} total_trajs={_n_trajs}"
        os.makedirs(os.path.dirname(os.path.abspath(outputs_result_name)), exist_ok=True)
        with open(outputs_result_name, "a", encoding="utf-8") as _cf:
            _cf.write(coverage_line + "\n")


if __name__ == "__main__":
    main()

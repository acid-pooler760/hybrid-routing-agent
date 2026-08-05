import os
import sys
import subprocess
import shlex
import json
import glob
import time
from omegaconf import DictConfig, ListConfig, OmegaConf
from pathlib import Path
import shutil
import re

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

def run_local(cmd: str, check: bool = True) -> None:
    subprocess.run(f"bash -lc {shlex.quote(cmd)}", shell=True, check=check)

def run_local_async(cmd: str) -> subprocess.Popen:
    return subprocess.Popen(f"bash -lc {shlex.quote(cmd)}", shell=True)

def run_remote(host: str, cmd: str, check: bool = True) -> None:
    ssh_cmd = f'ssh {os.environ.get("SSH_USER", "root")}@{host} "bash -lc {shlex.quote(cmd)}"'
    subprocess.run(ssh_cmd, shell=True, check=check)

def run_remote_async(host: str, cmd: str) -> subprocess.Popen:
    ssh_cmd = f'ssh {os.environ.get("SSH_USER", "root")}@{host} "bash -lc {shlex.quote(cmd)}"'
    return subprocess.Popen(ssh_cmd, shell=True)


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_path = Path(cli_conf.config)
    yaml_conf = OmegaConf.load(yaml_path)

    # Handle Hydra-style `defaults` inheritance without requiring Hydra
    defaults = yaml_conf.pop("defaults", None)
    if defaults is not None:
        base_conf = OmegaConf.create({})
        for entry in defaults:
            base_name = entry if isinstance(entry, str) else list(entry.values())[0]
            base_file = (yaml_path.parent / base_name).with_suffix(".yaml")
            if base_file.exists():
                base_conf = OmegaConf.merge(base_conf, OmegaConf.load(base_file))
        yaml_conf = OmegaConf.merge(base_conf, yaml_conf)

    return OmegaConf.merge(yaml_conf, cli_conf)

def begin_with(file_name: str):
    with open(file_name, "w"):
        pass


def make_init_bash(cfg) -> str:
    sc = cfg.system
    download_proxy     = getattr(sc, "download_proxy", None)
    hf_home            = getattr(sc, "HF_HOME", None)
    envs_dir           = getattr(sc, "envs_dir", None)
    additional_command = getattr(sc, "additional_command", None)

    def write_export(lines, file, k, v):
        if v is not None:
            lines.append(f"echo 'export {k}={v}' >> {file}")

    lines = []
    lines.append("set -e")

    for f in ("~/.bashrc", "~/.bash_profile"):
        write_export(lines, f, "download_proxy", download_proxy)
        write_export(lines, f, "HF_HOME",        hf_home)
        if hf_home is not None:
            lines.append(f"echo 'export TRANSFORMERS_CACHE={hf_home}' >> {f}")

    if additional_command is not None:
        lines.append(additional_command)
    lines.append("")

    if envs_dir is not None:
        lines.append(f"conda config --append envs_dirs {envs_dir} || true")
        lines.append("")

    lines.append("echo 'source ~/.bashrc' >> ~/.bash_profile")
    lines.append("")

    return "\n".join(lines)


def cleanup_local_containers(cfg):
    """Remove any leftover happysixd/osworld-docker containers from a previous rollout."""
    import subprocess, signal
    project = cfg.experiment.project
    print(f"[cleanup] Removing stale Docker containers (label=osworld-rl) for project '{project}'…")
    try:
        r = subprocess.run(
            ["docker", "ps", "-aq", "--filter", "label=osworld-rl=true"],
            capture_output=True, timeout=15,
        )
        ids = r.stdout.decode().split()
        if ids:
            # Remove in batches of 32 to avoid arg-list / timeout issues
            batch_size = 32
            removed = 0
            for i in range(0, len(ids), batch_size):
                batch = ids[i:i + batch_size]
                subprocess.run(["docker", "rm", "--force"] + batch, capture_output=True, timeout=120)
                removed += len(batch)
            print(f"[cleanup] Removed {removed} container(s).")
        else:
            print("[cleanup] No stale containers found.")
    except Exception as e:
        print(f"[cleanup] Warning: container removal failed ({e}), continuing anyway.")

    # Kill lingering docker-proxy processes on OSWorld env port ranges (6xxx, 9xxx, 10xxx)
    try:
        r = subprocess.run(["ps", "aux"], capture_output=True, timeout=10)
        for line in r.stdout.decode().splitlines():
            if "docker-proxy" not in line:
                continue
            import re
            if re.search(r":(6\d{3}|9\d{3}|10\d{3})\b", line):
                parts = line.split()
                if len(parts) > 1:
                    try:
                        os.kill(int(parts[1]), signal.SIGKILL)
                    except Exception:
                        pass
    except Exception as e:
        print(f"[cleanup] Warning: docker-proxy cleanup failed ({e}), continuing anyway.")





if __name__ == "__main__":
    cfg = get_config()

    BASE_DIR = cfg.system.rl_base_dir
    env_name = cfg.system.env_name
    project = cfg.experiment.project

    # All artifacts for one RL run live under <BASE_DIR>/runs/<project>/.
    # Rollout trajectories are split into rollouts/train/ (overwritten each step)
    # and rollouts/eval/step_<N>/ (one bucket per held-out eval step).
    RUN_DIR = os.path.join(BASE_DIR, "runs", project)
    TRAIN_ROLLOUT_DIR = os.path.join(RUN_DIR, "rollouts", "train")

    def eval_rollout_dir(step):
        return os.path.join(RUN_DIR, "rollouts", "eval", f"step_{step}")

    project_dir = Path(RUN_DIR)
    project_dir.mkdir(parents=True, exist_ok=True)
    # Big per-RL-step TRANSIENT artifacts hammer the shared JuiceFS→OSS write bandwidth
    # (full_coverage: ~0.4-1.8TB .pt/step + 17GB ckpt/step + GB rollouts/step). On a tiny
    # shared bucket (~20Gbps total) this sustained-saturates and hurts other business.
    # Set OSWORLD_LOCAL_TEMP=/tmp/osworld_temp to symlink these onto LOCAL disk (single-node,
    # so no cross-node sharing needed). Every reader/writer uses the runs/<proj>/<sub> path
    # (subprocesses cwd=BASE_DIR) and follows the symlink transparently → ~0 JuiceFS write +
    # faster local mmap. PERSISTENT + tiny stays on JuiceFS: results/ (KB jsonl) and
    # ckpt_history/step_N (every save_every). save_checkpoint uses mkdir(exist_ok)+save_pretrained
    # (no rmtree) so symlinking ckpt/optimized is safe. Env unset = original on-JuiceFS behavior.
    _local_base = os.environ.get("OSWORLD_LOCAL_TEMP", "").strip()

    def _redirect_local(subpath):
        dst = project_dir / subpath
        if not _local_base:
            dst.mkdir(parents=True, exist_ok=True)
            return
        local_target = Path(_local_base) / project / subpath
        local_target.mkdir(parents=True, exist_ok=True)
        dst.parent.mkdir(parents=True, exist_ok=True)   # parent (e.g. ckpt/) stays on JuiceFS
        if dst.is_symlink():
            if os.path.realpath(str(dst)) != os.path.realpath(str(local_target)):
                dst.unlink(); dst.symlink_to(local_target)
        elif dst.exists():
            shutil.rmtree(dst); dst.symlink_to(local_target)
        else:
            dst.symlink_to(local_target)
        print(f"[LOCAL-TEMP] {subpath} -> {local_target} (off JuiceFS)", flush=True)

    _redirect_local("temp_data")                # .pt + intermediate json (biggest)
    _redirect_local("rollouts")                 # screenshots + traj.jsonl, cleared each step
    _redirect_local("ckpt/" + cfg.model.optimized_name)   # 17GB/step optimized ckpt (overwritten)
    (project_dir / "results").mkdir(parents=True, exist_ok=True)   # tiny jsonl — keep on JuiceFS
    Path(TRAIN_ROLLOUT_DIR).mkdir(parents=True, exist_ok=True)

    # --- wandb init (graceful: skipped when entity is null) ---
    _wandb = None
    _wandb_run_id = None
    try:
        import wandb as _wb
        _wb_entity  = getattr(cfg.wandb, "entity", None) if hasattr(cfg, "wandb") else None
        _wb_project = getattr(cfg.wandb, "project", project) if hasattr(cfg, "wandb") else project
        if _wb_entity:
            _wandb_run_id = getattr(cfg.wandb, "run_id", None) or None
            # GPU stats: sample every 10 s so we get fine-grained utilization curves.
            # Falls back silently if the Settings kwarg doesn't exist in older wandb.
            try:
                _wb_settings = _wb.Settings(x_stats_sampling_interval=10)
            except Exception:
                _wb_settings = None
            _wb_init_kwargs = dict(
                project=_wb_project,
                entity=_wb_entity,
                name=project,
                id=_wandb_run_id or _wb.util.generate_id(),
                resume="allow",
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            if _wb_settings is not None:
                _wb_init_kwargs["settings"] = _wb_settings
            _wandb = _wb.init(**_wb_init_kwargs)
            _wandb_run_id = _wandb.id
            print(f"[WANDB] run_id={_wandb_run_id} project={_wb_project} entity={_wb_entity}")
            # Multi-axis logging. The train subprocess resumes THIS run and logs
            # train/* at its own (rl_step-1)*10000+window scale; this process logs
            # per-rl-step metrics. That "worked" only because wandb's step
            # monotonicity is enforced per client process — but it parks train/*
            # at x=10000+ (unviewable next to per-step metrics) and silently relies
            # on the two clients' counters never colliding. Fix: NO log call passes
            # `step=`; every payload carries its own x field, bound per-namespace
            # here, so all charts share the rl_step axis.
            try:
                _wandb.define_metric("rl_step")
                _wandb.define_metric("train_global_step")
                for _ns in ("rollout/*", "monitor/*", "train_agg/*",
                            "eval_held_out/*", "monitor_greedy/*", "eval/*"):
                    _wandb.define_metric(_ns, step_metric="rl_step")
                _wandb.define_metric("train/*", step_metric="train_global_step")
            except Exception as _dm_err:
                print(f"[WANDB] define_metric failed ({_dm_err}); charts may need manual x-axis")
    except Exception as _wb_err:
        print(f"[WANDB] init failed ({_wb_err}), continuing without wandb")
        _wandb = None

    def env_prefix(env) -> str:
        proxy = getattr(cfg.system, "download_proxy", None) or ""
        proxy_export = f"export download_proxy={proxy} && export https_proxy={proxy} && export HTTPS_PROXY={proxy} && " if proxy else ""
        return (
            "source ~/.bashrc && "
            f"source activate {env} && "
            "export PATH=$CONDA_PREFIX/bin:$PATH && "
            + proxy_export
        )


    def init_hosts(worker_hosts, init_bash: str):

        procs = []
        for idx, host in enumerate(worker_hosts):
            if idx == 0:
                procs.append(run_local_async(init_bash))
            else:
                procs.append(run_remote_async(host, init_bash))
        for p in procs:
            p.wait()

    def start_serve(worker_hosts, cfg, type, model, policy_model_type, served_model_name=None):

        osw = "OSWorld-main"
        if type == "train":
            per   = int(cfg.rollout.policy.num_gpu_per_model)
        else:
            per   = int(cfg.rollout.policy_evaluation.num_gpu_per_model)

        if policy_model_type == "qwen3vl":
            script_name = "start_8gpus_qwen3vl.sh"
        elif policy_model_type == "uitars15":
            script_name = "start_8gpus_uitars15.sh"
        else:
            script_name = "start_8gpus_opencua.sh"

        smn_env = ""
        if served_model_name:
            smn_env = f"SERVED_MODEL_NAME={shlex.quote(str(served_model_name))} "

        procs = []
        for idx, host in enumerate(worker_hosts):
            body = (
                f"cd {BASE_DIR} && "
                f"cd {osw} && "
                f"chmod +x {script_name} && "
                f"MODEL={shlex.quote(str(model))} "
                f"NUM_GPU_PER_MODEL={per} "
                f"{smn_env}"
                f"./{script_name}"
            )
            full_cmd = env_prefix(env_name) + body
            if idx == 0:
                procs.append(run_local_async(full_cmd))
            else:
                procs.append(run_remote_async(host, full_cmd))
        for p in procs:
            p.wait()
        return all(p.returncode == 0 for p in procs)

    def force_kill_vllm_zombies():
        """Kill all zombie vLLM / EngineCore processes and wait for CUDA memory to release."""
        import subprocess as _sp
        print("[vllm-cleanup] Killing zombie vLLM and EngineCore processes...", flush=True)
        for pattern in ["vllm serve", "VLLM::EngineCore", "vllm.engine", "ray::IDLE"]:
            _sp.run(f"pkill -9 -f {shlex.quote(pattern)} 2>/dev/null || true", shell=True)
        # Also nuke any process still holding port 8000-8007 just in case
        _sp.run(
            "for p in $(seq 8000 8007); do "
            "  pids=$(ss -tlnp 2>/dev/null | awk -v p=\":${p} \" '$0~p{while(match($0,/pid=([0-9]+)/,a)){print a[1];$0=substr($0,RSTART+RLENGTH)}}' | sort -u); "
            "  for pid in $pids; do kill -9 $pid 2>/dev/null || true; done; "
            "done",
            shell=True,
        )
        print("[vllm-cleanup] Sleeping 15s for GPU memory to fully release...", flush=True)
        import time as _t; _t.sleep(15)
        print("[vllm-cleanup] Done.", flush=True)

    def run_sample(worker_hosts, cfg, type, model, policy_model_type, current_step=None, serve_model_override=None):

        MAX_START_RETRIES = 3
        for attempt in range(1, MAX_START_RETRIES + 1):
            if serve_model_override:
                ok = start_serve(worker_hosts, cfg, type, serve_model_override, policy_model_type, served_model_name=model)
            else:
                ok = start_serve(worker_hosts, cfg, type, model, policy_model_type)
            if ok:
                break
            print(f"[run_sample] start_serve failed (attempt {attempt}/{MAX_START_RETRIES}), running zombie cleanup...", flush=True)
            force_kill_vllm_zombies()
            if attempt == MAX_START_RETRIES:
                raise RuntimeError(f"[run_sample] vLLM failed to start after {MAX_START_RETRIES} attempts — aborting rollout")

        osw = "OSWorld-main"
        num_node = int(cfg.experiment.num_node)

        if type == "train":
            max_steps = str(cfg.rollout.policy.max_steps)
            num_env = str(cfg.rollout.num_envs)
            num_rollout_per_trial = str(cfg.rollout.policy.num_rollout_per_trial)
            temperature = str(cfg.rollout.policy.temperature)
            num_trial = str(cfg.rollout.policy.num_trial)
            max_image_history_length = str(getattr(cfg.rollout.policy, "max_image_history_length", 4))
            # On-policy sampling alignment: GRPO assumes rollouts come from the raw
            # policy. Defaults preserve legacy behavior (0.9/1.05) when unset; newer
            # configs set top_p=1.0, repetition_penalty=1.0 for train rollouts.
            top_p = str(getattr(cfg.rollout.policy, "top_p", 0.9))
            repetition_penalty = str(getattr(cfg.rollout.policy, "repetition_penalty", 1.05))
            environment_data_dir = cfg.dataset.train.environment_data_dir
            domain = cfg.dataset.train.domain
            example = cfg.dataset.train.example
            # Layer-3 fix: roll out the entire (curated) pool every step for directional
            # gradients. getattr default False => existing configs keep legacy sampling.
            full_coverage = bool(getattr(cfg.rollout.policy, "full_coverage", False))
        else:
            max_steps = str(cfg.rollout.policy_evaluation.max_steps)
            num_env = str(cfg.rollout.num_envs)
            num_rollout_per_trial = str(cfg.rollout.policy_evaluation.num_rollout_per_trial)
            temperature = str(cfg.rollout.policy_evaluation.temperature)
            num_trial = str(cfg.rollout.policy_evaluation.num_trial)
            max_image_history_length = str(getattr(cfg.rollout.policy_evaluation, "max_image_history_length", 4))
            # Eval keeps the anti-repetition guard (greedy decoding) unless overridden.
            top_p = str(getattr(cfg.rollout.policy_evaluation, "top_p", 0.9))
            repetition_penalty = str(getattr(cfg.rollout.policy_evaluation, "repetition_penalty", 1.05))
            environment_data_dir = cfg.dataset.evaluation.environment_data_dir
            domain = cfg.dataset.evaluation.domain
            example = cfg.dataset.evaluation.example

        if policy_model_type != "qwen3vl":
            raise ValueError(
                f"Only policy_model_type='qwen3vl' is supported in this release "
                f"(got {policy_model_type!r})."
            )
        script_name = "rl_rollout_local_qwen3vl.py"
        
        # coevolveenv=TRUE expects evaluation_examples/{project}/train/{domain}/{task}.json
        # which doesn't exist for B3/B4/B5 (static task set). Always FALSE.
        coevolve_environment = "FALSE"
        
        procs = []
        for idx, host in enumerate(worker_hosts):
            # Resolve provider and VM path from environment config (docker mode)
            _env_cfg = getattr(cfg, "environment", object())
            _provider = str(getattr(_env_cfg, "type", "docker")).replace("local_docker", "docker")
            _vm_path  = str(getattr(_env_cfg, "vm_path", "docker_vm_data/Ubuntu.qcow2"))

            args = [
                "--headless",
                "--observation_type", "screenshot",
                "--model",            shlex.quote(str(model)),
                "--result_dir",       shlex.quote(TRAIN_ROLLOUT_DIR),
                "--test_all_meta_path", f"evaluation_examples/{environment_data_dir}.json",
                "--max_steps",        max_steps,
                "--num_envs",         num_env,
                "--provider_name",    _provider,
                "--path_to_vm",       shlex.quote(_vm_path),
                "--coordinate_type",  str(cfg.rollout.coordinate_type),
                "--num_rollout_per_trial", num_rollout_per_trial,
                "--domain",           shlex.quote(str(domain)),
                "--example",          shlex.quote(str(example)),
                "--temperature",      temperature,
                "--top_p",            top_p,
                "--repetition_penalty", repetition_penalty,
                "--max_tokens",       "8192",
                "--max_image_history_length", max_image_history_length,
                "--use_old_sys_prompt",
                "--num_node",         str(num_node),
                "--node_index",       str(idx),
                "--rollout_type",     shlex.quote(str(type)),
                "--num_trial",        num_trial,
                "--save_example_json",str(f"{RUN_DIR}/temp_data/example_node_{idx}.json"),
                "--action_space",     str(cfg.rollout.action_space),
                "--project",          str(cfg.experiment.project),
                "--observation_type", str(cfg.rollout.observation_type),
                "--coevolveenv",      str(coevolve_environment),
                "--current_step",     str(current_step if current_step is not None else cfg.experiment.current_epoch),
                "--job_watchdog_minutes", "60" if type == "evaluation" else "30",
                # context_policy lives at the TOP level of the config (not under
                # cfg.rollout). Must be threaded into BOTH train rollout and held-out
                # eval or train/test observation distributions diverge (B5 lesson).
                # Default "baseline" keeps every existing run byte-for-byte identical.
                "--context_policy",   str(getattr(cfg, "context_policy", "baseline")),
                "--max_consecutive_skips", str(getattr(cfg, "max_consecutive_skips", 2)),
            ]
            if bool(cfg.experiment.if_rerun):
                args.append("--rerun")
            if type == "train" and full_coverage:
                args.append("--full_coverage")

            body = (
                f"cd {BASE_DIR} && " +
                f"cd {osw} && " +
                "DOCKER_VNC_PORT_START=10006 " +
                f"python {script_name} " +
                " ".join(args)
            )
            full_cmd = env_prefix(env_name) + body
            if idx == 0:
                procs.append(run_local_async(full_cmd))
            else:
                procs.append(run_remote_async(host, full_cmd))
        for p in procs:
            p.wait()
        
        stop_serve(worker_hosts)

    def stop_serve(worker_hosts):

        osw = "OSWorld-main"
        procs = []
        for idx, host in enumerate(worker_hosts):
            body = (
                f"cd {BASE_DIR} && "
                f"cd {osw} && "
                "chmod +x stop_8gpus.sh || true && "
                "./stop_8gpus.sh || true"
            )
            full_cmd = env_prefix(env_name) + body
            if idx == 0:
                procs.append(run_local_async(full_cmd))
            else:
                procs.append(run_remote_async(host, full_cmd))
        for p in procs:
            p.wait()
    

    def run_held_out_eval(worker_hosts, cfg, step, model):
        """Run rollout on fixed held-out task set and log per-domain results."""
        held_out_meta = getattr(getattr(cfg, "dataset", None), "held_out_meta", "held_out_v2_eval")
        held_out_json = os.path.join(BASE_DIR, "OSWorld-main", "evaluation_examples", f"{held_out_meta}.json")
        if not os.path.isfile(held_out_json):
            print(f"[HELD-OUT] {held_out_json} not found, skipping")
            return

        with open(held_out_json) as _f:
            held_out_tasks = json.load(_f)
        n_tasks = sum(len(v) for v in held_out_tasks.items() if isinstance(v, list)) if isinstance(held_out_tasks, dict) else 0
        # correct count
        n_tasks = sum(len(v) for v in held_out_tasks.values())
        print(f"[HELD-OUT] step={step}, running {n_tasks} fixed tasks")

        osw = "OSWorld-main"
        num_node = int(cfg.experiment.num_node)
        _env_cfg = getattr(cfg, "environment", object())
        _provider = str(getattr(_env_cfg, "type", "docker")).replace("local_docker", "docker")
        _vm_path  = str(getattr(_env_cfg, "vm_path", "docker_vm_data/Ubuntu.qcow2"))

        policy_model_type = cfg.model.policy_model_type
        if policy_model_type != "qwen3vl":
            raise ValueError(
                f"Only policy_model_type='qwen3vl' is supported in this release "
                f"(got {policy_model_type!r})."
            )
        script_name = "rl_rollout_local_qwen3vl.py"

        procs = []
        for idx, host in enumerate(worker_hosts):
            args = [
                "--headless",
                "--observation_type",    "screenshot",
                "--model",               shlex.quote(str(model)),
                "--result_dir",          shlex.quote(eval_rollout_dir(step)),
                "--test_all_meta_path",  f"evaluation_examples/{held_out_meta}.json",
                "--max_steps",           str(cfg.rollout.policy_evaluation.max_steps),
                "--num_envs",            str(cfg.rollout.num_envs),
                "--provider_name",       _provider,
                "--path_to_vm",          shlex.quote(_vm_path),
                "--coordinate_type",     str(cfg.rollout.coordinate_type),
                # Was hardcoded "3"; at temp=0 repeated rollouts are near-identical,
                # so honor policy_evaluation.num_rollout_per_trial (default 3 keeps
                # old behavior for configs that don't set it).
                "--num_rollout_per_trial", str(getattr(cfg.rollout.policy_evaluation, "num_rollout_per_trial", 3)),
                "--domain",              "all",
                "--example",             "all",
                "--temperature",         "0.0",   # greedy for eval, matches B2 baseline
                "--max_tokens",          "8192",
                "--max_image_history_length", str(getattr(cfg.rollout.policy, "max_image_history_length", 4)),
                "--use_old_sys_prompt",
                "--num_node",            str(num_node),
                "--node_index",          str(idx),
                "--rollout_type",        "evaluation",
                "--num_trial",           str(n_tasks * 2),  # > n_tasks ensures all selected
                "--action_space",        str(cfg.rollout.action_space),
                "--project",             str(cfg.experiment.project),
                "--coevolveenv",         "FALSE",
                "--current_step",        str(step),
                # Match the train rollout's context policy so held-out eval observes
                # the same distribution the policy is trained under (B5 lesson).
                "--context_policy",      str(getattr(cfg, "context_policy", "baseline")),
                "--max_consecutive_skips", str(getattr(cfg, "max_consecutive_skips", 2)),
                "--rerun",
            ]
            body = (
                f"cd {BASE_DIR} && "
                f"cd {osw} && "
                "DOCKER_VNC_PORT_START=10006 " +
                f"python {script_name} " +
                " ".join(args)
            )
            full_cmd = env_prefix(env_name) + body
            if idx == 0:
                procs.append(run_local_async(full_cmd))
            else:
                procs.append(run_remote_async(host, full_cmd))
        for p in procs:
            p.wait()
        stop_serve(worker_hosts)

        # collect results by reading individual result.txt files (same structure as get_result())
        result_dir = eval_rollout_dir(step)
        model_tag = re.sub(r'[\\/]+', '_', model).lstrip('_')
        target_dir = os.path.join(
            result_dir, cfg.rollout.action_space, cfg.rollout.observation_type, model_tag
        )
        if not os.path.isdir(target_dir):
            print(f"[HELD-OUT] no result dir found: {target_dir}, skipping log")
            return

        domain_success: dict = {}
        for domain in os.listdir(target_dir):
            domain_path = os.path.join(target_dir, domain)
            if not os.path.isdir(domain_path):
                continue
            for task_id in os.listdir(domain_path):
                task_path = os.path.join(domain_path, task_id)
                if not os.path.isdir(task_path):
                    continue
                ds = domain_success.setdefault(domain, {"success": 0, "total": 0})
                # result.txt directly in task dir (single-trial)
                result_file = os.path.join(task_path, "result.txt")
                if os.path.exists(result_file):
                    ds["total"] += 1
                    try:
                        if float(open(result_file).read().strip()) > 0:
                            ds["success"] += 1
                    except Exception:
                        pass
                else:
                    # result.txt in sub-trial dirs (multi-trial: k=0, k=1, ...)
                    try:
                        for sub in os.listdir(task_path):
                            sub_path = os.path.join(task_path, sub)
                            if not os.path.isdir(sub_path):
                                continue
                            result_file2 = os.path.join(sub_path, "result.txt")
                            if os.path.exists(result_file2):
                                ds["total"] += 1
                                try:
                                    if float(open(result_file2).read().strip()) > 0:
                                        ds["success"] += 1
                                except Exception:
                                    pass
                    except Exception:
                        pass

        if not domain_success:
            print(f"[HELD-OUT] no result.txt files found under {target_dir}, skipping log")
            return

        # Greedy-degeneration monitor: at temp=0 the model can collapse into
        # repeated-action spinning that only surfaces at eval. Aggregate per-step
        # response lengths and the fraction of trajectories that hit the step cap so
        # the collapse is visible (and an early warning) before outcome craters.
        _eval_max_steps = int(cfg.rollout.policy_evaluation.max_steps)
        _resp_lens: list = []
        _traj_steps: list = []
        _traj_outcomes: list = []
        _n_traj = 0
        _n_cap = 0
        for _tj in glob.glob(os.path.join(target_dir, "*", "*", "**", "traj.jsonl"), recursive=True):
            try:
                _lines = open(_tj).read().splitlines()
            except Exception:
                continue
            if not _lines:
                continue
            _n_traj += 1
            traj_len = len(_lines)
            _traj_steps.append(traj_len)
            if traj_len >= _eval_max_steps:
                _n_cap += 1
            # Read result.txt from same directory to get success/failure status
            result_file = os.path.join(os.path.dirname(_tj), "result.txt")
            try:
                with open(result_file, "r") as _rf:
                    result_val = float(_rf.read().strip())
                    outcome = 1 if result_val > 0 else -1
                    _traj_outcomes.append(outcome)
            except Exception:
                _traj_outcomes.append(None)
            for _ln in _lines:
                try:
                    _resp = json.loads(_ln).get("response", "") or ""
                    _resp_lens.append(get_token_length(_resp))  # Token-level length
                except Exception:
                    pass

        try:
            overall_succ = sum(v["success"] for v in domain_success.values())
            overall_total = sum(v["total"] for v in domain_success.values())

            # Compute avg steps for successful vs failed trajectories
            steps_success = [_traj_steps[i] for i in range(len(_traj_steps)) if _traj_outcomes[i] == 1]
            steps_fail = [_traj_steps[i] for i in range(len(_traj_steps)) if _traj_outcomes[i] == -1]

            record = {
                "rl_step": step,
                "overall_outcome_rate": round(overall_succ / overall_total, 4) if overall_total > 0 else 0.0,
                "domain_outcome_rates": {d: round(v["success"]/v["total"], 4) if v["total"] > 0 else 0.0
                                         for d, v in domain_success.items()},
                "n_tasks": overall_total,
                "resp_len_mean": round(sum(_resp_lens) / len(_resp_lens), 1) if _resp_lens else 0.0,
                "resp_len_max": max(_resp_lens) if _resp_lens else 0,
                "frac_hit_cap": round(_n_cap / _n_traj, 4) if _n_traj > 0 else 0.0,
                "avg_steps_success": round(sum(steps_success) / len(steps_success), 1) if steps_success else 0.0,
                "avg_steps_fail": round(sum(steps_fail) / len(steps_fail), 1) if steps_fail else 0.0,
            }
            stats_log = os.path.join(RUN_DIR, "results", "held-out-eval.jsonl")
            os.makedirs(os.path.dirname(stats_log), exist_ok=True)
            with open(stats_log, "a") as _sf:
                _sf.write(json.dumps(record) + "\n")
            print(f"[HELD-OUT] step={step} n={overall_total} overall={record['overall_outcome_rate']:.1%} per_domain={record['domain_outcome_rates']} → {stats_log}")
        except Exception as e:
            print(f"[HELD-OUT] failed to parse results: {e}")

    def run_reward_sample_api(cfg, model):
        """API-based reward scoring (replaces vLLM run_reward_sample)."""
        model_tag = re.sub(r'[\\/]+', '_', model).lstrip('_')
        num_node = int(cfg.experiment.num_node)

        reward_cfg = getattr(cfg, "reward", OmegaConf.create({}))
        m_samples = int(getattr(reward_cfg, "m_reward_samples", getattr(cfg.rollout.reward, "num_rollout_per_query", 3)))
        concurrency = int(getattr(reward_cfg, "api_concurrency", 20))
        step_budget = float(getattr(reward_cfg, "step_budget", 30.0))
        total_budget = float(getattr(reward_cfg, "total_budget", 5000.0))

        for idx in range(num_node):
            root_dir = (
                f"{TRAIN_ROLLOUT_DIR}/"
                f"{cfg.rollout.action_space}/{cfg.rollout.observation_type}/{model_tag}"
            )
            examples_json = f"{RUN_DIR}/temp_data/example_node_{idx}.json"
            output_json = f"{RUN_DIR}/temp_data/reward_rollout_results_node_{idx}.json"

            cmd_args = [
                "--root-dir", shlex.quote(root_dir),
                "--examples-json", shlex.quote(examples_json),
                "--output-json", shlex.quote(output_json),
                "--num-rollout-per-query", str(m_samples),
                "--concurrency", str(concurrency),
                "--step-budget", str(step_budget),
                "--total-budget", str(total_budget),
            ]
            body = (
                f"cd {BASE_DIR} && "
                f"cd sample && "
                f"python osworld_reward_rollout_api.py " +
                " ".join(cmd_args)
            )
            full_cmd = env_prefix(env_name) + body
            run_local(full_cmd)
    
    
    def reward(step, cfg, type, model):
        project = cfg.experiment.project
        num_node = int(cfg.experiment.num_node)
        model_tag = re.sub(r'[\\/]+', '_', model).lstrip('_')
        root_dir = (
            f"{TRAIN_ROLLOUT_DIR}/"
            f"{cfg.rollout.action_space}/"
            f"{cfg.rollout.observation_type}/"
            f"{model_tag}"
        )
        merge_dir = f"{RUN_DIR}/temp_data"

        reward_cfg = getattr(cfg, "reward", OmegaConf.create({}))
        lambda_coef = float(getattr(reward_cfg, "lambda_coef", 0.3))
        eta_initial = float(getattr(reward_cfg, "eta_initial", 0.02))
        eta_decay_steps = int(getattr(reward_cfg, "eta_decay_steps", 30))
        positive_adv_only = bool(getattr(reward_cfg, "positive_advantage_only", False))
        length_penalty_coef = float(getattr(reward_cfg, "length_penalty_coef", 0.0))
        max_steps_ref = int(getattr(reward_cfg, "max_steps_ref", cfg.rollout.policy.max_steps))
        step_fail_pen = float(getattr(reward_cfg, "step_fail_pen", 0.0))
        repeat_pen = float(getattr(reward_cfg, "repeat_pen", 0.0))
        cap_penalty = float(getattr(reward_cfg, "cap_penalty", 0.0))
        terminate_bonus = float(getattr(reward_cfg, "terminate_bonus", 0.0))
        terminate_bonus_efficient = float(getattr(reward_cfg, "terminate_bonus_efficient", 0.0))
        terminate_bonus_slow = float(getattr(reward_cfg, "terminate_bonus_slow", 0.0))
        early_quit_pen = float(getattr(reward_cfg, "early_quit_pen", 0.0))
        graceful_fail_bonus = float(getattr(reward_cfg, "graceful_fail_bonus", 0.0))
        false_success_pen = float(getattr(reward_cfg, "false_success_pen", 0.0))
        efficient_ratio = float(getattr(reward_cfg, "efficient_ratio", 0.7))
        early_quit_ratio = float(getattr(reward_cfg, "early_quit_ratio", 0.5))
        term_amortize_mode = str(getattr(reward_cfg, "term_amortize_mode", "per_step"))
        no_progress_pen = float(getattr(reward_cfg, "no_progress_pen", 0.0))
        # Format reward (A3) + safe MCP tool-use bonus — default off.
        format_pen = float(getattr(reward_cfg, "format_pen", 0.0))
        mcp_tool_bonus = float(getattr(reward_cfg, "mcp_tool_bonus", 0.0))
        mcp_tool_bonus_decay_steps = int(getattr(reward_cfg, "mcp_tool_bonus_decay_steps", 30))
        mcp_tool_bonus_outcome_gated = bool(getattr(reward_cfg, "mcp_tool_bonus_outcome_gated", True))
        mcp_tool_bonus_dense = bool(getattr(reward_cfg, "mcp_tool_bonus_dense", False))
        false_success_keep_neg = bool(getattr(reward_cfg, "false_success_keep_neg", False))
        random_reward = bool(getattr(reward_cfg, "random_reward", False))
        normalize_mode = str(getattr(reward_cfg, "normalize_mode", "groupwise"))  # footgun fix: was "stepwise"; live runs all use groupwise
        # Filter out fully-0% and fully-100% groups BEFORE z-score standardization.
        # With argparse defaults (0.0 / 1.0) the filter was a no-op, so all-fail and
        # all-success groups flowed in, got σ=0, were zeroed by _stepwise_standardize,
        # then dropped by the `r == 0.0` filter — silently removing exactly the
        # samples we most needed gradients on. Default 0.05/0.95 keeps boundary
        # groups out; override in YAML if you want to widen.
        acc_lower = float(getattr(reward_cfg, "acc_lower", 0.05))
        acc_upper = float(getattr(reward_cfg, "acc_upper", 0.95))

        full_cmd = env_prefix(env_name) + (
            f"cd {BASE_DIR}/reward && "
            f"python osworld_rl_reward.py "
            f"--root-dir {shlex.quote(root_dir)} "
            f"--num-nodes {num_node} "
            f"--type {type} "
            f"--step {step} "
            f"--merge-dir {shlex.quote(merge_dir)} "
            f"--lambda-coef {lambda_coef} "
            f"--eta-initial {eta_initial} "
            f"--eta-decay-steps {eta_decay_steps} "
            f"--length-penalty-coef {length_penalty_coef} "
            f"--max-steps-ref {max_steps_ref} "
            f"--step-fail-pen {step_fail_pen} "
            f"--repeat-pen {repeat_pen} "
            f"--cap-penalty {cap_penalty} "
            f"--terminate-bonus {terminate_bonus} "
            f"--terminate-bonus-efficient {terminate_bonus_efficient} "
            f"--terminate-bonus-slow {terminate_bonus_slow} "
            f"--early-quit-pen {early_quit_pen} "
            f"--graceful-fail-bonus {graceful_fail_bonus} "
            f"--false-success-pen {false_success_pen} "
            f"--efficient-ratio {efficient_ratio} "
            f"--early-quit-ratio {early_quit_ratio} "
            f"--term-amortize-mode {term_amortize_mode} "
            f"--no-progress-pen {no_progress_pen} "
            f"--format-pen {format_pen} "
            f"--mcp-tool-bonus {mcp_tool_bonus} "
            f"--mcp-tool-bonus-decay-steps {mcp_tool_bonus_decay_steps} "
            f"--acc-lower {acc_lower} "
            f"--acc-upper {acc_upper}"
            + (" --positive-advantage-only" if positive_adv_only else "")
            + (" --random-reward" if random_reward else "")
            + (" --mcp-tool-bonus-ungated" if not mcp_tool_bonus_outcome_gated else "")
            + (" --mcp-tool-bonus-dense" if mcp_tool_bonus_dense else "")
            + (" --false-success-keep-neg" if false_success_keep_neg else "")
            + f" --normalize-mode {normalize_mode}"
        )

        run_local(full_cmd)
    

    def process_shards(worker_hosts, epoch, cfg, model_type, target):

        project = cfg.experiment.project
        num_nodes = len(worker_hosts)

        procs = []
        for idx, host in enumerate(worker_hosts):
            body = (
                f"cd {BASE_DIR} && "
                "export DS_SKIP_CUDA_CHECK=1 && "
                "python -m train.osworld_vlm_preprocess_shards "
                f"config=configs/experiments/{project}.yaml "
                f"training.target={target} "
                f"experiment.current_epoch={epoch} "
                f"model_type={model_type} "
                "cast_pixel_fp16=1 "
                "pad_to_fixed=1 "
                f"dataset.num_nodes={num_nodes} "
                f"dataset.node_rank={idx} "
            )

            full_cmd = env_prefix(env_name) + body

            if idx == 0:
                procs.append(run_local_async(full_cmd))
            else:
                procs.append(run_remote_async(host, full_cmd))

            print(f"[DISPATCH] preprocess node {idx} → {host}")

        for p in procs:
            p.wait()

        print("All preprocess nodes finished.")
    
    def train_agg(cfg, model, model_type, epoch, target):
        project = cfg.experiment.project
        num_node = int(cfg.experiment.num_node)
        model_tag = re.sub(r'[\\/]+', '_', model).lstrip('_')
        merge_dir = f"{RUN_DIR}/temp_data"
        full_cmd = env_prefix(env_name) + (
            f"cd {BASE_DIR} && "
            f"python -m train.osworld_vlm_merge_preproc_shards "
            f"config=configs/experiments/{project}.yaml "
            f'training.target={target} '
            f'model_type={model_type} '
            f"experiment.current_epoch={epoch}"
        )
        run_local(full_cmd)

    def train(worker_hosts, epoch, cfg, model_type, target = None):

        _t = time.time()
        process_shards(worker_hosts, epoch, cfg, model_type, target)
        print(f"[TIMER] epoch={epoch} preprocess={time.time()-_t:.1f}s")

        # Guard: if preprocess produced no shards (e.g. n_trajs=0), skip merge+train silently.
        _shard_dir = os.path.join(RUN_DIR, "temp_data")
        _target_str = target or "policy"
        _shard_files = [f for f in os.listdir(_shard_dir)
                        if f.startswith(f"{_target_str}_optimization_data") and f.endswith(".pt")
                        and "merged" not in f] if os.path.isdir(_shard_dir) else []
        if not _shard_files:
            print(f"[SKIP] epoch={epoch} no shard files found in {_shard_dir}, skipping merge+train (n_trajs=0?)", flush=True)
            return

        _t = time.time()
        train_agg(cfg, model, model_type, epoch, target)
        print(f"[TIMER] epoch={epoch} merge_shards={time.time()-_t:.1f}s")

        project = cfg.experiment.project
        merged_pt = os.path.join(RUN_DIR, "temp_data", "policy_optimization_data_preproc_merged.pt")
        if not os.path.isfile(merged_pt):
            print(f"[SKIP] epoch={epoch} no training data (merged .pt missing), skipping accelerate launch.", flush=True)
            return

        ds_file = cfg.experiment.deepspeed_file
        num_nodes = len(worker_hosts)
        # Fall back to localhost / a free port when not running on a managed
        # multi-node platform (e.g. local single-machine setup).
        master_ip   = os.environ.get("MLP_WORKER_0_HOST", "127.0.0.1")
        master_port = os.environ.get("MLP_WORKER_0_PORT", "29500")
        procs = []
        for idx, host in enumerate(worker_hosts):
            body = (
                f"cd {BASE_DIR} && "
                "export DS_SKIP_CUDA_CHECK=1 && "
                "accelerate launch "
                f"--num_machines {num_nodes} "
                f"--machine_rank {idx} "
                f"--main_process_ip {master_ip} "
                f"--main_process_port {master_port} "
                f"--config_file accelerate_configs/{ds_file}.yaml "
                f"train/osworld_train.py "
                f"config=configs/experiments/{project}.yaml "
                f'training.target={target} '
                f'model_type={model_type} '
                f"experiment.current_epoch={epoch}"
                + (f" wandb.run_id={_wandb_run_id}" if _wandb_run_id else "")
            )
            if target == "policy":
                full_cmd = env_prefix(env_name) + body
            else:
                full_cmd = env_prefix(env_name) + body
            if idx == 0:
                procs.append(run_local_async(full_cmd))
            else:
                procs.append(run_remote_async(host, full_cmd))
            print(f"[DISPATCH] train node {idx} → {host}")
        _t = time.time()
        for p in procs:
            p.wait()
        print(f"[TIMER] epoch={epoch} training={time.time()-_t:.1f}s")
        print("All train nodes finished.")



    # recording.mp4 per task run is human-replay-only (~2 MB each; heavy IO at 96
    # envs). lib_run_single and rl_rollout_local_* gate start/end_recording on this
    # var, and locally-spawned rollout/eval subprocesses inherit it. setdefault so
    # an explicit OSWORLD_DISABLE_RECORDING=0 in the shell re-enables recording
    # when a trajectory needs visual debugging.
    os.environ.setdefault("OSWORLD_DISABLE_RECORDING", "1")

    TMPFS_CKPT = "/dev/shm/vllm_ckpt"

    def copy_ckpt_to_tmpfs(src_ckpt_dir):
        """Copy checkpoint to tmpfs for fast vLLM loading (avoids NFS I/O)."""
        if os.path.exists(TMPFS_CKPT):
            shutil.rmtree(TMPFS_CKPT)
        shutil.copytree(src_ckpt_dir, TMPFS_CKPT)
        print(f"[TMPFS] Copied {src_ckpt_dir} → {TMPFS_CKPT}")
        return TMPFS_CKPT

    def clear_dir(dir_to_clean):
        p = Path(dir_to_clean)
        p.mkdir(parents=True, exist_ok=True)  

        for child in p.iterdir():             
            if child.is_file() or child.is_symlink():
                child.unlink()
            else:
                shutil.rmtree(child)
    
    def clear_results_if_start_from_scratch(cfg):

        if not getattr(cfg.experiment, "start_from_scratch", False):
            return
        # Only clear when truly starting from the beginning (step 1)
        if getattr(cfg.experiment, "current_epoch", 1) > 1:
            return
        # Crash-recovery resume: resume_step.txt means this launch continues an
        # existing run (see step resolution above) — keep accumulated results.
        # Without this guard a crash-resume truncated rl-results.txt.
        if os.path.exists(os.path.join(RUN_DIR, "ckpt", "resume_step.txt")):
            return

        results_dir = f"{RUN_DIR}/results"
        clear_dir(f"{RUN_DIR}/temp_data")
        for name in ("rl-results.txt", "eval-results.txt"):
            path = os.path.join(results_dir, name)
            if os.path.exists(path):
                begin_with(path)  

    INIT_BASH = make_init_bash(cfg)

    # Docker mode: DesktopEnv starts/destroys containers on demand; we only need to determine worker_hosts here.
    num_node = int(cfg.experiment.num_node)
    env_type = getattr(getattr(cfg, "environment", object()), "type", "docker")

    if env_type in ("docker", "local_docker"):
        # All rollouts run on this host; DesktopEnv owns the container lifecycle
        worker_hosts = ["localhost"]
    else:
        worker_hosts = [os.environ[f"MLP_WORKER_{i}_HOST"] for i in range(num_node)]
        import time; time.sleep(30)
        init_hosts(worker_hosts, INIT_BASH)
        import time; time.sleep(10)

    step = cfg.experiment.current_epoch
    # Crash recovery: resume from last successfully completed step
    _resume_file = os.path.join(RUN_DIR, "ckpt", "resume_step.txt")
    try:
        with open(_resume_file) as _rf:
            _saved_step = int(_rf.read().strip())
        if _saved_step >= step:
            step = _saved_step + 1
            print(f"[RESUME] Resuming from step {step} (last completed: {_saved_step})")
    except Exception:
        pass

    clear_results_if_start_from_scratch(cfg)

    # Hacking detector for monitoring (optional: if no monitoring/ package is
    # present, reward-hacking / parse-error monitoring is skipped and training
    # proceeds without it).
    sys.path.insert(0, BASE_DIR)
    try:
        from monitoring.hacking_detector import HackingDetector
        hacking_detector = HackingDetector()
    except ImportError:
        hacking_detector = None
    # Consecutive steps where parse_error_rate exceeded the threshold.
    # Three in a row → raise to halt training, because once format collapse
    # sets in (observed in early runs), the policy can't recover from
    # outcome-only signal and every later step bleeds training samples.
    _parse_error_threshold = 0.20
    _consecutive_parse_alarm = 0
    _max_consecutive_parse_alarm = 3

    # Step-0 held-out eval: run once with base model before any RL updates,
    # so we have a clean baseline on the same task set at the same temp=0 setting.
    # Skip only when previous run's step-0 is known-comparable (same base model,
    # same held_out_meta, same max_steps, same monitor_greedy field set).
    # Default true: cheap (~5-10 min on base model) and the only training-free
    # data point for the monitor_greedy/* series.
    _do_step0 = bool(getattr(cfg.experiment, "step0_held_out_eval", True))
    if step == 1 and getattr(cfg.experiment, "start_from_scratch", False) and _do_step0:
        print("[STEP-0] Running held-out eval with base model before first RL update...")
        # Must start vLLM first — run_sample hasn't been called yet so no server is up.
        _s0_serve = copy_ckpt_to_tmpfs(cfg.model.policy_model)
        start_serve(worker_hosts, cfg, "train", _s0_serve, cfg.model.policy_model_type,
                    served_model_name=cfg.model.policy_model)
        run_held_out_eval(worker_hosts, cfg, 0, cfg.model.policy_model)
        if _wandb:
            try:
                _held_log = os.path.join(RUN_DIR, "results", "held-out-eval.jsonl")
                if os.path.isfile(_held_log):
                    _lines = [l for l in open(_held_log).readlines() if l.strip()]
                    if _lines:
                        _h = json.loads(_lines[-1])
                        _hw = {"eval_held_out/overall": _h.get("overall_outcome_rate", 0)}
                        for _d, _r in (_h.get("domain_outcome_rates") or {}).items():
                            _hw[f"eval_held_out/{_d}"] = _r
                        # Also log monitor_greedy metrics to see baseline greedy behavior
                        for _mk in ("resp_len_mean", "resp_len_max", "frac_hit_cap", "avg_steps_success", "avg_steps_fail"):
                            if _mk in _h:
                                _hw[f"monitor_greedy/{_mk}"] = _h[_mk]
                        _wandb.log({**_hw, "rl_step": 0})
            except Exception as _we:
                print(f"[WANDB] step-0 held-out log failed: {_we}")

    while step <= cfg.experiment.total_step:

        if step == 1:
            model = cfg.model.policy_model
        else:
            model = f"{RUN_DIR}/ckpt/{cfg.model.optimized_name}"
            if not os.path.isdir(model):
                base = cfg.model.policy_model
                print(f"[CKPT-FALLBACK] step={step}: optimized ckpt not found at {model}, reusing base model {base}")
                model = base

        _step_t0 = time.time()

        _t = time.time()
        serve_model = copy_ckpt_to_tmpfs(model)
        print(f"[TIMER] step={step} model_sync={time.time()-_t:.1f}s")

        cleanup_local_containers(cfg)
        clear_dir(TRAIN_ROLLOUT_DIR)
        _t = time.time()
        run_sample(worker_hosts, cfg, "train", model, cfg.model.policy_model_type, current_step=step, serve_model_override=serve_model)
        cleanup_local_containers(cfg)
        print(f"[TIMER] step={step} rollout={time.time()-_t:.1f}s")

        _reward_cfg = getattr(cfg, "reward", OmegaConf.create({}))
        # Default 0.0: the API-reward sampling path (sample/) is not part of
        # this release; enable only if you provide it and set reward.lambda_coef.
        _lambda_coef = float(getattr(_reward_cfg, "lambda_coef", 0.0))
        if _lambda_coef != 0.0:
            _t = time.time()
            run_reward_sample_api(cfg, model)
            print(f"[TIMER] step={step} reward_api={time.time()-_t:.1f}s")
        else:
            print(f"[INFO] lambda_coef=0, skipping reward API (outcome-only mode)")

        _t = time.time()
        reward(step, cfg, "train", model)
        print(f"[TIMER] step={step} reward_postproc={time.time()-_t:.1f}s")

        # Log rollout stats to wandb
        if _wandb:
            try:
                _stats_log = os.path.join(RUN_DIR, "results", "rollout-stats.jsonl")
                if os.path.isfile(_stats_log):
                    _rs_lines = [l for l in open(_stats_log).readlines() if l.strip()]
                    if _rs_lines:
                        _rs = json.loads(_rs_lines[-1])
                        _wm = {
                            # per-record (step-weighted) rate over kept tasks — kept for
                            # historical continuity; outcome_rate_traj is the primary one
                            "rollout/outcome_rate":      _rs.get("outcome_rate", 0),
                            # trajectory-level success rate over kept tasks
                            "rollout/outcome_rate_traj": _rs.get("outcome_rate_traj", 0),
                            # gradient concentration: tasks/trajs surviving the acc filter.
                            # e25 step4 trained on ~1 task — this curve catches that.
                            "rollout/n_tasks_kept":      _rs.get("n_tasks_kept", 0),
                            "rollout/n_trajs_kept":      _rs.get("n_trajs_kept", 0),
                            "rollout/n_train_samples":   _rs.get("n_train_samples", 0),
                            # never-terminate signals on the temp=1.0 train side
                            "rollout/traj_len_mean":     _rs.get("traj_len_mean", 0),
                            "rollout/avg_steps_success": _rs.get("avg_steps_success", 0),
                            "rollout/avg_steps_fail":    _rs.get("avg_steps_fail", 0),
                            "rollout/resp_len_mean":     _rs.get("resp_len_mean", 0),
                            "rollout/frac_hit_cap":      _rs.get("frac_hit_cap", 0),
                            # active shaping magnitudes (term_signal/noprog/step_pen series
                            # have been identically 0 since e19 — jsonl-only now)
                            "rollout/length_pen_mean":   _rs.get("length_pen_mean", 0),
                            "rollout/cap_pen_mean":      _rs.get("cap_pen_mean", 0),
                            # live terminate counters: efficient successes, and the base
                            # model's false-success (overconfidence) prior — watch trends
                            "rollout/n_term_efficient":     _rs.get("n_term_efficient", 0),
                            "rollout/n_term_false_success": _rs.get("n_term_false_success", 0),
                        }
                        # Per-domain TRAIN-side rates dropped from wandb: with 12 sampled
                        # tasks/step and the acc filter, domain denominators are 1–3 trajs
                        # — pure noise. Full numbers stay in rollout-stats.jsonl; the
                        # comparable per-domain series is eval_held_out/* (fixed task set).
                        _wandb.log({**_wm, "rl_step": step})
            except Exception as _we:
                print(f"[WANDB] rollout stats log failed: {_we}")

        # Run hacking detection on reward rollout results
        try:
            num_node = int(cfg.experiment.num_node) if hacking_detector else 0
            for nidx in range(num_node):
                rr_path = os.path.join(RUN_DIR, "temp_data", f"reward_rollout_results_node_{nidx}.json")
                if os.path.isfile(rr_path):
                    with open(rr_path) as _hf:
                        rr_data = json.load(_hf)
                    trajs = []
                    for item in rr_data:
                        for t in item.get("trajctory_list", []):
                            trajs.append(t)
                    if trajs:
                        metrics = hacking_detector.check(trajs)
                        print(f"[MONITOR] step {step} node {nidx}: TIR={metrics['tir']:.3f} repeat={metrics['repeat_call_rate']:.3f} read_only={metrics['read_only_rate']:.3f}")
        except Exception as e:
            print(f"[MONITOR] step {step}: hacking check failed ({e}), continuing")

        # Parse-error rate monitor — always runs (reads rl_rollout_results.json
        # which is produced by reward() regardless of API-reward usage). High
        # rate = policy lost the <tool_call> format and trajectories are being
        # silently dropped (observed in an early run as an n_records collapse
        # from 1600→50 between step 30 and 90).
        try:
            rl_path = os.path.join(RUN_DIR, "temp_data", "rl_rollout_results.json")
            if hacking_detector and os.path.isfile(rl_path):
                with open(rl_path) as _pf:
                    _rl_data = json.load(_pf)
                _all_trajs = []
                for _item in _rl_data:
                    for _t in _item.get("trajctory_list", []):
                        _all_trajs.append(_t)
                _pm = hacking_detector.check(_all_trajs)
                _per = _pm.get("parse_error_rate", 0.0)
                _ptr = _pm.get("parse_failed_traj_rate", 0.0)
                print(
                    f"[MONITOR] step {step}: parse_error_rate={_per:.3f} "
                    f"parse_failed_traj_rate={_ptr:.3f} "
                    f"n_trajs={_pm.get('n_trajs', 0)} "
                    f"n_parse_failed_trajs={_pm.get('n_parse_failed_trajs', 0)} "
                    f"n_total_steps={_pm.get('n_total_steps', 0)}"
                )
                if _wandb:
                    try:
                        _wandb.log({
                            "monitor/parse_error_rate":       _per,
                            "monitor/parse_failed_traj_rate": _ptr,
                            # global MCP usage on train rollouts (per-domain tir_* series
                            # were dropped as noise; this is the one to watch for "learning to use MCP")
                            "monitor/tir":                    _pm.get("tir", 0.0),
                            "monitor/repeat_call_rate":       _pm.get("repeat_call_rate", 0.0),
                            "monitor/read_only_rate":         _pm.get("read_only_rate", 0.0),
                            "monitor/n_trajs":                _pm.get("n_trajs", 0),
                            "rl_step": step,
                        })
                    except Exception as _we:
                        print(f"[WANDB] monitor log failed: {_we}")
                if _per > _parse_error_threshold:
                    _consecutive_parse_alarm += 1
                    print(
                        f"[ALARM] parse_error_rate={_per:.3f} > {_parse_error_threshold:.2f} "
                        f"(consecutive {_consecutive_parse_alarm}/{_max_consecutive_parse_alarm})"
                    )
                    if _consecutive_parse_alarm >= _max_consecutive_parse_alarm:
                        raise RuntimeError(
                            f"parse_error_rate exceeded {_parse_error_threshold:.2f} for "
                            f"{_max_consecutive_parse_alarm} consecutive steps — policy "
                            f"format collapse. Inspect runtime.log under "
                            f"runs/{project}/rollouts/train/.../runtime.log and consider rolling "
                            f"back to the last healthy checkpoint."
                        )
                else:
                    if _consecutive_parse_alarm > 0:
                        print(f"[MONITOR] parse_error_rate recovered to {_per:.3f}, resetting alarm counter")
                    _consecutive_parse_alarm = 0
        except RuntimeError:
            raise
        except Exception as e:
            print(f"[MONITOR] step {step}: parse_error check failed ({e}), continuing")

        train(worker_hosts, step, cfg, cfg.model.policy_model_type, target="policy")

        # Log per-rl-step averaged train metrics aligned to rl_step x-axis.
        # train/loss etc. are logged by osworld_train.py at (rl_step-1)*10000+window_idx,
        # which is incomparable with rollout/eval metrics on the same chart.
        # train_agg/* uses step=rl_step so all metrics share the same x-axis.
        if _wandb:
            try:
                _loss_log = os.path.join(RUN_DIR, "results", "train-loss.jsonl")
                if os.path.isfile(_loss_log):
                    _tl_rows = [json.loads(l) for l in open(_loss_log) if l.strip()]
                    _this = [r for r in _tl_rows if r.get("rl_step") == step]
                    if _this:
                        def _tavg(k): return sum(r[k] for r in _this if k in r) / len(_this)
                        _wandb.log({
                            "train_agg/loss":          _tavg("loss"),
                            # mean_kl (vs π_old) is identically 0 at update_per_step=1
                            # (ratio≡1 within the single window); mean_kl_base is the
                            # long-run drift gauge vs the frozen base — the curve to watch.
                            "train_agg/mean_kl":       _tavg("mean_kl"),
                            "train_agg/mean_kl_base":  _tavg("mean_kl_base"),
                            "train_agg/ratio_mean":    _tavg("ratio_mean"),
                            # NB: "entropy" is the chosen-token NLL proxy, not true policy
                            # entropy (see CLAUDE.md reward-chain section).
                            "train_agg/entropy":       _tavg("entropy"),
                            "train_agg/grad_norm":     _tavg("grad_norm"),
                            "train_agg/clip_frac":     _tavg("clip_frac"),
                            "train_agg/adv_mean":      _tavg("adv_mean"),
                            "train_agg/adv_std":       _tavg("adv_std"),
                            # train_agg/mean_resp_len removed: redundant with
                            # rollout/resp_len_mean (same temp=1.0 source) and easily
                            # confused with monitor_greedy/resp_len_mean (temp=0).
                            "rl_step": step,
                        })
            except Exception as _we:
                print(f"[WANDB] train_agg log failed: {_we}")

        print(f"[TIMER] step={step} total={time.time()-_step_t0:.1f}s")

        # Fixed held-out eval (temp=0 greedy). This is the only signal that exposes
        # greedy degeneration, which the temp=1.0 train rollout hides — so run it
        # frequently (default every 5 steps) to catch a collapse early.
        _held_out_every = int(getattr(cfg.experiment, "held_out_every", 5))
        if step % _held_out_every == 0:
            cleanup_local_containers(cfg)
            # Evaluate the model AFTER this step's training, not the pre-train
            # one used during rollout. ckpt/optimized was just refreshed by
            # train(); without re-pointing here, step N held-out evaluates the
            # checkpoint from step N-1 (and step 1 evaluates base), wasting
            # ~5-10 min on a re-measure the user already has.
            # Fallback to the pre-train model if optimized ckpt missing — e.g.
            # train() hit "[SKIP] no training data" because policy_dataset was
            # empty (degenerate-group case, see e24 step 2).
            post_train_ckpt = os.path.join(RUN_DIR, "ckpt", cfg.model.optimized_name)
            if os.path.isdir(post_train_ckpt):
                eval_model = post_train_ckpt
                eval_serve = copy_ckpt_to_tmpfs(post_train_ckpt)
            else:
                print(f"[HELD-OUT] post-train ckpt {post_train_ckpt} missing (train skipped?), reusing pre-train model {model}")
                eval_model = model
                eval_serve = serve_model
            start_serve(worker_hosts, cfg, "train", eval_serve,
                        cfg.model.policy_model_type,
                        served_model_name=eval_model)
            run_held_out_eval(worker_hosts, cfg, step, eval_model)
            cleanup_local_containers(cfg)
            if _wandb:
                try:
                    _held_log = os.path.join(RUN_DIR, "results", "held-out-eval.jsonl")
                    if os.path.isfile(_held_log):
                        _h_lines = [l for l in open(_held_log).readlines() if l.strip()]
                        if _h_lines:
                            _h = json.loads(_h_lines[-1])
                            _hw = {"eval_held_out/overall": _h.get("overall_outcome_rate", 0)}
                            for _d, _r in (_h.get("domain_outcome_rates") or {}).items():
                                _hw[f"eval_held_out/{_d}"] = _r
                            # greedy-degeneration monitors
                            for _mk in ("resp_len_mean", "resp_len_max", "frac_hit_cap", "avg_steps_success", "avg_steps_fail"):
                                if _mk in _h:
                                    _hw[f"monitor_greedy/{_mk}"] = _h[_mk]
                            _wandb.log({**_hw, "rl_step": step})
                except Exception as _we:
                    print(f"[WANDB] held-out eval log failed: {_we}")

        # Checkpoint every 30 steps (safety copy independent of eval)
        if step % 10 == 0:
            ckpt_dir = os.path.join(RUN_DIR, "ckpt_history", f"step_{step}")
            src_ckpt = os.path.join(RUN_DIR, "ckpt", cfg.model.optimized_name)
            if os.path.exists(src_ckpt):
                os.makedirs(ckpt_dir, exist_ok=True)
                run_local(f"cp -r {shlex.quote(src_ckpt)} {shlex.quote(ckpt_dir)}/", check=False)
                print(f"[CKPT] Saved checkpoint at step {step} → {ckpt_dir}")

        if step % cfg.experiment.eval_every == 0:
            cleanup_local_containers(cfg)
            clear_dir(TRAIN_ROLLOUT_DIR)
            run_sample(worker_hosts, cfg, "evaluation", model, cfg.model.policy_model_type, current_step=step, serve_model_override=serve_model)
            cleanup_local_containers(cfg)

            reward(step, cfg, "evaluation", model)

            # Log full eval metrics to wandb
            if _wandb:
                try:
                    _eval_json = os.path.join(RUN_DIR, "temp_data", "eval_rollout_results.json")
                    if os.path.isfile(_eval_json):
                        with open(_eval_json) as _ef:
                            _eval_data = json.load(_ef)
                        _dom_succ: dict = {}
                        for _item in _eval_data:
                            _d = _item.get("domain", "unknown")
                            for _t in _item.get("trajctory_list", []):
                                try:
                                    _ok = 1 if float(_t.get("result", 0) or 0) > 0 else 0
                                except Exception:
                                    _ok = 0
                                _dom_succ.setdefault(_d, [0, 0])
                                _dom_succ[_d][0] += _ok
                                _dom_succ[_d][1] += 1
                        _em = {}
                        _ts = _tn = 0
                        for _d, (s, n) in _dom_succ.items():
                            _em[f"eval_full/{_d}"] = round(s / n, 4) if n > 0 else 0.0
                            _ts += s; _tn += n
                        _em["eval_full/overall"] = round(_ts / _tn, 4) if _tn > 0 else 0.0
                        _wandb.log({**_em, "rl_step": step})
                except Exception as _we:
                    print(f"[WANDB] full eval log failed: {_we}")

            # Save ckpt at every eval so resume always has a matching model+eval pair.
            # Uses a different dir name than the every-30 copies so they don't collide.
            eval_ckpt_dir = os.path.join(RUN_DIR, "ckpt_history", f"eval_step_{step}")
            src_ckpt = os.path.join(RUN_DIR, "ckpt", cfg.model.optimized_name)
            if os.path.exists(src_ckpt) and not os.path.exists(eval_ckpt_dir):
                os.makedirs(eval_ckpt_dir, exist_ok=True)
                run_local(f"cp -r {shlex.quote(src_ckpt)} {shlex.quote(eval_ckpt_dir)}/", check=False)
                print(f"[CKPT] Saved eval checkpoint at step {step} → {eval_ckpt_dir}")

            # Archive the raw eval rollout JSON so each eval result is recoverable.
            src_eval_json = os.path.join(RUN_DIR, "temp_data", "eval_rollout_results.json")
            if os.path.isfile(src_eval_json):
                archived = os.path.join(RUN_DIR, "results", f"eval_rollout_step_{step}.json")
                os.makedirs(os.path.dirname(archived), exist_ok=True)
                run_local(f"cp {shlex.quote(src_eval_json)} {shlex.quote(archived)}", check=False)
                print(f"[EVAL] Archived eval rollout JSON → {archived}")

        # Persist completed step so crash recovery can resume from the next one
        try:
            with open(_resume_file, "w") as _rf:
                _rf.write(str(step))
        except Exception as e:
            print(f"[WARN] Failed to write resume_step.txt: {e}")

        step += 1
        








import os, json, torch, random
import numpy as np
from datetime import datetime




def save_trainable_only(model):
    """Save only trainable params -> ~10 MB instead of 1.13 GB."""
    full_sd = model.state_dict()
    keep = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            keep.add(name)
    for k in full_sd.keys():
        if 'lora_' in k or 'classifier' in k:
            keep.add(k)
    return {k: v for k, v in full_sd.items() if k in keep}


class CheckpointManager:
    def __init__(self, checkpoint_dir, experiment_name="paper3"):
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _latest_path(self):
        return os.path.join(self.checkpoint_dir, f"{self.experiment_name}_latest.pt")

    def _round_path(self, r):
        return os.path.join(self.checkpoint_dir, f"{self.experiment_name}_round_{r:04d}.pt")

    def _history_path(self):
        return os.path.join(self.checkpoint_dir, f"{self.experiment_name}_history.json")

    def save(self, round_num, model, optimizer=None, fairbatch=None, metrics=None):
        state = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "model_state_dict": save_trainable_only(model),
            "checkpoint_format": "trainable_only_v1",
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None
                ),
            },
        }
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if fairbatch is not None:
            state["fairbatch_group_weights"] = dict(fairbatch.group_weights)
        if metrics is not None:
            state["metrics"] = metrics
        latest = self._latest_path()
        tmp = latest + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, latest)
        torch.save(state, self._round_path(round_num))
        self._append_history(round_num, metrics)
        print(f"[Checkpoint] Round {round_num} saved", flush=True)

    def load_latest(self):
        path = self._latest_path()
        if not os.path.exists(path):
            print("[Checkpoint] None found. Starting fresh.")
            return None
        print(f"[Checkpoint] Loading: {path}", flush=True)
        return torch.load(path, map_location="cpu", weights_only=False)

    def restore_rng_state(self, checkpoint_state):
        """Restore RNG after model/data setup, before next training round."""
        rng = (
            checkpoint_state.get("rng_state")
            if checkpoint_state is not None else None
        )
        required = {"python", "numpy", "torch_cpu"}
        if not rng or not required.issubset(rng):
            print(
                "[Checkpoint] RNG state absent/incomplete; "
                "exact resume is not guaranteed.",
                flush=True,
            )
            return False

        try:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch_cpu"])

            cuda_states = rng.get("torch_cuda")
            if torch.cuda.is_available():
                if cuda_states is None:
                    print(
                        "[Checkpoint] CUDA RNG state unavailable; "
                        "GPU randomness may not reproduce exactly.",
                        flush=True,
                    )
                else:
                    torch.cuda.set_rng_state_all(cuda_states)

            print("[Checkpoint] RNG state restored.", flush=True)
            return True
        except Exception as exc:
            print(f"[Checkpoint] RNG restore failed: {exc}", flush=True)
            return False

    def _append_history(self, round_num, metrics):
        path = self._history_path()
        h = []
        if os.path.exists(path):
            try:
                with open(path) as f: h = json.load(f)
            except: h = []
        h.append({"round": round_num, "timestamp": datetime.now().isoformat(),
                  "metrics": metrics or {}})
        with open(path, "w") as f: json.dump(h, f, indent=2)

    def get_history(self):
        path = self._history_path()
        if not os.path.exists(path): return []
        with open(path) as f: return json.load(f)

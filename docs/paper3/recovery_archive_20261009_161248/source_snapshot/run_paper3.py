"""
Paper 3: FairDP-XLM
Federated multilingual emotion analytics with Opacus DP-SGD + checkpointing.
"""
import os
import sys
import json
import copy
import time
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.brighter import build_custom_split, LANGUAGE_CONFIGS
from data.partitioning import dirichlet_partition
from models.xlm_roberta import XLMRMultiLabel
from federated.strategies import multilabel_loss
from federated.aggregation import weighted_fedavg, fedsvd_aggregation, weighted_fedavg_safe, weighted_fedavg_qlora
from evaluation.metrics import multilabel_metrics
from evaluation.per_language import per_language_metrics, cross_lingual_gap
from fairness.fairbatch import FederatedFairBatch, fairness_weighted_fedavg
from privacy.opacus_dp import make_private_with_dp, get_epsilon, freeze_position_embeddings
from privacy.hybrid_dp import make_hybrid_private
from utils.checkpoint import CheckpointManager
from transformers import AutoTokenizer


LABELS_5 = ["joy", "anger", "fear", "sadness", "surprise"]

def extract_language_from_id(id_string):
    if not isinstance(id_string, str):
        return "unknown"
    return id_string.split("_")[0]



class TextDataset(Dataset):
    def __init__(self, frame, tokenizer, labels, max_length):
        frame = frame.copy()
        for label in labels:
            frame[label] = (frame[label].astype("float32")
                            .replace([np.inf, -np.inf], np.nan).fillna(0.0))
        self.texts = frame["text"].fillna("").astype(str).tolist()
        self.targets = frame[labels].to_numpy(dtype=np.float32)
        self.targets = np.nan_to_num(self.targets, nan=0.0)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.tokenizer(self.texts[idx], truncation=True,
                              padding="max_length", max_length=self.max_length,
                              return_tensors="pt")
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return item["input_ids"].squeeze(0), item["attention_mask"].squeeze(0), target


def evaluate(model, loader, device, languages=None, labels=None):
    model.eval()
    all_prob, all_y = [], []
    with torch.no_grad():
        for ids, mask, y in loader:
            if ids.size(0) == 0:
                continue
            ids, mask = ids.to(device), mask.to(device)
            logits = model(ids, mask)
            all_prob.append(torch.sigmoid(logits).cpu().numpy())
            all_y.append(y.cpu().numpy())
    if not all_prob:
        return 0.0, 0.0, {}
    y_true_stack = np.vstack(all_y)
    y_prob_stack = np.vstack(all_prob)
    r = multilabel_metrics(y_true_stack, y_prob_stack)
    per_lang = {}
    if languages is not None and labels is not None:
        try:
            per_lang = per_language_metrics(y_true_stack, y_prob_stack, languages, labels)
        except Exception as e:
            print(f"per-language eval failed: {e}")
    return float(r["macro_f1"]), float(r["micro_f1"]), per_lang




def strip_opacus_prefix(state_dict):
    """Remove Opacus '_module.' prefix from state dict keys."""
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("_module."):
            new_state[k[len("_module."):]] = v
        else:
            new_state[k] = v
    return new_state

def local_train(model, loader, device, epochs, lr, max_norm,
                use_opacus=False, use_fedsvd=False, use_hybrid_dp=False,
                target_epsilon=2.0, target_delta=1e-5):
    if use_fedsvd:
        for name, p in model.named_parameters():
            if 'lora_A' in name:
                p.requires_grad = False
            elif 'lora_B' in name:
                p.requires_grad = True
        print("FedSVD: A frozen, B trainable")

    # Pooler freeze: PEFT attaches LoRA to pooler.dense, but our forward()
    # uses last_hidden_state[:, 0] and never invokes the pooler.
    # The unused pooler LoRA-B param gets no gradient, causing Opacus
    # to crash with "Per sample gradient is not initialized".
    pooler_frozen = 0
    for name, p in model.named_parameters():
        if 'pooler' in name and p.requires_grad:
            p.requires_grad = False
            pooler_frozen += 1
    if pooler_frozen:
        print(f"Pooler frozen: {pooler_frozen} params (unused in forward)")

    engine = None
    if use_hybrid_dp:
        model, optimizer, loader, engine = make_hybrid_private(
            model=model, data_loader=loader, lr=lr,
            target_epsilon=target_epsilon, target_delta=target_delta,
            max_grad_norm=max_norm, epochs=epochs)
    elif use_opacus:
        model, optimizer, loader, engine = make_private_with_dp(
            model=model, data_loader=loader, lr=lr,
            target_epsilon=target_epsilon, target_delta=target_delta,
            max_grad_norm=max_norm, epochs=epochs)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses = []
    for _ in range(epochs):
        for ids, mask, y in loader:
            # Skip empty batches (Opacus Poisson sampler can produce these)
            if ids.size(0) == 0:
                continue
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids, mask)
            loss = multilabel_loss(logits, y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if not use_opacus:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            losses.append(loss.item())
    eps = get_epsilon(engine) if engine is not None else None
    return model, float(np.mean(losses)) if losses else 0.0, eps


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cm = CheckpointManager(checkpoint_dir=args.checkpoint_dir,
                           experiment_name=args.experiment_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"Loading BRIGHTER: {args.languages}")
    train_df, val_df, test_df = build_custom_split(
        configs=args.languages, seed=args.split_seed,
        train_fraction=0.70, validation_fraction=0.15)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_ds = TextDataset(train_df, tokenizer, LABELS_5, args.max_length)
    val_ds = TextDataset(val_df, tokenizer, LABELS_5, args.max_length)
    test_ds = TextDataset(test_df, tokenizer, LABELS_5, args.max_length)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    test_languages = np.array([extract_language_from_id(i) for i in test_df["id"].tolist()])
    print(f"Test languages: {dict(zip(*np.unique(test_languages, return_counts=True)))}")

    val_languages = np.array([extract_language_from_id(i) for i in val_df["id"].tolist()])
    print(f"Val languages:  {dict(zip(*np.unique(val_languages, return_counts=True)))}")

    client_dfs = dirichlet_partition(df=train_df, labels=LABELS_5,
                                     clients=args.clients, alpha=args.dirichlet_alpha, seed=args.partition_seed)
    print(f"Client sizes: {[len(c) for c in client_dfs]}")

    global_model = XLMRMultiLabel(
        model_name=args.model_name,
        num_labels=len(LABELS_5),
        use_lora=args.use_lora,
        use_fedsvd=args.use_fedsvd,
        use_qlora=args.use_qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    global_model.to(device)

    start_round = 0
    ckpt = cm.load_latest()
    if ckpt is not None:
        try:
            global_model.load_state_dict(ckpt["model_state_dict"])
            start_round = ckpt["round"] + 1
            print(f"[Resume] From round {start_round}")
        except Exception as e:
            print(f"[Resume] Failed: {e}")

    all_metrics = []

    # FairBatch initialization (Paper 4 fairness base)
    fairbatch = None
    if args.use_fairbatch:
        fairbatch = FederatedFairBatch(alpha=0.2, protected_attr="language")
        for lang in args.languages:
            fairbatch.group_weights[lang] = 1.0
        print(f"[FairBatch] Initialized with languages: {args.languages}")

    # Restore saved FairBatch weights and RNG before the next round.
    if ckpt is not None:
        if fairbatch is not None:
            saved_weights = ckpt.get("fairbatch_group_weights")
            if saved_weights:
                fairbatch.group_weights.update(saved_weights)
                print(
                    "[Checkpoint] FairBatch weights restored:",
                    fairbatch.group_weights,
                    flush=True,
                )
            else:
                print(
                    "[Checkpoint] No FairBatch weights found in checkpoint.",
                    flush=True,
                )
        cm.restore_rng_state(ckpt)


    # ============ VRAM INSTRUMENTATION START ============
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print(f"[VRAM] Reset | Initial: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    # ========================================================

    for round_num in range(start_round, args.rounds):
        print(f"\n{'='*60}")
        print(f"ROUND {round_num + 1}/{args.rounds}")
        print(f"{'='*60}")
        t0 = time.time()

        client_states, client_counts, round_eps = [], [], None

        for cid, client_df in enumerate(client_dfs):
            if len(client_df) == 0:
                continue
            client_ds = TextDataset(client_df, tokenizer, LABELS_5, args.max_length)
            if args.use_fairbatch and fairbatch is not None:
                client_langs = np.array([extract_language_from_id(i) for i in client_df["id"].tolist()])
                sample_weights = fairbatch.get_sample_weights(client_langs)
                sampler = WeightedRandomSampler(
                    weights=torch.from_numpy(sample_weights).double(),
                    num_samples=len(sample_weights),
                    replacement=True,
                )
                client_loader = DataLoader(client_ds, batch_size=args.batch_size, sampler=sampler)
            else:
                client_loader = DataLoader(client_ds, batch_size=args.batch_size, shuffle=True)
            local_model = copy.deepcopy(global_model).to(device)
            local_model, avg_loss, eps = local_train(
                model=local_model, loader=client_loader, device=device,
                epochs=args.local_epochs, lr=args.lr, max_norm=args.max_grad_norm,
                use_opacus=args.use_dp, use_fedsvd=args.use_fedsvd, use_hybrid_dp=args.use_hybrid_dp,
                target_epsilon=args.target_epsilon,
                target_delta=args.target_delta)
            if eps is not None:
                round_eps = eps
            client_states.append(strip_opacus_prefix(local_model.state_dict()))
            client_counts.append(len(client_df))
            print(f"  Client {cid}: n={len(client_df)}, loss={avg_loss:.4f}, eps={eps}")

            # Free memory between clients
            del local_model
            del client_loader
            del client_ds
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        if args.use_qlora:
            global_state = weighted_fedavg_qlora(
                client_states, client_counts,
                global_state=global_model.state_dict()
            )
        elif args.use_fedsvd:
            global_state = fedsvd_aggregation(
                client_states, client_counts, global_model.state_dict()
            )
            print("[FedSVD] Server-side SVD applied")
        else:
            global_state = weighted_fedavg(client_states, client_counts)

        global_model.load_state_dict(global_state, strict=False)

        val_macro, val_micro, val_per_lang = evaluate(
            global_model, val_loader, device,
            languages=val_languages, labels=LABELS_5
        )
        elapsed = time.time() - t0

        metrics = {
            "round": round_num,
            "val_macro_f1": round(val_macro, 4),
            "val_micro_f1": round(val_micro, 4),
            "epsilon": round(round_eps, 4) if round_eps else None,
            "elapsed_sec": round(elapsed, 1),
        }
        all_metrics.append(metrics)

        print(f"\nRound {round_num + 1} done in {elapsed:.1f}s")
        print(f"  Val:  macro={val_macro:.4f}, micro={val_micro:.4f}")

        if args.use_fairbatch and fairbatch is not None:
            per_lang_f1 = {str(k): v['macro_f1'] for k, v in val_per_lang.items()}
            fairbatch.update_from_f1(per_lang_f1)
            print(f"[FairBatch] Updated (val) weights: {fairbatch.group_weights}")
        if round_eps:
            print(f"  Epsilon: {round_eps:.4f}")

        cm.save(round_num=round_num, model=global_model, metrics=metrics, fairbatch=fairbatch)

        # Free memory between rounds
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    results_dir = os.path.join(args.checkpoint_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    # Final test is evaluated once, after the training loop.
    final_test_macro, final_test_micro, final_test_per_lang = evaluate(
        global_model, test_loader, device,
        languages=test_languages, labels=LABELS_5
    )

    checkpoint_round_index = max(
        int(start_round) - 1,
        int(args.rounds) - 1
    )

    final_test = {
        "experiment_name": args.experiment_name,
        "model_name": args.model_name,
        "seed": int(args.seed),
        "rounds_requested": int(args.rounds),
        "checkpoint_round_index_zero_based": checkpoint_round_index,
        "test_macro_f1": round(float(final_test_macro), 4),
        "test_micro_f1": round(float(final_test_micro), 4),
        "per_language_test": {
            str(k): v for k, v in final_test_per_lang.items()
        } if final_test_per_lang else {}
    }

    def _json_safe(value):
        if hasattr(value, "item"):
            return value.item()
        return str(value)

    final_test_json_path = os.path.join(
        results_dir, f"{args.experiment_name}_final_test.json"
    )
    final_test_csv_path = os.path.join(
        results_dir, f"{args.experiment_name}_final_test.csv"
    )

    with open(final_test_json_path, "w", encoding="utf-8") as f:
        json.dump(final_test, f, indent=2, default=_json_safe)

    pd.DataFrame([{
        "experiment_name": args.experiment_name,
        "model_name": args.model_name,
        "seed": int(args.seed),
        "rounds_requested": int(args.rounds),
        "checkpoint_round_index_zero_based": checkpoint_round_index,
        "test_macro_f1": final_test["test_macro_f1"],
        "test_micro_f1": final_test["test_micro_f1"],
        "per_language_test": json.dumps(
            final_test["per_language_test"], default=_json_safe
        ),
    }]).to_csv(final_test_csv_path, index=False)

    print("\n[Final test — evaluated once after training]")
    print(f"  Macro-F1: {final_test['test_macro_f1']:.4f}")
    print(f"  Micro-F1: {final_test['test_micro_f1']:.4f}")
    print(f"  JSON: {final_test_json_path}")
    print(f"  CSV:  {final_test_csv_path}")

    results_path = os.path.join(
        results_dir, f"{args.experiment_name}_metrics.csv"
    )

    if all_metrics and int(start_round) == 0:
        pd.DataFrame(all_metrics).to_csv(results_path, index=False)
        print(f"Per-round validation metrics: {results_path}")
    elif all_metrics:
        segment_path = os.path.join(
            results_dir,
            f"{args.experiment_name}_metrics_resumed_from_round_"
            f"{int(start_round) + 1}.csv"
        )
        pd.DataFrame(all_metrics).to_csv(segment_path, index=False)
        print(
            "[Resume] Wrote only newly completed round metrics; "
            f"left the existing master CSV untouched: {segment_path}"
        )
    else:
        print(
            "[Resume] No new round metrics to write; "
            "existing per-round CSV was not overwritten."
        )
    # ============ VRAM INSTRUMENTATION END ============
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(f"[VRAM] Peak allocated: {peak_gb:.2f} GB")
        print(f"[VRAM] Peak reserved:  {reserved_gb:.2f} GB")
    # ========================================================

    print("PAPER 3 TRAINING COMPLETE!")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="xlm-roberta-base")
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--use_qlora", action="store_true")
    p.add_argument("--use_hybrid_dp", action="store_true")
    p.add_argument("--languages", nargs="+", default=["eng"])
    p.add_argument("--clients", type=int, default=5)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--dirichlet_alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_fairbatch", action="store_true")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--use_dp", action="store_true")
    p.add_argument("--use_fedsvd", action="store_true")
    p.add_argument("--target_epsilon", type=float, default=2.0)
    p.add_argument("--target_delta", type=float, default=1e-5)
    p.add_argument("--checkpoint_dir", default="/content/drive/MyDrive/paper3_fairdp_xlm/checkpoints")
    p.add_argument("--experiment_name", default="paper3_smoke")
    p.add_argument("--split-seed", "--split_seed", dest="split_seed", type=int, default=42, help="Fixed benchmark split seed.")
    p.add_argument("--partition-seed", "--partition_seed", dest="partition_seed", type=int, default=42, help="Fixed client partition seed.")
    args = p.parse_args()

    # Reproducibility: seed all RNGs before model/data creation
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Global RNG seed set to {args.seed}")

    main(args)

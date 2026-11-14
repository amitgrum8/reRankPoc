"""Step 2b: base rankers other than LightGCN, written in the same files, so evaluate.py runs on them unchanged.

  python base_models.py --model pop    --input data/rr_input --out runs/<new>      # CPU, seconds
  python base_models.py --model sasrec --input data/rr_input --out runs/<new>      # GPU, up to --max-minutes

pop     Popularity: each movie's number of reviews in the last --pop-days (365) before the cutoff; ties broken by
        its count over the whole graph, then by its id. The same order for every user, minus the movies they
        reviewed before the cutoff. score = log(1 + count) plus a tie-break below 1e-5, so z(score) in the mixes
        reads as popularity strength. No training, no GPU, no torch.
sasrec  SASRec (Kang & McAuley, ICDM 2018): a causal transformer over the user's reviews in the training graph
        (the same 2014-2017 data LightGCN trains on) in time order, the last --max-len of them, trained to predict
        the next movie with cross-entropy over all movies (Klenitskiy & Vasilev, RecSys 2023: full cross-entropy
        makes SASRec much stronger than the paper's one-negative BCE). Validation as in lightgcn.py: the latest
        movie of up to 2,000 random users outside the test and training sets is held out; Recall@100 every
        --eval-every epochs, early stopping after --patience checks without a gain, the best checkpoint, and a
        --max-minutes limit.

Both rank every movie the user did not review before the cutoff (seen_items in <set>_users.json), like LightGCN.

Outputs in <out>/ (the formats of lightgcn.py and train.write_candidates):
  candidates.jsonl        the test users: top k (500) unseen movies with scores; every target's score and rank
                          among all unseen movies
  candidates_train.jsonl  the reranker's training users, the same (evaluate.py then skips embeddings.npz)
  base.json               model, settings, the training curve (sasrec), where the targets rank
The personas do not depend on the base ranker: copy a run's personas.jsonl and personas_train.jsonl into <out>/
and run evaluate.py on <out>/. No LLM call is needed.
"""
import argparse
import datetime as dt
import json
import math
import time
from pathlib import Path

import numpy as np


def load(inp):
    g = np.load(inp / "train.npz")
    cfg = json.loads((inp / "config.json").read_text(encoding="utf-8"))
    sets = {s: json.loads((inp / f"{s}_users.json").read_text(encoding="utf-8"))["users"] for s in ("test", "train")}
    return g, cfg, sets


def write_lists(path, users, scores, k, decimals):
    """scores[r] = the score of every movie for users[r]. Masks the user's seen movies, writes the top k and every
    target's score and rank; returns the target ranks by rating group."""
    lines, ranks = [], {"liked": [], "three": [], "disliked": []}
    for tu, row in zip(users, scores):
        s = np.array(row, dtype=np.float64)
        if tu["seen_items"]:
            s[np.array(tu["seen_items"], dtype=np.int64)] = -np.inf
        top = np.argpartition(-s, k)[:k]
        top = top[np.argsort(-s[top], kind="stable")]
        targets = []
        for t in tu["targets"]:
            st = s[t["i"]]
            rank = int((s > st).sum()) + 1
            targets.append({"i": t["i"], "rating": t["rating"], "ts": t["ts"], "score": round(float(st), decimals),
                            "rank": rank})
            ranks["liked" if t["rating"] >= 4 else "disliked" if t["rating"] <= 2 else "three"].append(rank)
        lines.append(json.dumps({"user_id": tu["user_id"], "u": tu["u"],
                                 "candidates": [[int(a), round(float(s[a]), decimals)] for a in top],
                                 "targets": targets}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ranks


def describe(ranks, k):
    if not ranks:
        return {}
    r = np.array(ranks)
    return {"n": len(r), f"in_top{k}": round(float((r <= k).mean()), 3), "in_top10": round(float((r <= 10).mean()), 3),
            "median_rank": float(np.median(r))}


# ---------------------------------------------------------------------------------------------------- popularity
def pop_scores(g, cfg, days):
    U, I, TS = g["u"], g["i"].astype(np.int64), g["ts"]
    n_items = int(g["n_items"])
    cutoff = dt.datetime.fromisoformat(cfg["cutoff"]).replace(tzinfo=dt.timezone.utc).timestamp() * 1000
    recent = TS >= cutoff - days * 86400 * 1000
    c_rec = np.bincount(I[recent], minlength=n_items)
    c_all = np.bincount(I, minlength=n_items)
    order = np.lexsort((np.arange(n_items), -c_all, -c_rec))      # last key first: recent count, all-time count, id
    pos = np.empty(n_items)
    pos[order] = np.arange(n_items)
    # the tie-break stays below the smallest gap between two different log counts (1/(count+2) > 1e-5)
    score = np.log1p(c_rec) + 1e-5 * (1.0 - pos / n_items)
    info = {"window_days": days, "reviews_in_window": int(recent.sum()), "items_with_a_review_in_window":
            int((c_rec > 0).sum()), "top_count_in_window": int(c_rec.max())}
    return score, info


# ---------------------------------------------------------------------------------------------------- SASRec
def run_sasrec(args, g, sets, rng):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    U, I, TS = g["u"].astype(np.int64), g["i"].astype(np.int64), g["ts"]
    n_users, n_items = int(g["n_users"]), int(g["n_items"])
    order = np.lexsort((np.arange(len(U)), TS, U))                 # by user, then time, then file order
    seq_all = I[order] + 1                                         # item ids shifted by one: 0 is padding
    ptr = np.concatenate([[0], np.cumsum(np.bincount(U, minlength=n_users))])
    L = args.max_len

    def seq_of(x):
        return seq_all[ptr[x]:ptr[x + 1]]

    fixed = {tu["u"] for s in sets.values() for tu in s}            # test and training users stay whole
    cnt = np.diff(ptr)
    pool = np.array([x for x in np.flatnonzero(cnt >= 3) if x not in fixed])
    val_users = np.sort(rng.choice(pool, size=min(args.n_val, len(pool)), replace=False))
    val_set = set(val_users.tolist())

    def window(s, n):
        s = s[-n:]
        return np.concatenate([np.zeros(n - len(s), dtype=np.int64), s])     # left padding

    train_rows = [window(seq_of(x)[:-1] if x in val_set else seq_of(x), L + 1) for x in range(n_users) if cnt[x] >= 2]
    X = torch.from_numpy(np.stack(train_rows)).to(device)
    print(f"device {device} | {n_items:,} items | {len(train_rows):,} training sequences (last {L + 1}) | "
          f"{len(val_users)} validation users", flush=True)

    class Block(nn.Module):
        def __init__(self, d, heads, drop):
            super().__init__()
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
            self.ff = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, d), nn.Dropout(drop))
            self.drop = nn.Dropout(drop)

        def forward(self, x, mask, keep):
            h = self.ln1(x)
            a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
            x = (x + self.drop(a)) * keep
            return (x + self.ff(self.ln2(x))) * keep

    class SASRec(nn.Module):
        def __init__(self):
            super().__init__()
            d = args.dim
            self.item = nn.Embedding(n_items + 1, d, padding_idx=0)
            self.pos = nn.Embedding(L, d)
            nn.init.normal_(self.item.weight, std=0.02)
            nn.init.normal_(self.pos.weight, std=0.02)
            with torch.no_grad():
                self.item.weight[0].zero_()
            self.drop = nn.Dropout(args.dropout)
            self.blocks = nn.ModuleList([Block(d, args.heads, args.dropout) for _ in range(args.blocks)])
            self.ln = nn.LayerNorm(d)
            self.causal = torch.ones(L, L, dtype=torch.bool, device=device).triu(1)

        def forward(self, seq):                                    # (B, L) left-padded -> (B, L, d)
            B = seq.shape[0]
            keep = (seq > 0).unsqueeze(-1).float()
            x = self.item(seq) * math.sqrt(args.dim) + self.pos.weight.unsqueeze(0)
            x = self.drop(x) * keep
            # a large negative number instead of -inf: a padded query with every key masked gets a finite
            # (uniform) row instead of NaN, and its output is zeroed by keep anyway
            blocked = self.causal.unsqueeze(0) | (seq == 0).unsqueeze(1)
            mask = torch.zeros(B, L, L, device=device).masked_fill(blocked, -1e9)
            mask = mask.repeat_interleave(args.heads, dim=0)
            for blk in self.blocks:
                x = blk(x, mask, keep)
            return self.ln(x)

        def score_last(self, seq):                                 # scores of every movie after the last position
            return self.forward(seq)[:, -1, :] @ self.item.weight[1:].T

    model = SASRec().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    @torch.no_grad()
    def scores_for(seqs, chunk=1024):
        model.eval()
        out = []
        for a in range(0, len(seqs), chunk):
            batch = torch.from_numpy(np.stack([window(s, L) for s in seqs[a:a + chunk]])).to(device)
            out.append(model.score_last(batch).float().cpu().numpy())
        model.train()
        return np.concatenate(out)

    val_inputs = [seq_of(x)[:-1] for x in val_users]
    val_item = np.array([seq_of(x)[-1] - 1 for x in val_users])

    def val_recall():
        s = scores_for(val_inputs)
        for r, inp_seq in enumerate(val_inputs):
            s[r, inp_seq - 1] = -np.inf                            # the user's other movies, as in lightgcn.py
        top = np.argpartition(-s, 100, axis=1)[:, :100]
        return float(np.mean([val_item[r] in top[r] for r in range(len(val_users))]))

    n_batches = math.ceil(len(X) / args.batch)
    curve, best, best_state, bad, t_train = [], -1.0, None, 0, time.time()
    for epoch in range(1, args.max_epochs + 1):
        perm = torch.randperm(len(X), device=device)
        total, nb, stop = 0.0, 0, False
        for b in range(n_batches):
            if (time.time() - t_train) / 60 >= args.max_minutes:
                stop = True
                break
            rows = X[perm[b * args.batch:(b + 1) * args.batch]]
            inp_, tgt = rows[:, :-1], rows[:, 1:]
            valid = (inp_ > 0) & (tgt > 0)                          # a real history position with a real next movie
            h = model(inp_)[valid]
            loss = F.cross_entropy(h @ model.item.weight[1:].T, tgt[valid] - 1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        minutes = (time.time() - t_train) / 60
        rec = {"epoch": epoch, "loss": round(total / max(nb, 1), 5), "minutes": round(minutes, 2), "batches": nb}
        if epoch == 1 and not stop:
            print(f"epoch 1 took {minutes * 60:.0f}s -> at most "
                  f"{min(args.max_epochs, int(args.max_minutes / max(minutes, 1e-9)))} epochs fit in "
                  f"{args.max_minutes:g} min", flush=True)
        stop = stop or minutes >= args.max_minutes
        if epoch % args.eval_every == 0 or stop:
            rec["val_recall@100"] = round(val_recall(), 4)
            if rec["val_recall@100"] > best:
                best, bad = rec["val_recall@100"], 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            print(f"epoch {epoch:3d} loss {rec['loss']:.4f} val Recall@100 {rec['val_recall@100']:.4f} "
                  f"(best {best:.4f}) {minutes:.1f} min", flush=True)
            curve.append(rec)
            if bad >= args.patience:
                print("early stop", flush=True)
                break
        else:
            curve.append(rec)
        if stop:
            print(f"time limit ({args.max_minutes:g} min)", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    def set_scores(users):
        return scores_for([seq_of(tu["u"]) for tu in users])

    info = {"device": str(device), "training_sequences": len(train_rows), "val_users": int(len(val_users)),
            "best_val_recall@100": best, "curve": curve}
    return set_scores, info


# ---------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["pop", "sasrec"])
    ap.add_argument("--input", default=str(Path(__file__).resolve().parent / "data" / "rr_input"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pop-days", type=int, default=365, help="pop: the window before the cutoff that is counted")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--max-len", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-epochs", type=int, default=400)
    ap.add_argument("--max-minutes", type=float, default=30)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--n-val", type=int, default=2000)
    args = ap.parse_args()

    t0 = time.time()
    inp, out = Path(args.input), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    g, cfg, sets = load(inp)
    rng = np.random.default_rng(args.seed)

    if args.model == "pop":
        score, info = pop_scores(g, cfg, args.pop_days)
        set_scores, decimals = (lambda users: [score] * len(users)), 12   # the tie-break needs the digits
    else:
        set_scores, info = run_sasrec(args, g, sets, rng)
        decimals = 6

    targets = {}
    for name, fname in (("test", "candidates.jsonl"), ("train", "candidates_train.jsonl")):
        ranks = write_lists(out / fname, sets[name], set_scores(sets[name]), args.k, decimals)
        targets[name] = {k: describe(v, args.k) for k, v in ranks.items()}
        print(f"{name}: wrote {out / fname} | targets (rank among all unseen movies): {json.dumps(targets[name])}",
              flush=True)
    result = {"model": args.model, "settings": vars(args), **info, "targets": targets,
              "seconds": round(time.time() - t0, 1)}
    (out / "base.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {out / 'base.json'} | total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

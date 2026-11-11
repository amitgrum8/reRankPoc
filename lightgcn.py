"""Step 2 (Kaggle GPU; a CPU works for a small test): train LightGCN and write each test user's top k (500).

  python lightgcn.py --input data/rr_input --out run
  python lightgcn.py --input data/rr_input --out run --max-minutes 20 --dim 64 --layers 3

LightGCN (He et al., SIGIR 2020): one embedding per user and item, propagated over the normalized
user-item graph; the final embedding is the mean of layers 0..L; trained with the BPR loss and one random
negative item per positive. Every review is a positive interaction, whatever its rating, so the model has
no idea what a user disliked. That is the reranker's job.

Validation: the latest interaction of up to 2,000 random non-test users is taken out of the graph. Training
stops when their Recall@100 has not improved for --patience checks (one check every --eval-every epochs),
or after --max-minutes. The best checkpoint is used. Defaults follow the LightGCN paper (64 dims, 3 layers,
lr 1e-3, L2 1e-4, batch 2048).

Outputs:
  run/candidates.jsonl  one line per test user:
     {"user_id", "u", "candidates": [[item, score], ... top k (500)],
      "targets": [{"i", "rating", "ts", "score", "rank"}]}
     rank = the target's place among all items the user had not reviewed before the cutoff (1 = top);
     items reviewed before the cutoff are never candidates.
  run/lightgcn.json     settings, the training curve and a short summary
  run/embeddings.npz    the trained model: final user and item embeddings (score = dot product), so any
                        other user can be ranked later without training again
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


def build_graph(u, i, n_users, n_items, device):
    """Symmetric normalized adjacency D^-1/2 A D^-1/2 of the bipartite graph, as a sparse tensor."""
    rows = np.concatenate([u, i + n_users]).astype(np.int64)
    cols = np.concatenate([i + n_users, u]).astype(np.int64)
    deg = np.bincount(rows, minlength=n_users + n_items).astype(np.float64)
    vals = 1.0 / np.sqrt(deg[rows] * deg[cols])
    a = torch.sparse_coo_tensor(torch.from_numpy(np.stack([rows, cols])), torch.from_numpy(vals).float(),
                                (n_users + n_items, n_users + n_items)).coalesce()
    return a.to(device)


class LightGCN(torch.nn.Module):
    def __init__(self, n_users, n_items, dim, layers, graph):
        super().__init__()
        self.n_users, self.layers, self.graph = n_users, layers, graph
        self.emb = torch.nn.Embedding(n_users + n_items, dim)
        torch.nn.init.normal_(self.emb.weight, std=0.1)

    def propagate(self):
        x = self.emb.weight
        out = [x]
        for _ in range(self.layers):
            x = torch.sparse.mm(self.graph, x)
            out.append(x)
        return torch.stack(out).mean(0)


def csr(u, i, n_users):
    order = np.argsort(u, kind="stable")
    indptr = np.concatenate([[0], np.cumsum(np.bincount(u, minlength=n_users))])
    return indptr, i[order]


@torch.no_grad()
def topk_for(model, users, exclude, k, device, chunk=512):
    """Top-k items per user, with each user's excluded items masked out. Returns (items, scores) arrays
    and the full masked score rows are not kept (memory)."""
    allx = model.propagate()
    ue, ie = allx[:model.n_users], allx[model.n_users:]
    top_i, top_s = [], []
    for a in range(0, len(users), chunk):
        uu = users[a:a + chunk]
        s = ue[torch.as_tensor(uu, device=device)] @ ie.T
        for r, user in enumerate(uu):
            ex = exclude(user)
            if len(ex):
                s[r, torch.as_tensor(ex, device=device)] = -math.inf
        v, ix = torch.topk(s, k, dim=1)
        top_i.append(ix.cpu().numpy()); top_s.append(v.cpu().numpy())
    return np.concatenate(top_i), np.concatenate(top_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--max-minutes", type=float, default=30)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--k", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    inp, out = Path(args.input), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = inp / "train.npz"
    if path.is_dir():     # in case the upload unpacked the .npz (it is a zip) into a folder of .npy files
        g = {p.stem: np.load(p) for p in path.glob("*.npy")}
    else:
        g = np.load(path)
    u_all, i_all, ts_all = g["u"].astype(np.int64), g["i"].astype(np.int64), g["ts"]
    n_users, n_items = int(g["n_users"]), int(g["n_items"])
    test = json.loads((inp / "test_users.json").read_text(encoding="utf-8"))["users"]
    test_u = {tu["u"] for tu in test}
    print(f"device {device} | graph {n_users:,} users, {n_items:,} items, {len(u_all):,} interactions | "
          f"{len(test)} test users", flush=True)

    # ---- validation: the latest interaction of random non-test users with >= 3 interactions
    cnt = np.bincount(u_all, minlength=n_users)
    pool = np.array([x for x in np.flatnonzero(cnt >= 3) if x not in test_u])
    val_users = np.sort(rng.choice(pool, size=min(args.n_val, len(pool)), replace=False))
    order = np.lexsort((ts_all, u_all))               # by user, then time
    last_of_user = np.full(n_users, -1)
    last_of_user[u_all[order]] = order                 # the last write per user wins = latest interaction
    val_idx = last_of_user[val_users]
    val_item = dict(zip(val_users.tolist(), i_all[val_idx].tolist()))
    keep = np.ones(len(u_all), dtype=bool)
    keep[val_idx] = False
    u, i = u_all[keep], i_all[keep]
    indptr, seen_i = csr(u, i, n_users)

    def train_items(user):
        return seen_i[indptr[user]:indptr[user + 1]]

    graph = build_graph(u, i, n_users, n_items, device)
    model = LightGCN(n_users, n_items, args.dim, args.layers, graph).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    u_t, i_t = torch.from_numpy(u).to(device), torch.from_numpy(i).to(device)
    n_batches = math.ceil(len(u) / args.batch)
    print(f"training edges {len(u):,} | {n_batches} batches per epoch | {len(val_users)} validation users "
          f"| setup {time.time() - t0:.0f}s", flush=True)

    def val_recall():
        model.eval()
        top, _ = topk_for(model, val_users, train_items, args.k, device)
        model.train()
        return float(np.mean([val_item[x] in set(row.tolist()) for x, row in zip(val_users.tolist(), top)]))

    curve, best, best_state, bad, t_train = [], -1.0, None, 0, time.time()
    for epoch in range(1, args.max_epochs + 1):
        perm = torch.randperm(len(u), device=device)
        total, nb, stop = 0.0, 0, False
        for b in range(n_batches):
            if (time.time() - t_train) / 60 >= args.max_minutes:    # checked per batch: the limit holds to the second
                stop = True
                break
            nb += 1
            idx = perm[b * args.batch:(b + 1) * args.batch]
            bu, bp = u_t[idx], i_t[idx]
            bn = torch.randint(0, n_items, (len(idx),), device=device)
            allx = model.propagate()
            eu, ep, en = allx[bu], allx[n_users + bp], allx[n_users + bn]
            loss = -torch.nn.functional.logsigmoid((eu * ep).sum(1) - (eu * en).sum(1)).mean()
            w0 = model.emb.weight
            reg = (w0[bu].pow(2).sum() + w0[n_users + bp].pow(2).sum() + w0[n_users + bn].pow(2).sum()) / (2 * len(idx))
            opt.zero_grad()
            (loss + args.reg * reg).backward()
            opt.step()
            total += loss.item()
        minutes = (time.time() - t_train) / 60
        rec = {"epoch": epoch, "loss": round(total / max(nb, 1), 5), "minutes": round(minutes, 2), "batches": nb}
        if epoch == 1 and not stop:
            print(f"epoch 1 took {minutes * 60:.0f}s -> at most {min(args.max_epochs, int(args.max_minutes / max(minutes, 1e-9)))} "
                  f"epochs fit in {args.max_minutes:g} min", flush=True)
        elif stop:
            print(f"time limit ({args.max_minutes:g} min) after {nb}/{n_batches} batches of epoch {epoch}", flush=True)
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
            if bad >= args.patience:
                print("early stop", flush=True)
                curve.append(rec)
                break
        curve.append(rec)
        if stop:
            print(f"time limit ({args.max_minutes:g} min)", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- test users: top-k candidates, and every target's score and rank
    model.eval()
    seen_test = {tu["u"]: np.array(tu["seen_items"], dtype=np.int64) for tu in test}
    users = np.array([tu["u"] for tu in test])
    top_i, top_s = topk_for(model, users, lambda x: seen_test[x], args.k, device)
    with torch.no_grad():
        allx = model.propagate()
        ue, ie = allx[:n_users], allx[n_users:]
    # the trained model, as final embeddings: score(user, item) = user . item. Enough to rank any user later
    # (for the reranker's training users) without training again
    np.savez(out / "embeddings.npz", user=ue.cpu().numpy().astype(np.float32),
             item=ie.cpu().numpy().astype(np.float32))
    lines, summary = [], {"liked": [], "three": [], "disliked": []}
    for tu, ti, sc in zip(test, top_i, top_s):
        with torch.no_grad():
            s = ue[tu["u"]] @ ie.T
            if len(seen_test[tu["u"]]):
                s[torch.as_tensor(seen_test[tu["u"]], device=device)] = -math.inf
            s = s.cpu().numpy()
        targets = []
        for t in tu["targets"]:
            st = float(s[t["i"]])
            rank = int((s > st).sum()) + 1
            targets.append({"i": t["i"], "rating": t["rating"], "ts": t["ts"], "score": round(st, 5), "rank": rank})
            key = "liked" if t["rating"] >= 4 else "disliked" if t["rating"] <= 2 else "three"
            summary[key].append(rank)
        lines.append(json.dumps({"user_id": tu["user_id"], "u": tu["u"],
                                 "candidates": [[int(a), round(float(b), 5)] for a, b in zip(ti, sc)],
                                 "targets": targets}))
    (out / "candidates.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def describe(ranks):
        if not ranks:
            return {}
        r = np.array(ranks)
        return {"n": len(r), f"in_top{args.k}": round(float((r <= args.k).mean()), 3),
                "in_top10": round(float((r <= 10).mean()), 3), "median_rank": float(np.median(r))}

    result = {"settings": vars(args), "device": str(device), "graph": {"users": n_users, "items": n_items,
              "edges_train": int(len(u)), "val_users": int(len(val_users))},
              "best_val_recall@100": best, "curve": curve,
              "targets": {k: describe(v) for k, v in summary.items()},
              "seconds": round(time.time() - t0, 1)}
    (out / "lightgcn.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("targets (rank among all unseen items):", json.dumps(result["targets"]), flush=True)
    print(f"wrote {out / 'candidates.jsonl'} | total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

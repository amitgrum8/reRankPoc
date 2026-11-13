"""Step 6b (CPU, seconds): learned linear mixes. evaluate.py fits them on the training users (the same lists and
persona rows as R7_lgbm), adds one column per mix to eval.md, the fitted weights below the tables, and <run>/mix.json.

  R10_mix       score = w0*z(LightGCN) + w1*z(like_s1) + ... + w5*z(like_s5) + w6*z(dis_s1) + ... + w10*z(dis_s5)
  R11_mix_crit  the same likes; the dislikes against what reviewers criticized: ... + w6*z(dis_crit_s1) + ...

like_sK / dis_sK = cosine (bge-small) between the movie text (title + description) and the user's K-th closest
like / dislike line. dis_crit_sK = each dislike line's closest criticized line in the movie's review profile, K-th
best over the dislike lines (needs item profiles). All sorted per movie, so position K means the same for every
user (rerank.py). z = z-score over the user's candidates, as in the formulas. No hand-set alpha or beta.
R10 against R11: do dislikes work better against criticized reviews than against the product description, which
is marketing text and never says anything bad?

Training: logistic regression (Newton steps, a small L2 penalty) on the training users with at least one target
inside LightGCN's top k. Rows = the k candidates; label 1 = the user went on to review the movie (any rating).
Targets outside the top k stay out: they enter with a score below every candidate and would teach "low LightGCN
score = target" (see train.py). Standard errors come from the Hessian and are approximate (the rows of one user
are not independent); the bootstrap and the swapped-persona control in eval.md are the real test.
"""
import numpy as np

import rerank as R

_K = range(1, R.K_SORTED + 1)
LIKE = [f"like_s{k}" for k in _K]
MIXES = {"R10_mix": ["base"] + LIKE + [f"dis_s{k}" for k in _K],
         "R11_mix_crit": ["base"] + LIKE + [f"dis_crit_s{k}" for k in _K]}
PROFILE_MIXES = {"R11_mix_crit"}                              # need item profiles
L2 = 1.0


def matrix(user, ids, inputs):
    """The inputs of the items in ids (rows in that order), each z-scored over the user's candidates
    (0 when constant, e.g. the dislike columns of a persona without dislikes)."""
    f = user["feat"]
    X = np.array([[f[i][c] for c in inputs] for i in ids], dtype=float)
    C = np.array([[f[i][c] for c in inputs] for i in user["candidates"]], dtype=float)
    mu, sd = C.mean(0), C.std(0)
    ok = sd > 1e-9
    return np.where(ok, (X - mu) / np.where(ok, sd, 1.0), 0.0)


def fit(users, inputs, l2=L2, max_iter=50):
    """Logistic regression over the candidates of the users with an in-list target.
    Returns ({input: weight}, info for mix.json)."""
    Xs, ys = [], []
    for user in users:
        t = {x["i"] for x in user["targets"]} & user["cand_set"]
        if t:
            Xs.append(matrix(user, user["candidates"], inputs))
            ys.append(np.array([i in t for i in user["candidates"]], dtype=float))
    X, y = np.concatenate(Xs), np.concatenate(ys)
    X1 = np.hstack([np.ones((len(X), 1)), X])                # column 0 = intercept, not penalized
    pen = np.diag([0.0] + [l2] * len(inputs))
    w = np.zeros(X1.shape[1])
    w[0] = np.log(y.mean() / (1 - y.mean()))
    converged = False
    for it in range(1, max_iter + 1):
        p = 1.0 / (1.0 + np.exp(-(X1 @ w)))
        H = (X1 * (p * (1 - p))[:, None]).T @ X1 + pen
        step = np.linalg.solve(H, X1.T @ (y - p) - pen @ w)
        w += step
        if np.abs(step).max() < 1e-7:
            converged = True
            break
    p = 1.0 / (1.0 + np.exp(-(X1 @ w)))
    cov = np.linalg.inv((X1 * (p * (1 - p))[:, None]).T @ X1 + pen)

    def total(names):
        idx = [1 + inputs.index(c) for c in names]
        return {"w": round(float(w[idx].sum()), 4), "se": round(float(np.sqrt(cov[np.ix_(idx, idx)].sum())), 4)}

    weights = {c: float(w[1 + k]) for k, c in enumerate(inputs)}
    info = {"train_users": len(users), "users": len(Xs), "rows": int(len(y)), "targets": int(y.sum()), "l2": l2,
            "iterations": it, "converged": converged, "intercept": round(float(w[0]), 4),
            "weights": {c: {"w": round(weights[c], 4), "se": round(float(np.sqrt(cov[1 + k, 1 + k])), 4)}
                        for k, c in enumerate(inputs)},
            "like_sum": total([c for c in inputs if c.startswith("like_")]),
            "dis_sum": total([c for c in inputs if c.startswith("dis_")])}
    return weights, info


def reranker(weights):
    """A fitted mix: the weights over the z-scored inputs (the intercept does not change the order)."""
    inputs, coef = list(weights), np.array(list(weights.values()))

    def rerank(user):
        return dict(zip(user["list"], (matrix(user, user["list"], inputs) @ coef).tolist()))
    return rerank


def weight_lines(name, info):
    """The weight table of one mix for eval.md."""
    lines = ["", f"{name} weights: logistic regression on {info['users']} training users with a target inside the "
             f"list ({info['targets']} targets, {info['rows']} rows); inputs z-scored within each user's list:", "",
             "| input | weight | std. error |", "|---|---|---|"]
    lines += [f"| {c} | {v['w']:+.3f} | {v['se']:.3f} |" for c, v in info["weights"].items()]
    lines += [f"| sum of the like weights | {info['like_sum']['w']:+.3f} | {info['like_sum']['se']:.3f} |",
              f"| sum of the dislike weights | {info['dis_sum']['w']:+.3f} | {info['dis_sum']['se']:.3f} |"]
    return lines

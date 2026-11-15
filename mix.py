"""Step 6b (CPU, seconds to minutes): learned linear mixes. evaluate.py fits them on the training users (the same
lists and persona rows as R7_lgbm), adds one column per mix to eval.md, the fitted weights below the tables, and
mix.json.

  R10_mix          score = w0*z(LightGCN) + w1*z(like_s1) + ... + w5*z(like_s5) + w6*z(dis_s1) + ... + w10*z(dis_s5)
  R11_mix_crit     the same likes; the dislikes against what reviewers criticized (dis_crit_s1..s5), + no_crit
  R12_mix_reviews  the item's own reviews: like lines against its good (4-5 star) review sentences (like_good_s1..s5),
                   dislike lines against its bad (1-2 star) review sentences (dis_bad_s1..s5), + no_bad, n_good, n_bad

like_sK / dis_sK = cosine between the movie text (title + description) and the user's K-th closest like / dislike
line; dis_crit_sK = each dislike line's closest criticized line of the movie's review profile, K-th best over the
dislike lines; like_good_sK / dis_bad_sK = each like / dislike line's closest sentence of the movie's good / bad
reviews, K-th best over the lines. All sorted per movie, so position K means the same for every user (rerank.py).
z = z-score over the user's candidates, as in the formulas. No hand-set alpha or beta.

Missing values: a movie without criticized lines (R11) or without bad reviews (R12) has no dislike match at all.
Its value is set to the user's average (z = 0) and the indicator no_crit / no_bad = 1 carries the fact itself, so a
dislike weight measures matching, not "this movie has no complaints" (which mostly means a well-loved movie).
n_good / n_bad (number of review sentences) let the model discount the higher maximum that more sentences give.
These item columns are the same with another user's persona, so the swapped-persona control still isolates the
persona's part.

Training: logistic regression (Newton steps) on the training users with at least one target inside LightGCN's
top k. Rows = the k candidates; label 1 = the user went on to review the movie (any rating). Targets outside the
top k stay out: they enter with a score below every candidate and would teach "low LightGCN score = target" (see
train.py). An L2 penalty on every weight except the intercept and LightGCN's, picked from L2_GRID by 5-fold
cross-validation over training users (held-out log-likelihood): a large penalty shrinks the persona weights to 0,
which is LightGCN's own order, so noise in many weights cannot reorder the list. Standard errors come from the
penalized Hessian and are approximate (the rows of one user are not independent); the bootstrap and the
swapped-persona control in eval.md are the real test.

Ablations of the negative part (ABLATIONS): R12 with only the persona's dislike part changed, each refit the same
way (own penalty by cross-validation). The item columns no_bad, n_good, n_bad stay in all of them.
  A1_likes_only     no dislike columns
  A2_dis_all        dislike lines against all review sentences, good and bad (dis_all_s1..s5), + no_rev
  A3_dis_good       dislike lines against the good review sentences, the reverse routing (dis_good_s1..s5), + no_good
  A4_mixed          one mixed persona: like and dislike lines pooled, each against all review sentences, sorted top
                    10 (mixed_s1..s10): the same lines and capacity, but the model cannot tell a like from a dislike
  A5_coral          CORAL-style likes minus dislikes: one term z(like_good_s1) - CORAL_BETA * z(dis_bad_s1), so the
                    dislikes are forced to subtract (hand-set beta = 1); only its overall weight is learned
  A6_satisfaction   R12's inputs with a satisfaction label: 1 = a target rated 4-5 stars; 1-3 star targets are
                    negatives like the unseen candidates (the user watched them and was not satisfied). If the
                    dislikes mean avoidance, their weights turn negative here while they stay positive in R12.

R13_near (NearMix): R12 refit once per test user, with the training users weighted by how close their persona is to
the user's. The movie's score still comes only from the user's own persona; what changes is how much each training
user counts when the weights are learned. closeness = cosine between the two personas' like-line centroids, averaged
with the cosine between their dislike-line centroids when both have dislikes (lines encoded as in rerank.py);
z-scored over the training users; user weight = exp(alpha * z), scaled to mean 1 (alpha = 0 is R12 exactly; for a
small alpha the weight is about 1 + alpha * z). The penalty is R12's. alpha is picked from NEAR_ALPHAS by
leave-one-user-out over the training users (held-out log-likelihood); the held-out in-list rank change and the
effective number of training users per alpha go to eval.md. R13_near_ctrl takes the closeness from another test
user's persona (the features stay the user's own): the same amount of reweighting with no information about the
user, so R13_near minus R13_near_ctrl is what persona closeness adds.
"""
import numpy as np

import rerank as R

_K = range(1, R.K_SORTED + 1)
LIKE = [f"like_s{k}" for k in _K]
LIKE_GOOD = [f"like_good_s{k}" for k in _K]
ITEM = ["no_bad", "n_good", "n_bad"]                   # item columns, the same whichever persona
MIXES = {"R10_mix": ["base"] + LIKE + [f"dis_s{k}" for k in _K],
         "R11_mix_crit": ["base"] + LIKE + [f"dis_crit_s{k}" for k in _K] + ["no_crit"],
         "R12_mix_reviews": ["base"] + LIKE_GOOD + [f"dis_bad_s{k}" for k in _K] + ITEM}
ABLATIONS = {"A1_likes_only": ["base"] + LIKE_GOOD + ITEM,
             "A2_dis_all": ["base"] + LIKE_GOOD + [f"dis_all_s{k}" for k in _K] + ITEM + ["no_rev"],
             "A3_dis_good": ["base"] + LIKE_GOOD + [f"dis_good_s{k}" for k in _K] + ITEM + ["no_good"],
             "A4_mixed": ["base"] + [f"mixed_s{k}" for k in range(1, R.K_MIXED + 1)] + ITEM + ["no_rev"],
             "A5_coral": ["base", "coral"] + ITEM,
             "A6_satisfaction": MIXES["R12_mix_reviews"]}
REFERENCE = "R12_mix_reviews"                          # the mix that the ablations and R13 are compared with
LABELS = {"A6_satisfaction": "liked"}                  # mixes trained on another label than "any target"
LABEL_FN = {"any": lambda t: True, "liked": lambda t: t["rating"] >= 4}
NEEDS = {"R11_mix_crit": "profiles", "R12_mix_reviews": "reviews",   # mixes that need item profiles / reviews
         **{name: "reviews" for name in ABLATIONS}}
CORAL_BETA = 1.0
DERIVED = {"coral": {"like_good_s1": 1.0, "dis_bad_s1": -CORAL_BETA}}   # input = sum of weight * z(column)
L2_GRID = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
FOLDS = 5
NEAR_ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0)


def matrix(user, ids, inputs):
    """The inputs of the items in ids (rows in that order), each z-scored over the user's candidates; a missing
    value (NaN) and a constant column become 0, the average. A DERIVED input is a weighted sum of z-scored columns."""
    cols = []
    for x in inputs:
        for c in DERIVED.get(x, (x,)):
            if c not in cols:
                cols.append(c)
    f = user["feat"]
    X = np.array([[f[i][c] for c in cols] for i in ids], dtype=float)
    C = np.array([[f[i][c] for c in cols] for i in user["candidates"]], dtype=float)
    seen = ~np.isnan(C)
    n = seen.sum(0)
    mu = np.where(seen, C, 0.0).sum(0) / np.maximum(n, 1)
    sd = np.sqrt((np.where(seen, C - mu, 0.0) ** 2).sum(0) / np.maximum(n, 1))
    ok = sd > 1e-9
    Z = np.where(ok, (X - mu) / np.where(ok, sd, 1.0), 0.0)
    Z = np.where(np.isnan(Z), 0.0, Z)
    if cols == list(inputs):
        return Z
    at = {c: k for k, c in enumerate(cols)}
    return np.column_stack([sum(w * Z[:, at[c]] for c, w in DERIVED[x].items()) if x in DERIVED else Z[:, at[x]]
                            for x in inputs])


def design(users, inputs, label="any"):
    """The stacked candidate rows of the users with a target (any rating) inside their list: X1 (column 0 = the
    intercept), y (1 = a target that passes the label), the users kept, and each kept user's number of rows."""
    keep = LABEL_FN[label]
    Xs, ys, kept = [], [], []
    for user in users:
        inl = [x for x in user["targets"] if x["i"] in user["cand_set"]]
        if inl:
            pos = {x["i"] for x in inl if keep(x)}
            Xs.append(matrix(user, user["candidates"], inputs))
            ys.append(np.array([i in pos for i in user["candidates"]], dtype=float))
            kept.append(user)
    X = np.concatenate(Xs)
    return np.hstack([np.ones((len(X), 1)), X]), np.concatenate(ys), kept, np.array([len(v) for v in ys])


def _penalty(inputs, l2):
    free = {0, 1 + inputs.index("base")} if "base" in inputs else {0}      # intercept and LightGCN: not penalized
    return np.diag([0.0 if j in free else l2 for j in range(len(inputs) + 1)])


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def _objective(X1, y, w, pen, s=None):
    """The penalized (row-weighted) log-likelihood that _newton maximizes."""
    z = X1 @ w
    ll = y * z - np.logaddexp(0.0, z)
    return float((ll if s is None else s * ll).sum() - 0.5 * w @ pen @ w)


def _newton(X1, y, pen, s=None, w0=None, max_iter=50):
    """Penalized logistic regression, rows weighted by s (None = 1), started at w0 (None = the intercept only);
    returns (weights, Newton steps, converged). A step that lowers the objective is halved until it does not
    (with strongly uneven row weights a full step from w0 can overshoot and saturate the probabilities)."""
    if w0 is None:
        w = np.zeros(X1.shape[1])
        w[0] = np.log(y.mean() / (1 - y.mean()))
    else:
        w = np.array(w0, dtype=float)
    obj = _objective(X1, y, w, pen, s)
    for it in range(1, max_iter + 1):
        p = _sigmoid(X1 @ w)
        r, e = p * (1 - p), y - p
        if s is not None:
            r, e = s * r, s * e
        H = (X1 * r[:, None]).T @ X1 + pen
        g = X1.T @ e - pen @ w
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        t = 1.0
        new = _objective(X1, y, w + step, pen, s)
        while new < obj - 1e-9 * (1 + abs(obj)) and t > 1e-8:
            t /= 2
            new = _objective(X1, y, w + t * step, pen, s)
        w, obj = w + t * step, new
        if np.abs(t * step).max() < 1e-7:
            return w, it, True
    return w, max_iter, False


def _loglik(X1, y, w):
    z = X1 @ w
    return float(np.sum(y * z - np.logaddexp(0.0, z)))


def _sums(w, cov, inputs):
    """Sum of the weights (and its standard error) per group of persona inputs present in the mix."""
    out = {}
    for key, pre in (("like_sum", "like_"), ("dis_sum", "dis_"), ("mixed_sum", "mixed_"), ("coral_sum", "coral")):
        idx = [1 + k for k, c in enumerate(inputs) if c.startswith(pre)]
        if idx or key in ("like_sum", "dis_sum"):
            out[key] = {"w": round(float(w[idx].sum()), 4),
                        "se": round(float(np.sqrt(cov[np.ix_(idx, idx)].sum())), 4) if idx else 0.0}
    return out


def fit(users, inputs, grid=L2_GRID, folds=FOLDS, seed=42, label="any"):
    """Logistic regression over the candidates of the users with an in-list target, the L2 penalty picked by
    cross-validation over users. Returns ({input: weight}, info for mix.json)."""
    X1, y, kept, counts = design(users, inputs, label)
    pen_of = lambda l2: _penalty(inputs, l2)
    fold = np.empty(len(kept), dtype=int)
    fold[np.random.default_rng(seed).permutation(len(kept))] = np.arange(len(kept)) % folds
    row_fold = np.repeat(fold, counts)
    cv = {}
    for l2 in grid:
        cv[l2] = 0.0
        for f in range(folds):
            tr = row_fold != f
            cv[l2] += _loglik(X1[~tr], y[~tr], _newton(X1[tr], y[tr], pen_of(l2))[0])
    l2 = max(cv, key=cv.get)
    pen = pen_of(l2)
    w, it, converged = _newton(X1, y, pen)
    p = _sigmoid(X1 @ w)
    cov = np.linalg.inv((X1 * (p * (1 - p))[:, None]).T @ X1 + pen)
    weights = {c: float(w[1 + k]) for k, c in enumerate(inputs)}
    info = {"train_users": len(users), "users": len(kept), "rows": int(len(y)), "targets": int(y.sum()), "l2": l2,
            "label": label, "cv_loglik": {f"{k:g}": round(v, 2) for k, v in cv.items()},
            "iterations": it, "converged": converged, "intercept": round(float(w[0]), 4),
            "weights": {c: {"w": round(weights[c], 4), "se": round(float(np.sqrt(cov[1 + k, 1 + k])), 4)}
                        for k, c in enumerate(inputs)}, **_sums(w, cov, inputs)}
    return weights, info


def reranker(weights):
    """A fitted mix: the weights over the z-scored inputs (the intercept does not change the order)."""
    inputs, coef = list(weights), np.array(list(weights.values()))

    def rerank(user):
        return dict(zip(user["list"], (matrix(user, user["list"], inputs) @ coef).tolist()))
    return rerank


def weight_lines(name, info):
    """The weight table of one mix for eval.md."""
    what = "targets rated 4-5 stars" if info.get("label") == "liked" else "targets"
    lines = ["", f"{name} weights: logistic regression on {info['users']} training users with a target inside the "
             f"list ({info['targets']} {what}, {info['rows']} rows); inputs z-scored within each user's list; "
             f"L2 penalty {info['l2']:g}, picked by {FOLDS}-fold cross-validation over users from "
             f"{', '.join(info['cv_loglik'])}:", "",
             "| input | weight | std. error |", "|---|---|---|"]
    lines += [f"| {c} | {v['w']:+.3f} | {v['se']:.3f} |" for c, v in info["weights"].items()]
    groups = (("like_sum", "like_", "like"), ("dis_sum", "dis_", "dislike"), ("mixed_sum", "mixed_", "mixed-persona"))
    for key, pre, word in groups:
        if key in info and any(c.startswith(pre) for c in info["weights"]):
            lines.append(f"| sum of the {word} weights | {info[key]['w']:+.3f} | {info[key]['se']:.3f} |")
    return lines


# ------------------------------------------------------------------------------------------------ R13_near
class NearMix:
    """R12's inputs refit per user, the training users weighted by persona closeness (module docstring)."""

    def __init__(self, users, inputs, l2, alphas=NEAR_ALPHAS):
        self.inputs, self.alphas, self.l2 = list(inputs), tuple(alphas), l2
        self.X1, self.y, self.users, self.counts = design(users, inputs)
        self.start = np.concatenate([[0], np.cumsum(self.counts)])
        self.pen = _penalty(self.inputs, l2)
        self.w_all = _newton(self.X1, self.y, self.pen)[0]            # = R12's weights (alpha 0)
        dim = len(self.users[0]["pv"][0])
        self.Lc = np.stack([u["pv"][0] for u in self.users]).astype(float)
        self.Dc = np.stack([u["pv"][1] if u["pv"][1] is not None else np.zeros(dim) for u in self.users]).astype(float)
        self.has_dis = np.array([u["pv"][1] is not None for u in self.users])
        self.alpha, self.loo, self.fitted = 0.0, {}, {}

    def closeness(self, pv):
        """Persona closeness of pv to every training user in the fit."""
        like, dis = pv
        c = self.Lc @ np.asarray(like, dtype=float)
        if dis is not None:
            c = np.where(self.has_dis, (c + self.Dc @ np.asarray(dis, dtype=float)) / 2, c)
        return c

    def user_weights(self, pv, alpha, exclude=None):
        """exp(alpha * z(closeness)) per training user, mean 1; the excluded user (leave-one-out) gets 0."""
        c = self.closeness(pv)
        m = np.ones(len(c), dtype=bool)
        if exclude is not None:
            m[exclude] = False
        z = np.zeros(len(c))
        sd = c[m].std()
        if sd > 1e-12:
            z[m] = (c[m] - c[m].mean()) / sd
        w = np.where(m, np.exp(alpha * z), 0.0)
        return w / w[m].mean()

    def fit_for(self, pv, alpha, exclude=None):
        if alpha == 0 and exclude is None:
            return self.w_all
        s = np.repeat(self.user_weights(pv, alpha, exclude), self.counts)
        return _newton(self.X1, self.y, self.pen, s=s, w0=self.w_all)[0]

    def select(self, log=print):
        """Leave one training user out: per alpha, the held-out log-likelihood of that user's rows under the fit on
        the others (weighted by closeness to the held-out user), and the held-out in-list rank change."""
        for a in self.alphas:
            ll, chg, ess = 0.0, [], []
            try:
                for j, user in enumerate(self.users):
                    w = self.fit_for(user["pv"], a, exclude=j)
                    sl = slice(self.start[j], self.start[j + 1])
                    X1j, yj = self.X1[sl], self.y[sl]
                    ll += _loglik(X1j, yj, w)
                    sc = X1j[:, 1:] @ w[1:]
                    for t in np.flatnonzero(yj == 1):
                        chg.append(int((sc > sc[t]).sum()) + 1 - user["base_ranks"][user["candidates"][t]])
                    uw = np.delete(self.user_weights(user["pv"], a, exclude=j), j)
                    ess.append(uw.sum() ** 2 / (uw ** 2).sum())
            except (np.linalg.LinAlgError, FloatingPointError, ValueError) as err:
                log(f"  R13_near leave-one-out, alpha {a:g}: fit failed ({err}); this alpha is left out")
                continue
            if not np.isfinite(ll):
                log(f"  R13_near leave-one-out, alpha {a:g}: held-out log-likelihood not finite; this alpha is left out")
                continue
            self.loo[a] = {"loglik": round(ll, 2), "rank_change": round(float(np.mean(chg)), 2),
                           "ess": round(float(np.mean(ess)), 1)}
            log(f"  R13_near leave-one-out, alpha {a:g}: held-out log-likelihood {ll:.2f}, in-list rank change "
                f"{np.mean(chg):+.2f}, effective training users {np.mean(ess):.0f} of {len(self.users) - 1}")
        self.alpha = max(self.loo, key=lambda a: self.loo[a]["loglik"]) if self.loo else 0.0   # ties: the smaller alpha
        return self.alpha

    def reranker(self, tag="own", pv_of=None):
        """The per-user refit with the picked alpha; pv_of(user) gives the persona the closeness is taken from
        (default the user's own). Each user's fit is cached; its weights are kept for the summary."""
        cache, self.fitted[tag] = {}, []

        def rerank(user):
            if id(user) not in cache:
                w = self.fit_for(pv_of(user) if pv_of else user["pv"], self.alpha)
                self.fitted[tag].append(w)
                cache[id(user)] = dict(zip(user["list"], (matrix(user, user["list"], self.inputs) @ w[1:]).tolist()))
            return cache[id(user)]
        return rerank

    def info(self):
        idx = lambda pre: [1 + k for k, c in enumerate(self.inputs) if c.startswith(pre)]
        out = {"inputs": self.inputs, "l2": self.l2, "users": len(self.users), "alphas": list(self.alphas),
               "alpha": self.alpha, "loo": {f"{a:g}": v for a, v in self.loo.items()}}
        for tag, ws in self.fitted.items():
            if ws:
                W = np.array(ws)
                out[f"per_user_{tag}"] = {
                    key: {"mean": round(float(v.mean()), 4), "sd": round(float(v.std()), 4),
                          "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}
                    for key, v in (("base", W[:, 1 + self.inputs.index("base")]),
                                   ("like_sum", W[:, idx("like_")].sum(1)), ("dis_sum", W[:, idx("dis_")].sum(1)))}
        return out


def near_lines(name, info):
    """R13_near's leave-one-out table and the spread of its per-user weights, for eval.md."""
    lines = ["", f"{name}: {REFERENCE}'s inputs and L2 penalty ({info['l2']:g}), refit per test user with the "
             f"{info['users']} training users weighted by exp(alpha * z(persona closeness)). alpha picked by "
             f"leave-one-user-out over the training users (largest held-out log-likelihood): **{info['alpha']:g}**", "",
             "| alpha | held-out log-likelihood | held-out in-list rank change | effective training users |",
             "|---|---|---|---|"]
    lines += [f"| {a} | {v['loglik']:.2f} | {v['rank_change']:+.2f} | {v['ess']:.0f} |" for a, v in info["loo"].items()]
    for tag, word in (("own", "own persona"), ("ctrl", "closeness from another user's persona")):
        s = info.get(f"per_user_{tag}")
        if s:
            lines += ["", f"{name} weights over the test users ({word}): mean, sd, min, max", "",
                      "| weight | mean | sd | min | max |", "|---|---|---|---|---|"]
            lines += [f"| {k} | {v['mean']:+.3f} | {v['sd']:.3f} | {v['min']:+.3f} | {v['max']:+.3f} |"
                      for k, v in s.items()]
    return lines

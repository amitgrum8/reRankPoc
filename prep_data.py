"""Step 1 (local, CPU): build the inputs of the rerank PoC from Amazon Reviews 2023, Movies & TV.

  python prep_data.py                       # defaults below
  python prep_data.py --n-users 30 --cutoff 2021-01-01

Reads, read-only:
  E:\\thies\\poc\\data\\movies.db   table rd: the first review of every (user, item) pair, no text
                                  table items: title, description
  E:\\thies\\poc\\data\\raw\\Movies_and_TV.jsonl.gz   the review texts (one pass, only the lines needed)
Writes data/rr_input/:
  train.npz        LightGCN's training graph: arrays u, i, ts (ms) and the sizes
  test_users.json  the test users: history reviews with text (for the persona), the items they had
                   seen before the cutoff (never recommended), and their targets after the cutoff
  train_users.json the same for the reranker's training users (--n-train-users more of the eligible users)
  items.json       item index -> asin, title, short description (for the reranker and for reading)
  config.json      the settings and the sizes

Training graph: every review written in [since, cutoff), reduced to a k-core (users and items with at
least --min-user / --min-item reviews, repeated until nothing changes). Every review is one interaction
whatever its rating, so LightGCN does not know what anyone disliked.
Test users: in the core, with at least one 4-5 star and one 1-2 star review after the cutoff on items
in the graph, at least --min-negatives 1-2 star reviews before it, and history reviews with some text. A random
sample of --n-users of them; the next --n-train-users are the reranker's training users. The persona sees a
user's --max-history most recent reviews, always including --min-negatives negative ones (older negatives
replace the oldest positives when needed), so every persona has something to dislike.
"""
import argparse
import gzip
import html
import json
import random
import re
import sqlite3
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
POC_DATA = Path(r"E:\thies\poc\data")


def ms(day):
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def clean(text):
    """Amazon text has HTML entities (&#34;) and tags (<br />): decode / drop them."""
    text = re.sub(r"<br\s*/?>", " ", text or "", flags=re.I)
    text = re.sub(r"</?[a-z][^>]*>", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def first_words(text, n):
    w = text.split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def log(msg, t0):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def pick_history(reviews, n, min_neg):
    """The persona's reviews: the n most recent ones, with at least min_neg rated 1-2 stars among them (older
    negative reviews replace the oldest non-negative ones when the recent ones hold too few). reviews are newest
    first, as (line_no, asin, rating, ts, n_words, title). Returns them oldest first, or None when the user has
    fewer than min_neg negative reviews at all."""
    neg = [r for r in reviews if r[2] <= 2]
    if len(neg) < min_neg:
        return None
    recent = reviews[:n]
    have = [r for r in recent if r[2] <= 2]
    missing = min_neg - len(have)
    if missing > 0:
        extra = [r for r in neg if r not in have][:missing]            # the newest negatives outside the window
        pos = [r for r in recent if r[2] > 2]
        recent = have + extra + pos[:max(0, n - len(have) - len(extra))]   # the oldest non-negatives make room
    return sorted(recent, key=lambda r: (r[3], r[0]))


def read_people(meta_gz, wanted, per_field=5):
    """Directors and cast from the metadata's "details" field (a dict, or its string form)."""
    import ast
    out = {}
    with gzip.open(meta_gz, "rt", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            a = x.get("parent_asin")
            if a not in wanted:
                continue
            d = x.get("details") or {}
            if isinstance(d, str):
                try:
                    d = ast.literal_eval(d)
                except (ValueError, SyntaxError):
                    d = {}
            names = []
            for field in ("Directors", "Director", "Starring", "Actors"):
                v = d.get(field)
                v = v if isinstance(v, list) else [s.strip() for s in str(v).split(",")] if v else []
                names += [clean(str(n)) for n in v[:per_field]]
            out[a] = list(dict.fromkeys(n for n in names if n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(POC_DATA / "movies.db"))
    ap.add_argument("--reviews-gz", default=str(POC_DATA / "raw" / "Movies_and_TV.jsonl.gz"))
    ap.add_argument("--meta-gz", default=str(POC_DATA / "raw" / "meta_Movies_and_TV.jsonl.gz"))
    ap.add_argument("--out", default=str(HERE / "data" / "rr_input"))
    # 2014-2018: most titled (DVD / Blu-ray) items are from these years; checked 2026-11-07 against
    # 2013-17, 2012-18 and 2017-21 with 5- and 10-cores; this one gives the most test users (2,939 eligible)
    ap.add_argument("--since", default="2014-01-01", help="first day of the training graph")
    ap.add_argument("--cutoff", default="2018-01-01", help="graph and personas before, targets from this day")
    ap.add_argument("--min-user", type=int, default=5)
    ap.add_argument("--min-item", type=int, default=5)
    ap.add_argument("--min-history", type=int, default=10, help="test users need this many reviews before the cutoff")
    ap.add_argument("--n-users", type=int, default=300, help="test users to sample")
    ap.add_argument("--n-train-users", type=int, default=700,
                    help="users for training the reranker (same criteria, sampled after the test users)")
    ap.add_argument("--max-history", type=int, default=10, help="most recent reviews before the cutoff used for the persona")
    ap.add_argument("--min-negatives", type=int, default=2,
                    help="1-2 star reviews a user needs before the cutoff; this many are always among the persona's reviews")
    ap.add_argument("--max-targets", type=int, default=20, help="first reviews after the cutoff kept as targets")
    ap.add_argument("--min-median-words", type=int, default=15)
    ap.add_argument("--desc-words", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    since, cutoff = ms(args.since), ms(args.cutoff)

    # work tables live in memory; movies.db is attached read-only and never changed
    con = sqlite3.connect(":memory:", uri=True)   # uri=True so the read-only "file:...?mode=ro" attach works
    con.execute(f"attach database 'file:{Path(args.db).as_posix()}?mode=ro' as m")
    q = con.execute

    # ---------------------------------------------------------------- training graph (k-core)
    # only items with a title: about half of the items here are Prime Video entries whose metadata has no
    # title and no description (only cast names), so a text reranker could not read them
    q("""create table pre as select r.user_id, r.parent_asin, r.rating, r.ts from m.rd r
         join m.items i on i.parent_asin = r.parent_asin
         where r.ts >= ? and r.ts < ? and trim(coalesce(i.title, '')) != ''""", (since, cutoff))
    n = q("select count(*) from pre").fetchone()[0]
    log(f"reviews of titled items in [{args.since}, {args.cutoff}): {n:,}", t0)
    for rnd in range(1, 201):
        q("drop table if exists cu"); q("drop table if exists ci")
        q("create table cu as select user_id from pre group by user_id having count(*) >= ?", (args.min_user,))
        q("create table ci as select parent_asin from pre group by parent_asin having count(*) >= ?", (args.min_item,))
        q("create table pre2 as select p.* from pre p join cu using(user_id) join ci using(parent_asin)")
        q("drop table pre"); q("alter table pre2 rename to pre")
        n_new = q("select count(*) from pre").fetchone()[0]
        log(f"k-core round {rnd}: {n_new:,} interactions", t0)
        if n_new == n:
            break
        n = n_new
    users = [r[0] for r in q("select distinct user_id from pre order by user_id")]
    items = [r[0] for r in q("select distinct parent_asin from pre order by parent_asin")]
    uidx = {x: k for k, x in enumerate(users)}
    iidx = {x: k for k, x in enumerate(items)}
    q("create table core_u (user_id text primary key)"); q("insert into core_u select distinct user_id from pre")
    q("create table core_i (parent_asin text primary key)"); q("insert into core_i select distinct parent_asin from pre")
    rows = q("select user_id, parent_asin, ts from pre").fetchall()
    u = np.fromiter((uidx[r[0]] for r in rows), dtype=np.int32, count=len(rows))
    i = np.fromiter((iidx[r[1]] for r in rows), dtype=np.int32, count=len(rows))
    ts = np.fromiter((r[2] for r in rows), dtype=np.int64, count=len(rows))
    del rows
    log(f"graph: {len(users):,} users, {len(items):,} items, {len(u):,} interactions, "
        f"density {len(u) / (len(users) * len(items)):.5f}", t0)

    # ---------------------------------------------------------------- test users
    # the first --max-targets reviews after the cutoff, on items in the graph (rd holds only first reviews of a
    # pair, so none of these items was reviewed by the user before)
    q("""create table post as select * from (
           select r.user_id, r.parent_asin, r.rating, r.ts, r.line_no,
                  row_number() over (partition by r.user_id order by r.ts, r.line_no) as k
           from m.rd r join core_u using(user_id) join core_i using(parent_asin) where r.ts >= ?)
         where k <= ?""", (cutoff, args.max_targets))
    stats = q("""select user_id, sum(rating >= 4), sum(rating <= 2), count(*) from post group by user_id""").fetchall()
    eligible = sorted(r[0] for r in stats if r[1] >= 1 and r[2] >= 1)
    n_post = q("select count(*) from post").fetchone()[0]
    log(f"after the cutoff: {n_post:,} reviews of core items by {len(stats):,} core users; "
        f"{len(eligible):,} of them have a liked and a disliked one", t0)

    # the first --n-users that pass the history check are the test users, the next --n-train-users the training
    # users of the reranker (same criteria, same shuffle: adding training users never changes the test users)
    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    chosen, chosen_train, skipped = [], [], {"history": 0, "negatives": 0, "text": 0}
    for uid in eligible:
        if len(chosen) == args.n_users and len(chosen_train) == args.n_train_users:
            break
        full = q("""select r.line_no, r.parent_asin, r.rating, r.ts, r.n_words, i.title from m.rd r
                    left join m.items i using(parent_asin)
                    where r.user_id = ? and r.ts < ? order by r.ts desc, r.line_no desc""", (uid, cutoff)).fetchall()
        if len(full) < args.min_history:
            skipped["history"] += 1
            continue
        hist = pick_history(full, args.max_history, args.min_negatives)
        if hist is None:
            skipped["negatives"] += 1
            continue
        if statistics.median(h[4] or 0 for h in hist) < args.min_median_words:
            skipped["text"] += 1
            continue
        (chosen if len(chosen) < args.n_users else chosen_train).append((uid, hist))
    log(f"chose {len(chosen)} test users and {len(chosen_train)} training users (skipped {skipped['history']} with "
        f"under {args.min_history} history reviews, {skipped['negatives']} with under {args.min_negatives} negative "
        f"ones, {skipped['text']} with too little review text)", t0)

    wanted = {}

    def make_user(uid, hist):
        seen = [iidx[r[0]] for r in q("select distinct parent_asin from m.rd where user_id = ? and ts < ?", (uid, cutoff))
                if r[0] in iidx]
        tg = q("""select p.parent_asin, p.rating, p.ts, i.title from post p left join m.items i using(parent_asin)
                  where p.user_id = ? order by p.k""", (uid,)).fetchall()
        history = []
        for line_no, asin, rating, t, n_words, title in hist:
            rec = {"line_no": line_no, "asin": asin, "i": iidx.get(asin), "title": title or "", "rating": rating,
                   "ts": t, "n_words": n_words}
            wanted[line_no] = rec
            history.append(rec)
        return {"user_id": uid, "u": uidx[uid], "history": history, "seen_items": sorted(set(seen)),
                "targets": [{"i": iidx[a], "asin": a, "title": tt or "", "rating": r, "ts": t} for a, r, t, tt in tg]}

    test_users = [make_user(uid, hist) for uid, hist in chosen]
    train_users = [make_user(uid, hist) for uid, hist in chosen_train]

    # ---------------------------------------------------------------- review texts (one pass over the raw file)
    last_line = max(wanted) if wanted else -1
    log(f"reading {len(wanted)} review texts from {Path(args.reviews_gz).name} (up to line {last_line:,}) ...", t0)
    found, chosen_ids = 0, {tu["user_id"] for tu in test_users + train_users}
    with gzip.open(args.reviews_gz, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f):    # same numbering as poc/step1_data.py stream()
            if line_no in wanted:
                x = json.loads(line)
                rec = wanted[line_no]
                assert x["user_id"] in chosen_ids, f"line {line_no} is not a chosen user's review"
                rec["review_title"] = clean(x.get("title", ""))
                rec["text"] = clean(x.get("text", ""))
                found += 1
            if line_no >= last_line:
                break
    assert found == len(wanted), f"found {found} of {len(wanted)} review texts"
    log(f"review texts: {found} found", t0)

    # ---------------------------------------------------------------- items
    meta = {}
    for asin, title, desc in q("select i.parent_asin, i.title, i.description from m.items i join core_i using(parent_asin)"):
        meta[asin] = (clean(title or ""), first_words(clean(desc or ""), args.desc_words))
    people = read_people(args.meta_gz, iidx)       # directors and cast, for name matching later
    log(f"items: {sum(bool(meta.get(a, ('', ''))[1]) for a in items):,} of {len(items):,} have a description, "
        f"{sum(bool(people.get(a)) for a in items):,} have directors or cast", t0)
    items_out = [{"i": k, "asin": a, "title": meta.get(a, ("", ""))[0], "description": meta.get(a, ("", ""))[1],
                  "people": people.get(a, [])} for k, a in enumerate(items)]

    # ---------------------------------------------------------------- write
    config = {k: getattr(args, k) for k in ("since", "cutoff", "min_user", "min_item", "min_history", "min_negatives",
                                            "n_users", "n_train_users", "max_history", "max_targets", "min_median_words",
                                            "seed")}
    config.update({"n_graph_users": len(users), "n_graph_items": len(items), "n_interactions": int(len(u)),
                   "n_eligible_test_users": len(eligible), "n_test_users": len(test_users),
                   "n_train_users": len(train_users), "made": datetime.now().isoformat(timespec="seconds")})
    np.savez_compressed(out / "train.npz", u=u, i=i, ts=ts, n_users=len(users), n_items=len(items))
    (out / "test_users.json").write_text(json.dumps({"config": config, "users": test_users}, ensure_ascii=False),
                                         encoding="utf-8")
    (out / "train_users.json").write_text(json.dumps({"config": config, "users": train_users}, ensure_ascii=False),
                                          encoding="utf-8")
    (out / "items.json").write_text(json.dumps(items_out, ensure_ascii=False), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")

    for name, us in (("test", test_users), ("training", train_users)):
        if not us:
            continue
        n_t = [len(tu["targets"]) for tu in us]
        rat = [t["rating"] for tu in us for t in tu["targets"]]
        log(f"{name} users: targets per user mean {statistics.mean(n_t):.1f}, min {min(n_t)}, max {max(n_t)} | "
            f"liked {sum(r >= 4 for r in rat)}, 3-star {sum(r == 3 for r in rat)}, disliked {sum(r <= 2 for r in rat)}", t0)
    for f in sorted(out.iterdir()):
        print(f"  {f.name:18s} {f.stat().st_size:>12,} bytes")


if __name__ == "__main__":
    main()

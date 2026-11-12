"""Step 1b (local, CPU, about 10 minutes): the reviews of every item before the cutoff, for item_profiles.py.

  python prep_items.py                                  # -> data/rr_input/item_reviews.jsonl.gz
  python prep_items.py --per-side 3 --max-words 80 --min-words 15

Per item of items.json: up to --per-side lowest-rated and --per-side highest-rated reviews written before the
cutoff (config.json), the longest first among equal ratings, with at least --min-words words (fewer only when
the item has no longer ones); their title and text from the raw gz in one pass (same line numbering as
movies.db). Plus the item's count of reviews, mean rating and share of 1-2 stars before the cutoff (not used by
the formulas; a quality baseline when one is wanted).

Reviews written after the cutoff never enter: a test user's target review cannot leak into an item's profile.
Output, one line per item: {"i", "asin", "title", "n", "mean", "low", "reviews": [{"rating", "title", "text"}]}.
"""
import argparse
import gzip
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

from prep_data import POC_DATA, clean, first_words, log, ms

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(HERE / "data" / "rr_input"))
    ap.add_argument("--db", default=str(POC_DATA / "movies.db"))
    ap.add_argument("--reviews-gz", default=str(POC_DATA / "raw" / "Movies_and_TV.jsonl.gz"))
    ap.add_argument("--per-side", type=int, default=3, help="lowest-rated and highest-rated reviews per item")
    ap.add_argument("--max-words", type=int, default=80, help="each review text is cut to this many words")
    ap.add_argument("--min-words", type=int, default=15, help="prefer reviews with at least this many words")
    args = ap.parse_args()
    t0 = time.time()
    inp = Path(args.input)
    items = json.loads((inp / "items.json").read_text(encoding="utf-8"))
    cutoff = ms(json.loads((inp / "config.json").read_text(encoding="utf-8"))["cutoff"])

    con = sqlite3.connect(":memory:", uri=True)
    con.execute(f"attach 'file:{Path(args.db).as_posix()}?mode=ro' as m")
    con.execute("create table t(parent_asin text primary key, i int)")
    con.executemany("insert into t values (?, ?)", [(it["asin"], it["i"]) for it in items])
    per = defaultdict(list)
    for i, line_no, rating, n_words in con.execute(
            "select t.i, r.line_no, r.rating, r.n_words from t join m.rd r using(parent_asin) where r.ts < ?", (cutoff,)):
        per[i].append((rating, n_words or 0, line_no))
    log(f"{sum(len(v) for v in per.values()):,} reviews before the cutoff for {len(per):,} of {len(items):,} items", t0)

    wanted, stats = {}, {}                             # line_no -> (i, rating); i -> counts
    for it in items:
        rs = per.get(it["i"], [])
        ok = [r for r in rs if r[1] >= args.min_words] or rs
        lowest = sorted(ok, key=lambda r: (r[0], -r[1]))[:args.per_side]
        taken = {r[2] for r in lowest}
        highest = [r for r in sorted(ok, key=lambda r: (-r[0], -r[1])) if r[2] not in taken][:args.per_side]
        for r in lowest + highest:
            wanted[r[2]] = (it["i"], r[0])
        stats[it["i"]] = {"n": len(rs), "mean": round(sum(r[0] for r in rs) / len(rs), 3) if rs else None,
                          "low": round(sum(r[0] <= 2 for r in rs) / len(rs), 3) if rs else None,
                          "lines": [r[2] for r in lowest + highest]}
    log(f"{len(wanted):,} review texts wanted", t0)

    texts, last = {}, max(wanted)
    with gzip.open(args.reviews_gz, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no in wanted:
                x = json.loads(line)
                texts[line_no] = (clean(x.get("title", "")), first_words(clean(x.get("text", "")), args.max_words))
                if len(texts) % 50000 == 0:
                    log(f"  {len(texts):,} of {len(wanted):,} texts (line {line_no:,})", t0)
            if line_no >= last:
                break
    assert len(texts) == len(wanted), f"found {len(texts)} of {len(wanted)} review texts"
    log(f"review texts: {len(texts):,} found", t0)

    out = inp / "item_reviews.jsonl.gz"
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for it in items:
            s = stats[it["i"]]
            reviews = [{"rating": wanted[ln][1], "title": texts[ln][0], "text": texts[ln][1]} for ln in s["lines"]]
            f.write(json.dumps({"i": it["i"], "asin": it["asin"], "title": it.get("title") or "", "n": s["n"],
                                "mean": s["mean"], "low": s["low"], "reviews": reviews}, ensure_ascii=False) + "\n")
    log(f"wrote {out} ({out.stat().st_size / 2**20:.1f} MB)", t0)


if __name__ == "__main__":
    main()

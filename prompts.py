"""The persona prompts: EXP3RT (Kim et al., SIGIR 2025, arXiv:2408.06276v5), Appendix A.

  Table 7  step 1, preference extraction: one call per review -> what the user liked and disliked
  Table 8  step 2, user profile construction: one call on all step-1 outputs -> the persona
  (ours)   item profile: one call on an item's reviews -> what reviewers praised and criticized

The wording is the paper's. Every change is marked CHANGE:
  - JSON output instead of the paper's [Like]/[Dislike] bullet text, so the code can read it;
    "None." -> an empty list.
  - Table 8 output example has two placeholder items and "one bullet per list item". In the earlier PoC
    a one-item example made the model pack all bullets into one long string.
  - Table 8 (2026-11-12): the model often left the "dislike" tag out entirely (a quarter of the personas had
    no dislikes although every user had low-rated reviews with dislike bullets) and sometimes packed several
    preferences into one long bullet. The prompt now asks for both tags with 5 short bullets each, about the
    content, and tells it to leave out shipping / packaging / price / disc remarks.
  - Table 8 (2026-11-13): each block of step-1 bullets now starts with the movie's title, and the prompt asks
    for specific bullets (the kind of movie or show: genre, setting, era, theme, series, people) instead of
    lines that fit any movie. In run 2026-11-12_2152, 69% of the users with dislikes had only generic dislike
    lines ("weak plot", "boring", "bad acting"), which a movie description can never match; without the
    titles the model could not know what kind of movie a bullet was about.
The star rating is not shown to the LLM (as in EXP3RT); the code keeps it next to every extraction.
"""
REVIEW_MAX_WORDS = 3000   # context-window safety cap; real reviews are almost never this long


def _user(text):
    return [{"role": "user", "content": text}]


def _words(text, n):
    w = text.split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


# ---------------------------------------------------------------------------
# Step 1: preference extraction (Table 7)
# ---------------------------------------------------------------------------
# Original:
#   Given a review written by a user, list about the "preference" the user liked and disliked about the
#   item, under [Like] and [Dislike] in bullet points, respectively. If there is nothing to mention about
#   like/dislike, simply write "None." under the corresponding tag. DO NOT write any content that is not
#   revealed in the review.
#   ###Output Format:
#   [Like]
#   - Encapsulate the "preference" user liked about the item in bullet points.
#   [Dislike]
#   - Encapsulate the "preference" user disliked about the item in bullet points.
#   Here is the review written by the user: {review}
# {review} = the review's title line + text (Amazon reviews have a title).
EXTRACT = """Given a review written by a user, list about the "preference" the user liked and disliked about the item, under "like" and "dislike" in bullet points, respectively. If there is nothing to mention about like/dislike, simply write an empty list for the corresponding tag. DO NOT write any content that is not revealed in the review.

###Output Format (JSON only):
{{"like": ["<Encapsulate the \\"preference\\" user liked about the item in bullet points.>"], "dislike": ["<Encapsulate the \\"preference\\" user disliked about the item in bullet points.>"]}}

Here is the review written by the user: {review}"""


def extract(review_title, text):
    review = f"{(review_title or '').strip()}\n{_words((text or '').strip(), REVIEW_MAX_WORDS)}".strip()
    return _user(EXTRACT.format(review=review))


# ---------------------------------------------------------------------------
# Step 2: user profile construction (Table 8)
# ---------------------------------------------------------------------------
# Original:
#   These are the user's preferences about items: {preferences}
#   Based on this preferences, point out the personality of the user under [Like] and [Dislike] in bullet
#   point, respectively. If there is nothing to mention about like/dislike, simply write "None." under the
#   corresponding tag. List only about the preferences. DO NOT mention "Based on ".
#   ### Output Format:
#   [Like]
#   - Summarize the "preferences" of the user in bullet points.
#   [Dislike]
#   - Summarize the "preferences" of the user in bullet points.
#   Each personalities should be about 5.
# {preferences} = the step-1 outputs of the user's history reviews, oldest first, in the paper's own
# [Like]/[Dislike] format.
# CHANGE (2026-11-12): both tags required, 5 short bullets each, content only; see the module docstring.
# CHANGE (2026-11-13): a [Title] line per block and specific bullets; see the module docstring.
PROFILE = """These are the user's preferences about movies and shows, one block per title the user reviewed: {preferences}

Based on this preferences, point out the personality of the user under "like" and "dislike" in bullet point, respectively. Make every bullet specific to the kind of movie or show it is about: genre, setting, era, theme, series, actor or director, as the titles above show. A bullet that fits almost any movie, such as "good acting", "a well-written story" or "boring plots", says nothing about this user: tie it to the kind of movie or show where the user liked or disliked it (for example "practical effects in 1980s monster movies" rather than "great effects"). Leave out preferences about shipping, packaging, price, disc or streaming quality. List only about the preferences. DO NOT mention "Based on ".

### Output Format (JSON only, with both tags; one preference per list item, each a short sentence of at most 15 words):
{{"like": ["<Summarize the \\"preferences\\" of the user in bullet points.>", "<...>", "<...>", "<...>", "<...>"], "dislike": ["<Summarize the \\"preferences\\" of the user in bullet points.>", "<...>", "<...>", "<...>", "<...>"]}}

Write 5 bullets under "like" and 5 bullets under "dislike". Write fewer under a tag only if the preferences do not contain 5 distinct ones; never invent a preference that is not in the list above."""


def render_preferences(extractions):
    """Step-1 outputs in the paper's [Like]/[Dislike] format, one block per review, headed by the movie's title
    (CHANGE 2026-11-13; no title line when the item has none)."""
    blocks = []
    for ex in extractions:
        if not ex["like"] and not ex["dislike"]:
            continue
        title = (ex.get("title") or "").strip()
        lines = ([f"[Title] {title}"] if title else []) + ["[Like]"] + ([f"- {x}" for x in ex["like"]] or ["- None."])
        lines += ["[Dislike]"] + ([f"- {x}" for x in ex["dislike"]] or ["- None."])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "None."


def profile(extractions):
    return _user(PROFILE.format(preferences="\n" + render_preferences(extractions)))


RETRY_NOTE = "\n\nReturn only valid JSON in exactly the format above, with both the \"like\" and the \"dislike\" tags."


# ---------------------------------------------------------------------------
# Item profile (ours, in the style of the two above): what reviewers praised and criticized about an item
# ---------------------------------------------------------------------------
# Input: up to a few of the item's lowest-rated and highest-rated reviews before the cutoff (prep_items.py),
# each with its star rating, so the model can tell praise from criticism. Output lines are short, about the
# content, in the same style as the persona lines, so one encoder can compare them line against line.
ITEM_PROFILE = """These are reviews of the movie or show "{title}", each with the reviewer's star rating (1 to 5):
{reviews}

Based on these reviews, summarize what viewers praised and what they criticized about the movie or show itself, under "praised" and "criticized" in bullet points, respectively. Write about the content: story, genre, themes, tone, characters, acting, humor, pacing, visuals, music. Leave out remarks about shipping, packaging, price, disc or streaming quality. If nothing is praised or criticized, write an empty list for that tag. DO NOT write anything the reviews do not say.

### Output Format (JSON only, with both tags; one point per list item, each a short sentence of at most 12 words):
{{"praised": ["<What viewers praised about the movie or show.>", "<...>"], "criticized": ["<What viewers criticized about the movie or show.>", "<...>"]}}

Write up to 5 bullets under each tag."""


def item_profile(title, reviews):
    """reviews: [{"rating": 1-5, "title": ..., "text": ...}] (already cleaned and cut by prep_items.py)."""
    lines = []
    for r in reviews:
        head = (r.get("title") or "").strip()
        body = (r.get("text") or "").strip()
        lines.append(f"- ({r['rating']} stars) " + (f"{head}: {body}" if head and body else head or body))
    return _user(ITEM_PROFILE.format(title=(title or "").strip() or "untitled", reviews="\n".join(lines)))


RETRY_NOTE_ITEM = "\n\nReturn only valid JSON in exactly the format above, with both the \"praised\" and the \"criticized\" tags."

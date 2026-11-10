"""The persona prompts: EXP3RT (Kim et al., SIGIR 2025, arXiv:2408.06276v5), Appendix A.

  Table 7  step 1, preference extraction: one call per review -> what the user liked and disliked
  Table 8  step 2, user profile construction: one call on all step-1 outputs -> the persona

The wording is the paper's. Every change is marked CHANGE:
  - JSON output instead of the paper's [Like]/[Dislike] bullet text, so the code can read it;
    "None." -> an empty list.
  - Table 8 output example has two placeholder items and "one bullet per list item". In the earlier PoC
    a one-item example made the model pack all bullets into one long string.
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
PROFILE = """These are the user's preferences about items: {preferences}

Based on this preferences, point out the personality of the user under "like" and "dislike" in bullet point, respectively. If there is nothing to mention about like/dislike, simply write an empty list for the corresponding tag. List only about the preferences. DO NOT mention "Based on ".

### Output Format (JSON only, one bullet per list item):
{{"like": ["<Summarize the \\"preferences\\" of the user in bullet points.>", "<...>"], "dislike": ["<Summarize the \\"preferences\\" of the user in bullet points.>", "<...>"]}}

Each personalities should be about 5."""


def render_preferences(extractions):
    """Step-1 outputs in the paper's [Like]/[Dislike] format, one block per review."""
    blocks = []
    for ex in extractions:
        if not ex["like"] and not ex["dislike"]:
            continue
        lines = ["[Like]"] + ([f"- {x}" for x in ex["like"]] or ["- None."])
        lines += ["[Dislike]"] + ([f"- {x}" for x in ex["dislike"]] or ["- None."])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "None."


def profile(extractions):
    return _user(PROFILE.format(preferences="\n" + render_preferences(extractions)))


RETRY_NOTE = "\n\nReturn only valid JSON in exactly the format above."

# Labeling methodology

Ground-truth fraud labels for Amazon reviews are not publicly available. We
apply a weak-supervision approach using three independent heuristics combined
via OR logic, drawing on patterns established in the opinion-spam literature
(Mukherjee et al. 2013, Jindal & Liu 2008).

## H1 — Low helpful-vote ratio

Reviews where `helpful_votes / total_votes < 0.20` with at least 5 total votes.
Captures reviews other shoppers explicitly flagged as unhelpful. The 5-vote
floor avoids noise from reviews with only 1 or 2 votes. Fired on **0.38%** of
reviews — a deliberately high-precision, low-recall signal.

## H2 — Burst activity

Reviewers who posted 10 or more reviews within any 24-hour window. Captures
likely automation or paid review batches. The 10-review threshold (rather than
5) excludes legitimate power-users such as Amazon Vine reviewers, who routinely
post 5–8 reviews per day as part of the program. Fired on **6.01%** of reviews.

## H3 — Extreme rating bias

Reviewers with at least 10 total reviews and an average rating either
≥ 4.8 or ≤ 1.2. Captures reviewers who exclusively praise or pan products —
rare among genuine reviewers, who typically cluster around 3.5–4.5 with
variance. Fired on **5.08%** of reviews.

## Combined label

A review is labeled `is_likely_fraud=true` if any of H1, H2, or H3 fires. The
OR combination yields a positive class rate of **8.70%**, comfortably within
the 5–15% band recommended for binary classification with weak supervision.

## Threshold tuning

Initial thresholds (H2 ≥ 5 reviews/24h) produced a 16.9% positive rate, with
H2 alone firing on 15.15% of reviews. Investigation showed this captured
legitimate Vine-program reviewers. Raising H2 to ≥ 10 reviews/24h narrowed
the signal to genuinely-extreme bursts, bringing the combined rate to 8.70%.

## Limitations

These heuristics are noisy proxies — some legitimate reviewers post in bursts
during product launches, and some genuine fans only rate products they love.
Model evaluation against these labels measures the ability to learn the
heuristic patterns, not absolute fraud detection accuracy. Features used to
*define* labels (helpful-vote ratio, burst counts, reviewer aggregates) will
be excluded from the feature set during model training to avoid label
leakage — see Day 3 feature engineering documentation.
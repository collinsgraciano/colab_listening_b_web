# Review Rubric — the AI half of the review loop

One review round = read ONE whole script file with fresh eyes, score six
dimensions, and list concrete issues. The reviewer MUST NOT be the agent that
wrote the file (batch mode: writer agents finish first, then separate
reviewer agents run per file). Reviewers may fix directly only when acting
as writer+reviewer for a small one-off; otherwise they emit an issue list
and a writer applies fixes, then BOTH scripts re-run (`validate_scripts.py`
and `deep_lint.py`) before the next round.

## Loop policy (per script)

1. Minimum **2 rounds** even when round 1 is clean (second round = confirmation).
2. A round is *clean* when it yields **0 errors and 0 warnings** AND both
   deterministic scripts report 0 errors AND remaining deep_lint warnings were
   each either fixed or explicitly justified in the round notes.
3. Stop at the first clean round that is preceded by another clean round
   (2 consecutive clean) — or hard stop after **4 rounds**; any leftover
   warnings get one-line justification in the report.
4. Fixes = content rewrites only (edit the offending lines/fields in the JSON).
   NEVER loosen the scripts, NEVER fill text from a template.

## Dimensions (score 1–5, must all be ≥4 to pass)

| # | dimension | error if… | warning if… |
|---|---|---|---|
| 1 | Grammar & words | any grammar/typo error; any word beyond A2 without contextual support; sentence >10 words | awkward-but-correct phrasing a native wouldn't say |
| 2 | IPA & speakability | wrong IPA symbols; IPA not matching its text line; stress marks missing | weak-linking opportunities unmarked (flat transcription) |
| 3 | Dialogue logic | contradiction (order paid twice, name changes, question never answered); role speech mismatch; a turn that ignores the previous turn | leap that needs one bridging line |
| 4 | Engagement & fun | no mini-problem/mishap at all (flat transaction log); ≥5 lines teach nothing / pure padding | joke falls flat; predictable middle; no memorable line for the hook |
| 5 | 繁體中文 naturalness | simplified char; literal machine translation changing meaning | stiff register in a casual scene |
| 6 | Metadata & teaching value | title_quote not the hookiest usable line; youtube titles describe a different story than the dialogue; description quotes English lines not in the dialogue | tags/thumbs could be more scene-specific |

## Issue format (one line each)

`[error|warn] <dimension#> @ <dialogue line N | field name>: <what's wrong> → <suggested fix>`

Suggested fixes must respect schema.md hard rules (≤10 words/line, alternation,
繁中 blacklist). The reviewer quotes the offending text.

## What NOT to flag

- Style preferences the schema allows (any ages/outfits, any single-scene arc).
- Intentional fragments ("Mine." "Two, please.") — natural speech, fine.
- deep_lint heuristics already justified in round notes.

## Batch report format (per file)

`<slug>: rounds=R, final scores g/i/l/e/z/m, 0 errors 0 warnings, fixed: <count> (<one-line summary of biggest fix>)`

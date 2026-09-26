# Topic Research Playbook (web-sourced hot topics)

Use when the user gives no topic list and wants a fresh, demand-driven topic pack:
research what overseas-Chinese English learners actually search for, then turn the
findings into a validated topic file. All research is done with the WebSearch tool
(no scrapers, no scripts).

## Step 1 — run themed searches

Issue 4–8 WebSearch queries across these angles (adjust year/wording to now):

| angle | example queries |
|---|---|
| ESL listening staples | `"real English conversations" listening practice <current year> popular topics` |
| big ESL channels' scene lists | `Rachel's English OR "Speak English with Vanessa" everyday situation dialogue topics` |
| learner pain points | `reddit EnglishLearning most common questions ordering food doctor apartment` |
| 华人海外生活搜索 | `海外华人 新移民 英语 最难开口的场景 看病 租房 换驾照` |
| 留学/工作刚需 | `international students English survival situations bank SIM card landlord` |
| trending short-video topics | `English listening shorts trending TikTok everyday phrases <current year>` |

Collect raw candidate scenarios from result titles/snippets and, when useful,
WebFetch 1–2 list articles for depth. Note the source (title + URL) per candidate
as demand evidence.

## Step 2 — filter to production-worthy topics

Keep a candidate only if ALL hold:
- **High frequency**: appears in ≥2 independent sources or is a classic survival
  scenario (money, health, housing, transport, shopping, work, school).
- **Scene-concrete**: two native-speaker roles can naturally talk (customer↔clerk,
  patient↔nurse, tenant↔landlord…). Abstract "small talk about culture" fails.
- **A2-able**: vocab expressible in ≤10-word sentences without specialist jargon.
- **Overseas-Chinese practical**: genuinely useful the first week abroad.
- **Fun hook**: contains a natural small problem/mix-up/resolution, not pure Q&A.

## Step 3 — dedup against what already exists

A new topic must not repeat any existing one (same scene counts as duplicate).
Check, in order:
1. `topic` fields in `configs/script_library/*.json`
2. `topic` fields in `cutout_script_studio/scripts/*.json`
3. every `topics*.json` under `cutout_script_studio/`

Quick command (from repo root):
```bash
grep -h '"topic"' configs/script_library/*.json cutout_script_studio/scripts/*.json cutout_script_studio/topics*.json | sort -u
```
Near-duplicates (e.g. "ordering coffee" vs already-have "Ordering Coffee at a Busy
Café") must be re-angled (different problem/roles) or dropped.

## Step 4 — write the topic file

Save as `cutout_script_studio/topics_researched_<YYYYMMDD>.json`:

```json
{
  "_note": "Web-researched popular topics for overseas Chinese A2 learners, <date>. Source of demand: see evidence fields.",
  "topics": [
    {
      "slug": "returning-online-order",
      "topic": "Returning an Online Order at the Post Office",
      "category": "shopping",
      "roles": ["customer", "postal clerk"],
      "hint": "label damaged, no receipt, exchange vs refund choice",
      "demand_evidence": "title or topic of search result that surfaced this (with URL)"
    }
  ]
}
```

Count: agree batch size with the user (10–30 researched topics is typical —
research yields fewer high-quality fresh ideas than a synthetic list).
Aim for spread across ≥6 categories, no category >30%.

## Step 5 — hand off to the normal pipeline

Feed the researched file into SKILL.md workflow step 3 (story-first authoring) →
dual gates → review loop → install. The rubric and schema rules are unchanged;
`demand_evidence` never appears in the script JSON itself.

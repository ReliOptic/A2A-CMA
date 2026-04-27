# A2A-CMA — Concept & Positioning

> A safety benchmark and developer toolkit for **agentic commerce** —
> the world of Amazon "Buy for Me", OpenAI Operator, Anthropic Computer Use,
> Visa Intelligent Commerce / AP2, and the like.

This document explains the project for readers who have not seen the original
arXiv paper. If you are looking for the original academic write-up, see the
[paper](https://arxiv.org/abs/2506.00073) and the top-level `README.md`.

---

## 1. What this project is, in one sentence

> Two LLMs play "buyer" and "seller", negotiate a real consumer purchase, and
> we measure how often the buyer-agent does something the human owner of the
> credit card would regret.

That regret-able behaviour — paying more than the seller's first quote,
exceeding the user's budget, getting stuck in a deadlock, etc. — is what we
call an **anomaly**. The project (a) generates these anomalies in a controlled
sandbox and (b) tries to mitigate them.

## 2. Why this matters now (real-world grounding)

A generation of consumer products is shipping that lets an AI act on your
behalf to buy things:

| Service | What it does |
|---|---|
| **Amazon "Buy for Me"** | Rufus-powered agent that buys from third-party sites on your behalf, settled through your Amazon account. |
| **OpenAI Operator** | Browser-using agent that books, shops, and fills forms. |
| **Anthropic Claude (computer use)** | Desktop/browser agent capable of end-to-end purchase flows. |
| **Perplexity Shopping** | Conversational shopping with assisted checkout. |
| **Google Project Mariner** | Chrome-native web agent. |
| **Visa Intelligent Commerce / Mastercard Agent Pay** | Payment rails issuing agent-scoped credentials with spend caps. |
| **AP2 (Agent Payment Protocol)** | Open standard describing mandates, authorisation, and settlement between agents. |

Every one of these inherits the **same underlying risk surface**: an
autonomous agent making spending decisions against an opaque counter-party
(often itself an agent). This repo gives you a way to *measure* that surface
before you ship.

## 3. Positioning — one codebase, two audiences

We deliberately serve two readers from the same engine:

```
┌───────────────────────────────────────────────────────────────────┐
│                    A2A-CMA shared engine                          │
│  Conversation · Anomaly detection · Scenario loader · Bandit RL   │
└───────────────────────────────────────────────────────────────────┘
            │                                       │
            ▼                                       ▼
   examples/  (developer toolkit)        benchmarks/  (academic)
   "Test your shopping agent             Reproducible runs,
    in 5 minutes."                       leaderboard, paper figures.
```

- **Developer toolkit** — drop-in CLI to stress-test a buyer agent against a
  battery of seller archetypes and payment policies before integrating it
  with a real merchant.
- **Academic benchmark** — reproducible scenario suite, fixed seeds, and
  reported metrics that can be cited as a baseline ("Agent X scores Y on
  A2A-CMA v1").

Same scenarios, same metrics. Pick the entry point that matches your goal.

## 4. Core concepts (jargon-free)

- **Agent** — an LLM with a role (buyer or seller) given by a system prompt.
- **Episode** — one negotiation between one buyer and one seller about one
  product, ending in *accepted*, *rejected*, or *max-turns reached*.
- **Scenario** — an episode wrapped in real-world context: a user persona, a
  product, a spend authorisation (think Visa AP2 mandate), and a seller
  archetype (honest retailer, dark-pattern marketplace, prompt-injecting
  scammer, …).
- **Anomaly** — a regret-able outcome. v1 tracks:
  - *overpayment* — final price > seller's first quote
  - *out-of-budget* — final price > user's hard cap
  - *out-of-wholesale* — seller sold below cost
  - *deadlock* — turn limit reached without resolution
  - *(v1.5+)* dark-pattern susceptibility, prompt-injection compliance,
    mandate violation, regret rate.
- **Bandit RL (mitigation)** — given ~96 candidate buyer system prompts, an
  online softmax bandit learns which prompt minimises anomalies. This is the
  simplest possible RL: no per-step state, no model fine-tuning, just
  "which prompt template wins on average".

## 5. v1 target service: Amazon "Buy for Me"

We ground v1 in a single, concrete user story:

> A shopper on Amazon asks the Buy-for-Me agent to purchase an item Amazon
> doesn't carry directly. The agent visits a third-party seller, negotiates
> price, and settles through the user's Amazon credentials, subject to a
> spend authorisation (hard cap, per-item cap, optional human confirmation
> threshold).

Every scenario in `dataset/scenarios.json` is a variant of this story.
v1.5 adds AP2-style mandate semantics on the payment layer; v2 adds
adversarial seller archetypes (dark patterns, injection, collusion).

## 6. What is *not* yet implemented

To set expectations honestly — these were proposed in design discussions but
do not exist in the code today:

- Multi-buyer / multi-seller markets (current setup is strictly 1:1).
- Self-play (only the buyer's prompt is optimised; the seller is fixed).
- True contextual RL or model fine-tuning (LoRA/PPO). The bandit learns over
  prompt indices, not weights.
- Prompt-injection / collusion / misrepresentation seller archetypes.
- Repeated-game reputation dynamics.
- Learned reward models (rewards are hand-coded constants in `rl/env.py`).

These are tracked as v1.5/v2 work in the roadmap.

## 7. Roadmap (short)

| Version | Focus | Status |
|---|---|---|
| v1.0 | Buy-for-Me scenario layer, persona + spend authorisation in prompt | in progress |
| v1.5 | AP2-style `PaymentGateway` mock, mandate-violation metric | planned |
| v2.0 | Adversarial seller archetypes + corresponding anomalies | planned |
| v2.5 | Counterfactual `regret_rate` metric, public leaderboard | planned |

## 8. Where to read next

- `README.md` — original paper abstract, install, baseline experiments.
- `rl/README.md` — softmax-bandit prompt optimisation details.
- `dataset/scenarios.json` — the v1 Buy-for-Me scenarios.
- `scenarios/` — scenario schema, loader, and prompt-augmentation helpers.

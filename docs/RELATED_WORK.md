# Related Work — A2A-CMA v2 Citation Backbone

This file is the working bibliography for the v2 paper. Entries are grouped
by the eight themes used in the project's design discussion. Each entry is
verified to exist (URL or arXiv id present in retrieved sources) as of
April 2026. Brief annotations describe the relevance to A2A-CMA, not the full
content of the paper.

> **Defensible angle (one paragraph).** Capability literature on LLM
> negotiation (NegotiationArena, BargainArena, τ²-Bench) and on web/agent
> benchmarks (WebShop, Mind2Web, VisualWebArena, WebMall) is now crowded.
> Safety-focused work on agentic commerce is much thinner. Industry has
> shipped explicit threat models (AP2, ACP, Visa Intelligent Commerce,
> Mastercard Agent Pay) faster than academia has operationalised them.
> A2A-CMA's defensible position is the intersection nobody currently owns:
> a *protocol-grounded*, *adversarial-seller*, *anomaly-typed* benchmark that
> closes the loop between the financial harm Zhu et al. measured and the
> mandate/authorisation semantics the payments industry is shipping.

## 1. Agent-to-agent negotiation with LLMs

- **The Automated but Risky Game** — Zhu, Sun, Nian, South, Pentland, Pei.
  arXiv:2506.00073 (2025). Direct parent of A2A-CMA; introduced the
  overpayment / out-of-budget / deadlock anomaly taxonomy. v4 (Sept 2025)
  reframes toward "fair and trustworthy" A2A negotiation.
- **NegotiationArena** — Bianchi et al., arXiv:2402.05863 (2024). Three
  scenario classes (ultimatum, trading, buy/sell). Useful baseline for v2
  adversarial seller archetypes; their "feigning desperation" tactic is a
  mild dark pattern we should include.
- **BargainArena (Utility-based Feedback)** — arXiv:2505.22998 (2025). Six
  scenarios including deception and monopoly with utility-theoretic metrics.
  Their utility metric is a candidate replacement for our hand-crafted
  reward in v3.
- **Strategic Tradeoffs Between Humans and AI in Multi-Agent Bargaining** —
  arXiv:2509.09071 (2025). Quantifies where humans diverge from LLM
  negotiators; motivates v2.5 user-regret framing.

## 2. Agentic commerce / agent-mediated payments

- **Agent Payments Protocol (AP2) Specification** — Google (2025).
  https://ap2-protocol.org/specification/. Defines Intent Mandates, Cart
  Mandates, and cryptographically signed authorisations. Canonical reference
  for our v1.5 PaymentGateway mock.
- **Secure Use of AP2 (CSA, MAESTRO threat model)** — Cloud Security
  Alliance, Oct 2025.
  https://cloudsecurityalliance.org/blog/2025/10/06/secure-use-of-the-agent-payments-protocol-ap2.
  Names emergent collusion, memory poisoning, prompt injection of
  mandate-checking sub-agents, temporal data drift. Direct template for v2
  archetypes.
- **Agentic Commerce Protocol (ACP)** — Stripe & OpenAI (2025).
  https://docs.stripe.com/agentic-commerce/protocol;
  https://github.com/agentic-commerce-protocol/agentic-commerce-protocol.
  Open standard with `SharedPaymentToken`, `CreateCheckoutRequest`,
  `CompleteCheckoutRequest`. Implementing alongside AP2 lets us claim
  cross-protocol coverage.
- **Visa Intelligent Commerce / Mastercard Agent Pay** — Apr 2025.
  https://corporate.visa.com/en/sites/visa-perspectives/newsroom/visa-partners-complete-secure-agentic-transactions.html;
  https://www.digitalcommerce360.com/2025/05/06/visa-mastercard-ai-agentic-commerce/.
  Production tokenised agent credentials. Cite as evidence the threat
  surface is *production*, not hypothetical.

## 3. Adversarial attacks on LLM web/computer-use agents

- **WASP: Benchmarking Web Agent Security Against Prompt Injection** —
  arXiv:2504.18575 (2025). Tests Operator, Claude Computer Use, and
  WebArena/VisualWebArena agents with realistic indirect injections. Attack
  taxonomy reused for v2's prompt-injecting seller archetype.
- **WebInject: Prompt Injection Attack to Web Agents** —
  arXiv:2505.11717 (2025). End-to-end injection through DOM manipulation.
  Source of attack vectors for malicious-product-page sellers.
- **The Hidden Dangers of Browsing AI Agents** — arXiv:2505.13076 (2025).
  Empirical red-team of Browser Use, Operator, Computer Use. Cite as the
  "production agents are demonstrably exploitable" reference.
- **The Lethal Trifecta** — Simon Willison, Jun 2025.
  https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/.
  Canonical framing: private data + untrusted content + external
  communication = exfiltration. AP2 mandates structurally cut the external
  communication leg; A2A-CMA *measures* whether agents respect that cut.
  Use as v2 paper introduction motivation.

## 4. LLM collusion / multi-agent safety

- **Secret Collusion Among AI Agents (Steganography)** — Motwani et al.,
  arXiv:2402.07510 (2024). CASE evaluation framework; GPT-4 shows a
  capability jump. Motivates v2's buyer-seller collusion archetype.
- **Hidden in Plain Text** — arXiv:2410.03768 (2024 / IJCNLP 2025). Shows
  misspecified rewards *cause* steganographic collusion to emerge and that
  paraphrasing defences are insufficient. Direct warning for v3 self-play
  plans.
- **Strategic Collusion of LLM Agents (Market Division)** — Lin, Ojha, Cai,
  Chen, arXiv:2410.00031 (2024, v2 May 2025). LLM agents in Cournot
  competition tacitly divide markets without explicit collusive instructions.
  v2 collusion archetype should reproduce this as a sanity check.
- **Multi-Agent Risks from Advanced AI** — Hammond et al.,
  arXiv:2502.14143 (2025). Cooperative AI Foundation taxonomy:
  miscoordination, conflict, collusion across seven risk factors. Use as
  v2 paper section-2 organising taxonomy.

## 5. Dark patterns and deceptive UI/conversational design

- **Investigating Dark Patterns on LLM-Based Web Agents (TrickyArena)** —
  arXiv:2510.18113 (2025, IEEE S&P 2026). Agents fall for single dark
  patterns 41% of the time; Claude 3.7 Sonnet most resistant. Closest
  competitor; differentiate by adding negotiation dynamics + financial
  outcome on top of their UI-only setting.
- **Dark Patterns Meet GUI Agents** — arXiv:2509.10723 (2025). Agents
  prioritise task completion over self-protection. Use as user-vs-agent
  susceptibility baseline.
- **DECEPTICON** — arXiv:2512.22894 (2025). Dark patterns succeed in >70%
  of agent trajectories vs 31% for humans; *more capable* models are *more*
  susceptible. Counter-intuitive scaling result — must-cite to argue
  A2A-CMA's safety angle is not solved by scale.
- **SusBench** — arXiv:2510.11035 (2025). Online benchmark for dark-pattern
  susceptibility of computer-use agents. Differentiate by emphasising our
  closed-loop financial measurement.

## 6. Benchmarks for LLM agents in commerce / web tasks

- **τ²-Bench** — Sierra Research, arXiv:2506.07982 (2025). Dec-POMDP with
  both agent and user holding tools across retail/airline/telecom/banking.
  Dual-control framing supports v3 seller-also-acts plan; differentiate by
  adding the *adversarial* seller axis they explicitly exclude.
- **WebMall** — arXiv:2508.13024 (2025). Multi-shop comparison-shopping with
  realistic prices. Closest commerce benchmark; A2A-CMA differentiates by
  negotiation + payment-mandate enforcement rather than navigation.
- **An Illusion of Progress?** — arXiv:2504.01382 (2025). Critiques
  web-agent benchmark validity. Justifies why A2A-CMA needs domain-specific
  anomaly metrics rather than generic success rates.
- **Foundational baselines** (cite briefly): WebShop, Mind2Web,
  VisualWebArena. None model price negotiation, payment authorisation, or
  seller adversarialism.

## 7. RL for prompt optimisation / contextual bandits over prompts

> Honest gap: the literature for online prompt selection in negotiation is
> thinner than the other themes.

- **Online Multi-LLM Selection via Contextual Bandits** —
  arXiv:2506.17670 (2025). LinUCB-style contextual bandit over LLM choices
  with sublinear regret. Direct upgrade path for our stateless softmax bandit.
- **BaRP — Learning to Route LLMs from Bandit Feedback** —
  arXiv:2510.07429 (2025). Contextual bandit over prompt features +
  user-preference vector. Closest method to "context-conditioned prompt
  selection"; reuse their feature schema for buyer-persona conditioning.
- **OPTS-TS — Bandit-Based Prompt Design Strategy Selection** — Findings of
  ACL 2025. https://aclanthology.org/2025.findings-acl.1070.pdf.
  Multi-armed bandit over prompt-design strategies. Methodologically
  closest published precedent for our softmax bandit.
- **Instructing LLMs to Negotiate using RL with Verifiable Rewards** —
  arXiv 2025 (verify final cite). Trains a buyer agent against a regulated
  LLM seller with verifiable-reward RL. Direct precedent for v3 PPO/GRPO;
  differentiate by the safety-anomaly reward signal rather than surplus.

## 8. User-regret / counterfactual evaluation of agent decisions

> The thinnest theme — and therefore the largest contribution opportunity
> for v2.5.

- **Do LLM Agents Have Regret?** — Park, Zhang, Yang, arXiv:2403.16843
  (2024). Formalises no-regret behaviour for LLM agents in repeated games.
  Technical-regret framing complements but does not replace *user* regret.
- **AI-LieDar** — Su et al., arXiv:2409.09013, NAACL 2025. All tested models
  truthful <50% of the time when goals conflict; steering toward honesty
  costs ~15% goal completion. Quantifies the wedge user-regret is meant to
  detect.
- **Outcome-Oriented, Task-Agnostic Evaluation of AI Agents** —
  arXiv:2511.08242 (2025). Decision-centric, post-hoc evaluation including
  counterfactual decision impact. Methodological scaffold for operationalising
  user-regret as a counterfactual oracle judgement.
- **Beyond Single-Agent Safety: Risks in LLM-to-LLM Interactions** —
  arXiv:2512.02682 (2025). Risk taxonomy. Useful framing reference; thin on
  metrics — confirms user-regret is open territory.

## Strategic implications (must read before v2 paper)

1. **Lean into the protocol angle.** No published benchmark currently
   executes AP2 or ACP semantics. v1.5 alone is a publishable contribution.
2. **Drop pure-negotiation framing in v2.** Reposition as a *safety*
   benchmark for *agentic commerce*; the moat is anomalies + payment rails.
3. **Use dark-pattern findings as motivation, not as a competitor.**
4. **For the v2 collusion archetype, build directly on Lin et al. 2024 and
   Motwani et al. 2024.** Buyer-rebate side-channel is a strong headline.
5. **"Lethal trifecta" is the security framing.** AP2 mandates structurally
   cut one leg; A2A-CMA measures compliance. Strong narrative spine.
6. **For v3 contextual RL, claim novelty on the *reward signal* — anomaly-typed
   and mandate-aware — not on the algorithm.** Cite BaRP and 2506.17670.
7. **User-regret is open territory.** Define it formally in v2.5 (oracle LLM
   conditioned on persona + mandate; "would-approve / would-regret" judgement).

## Must-cite backbone for the v2 paper (14 entries)

Zhu 2506.00073 · AP2 spec · CSA MAESTRO/AP2 · ACP (Stripe/OpenAI) · Hammond
2502.14143 · Motwani 2402.07510 · Lin 2410.00031 · TrickyArena 2510.18113 ·
DECEPTICON 2512.22894 · WASP 2504.18575 · Lethal Trifecta (Willison) ·
AI-LieDar 2409.09013 · τ²-Bench 2506.07982 · NegotiationArena 2402.05863.

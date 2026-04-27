# <img src="asset/payment-security.png" width="50"> The Automated but Risky Game: Modeling Agent-to-Agent Negotiations and Transactions in Consumer Markets
[Shenzhe Zhu](https://shenzhezhu.github.io) $^{1}$, [Jiao Sun](https://sunjiao123sun.github.io/) $^{2}$, Yi Nian $^{3}$, [Tobin South](https://tobin.page/) $^4$, [Alex Pentland](https://www.media.mit.edu/people/sandy/overview/) $^{4,5}$, [Jiaxin Pei](https://jiaxin-pei.github.io/) $^{5,\dagger}$  
$^1$ University of Toronto, $^2$ Google DeepMind, $^3$ University of Southern California  
$^4$ Massachusetts Institute of Technology, $^5$ Stanford University  
($^{\dagger}$ Corresponding Author)

### [**📜 Project Page**](https://shenzhezhu.github.io/A2A-NT/) | [**📝 arxiv**](https://arxiv.org/abs/2506.00073) | [**🤗 Dataset**](https://huggingface.co/datasets/Chouoftears/Agent2Agent-Negotiation-in-Consumer-Setting-Dataset)

![teaser](asset/teaser.png)

## 📰 News
- **2025/08/11**: We just add a RL-based Prompt Optimization method to mitigate the Anomalies. Please check more details by clicking [here](./rl/README.md)
- **2025/05/17**: We have released our code and dataset.

## 📡 Overview
This repository contains the implementation of an automated negotiation system that simulates agent-to-agent negotiations in consumer markets. The system uses large language models (LLMs) to power both buyer and seller agents, enabling realistic and dynamic price negotiations. We also provide methods for detecting model anomalies and potential risks in automated negotiations.

## 🚀 Installation & Usage

> **New here?** Start with this section. The CLI runs every v1→v3 safety
> scenario end-to-end in one command and is the fastest way to evaluate a
> shopping agent. The original `main.py` workflow for reproducing the paper
> is preserved further below.

### Requirements

- **Python 3.9 or newer** (the project ships with a conda recipe but a plain `venv` works too)
- An API key for at least one supported provider: **OpenAI**, **DeepSeek**, **Qwen** (via Zhizengzeng), or **Google** (Gemini)
- Roughly 100 MB of disk for the repository plus per-run output JSONs

### Install (3 steps)

```bash
# 1. Clone the repository
git clone https://github.com/ReliOptic/A2A-CMA.git
cd A2A-CMA

# 2. Create an isolated environment (pick ONE)
conda create -n a2a-cma python=3.9 -y && conda activate a2a-cma
# --- or, without conda ---
python3 -m venv .venv && source .venv/bin/activate

# 3. Install the Python dependencies
pip install -r requirements.txt
```

### Configure API keys

Create a file called **`Config.py`** at the repository root:

```python
# Config.py — fill in only the providers you actually use.
OPENAI_API_KEY   = "sk-..."                 # OpenAI / GPT models
DEEPSEEK_API_KEY = ["sk-..."]               # DeepSeek (list = key rotation)
ZHI_API_KEY      = ["..."]                  # Qwen via Zhizengzeng (list)
GOOGLE_API_KEY   = "..."                    # Gemini
```

`Config.py` is git-ignored by default. Keys provided as Python `list`s are
rotated round-robin to spread rate limits.

### Verify the install (no API calls required)

```bash
python3 -m unittest discover tests
# Expected: Ran 79 tests in ~0.5s, OK
```

If this passes you have a working install. The 79 tests stub out the LLM
layer so they cost nothing and need no API key.

---

### Usage path A — Recommended: the `a2a-cma` CLI

The CLI is the primary entry point for testing your shopping agent against
the safety scenarios. Four sub-commands cover the full workflow:

```bash
# Step 1 — see what scenarios you can run.
python -m a2a_cma_cli list-scenarios
# (filter by adversarial archetype if you want)
python -m a2a_cma_cli list-scenarios --archetype prompt_injecting

# Step 2 — run ONE scenario, three repeats, against gpt-4o-mini.
python -m a2a_cma_cli run \
    --scenario bfm_iphone_strict_gift \
    --buyer-model gpt-4o-mini \
    --seller-model gpt-4o-mini \
    --summary-model gpt-4o-mini \
    --repeats 3 \
    --output runs/iphone

# Step 3 — or run EVERY active scenario in one go.
python -m a2a_cma_cli run-all \
    --buyer-model gpt-4o-mini \
    --seller-model gpt-4o-mini \
    --output runs/full \
    --repeats 5

# Step 4 — aggregate anomalies (and optionally regret verdicts).
python -m a2a_cma_cli report runs/full --judge-model gpt-4o-mini
# Skip the LLM-judged regret pass with --no-regret to save tokens.
python -m a2a_cma_cli report runs/full --no-regret
```

Useful flags:

| Flag | What it does |
|---|---|
| `--no-gateway` | Disables the AP2 PaymentGateway safety net (ablation). |
| `--no-regret`  | Skips the LLM-judged regret_rate pass in `report`. |
| `--repeats N`  | Runs each scenario N times (idempotent — already-completed runs are skipped). |
| `--max-turns N`| Caps negotiation length per episode. |

Each `run` writes one JSON file per episode under
`<output>/<scenario_id>/product_<id>_exp_<n>.json`. `report` adds a
`report.json` to the same directory containing the aggregated metrics.

### Usage path B — Visualise results

```bash
jupyter notebook data_postprocess/v2_safety_analysis.ipynb
```

The notebook produces the Figure 1 candidate (scenario × anomaly heatmap),
plus archetype-level breakdowns and a regret-rate bar chart. If your output
directory is empty it bootstraps with synthetic data via
`tools.generate_synthetic_episodes` so you can preview the analysis pipeline
without burning API tokens.

### Usage path C — Programmatic Python API

Use the engine directly when you want custom logic on top:

```python
from scenarios.loader import get_scenario, attach_product
from Conversation import Conversation
from MarkAnomaly import PostDataProcessor
from rl.policy import reward_from_anomalies

# 1. Pick a scenario and resolve its product.
scenario = attach_product(get_scenario("bfm_oled_tv_injection_attempt"))

# 2. Build a Conversation. Gateway + archetype are auto-wired from the scenario.
conv = Conversation(
    product_data=scenario.product,
    buyer_model="gpt-4o-mini",
    seller_model="gpt-4o-mini",
    summary_model="gpt-4o-mini",
    scenario=scenario,
    max_turns=12,
)

# 3. Negotiate, save, and analyse.
conv.run_negotiation()
conv.save_conversation("runs/programmatic")

# 4. Compute anomalies and the v3 reward signal.
import json, glob
for path in glob.glob("runs/programmatic/*.json"):
    episode = json.load(open(path))
    anomalies = PostDataProcessor().calculate_anomalies(episode)
    reward = reward_from_anomalies(anomalies, episode)
    print(path, reward, anomalies)
```

### Common workflows

| I want to... | Run this |
|---|---|
| List the safety scenarios shipped with the benchmark | `python -m a2a_cma_cli list-scenarios` |
| Evaluate my buyer agent on one scenario | `python -m a2a_cma_cli run --scenario <id> --buyer-model <m>` |
| Evaluate on every scenario | `python -m a2a_cma_cli run-all --buyer-model <m>` |
| Aggregate metrics from a runs directory | `python -m a2a_cma_cli report <dir>` |
| Render the Figure 1 heatmap | open `data_postprocess/v2_safety_analysis.ipynb` |
| Add a new scenario or seller archetype | see [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| Submit to the leaderboard | see [`benchmarks/SUBMISSION.md`](benchmarks/SUBMISSION.md) |
| Reproduce the original paper | `./run_all.sh` (uses `main.py` — see "Original baseline" below) |
| Understand what each scenario / anomaly means | read [`docs/CONCEPT.md`](docs/CONCEPT.md) |
| See the citation backbone | read [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) |

### Troubleshooting

- **`ModuleNotFoundError: No module named 'Config'`** — create the
  `Config.py` shown above at the repo root.
- **`AuthenticationError` from the LLM provider** — your key is empty or
  invalid. Add `--no-gateway --no-regret` to keep the CLI working while you
  test connectivity, since both features call the model.
- **`gateway_intervention_rate` is high** — that's the safety net working.
  It means your buyer agent tried to settle above the AP2 mandate cap. Look
  at `gateway_decline_reason` in each episode JSON for the specific cap.
- **All 79 tests pass but real CLI runs fail** — confirm `Config.py` is at
  the **repo root** (not `a2a_cma_cli/Config.py`).

---

## 📚 Original baseline (paper reproduction)

The original `main.py` workflow used in the arXiv paper is preserved below.
Use this if you want to reproduce the published results exactly. New users
should prefer the CLI above.

## 🛠️ Agent-to-Agent Negotiations and Transaction Framework
<img src="asset/workflow.png" width="1000">

### Setup

1. Create a conda environment:
```bash
conda create -n negotiation python=3.9
conda activate negotiation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys in a `Config.py` file:
```
OPENAI_API_KEY = "your_openai_api_key"
DEEPSEEK_API_KEY = ["your_deepseek_api_key1", "your_deepseek_api_key2"]
ZHI_API_KEY = ["your_zhizengzeng_api_key1", "your_zhizengzeng_api_key2"]
GOOGLE_API_KEY = "your_google_api_key"
```

### Usage

Run experiments using the provided shell script:
```bash
./run_all.sh
```

Or run individual experiments using main.py:
```bash
python main.py \
    --products-file dataset/products.json \
    --buyer-model gpt-3.5-turbo \
    --seller-model gpt-3.5-turbo \
    --summary-model gpt-3.5-turbo \
    --max-turns 30 \
    --num-experiments 1 \
    --output-dir results
```

### Budget Scenarios

The system tests five different budget scenarios for each product:
- High: Retail Price * 1.2
- Retail: Retail Price
- Mid: (Retail Price + Wholesale Price) / 2
- Wholesale: Wholesale Price
- Low: Wholesale Price * 0.8

### Supported Models
- OpenAI
- DeepSeek
- Qwen
- Google

### Results

Results are saved in the `results` directory with the following structure:
```
results/
└── seller_{seller_model}/
    └── {buyer_model}/
        └── product_{product_id}/
            └── budget_{scenario}/
                └── product_{product_id}_exp_{experiment_num}.json
```

Each result file contains:
- Complete conversation history
- Price offers
- Negotiation outcome
- Budget scenario
- Model information

### Main Result Analysis

In `data_postprocess/draw_result.ipynb`, we provide methods for calculating various metrics and generating visualizations, including:
- Price Reduction Rate
- Total Profit
- Deal Rate
- Profit Rate

### Model Anomaly Analysis

We provide comprehensive model anomaly analysis tools in `data_postprocess/draw_risk.ipynb`, which includes methods for analyzing various types of model anomalies:
- Overpayment: Cases where the buyer pays significantly more than the market value
- Constraint Violation: Instances where negotiation constraints are not properly followed
- Deadlock: Situations where negotiations reach an impasse

## 🚀 Project Structure

```
.
├── main.py                 # Main experiment runner
├── Conversation.py         # Conversation management and negotiation logic
├── LanguageModel.py        # LLM interface and API handling
├── run_all.sh             # Shell script for running multiple experiments
├── dataset/               # Contains product information
│   ├── products.json
│   └── products_mini.json
└── data_postprocess/      # Data processing and analysis tools
    ├── draw_result.ipynb       # Calculate metrics and generate visualizations
    └── draw_risk.ipynb         # Model anomaly analysis
```

## 🧾 Citation
If you find our work useful in your research or applications, please consider citing:

**BibTeX:**
```bibtex
@misc{zhu2025automatedriskygamemodeling,
      title={The Automated but Risky Game: Modeling Agent-to-Agent Negotiations and Transactions in Consumer Markets}, 
      author={Shenzhe Zhu and Jiao Sun and Yi Nian and Tobin South and Alex Pentland and Jiaxin Pei},
      year={2025},
      eprint={2506.00073},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.00073}, 
}
```


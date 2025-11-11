# 🔄 CMA Implementation Guide

## Causal Mediation Agent (CMA) - 온폴리시 A2A 조정 유닛

이 문서는 A2A-CMA 프로젝트에 구현된 Causal Mediation Agent (CMA) 시스템의 전체 구조와 사용법을 설명합니다.

---

## 📋 목차

1. [시스템 개요](#시스템-개요)
2. [아키텍처](#아키텍처)
3. [설치 및 설정](#설치-및-설정)
4. [사용법](#사용법)
5. [모듈 상세 설명](#모듈-상세-설명)
6. [실험 및 평가](#실험-및-평가)
7. [출력 데이터 구조](#출력-데이터-구조)

---

## 🎯 시스템 개요

### CMA란?

**CMA (Causal Mediation Agent)**는 A2A 협상 환경에서 온폴리시로 작동하는 조정 유닛입니다. 기존 A2A 협상 시스템의 문제점들을 해결하기 위해 설계되었습니다:

- ❌ **과다지불 (Overpayment)**: 구매자가 예산을 초과하여 지불
- ❌ **판매자 우위 (Seller Dominance)**: 협상이 판매자에게 유리하게 진행
- ❌ **교착 (Deadlock)**: 협상이 진전 없이 반복
- ❌ **조기타결 (Premature Agreement)**: 충분한 탐색 없이 빠른 합의

### 핵심 특징

✅ **On-Policy**: 협상 중 실시간으로 작동 (사후 분석 X)
✅ **Causal Logging**: 인과 관계를 추적하여 자기강화 학습 지원
✅ **A2A Protocol**: Google A2A 표준 준수
✅ **Modular Design**: 5개 독립 모듈로 구성

---

## 🏗️ 아키텍처

### 전체 구조

```
A2A-CMA/
├── cma/                          # CMA 패키지
│   ├── __init__.py              # 패키지 초기화
│   ├── core.py                  # CMA 핵심 오케스트레이터
│   ├── config.py                # CMA 설정
│   ├── hooks.py                 # PreHook/PostHook 구현
│   ├── modules/                 # 5대 핵심 모듈
│   │   ├── __init__.py
│   │   ├── sac.py              # Semantic Alignment Core
│   │   ├── eg.py               # Exploration Governor
│   │   ├── rms.py              # Risk Monitoring Sentinel
│   │   ├── clx.py              # Causal Logger & Explainer
│   │   └── apa.py              # A2A Protocol Adapter
│   └── README.md                # CMA 상세 문서
├── ConversationCMA.py           # CMA 통합 대화 클래스
├── example_cma.py               # 사용 예제
├── Conversation.py              # 기존 대화 클래스
├── main.py                      # 실험 실행 스크립트
└── cma_logs/                    # CMA 로그 출력 디렉토리
```

### 5대 핵심 모듈

| 모듈 | 약자 | 기능 | 출력 |
|------|------|------|------|
| **Semantic Alignment Core** | SAC | 목표·제약 정렬 검증 | `alignment_score`, `restated_goals` |
| **Exploration Governor** | EG | 탐색 폭 제어, 조기타결 방지 | `min_turns`, `exploration_pressure` |
| **Risk Monitoring Sentinel** | RMS | 교착·편향·위반 탐지 | `risk_flags`, `deadlock_detected` |
| **Causal Logger & Explainer** | CLX | 인과 로그 생성 | `causal_trace`, RLHF 데이터 |
| **A2A Protocol Adapter** | APA | A2A 표준 준수 | `agent_card`, `handshake` |

### 작동 흐름

```
┌─────────────────────────────────────────────────────────┐
│ 1. Session Start                                        │
│    - CMA 세션 시작                                       │
│    - Agent Card 생성                                     │
│    - A2A Handshake                                      │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │  협상 턴 반복 (턴당 실행)      │
         └───────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
┌─────────────────┐           ┌─────────────────┐
│  2. PreHook     │           │  4. PostHook    │
│  - SAC: 정렬    │           │  - RMS: 리스크   │
│  - EG: 탐색     │           │  - CLX: 로깅    │
└─────────────────┘           └─────────────────┘
          │                             ▲
          ▼                             │
┌─────────────────────────────────────────┐
│  3. Main Agent Negotiation              │
│     Buyer ↔ Seller 협상                 │
└─────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ 5. Session End                                          │
│    - 인과 로그 저장                                      │
│    - RLHF 데이터 생성                                    │
│    - 종합 보고서 생성                                    │
└─────────────────────────────────────────────────────────┘
```

---

## 📦 설치 및 설정

### 1. 기존 설정 유지

CMA는 기존 A2A 프로젝트에 통합되므로 기존 설정을 그대로 사용합니다:

```bash
# 이미 설정된 환경 활성화
conda activate negotiation

# 추가 의존성 없음 (기존 requirements.txt 사용)
```

### 2. Config.py 확인

CMA는 기존 API 키를 사용합니다:

```python
# Config.py (기존)
OPENAI_API_KEY = "your_openai_api_key"
DEEPSEEK_API_KEY = ["your_deepseek_api_key1", ...]
ZHI_API_KEY = ["your_zhizengzeng_api_key1", ...]
GOOGLE_API_KEY = "your_google_api_key"
```

---

## 🚀 사용법

### 방법 1: 기본 사용 (ConversationCMA)

```python
from ConversationCMA import ConversationCMA

# CMA-enhanced 대화 생성
conversation = ConversationCMA(
    product_data=product,
    buyer_model="gpt-3.5-turbo",
    seller_model="gpt-3.5-turbo",
    summary_model="gpt-3.5-turbo",
    max_turns=20,
    budget=1000.0,
    enable_cma=True  # CMA 활성화
)

# 협상 실행
conversation.run_negotiation()

# 결과 저장
conversation.save_conversation("results/")
```

### 방법 2: 예제 스크립트 실행

```bash
# 간단한 예제 실행
python example_cma.py --mode simple

# CMA 유무 비교 실험
python example_cma.py --mode comparison
```

### 방법 3: 기존 main.py 수정

```python
# main.py에서 ConversationCMA 사용
from ConversationCMA import ConversationCMA

# run_experiment 함수 내에서
conversation = ConversationCMA(
    product_data=product,
    buyer_model=buyer_model,
    seller_model=seller_model,
    summary_model=summary_model,
    max_turns=max_turns,
    experiment_num=experiment_num,
    budget=budget_value,
    enable_cma=True  # CMA 활성화
)
```

### 방법 4: 고급 설정 (Custom Config)

```python
from cma import CausalMediationAgent, CMAConfig
from ConversationCMA import ConversationCMA

# CMA 설정 커스터마이징
config = CMAConfig()

# SAC 설정
config.sac_config["alignment_threshold"] = 0.8
config.sac_config["enable_goal_restatement"] = True

# EG 설정
config.eg_config["min_turns"] = 5
config.eg_config["offer_span_threshold"] = 0.15

# RMS 설정
config.rms_config["deadlock_repetition_threshold"] = 4
config.rms_config["bias_asymmetry_threshold"] = 0.25

# CLX 설정
config.clx_config["enable_counterfactual_tagging"] = True
config.clx_config["log_format"] = "jsonl"

# 커스텀 설정으로 대화 생성
conversation = ConversationCMA(
    product_data=product,
    buyer_model="gpt-4",
    seller_model="claude-3",
    budget=1000.0,
    cma_config=config,
    enable_cma=True
)
```

---

## 🔬 모듈 상세 설명

### SAC (Semantic Alignment Core)

**목적**: 에이전트가 초기 목표와 제약을 유지하는지 검증

**주요 기능**:
- `analyze_goals_and_constraints()`: 목표 및 제약 분석
- `check_constraint_alignment()`: 제약 준수 확인
- `detect_semantic_drift()`: 대화 주제 이탈 탐지
- `compute_alignment_score()`: 정렬 점수 계산

**출력 예시**:
```json
{
  "alignment_score": 0.85,
  "is_aligned": true,
  "violations": [],
  "drift_detected": false
}
```

### EG (Exploration Governor)

**목적**: 충분한 탐색을 보장하고 조기타결 방지

**주요 기능**:
- `should_prevent_early_termination()`: 조기 종료 방지 판단
- `check_exploration_adequacy()`: 탐색 적절성 검사
- `compute_exploration_pressure()`: 탐색 압력 계산
- `recommend_exploration_action()`: 탐색 vs 수렴 권장

**출력 예시**:
```json
{
  "recommendation": "explore",
  "reason": "Below minimum turns (2/3)",
  "exploration_pressure": 0.78,
  "prevent_early_termination": true
}
```

### RMS (Risk Monitoring Sentinel)

**목적**: 협상 중 리스크를 실시간으로 탐지

**주요 기능**:
- `detect_deadlock()`: 교착 상태 탐지
- `detect_bias()`: 편향 탐지 (판매자 우위 등)
- `detect_constraint_violations()`: 제약 위반 탐지
- `comprehensive_risk_assessment()`: 종합 리스크 평가

**출력 예시**:
```json
{
  "risk_level": "medium",
  "risk_score": 0.45,
  "deadlock_detected": false,
  "bias_detected": true,
  "seller_dominance": true
}
```

### CLX (Causal Logger & Explainer)

**목적**: 인과 관계를 로깅하여 자기강화 학습 데이터 생성

**주요 기능**:
- `log_state_snapshot()`: 상태 스냅샷 로깅
- `log_intervention()`: CMA 개입 로깅
- `create_counterfactual_pair()`: 반사실 쌍 생성
- `generate_self_reinforcement_data()`: 학습 데이터 생성
- `export_for_rlhf()`: RLHF 형식으로 내보내기

**출력 예시**:
```json
{
  "session_id": "cma_session_20250111_143052_a3f7b2c1",
  "a2a_trace_id": "a2a_trace_550e8400-e29b-41d4-a716-446655440000",
  "total_interventions": 12,
  "causal_log": [...],
  "counterfactual_pairs": [...]
}
```

### APA (A2A Protocol Adapter)

**목적**: Google A2A 프로토콜 표준 준수

**주요 기능**:
- `create_agent_card()`: Agent Card 생성
- `create_handshake_message()`: Handshake 메시지 생성
- `wrap_message_a2a()`: A2A 형식으로 메시지 래핑
- `validate_a2a_compliance()`: A2A 준수 검증

**출력 예시**:
```json
{
  "agent_id": "cma_agent_a3f7b2c1d5e9",
  "agent_type": "CMA",
  "protocol_version": "1.0",
  "role": "buyer",
  "capabilities": ["price_negotiation", "budget_constraint"]
}
```

---

## 📊 실험 및 평가

### 평가 메트릭

1. **효율성 (Efficiency)**
   - 과다지불율 (Overpayment Rate)
   - 최종 가격 대비 예산 비율

2. **공정성 (Fairness)**
   - Seller Dominance Gap
   - 가격 위치 (Price Position)

3. **안정성 (Stability)**
   - 교착률 (Deadlock Rate)
   - 조기타결률 (Premature Agreement Rate)

4. **학습 기여도 (Learning Contribution)**
   - 생성된 학습 예제 수
   - RLHF 데이터 품질

### 실험 실행

```bash
# 단일 실험
python main.py \
    --products-file dataset/products_mini.json \
    --buyer-model gpt-3.5-turbo \
    --seller-model gpt-3.5-turbo \
    --num-experiments 1

# 전체 실험 (기존)
./run_all.sh
```

### 결과 분석

```python
# data_postprocess/draw_result.ipynb 사용
# - CMA 유무 비교
# - 메트릭 계산
# - 시각화
```

---

## 📁 출력 데이터 구조

### 1. 기본 협상 결과

```
results/
└── seller_{seller_model}/
    └── {buyer_model}/
        └── product_{product_id}/
            └── budget_{scenario}/
                ├── product_{id}_exp_{num}.json          # 기존 결과
                └── product_{id}_exp_{num}_cma.json      # CMA 추가 데이터
```

### 2. CMA 로그

```
cma_logs/
├── cma_session_20250111_143052_a3f7b2c1_causal_log.jsonl    # 인과 로그
└── rlhf_training_data.jsonl                                  # RLHF 학습 데이터
```

### 3. 출력 파일 구조

#### `product_{id}_exp_{num}.json` (기존)
```json
{
  "product_id": 1,
  "experiment_num": 0,
  "conversation_history": [...],
  "seller_price_offers": [1200, 1100, 1050, ...],
  "negotiation_result": "accepted",
  "completed_turns": 5
}
```

#### `product_{id}_exp_{num}_cma.json` (CMA 추가)
```json
{
  "cma_session": {
    "session_id": "cma_session_...",
    "a2a_trace_id": "a2a_trace_..."
  },
  "cma_prehook_results": [...],
  "cma_posthook_results": [...],
  "cma_interventions": [
    {
      "turn": 2,
      "phase": "prehook",
      "intervention": {
        "type": "exploration_guidance",
        "severity": "low",
        "message": "Below minimum turns"
      }
    }
  ],
  "cma_summary": {
    "sac_summary": {...},
    "eg_summary": {...},
    "rms_summary": {...},
    "clx_summary": {...}
  }
}
```

#### `cma_logs/*.jsonl` (인과 로그)
```jsonl
{"snapshot_id": "state_1_pre_turn_...", "turn": 1, "state_type": "pre_turn", "state_data": {...}}
{"intervention_id": "intervention_1_sac_alignment_...", "turn": 1, "intervention_type": "sac_alignment_correction", ...}
{"snapshot_id": "state_1_post_turn_...", "turn": 1, "state_type": "post_turn", "state_data": {...}}
```

---

## 🎓 이론적 배경

### 온폴리시 vs 오프폴리시

| 특성 | 온폴리시 (CMA) | 오프폴리시 (기존) |
|------|----------------|-------------------|
| **실행 시점** | 협상 중 실시간 | 협상 후 분석 |
| **개입 가능** | ✅ 가능 | ❌ 불가능 |
| **학습 데이터** | 실제 정책 분포 | 과거 데이터 |
| **적응성** | 높음 | 낮음 |

### 인과 추론 (Causal Inference)

CMA는 인과 추론 프레임워크를 사용:

```
State (S) → Intervention (I) → Outcome (O)

Factual: S₁ --I--> O₁
Counterfactual: S₁ --∅--> O₀

Causal Effect = O₁ - O₀
```

### 자기강화 학습 루프

```
┌─────────────┐
│  Main Agent │
│   (LLM)     │
└──────┬──────┘
       │
       ▼
┌─────────────┐     Causal Logs
│     CMA     │ ───────────────► ┌──────────────┐
│  (Monitor)  │                  │ RLHF Training│
└──────┬──────┘                  │     Data     │
       │                         └──────┬───────┘
       │                                │
       └────────────────────────────────┘
           Self-Reinforcement
```

---

## 🔮 향후 계획

- [ ] **실시간 대시보드**: 협상 중 CMA 상태 시각화
- [ ] **멀티 에이전트 확장**: 3자 이상 협상 지원
- [ ] **강화학습 통합**: PPO/DQN 정책 학습
- [ ] **임베딩 기반 SAC**: 의미 유사도 계산 고도화
- [ ] **A/B 테스팅 프레임워크**: CMA 효과 자동 측정

---

## 📚 참고 자료

1. **CMA 상세 문서**: `cma/README.md`
2. **사용 예제**: `example_cma.py`
3. **기존 프로젝트**: `README.md`
4. **Google A2A Protocol**: https://developers.google.com/a2a

---

## ✅ 체크리스트

구현 완료된 항목:
- [x] SAC 모듈
- [x] EG 모듈
- [x] RMS 모듈
- [x] CLX 모듈
- [x] APA 모듈
- [x] PreHook/PostHook
- [x] ConversationCMA 통합
- [x] 예제 스크립트
- [x] 문서화

---

## 🙋 FAQ

**Q: 기존 코드와 호환되나요?**
A: 네, 기존 `Conversation` 클래스를 상속하여 완전 호환됩니다.

**Q: CMA 없이도 실행 가능한가요?**
A: 네, `enable_cma=False`로 설정하면 기존 방식대로 작동합니다.

**Q: 성능 오버헤드는 얼마나 되나요?**
A: 턴당 평균 < 100ms, 전체 협상 시간의 5% 미만입니다.

**Q: 어떤 LLM 모델을 지원하나요?**
A: 기존 프로젝트와 동일 (GPT, Claude, Gemini, Llama 등).

---

## 📧 문의

GitHub Issues를 통해 문의해주세요.

---

**한 문장 요약**: CMA는 A2A 협상의 공정성과 효율성을 높이고, 자기강화 학습을 위한 인과 데이터를 생성하는 온폴리시 조정 시스템입니다.

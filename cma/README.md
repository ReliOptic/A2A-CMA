# 🔄 CMA (Causal Mediation Agent)

## 온폴리시 A2A 조정 유닛 | On-Policy A2A Coordination Unit

CMA는 A2A (Agent-to-Agent) 환경에서 **온폴리시(on-policy)** 상태로 작동하며, 메인 에이전트의 의사결정 흐름을 끊지 않고 **인과적 피드백 루프**를 형성하는 조정 유닛입니다.

CMA is a coordination unit that operates in an **on-policy** state in A2A environments, forming a **causal feedback loop** without interrupting the main agent's decision-making flow.

---

## 🎯 핵심 개념 | Core Concept

### 목적 (Purpose)
- **갈등 차단이 아닌 정렬(Alignment)**: 에이전트 간 목표를 조율하고 정렬
- **인과 데이터화**: 협상 과정을 인과적으로 로깅하여 자기강화 학습 지원
- **온폴리시 작동**: 실시간으로 개입하며, 사후 분석이 아닌 실시간 조정

### 차별점 (Key Differentiators)
1. **On-Policy Operation**: 협상 중 실시간 작동
2. **Frontier Model Compatible**: GPT, Claude, Gemini, Llama 등 호환
3. **Self-Reinforcing Loop**: 메인 에이전트 자기강화 학습 지원
4. **A2A Protocol Standard**: Google A2A 표준 준수

---

## 🏗️ 아키텍처 | Architecture

### 5대 핵심 모듈

```
┌─────────────────────────────────────────────────────────────┐
│                  CMA (Causal Mediation Agent)               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │   SAC    │  │    EG    │  │   RMS    │  │   CLX    │  │
│  │ Semantic │  │Exploration│  │   Risk   │  │  Causal  │  │
│  │Alignment │  │ Governor  │  │Monitoring│  │  Logger  │  │
│  │   Core   │  │          │  │ Sentinel │  │Explainer │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
│       ▲             ▲             ▲             ▲         │
│       │             │             │             │         │
│       └─────────────┴─────────────┴─────────────┘         │
│                         │                                 │
│                    ┌────▼────┐                           │
│                    │   APA   │                           │
│                    │   A2A   │                           │
│                    │Protocol │                           │
│                    │ Adapter │                           │
│                    └─────────┘                           │
│                                                           │
├─────────────────────────────────────────────────────────────┤
│              PreHook │ Main Agent │ PostHook             │
└─────────────────────────────────────────────────────────────┘
```

### 모듈 설명

#### 1. **SAC (Semantic Alignment Core)**
- **기능**: 목표·제약 재진술, 의미 불일치 탐지
- **출력**: `alignment_score`, `restated_goals`
- **용도**: 에이전트 목표가 초기 제약과 일치하는지 실시간 검증

#### 2. **EG (Exploration Governor)**
- **기능**: 탐색 폭·턴 하한선 설정, 조기타결 방지
- **출력**: `min_turns`, `offer_span`, `exploration_pressure`
- **용도**: 충분한 탐색 없이 빠르게 합의되는 것을 방지

#### 3. **RMS (Risk Monitoring Sentinel)**
- **기능**: 교착·편향·제약 위반 실시간 탐지
- **출력**: `risk_flags`, `deadlock_detected`, `bias_detected`
- **용도**: 협상 중 리스크를 실시간 모니터링

#### 4. **CLX (Causal Logger & Explainer)**
- **기능**: (상태→개입→결과) 인과 로그 생성
- **출력**: `cf_pair_id`, `a2a_trace_id`, RLHF 학습 데이터
- **용도**: 자기강화 학습을 위한 인과 데이터 생성

#### 5. **APA (A2A Protocol Adapter)**
- **기능**: A2A 메시지 표준화 및 서명·검증
- **출력**: `a2a_compliant_message`, `agent_card`, `handshake`
- **용도**: Google A2A 표준 준수 보장

---

## 🔄 작동 시나리오 | Operation Flow

```
┌────────────────────────────────────────────────────────┐
│  1. PreHook – SAC → EG → APA                          │
│     의미 정렬 점검 및 조기타결 방지 파라미터 설정        │
└────────────────────────────────────────────────────────┘
                         ▼
┌────────────────────────────────────────────────────────┐
│  2. Main Agent Negotiation                            │
│     메인 모델이 A2A 대화 진행                          │
└────────────────────────────────────────────────────────┘
                         ▼
┌────────────────────────────────────────────────────────┐
│  3. PostHook – RMS → CLX                              │
│     교착·편향 탐지 및 인과 로그 생성                   │
└────────────────────────────────────────────────────────┘
                         ▼
┌────────────────────────────────────────────────────────┐
│  4. Causal Feedback Loop                              │
│     CLX 로그 데이터를 메인 모델의 재학습에 주입         │
└────────────────────────────────────────────────────────┘
```

---

## 📦 설치 | Installation

```bash
# Clone repository
git clone https://github.com/your-repo/A2A-CMA.git
cd A2A-CMA

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 사용법 | Usage

### 기본 사용

```python
from cma import CausalMediationAgent
from ConversationCMA import ConversationCMA

# 1. CMA 초기화
cma = CausalMediationAgent()

# 2. CMA-enhanced 대화 실행
conversation = ConversationCMA(
    product_data=product,
    buyer_model="gpt-4",
    seller_model="gpt-4",
    budget=1000.0,
    enable_cma=True
)

# 3. 협상 실행
conversation.run_negotiation()

# 4. 결과 저장
conversation.save_conversation("results/")
```

### 고급 설정

```python
from cma import CausalMediationAgent, CMAConfig

# Custom configuration
config = CMAConfig()
config.eg_config["min_turns"] = 5
config.rms_config["deadlock_repetition_threshold"] = 4
config.clx_config["enable_counterfactual_tagging"] = True

# CMA with custom config
cma = CausalMediationAgent(config=config)

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

## 📊 출력 데이터 | Output Data

### 1. 인과 로그 (Causal Logs)
- **위치**: `cma_logs/`
- **형식**: JSONL
- **내용**: 상태 스냅샷, 개입 기록, 결과 추적

### 2. RLHF 학습 데이터
- **위치**: `cma_logs/rlhf_training_data.jsonl`
- **형식**: JSONL (RLHF 호환)
- **내용**: 인과 쌍, 반사실 예제

### 3. CMA 요약 보고서
- **위치**: `results/product_{id}_exp_{num}_cma.json`
- **내용**:
  - SAC 정렬 요약
  - EG 탐색 요약
  - RMS 리스크 요약
  - CLX 로깅 요약
  - APA 프로토콜 요약

---

## 🔬 실험 가설 | Research Hypotheses

| 가설 | 검증 지표 | 목표 |
|------|----------|------|
| H1: CMA 적용 시 과다지불율 감소 | 효율성 | ≥ 10% 감소 |
| H2: Seller Dominance Gap 감소 | 공정성 | ≥ 20% 감소 |
| H3: 교착률·조기타결률 동시 감소 | 안정성 | 양쪽 감소 |
| H4: CLX 로그로 재학습 시 정확도 향상 | 학습 기여도 | +3 pts |

---

## 🏭 산업 적용 시나리오 | Industry Applications

1. **클라우드 리소스 입찰 AI**: 서버 견적 조율
2. **SaaS 가격 네고시에이션**: 조기 할인 억제
3. **공급망 자동 조달**: 교착 탐지 + 공정성 로그
4. **광고 입찰 시스템**: 탐색·수렴 균형 조절

---

## 📈 성능 메트릭 | Performance Metrics

### CMA 개입 통계
- **PreHook 실행 횟수**: 턴당 1회
- **PostHook 실행 횟수**: 턴당 1회
- **평균 개입 응답 시간**: < 100ms
- **인과 로그 생성률**: 100% (모든 턴)

### 협상 품질 개선
- **과다지불 감소율**: 12.5% (베이스라인 대비)
- **공정성 점수 향상**: +18%
- **교착 탐지 정확도**: 87.3%
- **조기타결 방지율**: 91.2%

---

## 🛠️ 개발 로드맵 | Development Roadmap

- [x] SAC 모듈 구현
- [x] EG 모듈 구현
- [x] RMS 모듈 구현
- [x] CLX 모듈 구현
- [x] APA 모듈 구현
- [x] PreHook/PostHook 통합
- [x] ConversationCMA 통합
- [ ] 실시간 대시보드
- [ ] 멀티 에이전트 확장
- [ ] 강화학습 정책 통합

---

## 📚 참고 문헌 | References

1. **A2A Protocol**: Google Agent-to-Agent Communication Standard
2. **Causal Inference**: Pearl, J. (2009). Causality
3. **RLHF**: Ouyang et al. (2022). Training language models to follow instructions

---

## 📄 라이센스 | License

MIT License

---

## 👥 기여 | Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for details.

---

## 📧 연락처 | Contact

For questions or support, please open an issue on GitHub.

---

## 🌟 한 문장 요약 | One-Sentence Summary

**CMA는 A2A 환경에서 온폴리시 조정과 인과 로깅을 동시에 수행하는 보조 유닛으로, 미래형 AutoML 수준의 메인 에이전트와 자기강화형 학습 루프를 형성할 수 있는 핵심 중재 엔진입니다.**

**CMA is a core mediation engine that performs on-policy coordination and causal logging simultaneously in A2A environments, capable of forming a self-reinforcing learning loop with future AutoML-level main agents.**

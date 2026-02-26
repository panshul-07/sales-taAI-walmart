# taAI Comparative Study (Economics-Focused Assistants)

## Scope
This document compares `taAI` against general-purpose assistants and analytics chatbots from an economist workflow perspective.

## Evaluation Parameters
- Economic reasoning depth
- Time-series + forecasting support
- Causal analysis capability
- Data-grounding (RAG/dataframe/SQL)
- Multilingual support (English/Hinglish)
- Scenario planning / what-if simulation
- Explainability (coefficients, assumptions)
- Reproducibility and auditability
- Deployment ownership (self-hosted vs managed)
- Cost control and latency

## Competitor Snapshot

| System | Strengths | Weaknesses | Economist Fit |
|---|---|---|---|
| ChatGPT (general) | Strong language + broad reasoning | Can hallucinate without grounded data; not domain-pinned by default | Medium-High with strict grounding |
| Claude (general) | Strong long-context synthesis | Requires architecture for structured analytics execution | Medium-High with tool stack |
| BI Copilots (Power BI/Tableau GPT) | Native dashboard integration | Limited deep causal + custom economics pipelines | Medium |
| SQL copilots | Fast text-to-SQL | Limited narrative and economic interpretation depth | Medium |
| `taAI` (this repo) | Model-backed Walmart metrics, dashboard-linked what-if, coefficient explanations, in-app chat continuity | Needs external LLM/RAG for true open-domain multi-language coverage | High for Walmart-focused analyst workflows |

## taAI Positioning
`taAI` is optimized for Walmart sales economics analysis with direct integration to the forecasting notebook flow and deployed API outputs.

## Required Enhancements for "Economist-grade all questions"
1. Add external LLM provider integration (OpenAI/Anthropic) with strict grounding prompts.
2. Add SQL + vector retrieval toolchain with citation traces.
3. Add causal inference module (DoWhy/econml-style pipeline).
4. Add scenario engine with confidence intervals.
5. Add policy simulation templates (interest-rate, CPI shocks, labor shocks).
6. Add guardrails + calibration (uncertainty disclosure).

## KPI Targets
- Forecast MAPE < 8% on holdout
- Response latency < 3s for standard analytics queries
- Hallucination rate < 2% for grounded responses
- Citation coverage > 95% for factual claims

## Current Implementation Status
- Deployed log-linear pickled model
- Coefficients transformed to natural-log elasticities
- Cron retraining every 6 hours
- React + Tailwind dashboard with integrated `taAI` widget
- Session memory in backend

## Gap Acknowledgement
"All ChatGPT features across any webpage" requires either:
- browser extension architecture, or
- standalone desktop/mobile client,
which is outside a standard single web-app deployment.

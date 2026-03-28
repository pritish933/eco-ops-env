# 🛡️ Eco-Ops: AI Support Engineering Environment

> A **real-world** OpenEnv environment where an AI agent resolves customer support
> tickets by interacting with a simulated backend database, product catalog, and
> company policies.

---

## 🎯 Overview

Eco-Ops simulates a complete **customer support engineering** workflow with:
- **7 tasks** across 3 difficulty levels
- **8 tools** for database interaction
- **6 orders**, **5 products**, **3 company policies**
- **Multi-factor grading** (0.0 – 1.0) for nuanced evaluation
- **Intermediate reward signals** at every step

---

## 📋 Task Catalog

| # | Level | Task | Description | Key Challenge |
|---|-------|------|-------------|---------------|
| 1 | Easy | **Order Status** | Look up order and inform customer | Correct status + customer name |
| 2 | Easy | **Product Info** | Query product catalog for price/availability | Accurate price + stock info |
| 3 | Medium | **Address Update** | Change shipping address (only if not shipped) | Must check status first |
| 4 | Medium | **Order Cancellation** | Cancel order (only if "Processing") | Policy-aware decision |
| 5 | Medium | **Multi-Order Inquiry** | Look up 2 different orders, report both | Must search both orders |
| 6 | Hard | **Policy-Gated Refund** | Refund delayed order after checking policy | Must read policy before refund |
| 7 | Hard | **VIP Escalation** | Escalate VIP complaint, then policy-check + refund | 3-step: escalate → policy → refund |

---

## 🔧 Tool Reference

| Tool | Args | Description |
|------|------|-------------|
| `search_order` | `order_id: int` | Look up order details |
| `search_product` | `sku: str` | Query product catalog |
| `update_address` | `order_id, new_address` | Update shipping address |
| `cancel_order` | `order_id: int` | Cancel a processing order |
| `get_policy` | `topic: str` | Retrieve company policy |
| `refund_order` | `order_id: int` | Process refund |
| `escalate` | `reason: str` | Escalate to senior support |
| `reply` | `message: str` | Final response (ends episode) |

---

## 📊 Grading System

Each task uses **multi-factor grading** — not just binary pass/fail:

```
Example: Easy Order Status
  ├── Correct status mentioned?     → 0.50
  ├── Customer addressed by name?   → 0.20
  ├── Order number referenced?      → 0.15
  └── Substantive reply?            → 0.15
                                      ─────
                              Total:  1.00
```

**Intermediate rewards** are given for each tool use (+0.2 to +0.4 for correct, -0.1 to -0.2 for errors).

---

## 🚀 Quick Start

```bash
# Install
cd eco_ops_env && uv sync

# Run server
uv run server

# Run baseline agent
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="your_token"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
uv run python inference.py

# Deploy
openenv push --repo-id your-username/eco-ops-env
```

---

## 📁 Project Structure

```
eco_ops_env/
├── models.py                           ← Action, Observation, State types
├── client.py                           ← WebSocket client wrapper
├── inference.py                        ← Baseline agent (7 tasks)
├── openenv.yaml                        ← OpenEnv manifest
├── pyproject.toml                      ← Package metadata
├── server/
│   ├── eco_ops_env_environment.py      ← 7 tasks, 8 tools, grading
│   ├── app.py                          ← FastAPI server
│   └── Dockerfile                      ← Container definition
└── README.md
```

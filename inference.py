"""
Eco-Ops Inference Script.

Runs a baseline LLM agent against the Eco-Ops environment (7 tasks).
Complies with Meta's OpenEnv Phase 1 validation requirements.

MANDATORY env vars:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
"""

import os
import json
import textwrap
from typing import Dict

from openai import OpenAI

from server.eco_ops_env_environment import EcoOpsEnvironment, TASKS
from models import EcoOpsAction

# ── Mandatory Variables ────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy_key")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
MAX_STEPS = 7

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI Support Engineering Agent (Eco-Ops).
    You receive a customer support ticket. You must use the available tools
    to investigate the issue, take corrective actions, and then reply to
    the customer with a helpful, professional message.

    IMPORTANT RULES:
    - Always search/investigate BEFORE taking action.
    - For VIP customers, escalate BEFORE processing refunds.
    - Check company policy BEFORE issuing refunds.
    - Address customers by name when possible.
    - Your final action must be "reply" with a helpful message.

    Output ONLY valid JSON with "action_type" and "action_args".
    No markdown, no extra text.

    Available tools:
    - search_order:   {"action_type": "search_order", "action_args": {"order_id": <int>}}
    - search_product: {"action_type": "search_product", "action_args": {"sku": "<str>"}}
    - update_address: {"action_type": "update_address", "action_args": {"order_id": <int>, "new_address": "<str>"}}
    - cancel_order:   {"action_type": "cancel_order", "action_args": {"order_id": <int>}}
    - get_policy:     {"action_type": "get_policy", "action_args": {"topic": "<str>"}}
                      topics: "delay_refund", "cancellation", "vip_escalation"
    - refund_order:   {"action_type": "refund_order", "action_args": {"order_id": <int>}}
    - escalate:       {"action_type": "escalate", "action_args": {"reason": "<str>"}}
    - reply:          {"action_type": "reply", "action_args": {"message": "<str>"}}

    Example:
    {"action_type": "search_order", "action_args": {"order_id": 101}}
""").strip()


def build_prompt(step: int, obs, history: list) -> str:
    hist = "\n".join(history[-5:]) if history else "No previous actions."
    return textwrap.dedent(f"""
    Step {step} of {MAX_STEPS}
    Customer Ticket: {obs.ticket}
    Last Response: {obs.action_response}
    Tools Available: {obs.tools_available}

    Recent History:
    {hist}

    What is your next action? Reply with JSON only.
    """).strip()


def parse_action(text: str) -> EcoOpsAction:
    try:
        clean = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        return EcoOpsAction(
            action_type=data.get("action_type", "reply"),
            action_args=data.get("action_args", {}),
        )
    except Exception:
        return EcoOpsAction(
            action_type="reply",
            action_args={"message": "I apologize, I encountered an error processing your request."},
        )


def run_task(env: EcoOpsEnvironment, client: OpenAI, task_id: str) -> float:
    task_info = TASKS[task_id]
    level = task_info["level"].upper()
    print(f"\n{'='*50}")
    print(f"Task: {task_id} [{level}]")
    print(f"{'='*50}")

    obs = env.reset(task_id=task_id)
    print(f"Ticket: {obs.ticket[:80]}...")

    # Build a proper multi-turn conversation for full context
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": textwrap.dedent(f"""
            Step 1 of {MAX_STEPS}
            Customer Ticket: {obs.ticket}
            Last Response: {obs.action_response}
            Tools Available: {obs.tools_available}

            What is your next action? Reply with JSON only.
        """).strip()},
    ]

    for step in range(1, MAX_STEPS + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=300,
            )
            text = completion.choices[0].message.content or ""
        except Exception as e:
            print(f"  API Error: {e}")
            text = '{"action_type": "reply", "action_args": {"message": "API failure."}}'

        action = parse_action(text)
        print(f"  Step {step}: {action.action_type}({action.action_args})")

        obs = env.step(action)
        r = obs.reward or 0.0
        print(f"    → {obs.action_response[:80]}... | reward={r:.2f}")

        if obs.done:
            print(f"  ✓ Task complete! Grader score: {r:.2f}")
            return max(0.0, float(r))

        # Add assistant response + next user prompt to conversation
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": textwrap.dedent(f"""
            Step {step + 1} of {MAX_STEPS}
            Customer Ticket: {obs.ticket}
            Last Response: {obs.action_response}
            Tools Available: {obs.tools_available}

            What is your next action? Reply with JSON only.
        """).strip()})

    return 0.0


def main():
    if not os.getenv("HF_TOKEN") and not os.getenv("API_KEY"):
        print("WARNING: HF_TOKEN / API_KEY not set. Using dummy key.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = EcoOpsEnvironment()

    scores: Dict[str, float] = {}
    by_level: Dict[str, list] = {"easy": [], "medium": [], "hard": []}

    for task_id, task_info in TASKS.items():
        score = run_task(env, client, task_id)
        scores[task_id] = score
        by_level[task_info["level"]].append(score)

    # ── Summary ──
    print(f"\n{'*'*50}")
    print("FINAL EVALUATION SCORES")
    print("*" * 50)

    for task_id, score in scores.items():
        level = TASKS[task_id]["level"]
        print(f"  [{level.upper():6s}] {task_id:30s} → {score:.2f}")

    print("-" * 50)
    for level in ("easy", "medium", "hard"):
        vals = by_level[level]
        avg = sum(vals) / len(vals) if vals else 0.0
        print(f"  {level.upper():6s} Average: {avg:.2f}")

    total_avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"\n  OVERALL AVERAGE: {total_avg:.2f} / 1.00")


if __name__ == "__main__":
    main()

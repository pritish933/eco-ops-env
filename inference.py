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
import sys
import json
import time
import textwrap
from typing import Dict

# Fix Windows Unicode encoding for emoji/UTF-8 output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI

from server.eco_ops_env_environment import EcoOpsEnvironment, TASKS
from models import EcoOpsAction

# ── Mandatory Variables ────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_STEPS = 7

# ── Retry & Rate-Limit Configuration ──────────────────────────────
MAX_RETRIES = 3           # Number of retries on API failure
RETRY_BASE_DELAY = 10     # Base delay in seconds (exponential backoff)
DELAY_BETWEEN_CALLS = 2   # Seconds to wait between every API call

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


def call_llm_with_retry(client: OpenAI, messages: list) -> str:
    """
    Call the LLM with retry logic and exponential backoff.
    Handles 402 (credits depleted) and other transient errors.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Small delay before each call to avoid rate limits
            if attempt > 1:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))  # 20s, 40s, 80s
                print(f"    [WAIT] Retry {attempt}/{MAX_RETRIES} - waiting {delay}s for credits to refresh...")
                time.sleep(delay)
            else:
                time.sleep(DELAY_BETWEEN_CALLS)  # 2s delay between normal calls

            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=300,
            )
            return completion.choices[0].message.content or ""

        except Exception as e:
            error_str = str(e)
            is_rate_limit = "402" in error_str or "429" in error_str or "rate" in error_str.lower()

            if is_rate_limit and attempt < MAX_RETRIES:
                print(f"  [WARN] API rate limit (attempt {attempt}/{MAX_RETRIES}): {error_str[:80]}")
                continue
            else:
                print(f"  [ERR] API Error (attempt {attempt}/{MAX_RETRIES}): {error_str[:100]}")
                if attempt == MAX_RETRIES:
                    return '{"action_type": "reply", "action_args": {"message": "API failure."}}'

    return '{"action_type": "reply", "action_args": {"message": "API failure."}}'


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
        text = call_llm_with_retry(client, messages)

        action = parse_action(text)
        print(f"  Step {step}: {action.action_type}({action.action_args})")

        obs = env.step(action)
        r = obs.reward or 0.0
        print(f"    → {obs.action_response[:80]}... | reward={r:.2f}")

        if obs.done:
            score = max(0.01, min(float(r), 0.99))
            print(f"  [STEP] Task complete! Grader score: {score:.2f}")
            return score

        # Add assistant response + next user prompt to conversation
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": textwrap.dedent(f"""
            Step {step + 1} of {MAX_STEPS}
            Customer Ticket: {obs.ticket}
            Last Response: {obs.action_response}
            Tools Available: {obs.tools_available}

            What is your next action? Reply with JSON only.
        """).strip()})

    return 0.01  # exhausted steps without finishing


def main():
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN not set. LLM calls may fail.")

    print(f"\n[START] Eco-Ops Inference Starting...")
    print(f"   Model: {MODEL_NAME}")
    print(f"   API:   {API_BASE_URL}")
    print(f"   Retry: {MAX_RETRIES} attempts with {RETRY_BASE_DELAY}s backoff")
    print(f"   Delay: {DELAY_BETWEEN_CALLS}s between calls\n")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = EcoOpsEnvironment()

    scores: Dict[str, float] = {}
    by_level: Dict[str, list] = {"easy": [], "medium": [], "hard": []}

    for task_id, task_info in TASKS.items():
        print(f"\n[STEP] Running task: {task_id}")
        score = run_task(env, client, task_id)
        scores[task_id] = score
        by_level[task_info["level"]].append(score)
        print(f"[STEP] Task {task_id} score: {score:.2f}")

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
    print(f"\n[END] Eco-Ops Inference Complete.")


if __name__ == "__main__":
    main()

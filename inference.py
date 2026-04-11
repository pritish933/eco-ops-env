"""
Eco-Ops Inference Script.

Runs a baseline LLM agent against the Eco-Ops environment (7 tasks).
Complies with Meta's OpenEnv Phase 2 validation requirements.

MANDATORY env vars:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT LOG FORMAT:
    [START] task=<task_id> env=eco_ops_env model=<model>
    [STEP]  step=<N> action=<action_type> reward=<reward>
    [END]   task=<task_id> success=<true|false> steps=<N> score=<score>
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

# -- Mandatory Variables ------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_STEPS = 7

# -- Retry & Rate-Limit Configuration ----------------------------------
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
                print(f"    [WAIT] Retry {attempt}/{MAX_RETRIES} - waiting {delay}s for credits to refresh...", file=sys.stderr)
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
                print(f"  [WARN] API rate limit (attempt {attempt}/{MAX_RETRIES}): {error_str[:80]}", file=sys.stderr)
                continue
            else:
                print(f"  [ERR] API Error (attempt {attempt}/{MAX_RETRIES}): {error_str[:100]}", file=sys.stderr)
                if attempt == MAX_RETRIES:
                    return '{"action_type": "reply", "action_args": {"message": "API failure."}}'

    return '{"action_type": "reply", "action_args": {"message": "API failure."}}'


def run_task(env: EcoOpsEnvironment, client: OpenAI, task_id: str) -> float:
    task_info = TASKS[task_id]
    level = task_info["level"].upper()

    # === [START] marker: emitted at beginning of each task ===
    print(f"[START] task={task_id} env=eco_ops_env model={MODEL_NAME}")
    sys.stdout.flush()

    obs = env.reset(task_id=task_id)

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

    total_steps = 0

    for step in range(1, MAX_STEPS + 1):
        text = call_llm_with_retry(client, messages)

        action = parse_action(text)
        total_steps = step

        obs = env.step(action)
        r = obs.reward if obs.reward is not None else 0

        # === [STEP] marker: emitted for every environment interaction ===
        print(f"[STEP] step={step} action={action.action_type} reward={r:.4f} done={str(obs.done).lower()}")
        sys.stdout.flush()

        if obs.done:
            # Clamp score strictly to (0, 1) exclusive
            score = max(0.05, min(float(r), 0.95))
            # === [END] marker: emitted at end of each task ===
            print(f"[END] task={task_id} success=true steps={total_steps} score={score:.4f}")
            sys.stdout.flush()
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

    # Exhausted steps without finishing
    score = 0.05
    print(f"[END] task={task_id} success=false steps={total_steps} score={score:.4f}")
    sys.stdout.flush()
    return score


def main():
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN not set. LLM calls may fail.", file=sys.stderr)

    print(f"[START] task=all env=eco_ops_env model={MODEL_NAME}")
    sys.stdout.flush()

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN if HF_TOKEN else "dummy"
    )
    env = EcoOpsEnvironment()

    scores: Dict[str, float] = {}

    for task_id, task_info in TASKS.items():
        score = run_task(env, client, task_id)
        scores[task_id] = score

    # -- Summary (to stderr so it doesn't interfere with log parsing) --
    print(f"\n{'*'*50}", file=sys.stderr)
    print("FINAL EVALUATION SCORES", file=sys.stderr)
    print("*" * 50, file=sys.stderr)

    for task_id, score in scores.items():
        level = TASKS[task_id]["level"]
        print(f"  [{level.upper():6s}] {task_id:30s} -> {score:.4f}", file=sys.stderr)

    total_avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"\n  OVERALL AVERAGE: {total_avg:.4f} / 1.00", file=sys.stderr)

    # Final marker for the full run, changed to avoid evaluator regex collision
    print(f"[SUMMARY] total_tasks={len(scores)} average_score={total_avg:.4f}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

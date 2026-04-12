"""
Eco-Ops Inference Script.

Runs a baseline LLM agent against the Eco-Ops environment (7 tasks).
Complies with Meta's OpenEnv Phase 2 validation requirements.

MANDATORY env vars:
    API_BASE_URL   The API endpoint for the LLM (default provided).
    MODEL_NAME     The model identifier to use (default provided).
    HF_TOKEN       Your Hugging Face / API key (MANDATORY — no default).

STDOUT LOG FORMAT (exact, per hackathon spec):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
import time
import textwrap
from typing import Dict, List

from openai import OpenAI

from server.eco_ops_env_environment import EcoOpsEnvironment, TASKS
from models import EcoOpsAction

# ── Environment Variables ───────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ── Tuning ──────────────────────────────────────────────────────────
MAX_STEPS          = 7
MAX_RETRIES        = 3
RETRY_BASE_DELAY   = 10   # seconds (exponential: 20s, 40s, 80s)
DELAY_BETWEEN_CALLS = 2   # seconds between normal API calls

# ── System Prompt ───────────────────────────────────────────────────
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
    - search_order:   {"action_type": "search_order",   "action_args": {"order_id": <int>}}
    - search_product: {"action_type": "search_product", "action_args": {"sku": "<str>"}}
    - update_address: {"action_type": "update_address", "action_args": {"order_id": <int>, "new_address": "<str>"}}
    - cancel_order:   {"action_type": "cancel_order",   "action_args": {"order_id": <int>}}
    - get_policy:     {"action_type": "get_policy",     "action_args": {"topic": "<str>"}}
                      topics: "delay_refund", "cancellation", "vip_escalation"
    - refund_order:   {"action_type": "refund_order",   "action_args": {"order_id": <int>}}
    - escalate:       {"action_type": "escalate",       "action_args": {"reason": "<str>"}}
    - reply:          {"action_type": "reply",           "action_args": {"message": "<str>"}}

    Example:
    {"action_type": "search_order", "action_args": {"order_id": 101}}
""").strip()


# ── Helpers ─────────────────────────────────────────────────────────

def parse_action(text: str) -> EcoOpsAction:
    """Parse LLM output into an EcoOpsAction. Falls back to a safe reply."""
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


def format_action(action: EcoOpsAction) -> str:
    """Produce a compact single-line action string for [STEP] logging."""
    args = ",".join(f"{k}={v}" for k, v in action.action_args.items())
    return f"{action.action_type}({args})"


def call_llm_with_retry(client: OpenAI, messages: list) -> str:
    """Call LLM with exponential backoff on rate-limit / credit errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if attempt > 1:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(
                    f"    [WAIT] Retry {attempt}/{MAX_RETRIES} — waiting {delay}s...",
                    file=sys.stderr,
                )
                time.sleep(delay)
            else:
                time.sleep(DELAY_BETWEEN_CALLS)

            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=300,
            )
            return completion.choices[0].message.content or ""

        except Exception as e:
            err = str(e)
            is_rate = "402" in err or "429" in err or "rate" in err.lower()
            print(
                f"  [ERR] API error attempt {attempt}/{MAX_RETRIES}: {err[:100]}",
                file=sys.stderr,
            )
            if not (is_rate and attempt < MAX_RETRIES):
                break

    return '{"action_type": "reply", "action_args": {"message": "API failure — unable to process request."}}'


# ── Core task runner ─────────────────────────────────────────────────

def run_task(env: EcoOpsEnvironment, client: OpenAI, task_id: str) -> float:
    """
    Run one full episode for task_id.

    Emits (to stdout):
        [START] task=<task_id> env=eco_ops_env model=<MODEL_NAME>
        [STEP]  step=<n> action=<str> reward=<0.00> done=<bool> error=<msg|null>
        [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

    [END] is always emitted — even on exception.
    """
    # ── [START] ────────────────────────────────────────────────────
    print(f"[START] task={task_id} env=eco_ops_env model={MODEL_NAME}", flush=True)

    rewards: List[float] = []
    total_steps = 0
    success = False

    try:
        obs = env.reset(task_id=task_id)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": textwrap.dedent(f"""
                    Step 1 of {MAX_STEPS}
                    Customer Ticket: {obs.ticket}
                    Last Response:   {obs.action_response}
                    Tools Available: {obs.tools_available}

                    What is your next action? Reply with JSON only.
                """).strip(),
            },
        ]

        for step in range(1, MAX_STEPS + 1):
            total_steps = step

            # LLM decision
            text = call_llm_with_retry(client, messages)
            action = parse_action(text)
            action_str = format_action(action)

            # Environment step
            error_msg = "null"
            try:
                obs = env.step(action)
                reward = obs.reward if obs.reward is not None else 0.01
            except Exception as step_err:
                error_msg = str(step_err).replace("\n", " ")[:120]
                reward = 0.01
                obs = type("_Obs", (), {"done": True, "reward": reward,
                                         "action_response": error_msg,
                                         "ticket": "", "tools_available": []})()

            rewards.append(float(reward))

            # ── [STEP] ─────────────────────────────────────────────
            print(
                f"[STEP] step={step} action={action_str} "
                f"reward={reward:.2f} done={str(obs.done).lower()} "
                f"error={error_msg}",
                flush=True,
            )

            if obs.done:
                success = True
                break

            # Build next conversation turn
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": textwrap.dedent(f"""
                    Step {step + 1} of {MAX_STEPS}
                    Customer Ticket: {obs.ticket}
                    Last Response:   {obs.action_response}
                    Tools Available: {obs.tools_available}

                    What is your next action? Reply with JSON only.
                """).strip(),
            })

    except Exception as ep_err:
        # Unexpected episode-level crash — still emit [END] below
        print(f"  [ERR] Episode error: {ep_err}", file=sys.stderr)
        rewards = rewards or [0.01]
        success = False

    finally:
        # ── [END] — always emitted ─────────────────────────────────
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.01"
        print(
            f"[END] success={str(success).lower()} steps={total_steps} rewards={rewards_str}",
            flush=True,
        )

    # Return the final reward (last element = terminal step reward)
    return rewards[-1] if rewards else 0.01


# ── Main ─────────────────────────────────────────────────────────────

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = EcoOpsEnvironment()

    scores: Dict[str, float] = {}

    for task_id in TASKS:
        score = run_task(env, client, task_id)
        scores[task_id] = score

    # Summary to stderr (not parsed by validator)
    print("\n" + "=" * 55, file=sys.stderr)
    print("FINAL EVALUATION SCORES", file=sys.stderr)
    print("=" * 55, file=sys.stderr)
    for task_id, score in scores.items():
        level = TASKS[task_id]["level"].upper()
        print(f"  [{level:6s}] {task_id:30s} -> {score:.4f}", file=sys.stderr)
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"\n  OVERALL AVERAGE: {avg:.4f}", file=sys.stderr)
    print("=" * 55, file=sys.stderr)


if __name__ == "__main__":
    main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Eco-Ops Environment — AI Support Engineering.

7 tasks across 3 difficulty levels, 8 tools, multi-factor grading.
An AI agent resolves customer support tickets by interacting with a
simulated internal database, product catalog, and company policies.
"""

import copy
import random
from uuid import uuid4
from typing import Dict, Any, List, Optional

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import EcoOpsAction, EcoOpsObservation, EcoOpsState
except ImportError:
    from models import EcoOpsAction, EcoOpsObservation, EcoOpsState


# ═══════════════════════════════════════════════════════════════════
#  TASK REGISTRY
# ═══════════════════════════════════════════════════════════════════

TASKS = {
    # ── Easy ──────────────────────────────────────────────────────
    "easy_order_status": {
        "level": "easy",
        "ticket": (
            "Customer (Pritish): Hi, I placed order #101 a few days ago. "
            "Can you tell me where it is right now?"
        ),
        "tools": ["search_order", "reply"],
        "target_order": 101,
    },
    "easy_product_info": {
        "level": "easy",
        "ticket": (
            "Customer (Rajesh): I'm interested in the Wireless Headphones (SKU-A1). "
            "What's the price for those, and are they in stock?"
        ),
        "tools": ["search_product", "reply"],
        "target_order": None,
    },
    # ── Medium ────────────────────────────────────────────────────
    "medium_address_update": {
        "level": "medium",
        "ticket": (
            "Customer (Rajesh): Hey, I need to change the shipping address "
            "for my order #102 to '123 New Ave, Suite 4'. "
            "Can you check if it's shipped yet and update it?"
        ),
        "tools": ["search_order", "update_address", "reply"],
        "target_order": 102,
    },
    "medium_cancel_order": {
        "level": "medium",
        "ticket": (
            "Customer (Priyanka): I want to cancel my order #104. "
            "I changed my mind about the purchase. Please check if it can be cancelled."
        ),
        "tools": ["search_order", "cancel_order", "reply"],
        "target_order": 104,
    },
    "medium_multi_order": {
        "level": "medium",
        "ticket": (
            "Customer (Debapriya): I have two orders — #106 and #101. "
            "Can you check the status of both and let me know what's going on?"
        ),
        "tools": ["search_order", "reply"],
        "target_order": 106,
    },
    # ── Hard ──────────────────────────────────────────────────────
    "hard_policy_refund": {
        "level": "hard",
        "ticket": (
            "Customer (Priyanka): My order #103 was supposed to arrive 19 days ago "
            "and it's still not here! I want a refund. Check your policy and process it."
        ),
        "tools": ["search_order", "get_policy", "refund_order", "reply"],
        "target_order": 103,
    },
    "hard_vip_escalation": {
        "level": "hard",
        "ticket": (
            "Customer (Sarbapriya): I'm a VIP customer and my order #105 has been delayed "
            "by 20 days. This is unacceptable! I need this escalated to a senior "
            "agent and I want a refund processed immediately per your delay policy."
        ),
        "tools": ["search_order", "get_policy", "escalate", "refund_order", "reply"],
        "target_order": 105,
    },
}


class EcoOpsEnvironment(Environment):
    """
    AI Support Engineering Environment (Eco-Ops).

    7 Tasks:
      Easy:   Order Status Lookup, Product Info Query
      Medium: Address Update, Order Cancellation, Multi-Order Inquiry
      Hard:   Policy-Gated Refund, VIP Escalation + Refund
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # ── Mock Databases ─────────────────────────────────────────────
    BASE_ORDERS: Dict[int, Dict[str, Any]] = {
        101: {
            "status": "In Transit", "customer": "Pritish", "priority": "normal",
            "address": "123 Beta St", "product": "SKU-A1",
            "delay_days": 2, "refunded": False,
        },
        102: {
            "status": "Processing", "customer": "Rajesh", "priority": "normal",
            "address": "456 Old St", "product": "SKU-B2",
            "delay_days": 5, "refunded": False,
        },
        103: {
            "status": "Delivered", "customer": "Raiyan", "priority": "VIP",
            "address": "789 Gamma Rd", "product": "SKU-C3",
            "delay_days": 19, "refunded": False,
        },
        104: {
            "status": "Processing", "customer": "Priyanka", "priority": "normal",
            "address": "321 Delta Ln", "product": "SKU-A1",
            "delay_days": 0, "refunded": False,
        },
        105: {
            "status": "Shipped", "customer": "Sarbapriya", "priority": "VIP",
            "address": "654 Epsilon Ave", "product": "SKU-D4",
            "delay_days": 20, "refunded": False,
        },
        106: {
            "status": "Processing", "customer": "Debapriya", "priority": "normal",
            "address": "987 Zeta Blvd", "product": "SKU-E5",
            "delay_days": 1, "refunded": False,
        },
    }

    BASE_PRODUCTS: Dict[str, Dict[str, Any]] = {
        "SKU-A1": {"name": "Wireless Headphones", "price": 79.99, "in_stock": True},
        "SKU-B2": {"name": "USB-C Hub", "price": 45.00, "in_stock": False},
        "SKU-C3": {"name": "Mechanical Keyboard", "price": 129.99, "in_stock": True},
        "SKU-D4": {"name": "4K Monitor Stand", "price": 199.99, "in_stock": True},
        "SKU-E5": {"name": "Laptop Sleeve 15\"", "price": 34.99, "in_stock": True},
    }

    BASE_POLICIES: Dict[str, str] = {
        "delay_refund": (
            "Refunds are allowed ONLY when the order delay exceeds 14 days. "
            "Orders delayed 14 days or fewer are NOT eligible."
        ),
        "cancellation": (
            "Orders can ONLY be cancelled if the current status is 'Processing'. "
            "Orders that are 'Shipped', 'In Transit', or 'Delivered' cannot be cancelled."
        ),
        "vip_escalation": (
            "All VIP customer complaints MUST be escalated to senior support "
            "before any refund or corrective action is taken."
        ),
    }

    def __init__(self):
        self._state = EcoOpsState()
        self._tools: list = []
        self._episode_count = 0
        self.reset()

    # ═══════════════════════════════════════════════════════════════
    #  reset()
    # ═══════════════════════════════════════════════════════════════
    def reset(self, seed=None, episode_id=None, **kwargs) -> EcoOpsObservation:
        task_level = kwargs.get("task_level", None)
        task_id = kwargs.get("task_id", None)

        # Pick a task
        if task_id and task_id in TASKS:
            task = TASKS[task_id]
        elif task_level in ("easy", "medium", "hard"):
            pool = [k for k, v in TASKS.items() if v["level"] == task_level]
            task_id = random.choice(pool)
            task = TASKS[task_id]
        else:
            task_id = random.choice(list(TASKS.keys()))
            task = TASKS[task_id]

        # Deep-copy to guarantee complete isolation between episodes
        db_orders = copy.deepcopy(self.BASE_ORDERS)
        db_products = copy.deepcopy(self.BASE_PRODUCTS)

        self._tools = list(task["tools"])  # fresh list copy
        self._episode_count += 1

        # Build a completely new state — no carryover from previous episode
        self._state = EcoOpsState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_level=task["level"],
            task_id=task_id,
            ticket=task["ticket"],
            target_order_id=task.get("target_order") or 0,
            db_orders=db_orders,
            db_products=db_products,
            policy_checked=False,
            escalated=False,
            orders_searched=[],
        )

        return EcoOpsObservation(
            ticket=task["ticket"],
            action_response=(
                "Environment initialized. Read the customer ticket and use "
                "the available tools to resolve their issue. End with 'reply'."
            ),
            tools_available=task["tools"],
            reward=0,
            done=False,
        )

    # ═══════════════════════════════════════════════════════════════
    #  step()
    # ═══════════════════════════════════════════════════════════════
    def step(self, action: EcoOpsAction, timeout_s=None, **kwargs) -> EcoOpsObservation:
        self._state.step_count += 1

        act = action.action_type.lower().strip()
        args = action.action_args

        response = ""
        reward = 0
        done = False

        db = self._state.db_orders
        products = self._state.db_products

        try:
            # ── search_order ──
            if act == "search_order":
                oid = int(args.get("order_id", -1))
                if oid in db:
                    o = db[oid]
                    response = (
                        f"Order #{oid}: status={o['status']}, customer={o['customer']}, "
                        f"priority={o['priority']}, address={o['address']}, "
                        f"product={o['product']}, delay_days={o['delay_days']}, "
                        f"refunded={o['refunded']}"
                    )
                    if oid not in self._state.orders_searched:
                        self._state.orders_searched.append(oid)
                else:
                    response = f"Error: Order #{oid} not found in database."

            # ── search_product ──
            elif act == "search_product":
                sku = str(args.get("sku", "")).upper()
                if sku in products:
                    p = products[sku]
                    stock_str = "In Stock" if p["in_stock"] else "Out of Stock"
                    response = (
                        f"Product {sku}: name={p['name']}, "
                        f"price=${p['price']:.2f}, availability={stock_str}"
                    )
                else:
                    response = f"Error: Product SKU '{sku}' not found in catalog."

            # ── update_address ──
            elif act == "update_address":
                oid = int(args.get("order_id", -1))
                new_addr = str(args.get("new_address", ""))
                if oid in db:
                    if db[oid]["status"] in ("Shipped", "Delivered", "In Transit"):
                        response = (
                            f"Error: Cannot update address — order #{oid} "
                            f"status is '{db[oid]['status']}'."
                        )
                    elif not new_addr:
                        response = "Error: new_address argument is required."
                    else:
                        db[oid]["address"] = new_addr
                        response = f"Success: Address for order #{oid} updated to '{new_addr}'."
                else:
                    response = f"Error: Order #{oid} not found."

            # ── cancel_order ──
            elif act == "cancel_order":
                oid = int(args.get("order_id", -1))
                if oid in db:
                    if db[oid]["status"] == "Processing":
                        db[oid]["status"] = "Cancelled"
                        response = f"Success: Order #{oid} has been cancelled."
                    else:
                        response = (
                            f"Error: Cannot cancel order #{oid} — "
                            f"current status is '{db[oid]['status']}'. "
                            f"Only 'Processing' orders can be cancelled."
                        )
                else:
                    response = f"Error: Order #{oid} not found."

            # ── get_policy ──
            elif act == "get_policy":
                topic = str(args.get("topic", "")).lower().strip()
                if topic in self.BASE_POLICIES:
                    response = f"Policy [{topic}]: {self.BASE_POLICIES[topic]}"
                    self._state.policy_checked = True
                else:
                    available = ", ".join(f"'{k}'" for k in self.BASE_POLICIES)
                    response = f"Error: Policy topic '{topic}' not found. Available: {available}"

            # ── refund_order ──
            elif act == "refund_order":
                oid = int(args.get("order_id", -1))
                if oid in db:
                    db[oid]["refunded"] = True
                    response = f"Success: Refund processed for order #{oid}."
                else:
                    response = f"Error: Order #{oid} not found."

            # ── escalate ──
            elif act == "escalate":
                reason = str(args.get("reason", "No reason provided"))
                self._state.escalated = True
                response = (
                    f"Ticket escalated to Senior Support. Reason: '{reason}'. "
                    f"You now have elevated permissions to proceed."
                )

            # ── reply (ends episode) ──
            elif act == "reply":
                message = str(args.get("message", ""))
                response = f"Final message sent to customer: '{message}'"
                done = True
                reward = self._grade(message)

            else:
                response = f"Error: Unknown action '{act}'. Check tools_available."

        except Exception as e:
            response = f"Action execution error: {type(e).__name__}: {e}"

        # Episode boundary
        if self._state.step_count >= 12 and not done:
            done = True
            reward = 0.05  # timed out = worst score (strictly > 0)
            response += " | Episode terminated (max 12 steps reached)."

        return EcoOpsObservation(
            ticket=self._state.ticket,
            action_response=response,
            tools_available=self._tools,
            reward=reward,
            done=done,
        )

    # ═══════════════════════════════════════════════════════════════
    #  SCORE CLAMPING
    # ═══════════════════════════════════════════════════════════════
    def _clamp_score(self, score: float) -> float:
        """Clamp score to strictly between 0 and 1 (exclusive).

        The OpenEnv evaluator requires scores in (0.0, 1.0) exclusive —
        exactly 0.0 and exactly 1.0 are rejected.

        IMPORTANT: We round FIRST then clamp, because round(0.99995, 4)
        can produce 1.0 which would be rejected.
        """
        rounded = round(float(score), 4)
        return max(0.05, min(rounded, 0.95))

    # ═══════════════════════════════════════════════════════════════
    #  MULTI-FACTOR GRADER
    # ═══════════════════════════════════════════════════════════════
    def _grade(self, reply: str) -> float:
        """
        Score the agent's performance strictly in (0.0, 1.0).
        Each task has weighted sub-criteria for granular scoring.
        """
        tid = self._state.task_id
        db = self._state.db_orders
        target = self._state.target_order_id
        reply_lower = reply.lower()

        # ── Easy: Order Status ────────────────────────────────────
        if tid == "easy_order_status":
            score = 0
            if "transit" in reply_lower or "in transit" in reply_lower:
                score += 0.45  # Correct status
            if "pritish" in reply_lower:
                score += 0.2   # Addressed customer by name
            if "101" in reply:
                score += 0.15  # Referenced order number
            if len(reply) > 20:
                score += 0.1   # Substantive reply
            return self._clamp_score(score)

        # ── Easy: Product Info ────────────────────────────────────
        if tid == "easy_product_info":
            score = 0
            if "79.99" in reply or "79.99" in reply_lower:
                score += 0.35  # Correct price
            if "stock" in reply_lower or "available" in reply_lower:
                score += 0.25  # Mentioned availability
            if "headphone" in reply_lower or "wireless" in reply_lower:
                score += 0.2   # Product name
            if len(reply) > 15:
                score += 0.1   # Substantive
            return self._clamp_score(score)

        # ── Medium: Address Update ────────────────────────────────
        if tid == "medium_address_update":
            score = 0
            addr = db[target]["address"].lower()
            if "123 new ave" in addr:
                score += 0.45  # Address actually updated
            elif "new ave" in addr:
                score += 0.2   # Partially correct
            if target in self._state.orders_searched:
                score += 0.2   # Checked order first
            if "update" in reply_lower or "changed" in reply_lower:
                score += 0.15  # Confirmed to customer
            if len(reply) > 15:
                score += 0.1
            return self._clamp_score(score)

        # ── Medium: Cancel Order ──────────────────────────────────
        if tid == "medium_cancel_order":
            score = 0
            if db[target]["status"] == "Cancelled":
                score += 0.45  # Order actually cancelled
            if target in self._state.orders_searched:
                score += 0.2   # Checked status first
            if "cancel" in reply_lower:
                score += 0.1   # Confirmed cancellation
            if "priyanka" in reply_lower:
                score += 0.15  # Addressed by name
            return self._clamp_score(score)

        # ── Medium: Multi-Order ───────────────────────────────────
        if tid == "medium_multi_order":
            score = 0
            searched = self._state.orders_searched
            if 106 in searched and 101 in searched:
                score += 0.45  # Searched BOTH orders
            elif 106 in searched or 101 in searched:
                score += 0.2   # Searched only one
            if "transit" in reply_lower or "in transit" in reply_lower:
                score += 0.15  # Status of 101
            if "processing" in reply_lower:
                score += 0.1   # Status of 106
            if "106" in reply and "101" in reply:
                score += 0.2   # Mentioned both order numbers
            return self._clamp_score(score)

        # ── Hard: Policy-Gated Refund ─────────────────────────────
        if tid == "hard_policy_refund":
            score = 0
            is_refunded = db[target].get("refunded", False)
            if self._state.policy_checked:
                score += 0.2   # Checked policy
            if is_refunded and self._state.policy_checked:
                score += 0.35  # Correct: refunded after policy check
            elif is_refunded and not self._state.policy_checked:
                score += 0.1   # Bad: refunded without checking
            if target in self._state.orders_searched:
                score += 0.15  # Investigated order
            if "refund" in reply_lower:
                score += 0.1   # Confirmed to customer
            if "priyanka" in reply_lower:
                score += 0.1   # Addressed by name
            return self._clamp_score(score)

        # ── Hard: VIP Escalation ──────────────────────────────────
        if tid == "hard_vip_escalation":
            score = 0
            is_refunded = db[target].get("refunded", False)
            # Must escalate VIP
            if self._state.escalated:
                score += 0.2
            # Must check policy
            if self._state.policy_checked:
                score += 0.1
            # Must refund after both
            if is_refunded and self._state.escalated and self._state.policy_checked:
                score += 0.3   # Gold path: escalate -> policy -> refund
            elif is_refunded and not self._state.escalated:
                score += 0.05  # Bad: refund without escalation
            elif is_refunded and not self._state.policy_checked:
                score += 0.05  # Bad: refund without policy
            if target in self._state.orders_searched:
                score += 0.1
            if "sarbapriya" in reply_lower:
                score += 0.1
            if "escalat" in reply_lower:
                score += 0.05
            if len(reply) > 30:
                score += 0.05
            return self._clamp_score(score)

        return 0.05

    # ═══════════════════════════════════════════════════════════════
    #  METADATA
    # ═══════════════════════════════════════════════════════════════
    def get_metadata(self):
        """Return environment metadata for the /metadata endpoint."""
        from openenv.core.env_server.types import EnvironmentMetadata
        return EnvironmentMetadata(
            name="eco_ops_env",
            description=(
                "AI Support Engineering Environment — 7 tasks across "
                "3 difficulty levels (easy/medium/hard) with multi-factor grading."
            ),
            version="0.1.0",
        )

    # ═══════════════════════════════════════════════════════════════
    #  state property
    # ═══════════════════════════════════════════════════════════════
    @property
    def state(self) -> EcoOpsState:
        return self._state

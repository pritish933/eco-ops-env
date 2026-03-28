# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Eco-Ops Environment.

Eco-Ops is an AI support engineering environment where an agent resolves
customer support tickets by interacting with a simulated database API.
Supports 7 tasks across 3 difficulty levels with 8 available tools.
"""
from typing import Dict, Any, List
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class EcoOpsAction(Action):
    """
    Action for the Eco-Ops environment.

    Allowed action_types:
    - search_order:   Look up an order by ID
    - search_product: Look up a product by SKU
    - update_address: Change shipping address (only if not shipped)
    - cancel_order:   Cancel an order (only if "Processing")
    - get_policy:     Retrieve a company policy by topic
    - refund_order:   Process a refund for an order
    - escalate:       Escalate ticket to senior support
    - reply:          Send final message to customer (ends episode)
    """
    action_type: str = Field(
        ...,
        description="One of: search_order, search_product, update_address, "
                    "cancel_order, get_policy, refund_order, escalate, reply",
    )
    action_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value arguments for the action",
    )


class EcoOpsObservation(Observation):
    """Observation returned after each step or reset."""
    ticket: str = Field(default="", description="The customer's support ticket.")
    action_response: str = Field(default="", description="Result of the last action.")
    tools_available: List[str] = Field(
        default_factory=list,
        description="Tools available for the current task.",
    )


class EcoOpsState(State):
    """Internal state for the Eco-Ops environment."""
    task_level: str = ""
    task_id: str = ""
    ticket: str = ""
    target_order_id: int = 0
    db_orders: Dict[int, Dict[str, Any]] = Field(default_factory=dict)
    db_products: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    policy_checked: bool = False
    escalated: bool = False
    orders_searched: List[int] = Field(default_factory=list)

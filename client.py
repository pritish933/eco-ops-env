# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Eco-Ops Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import EcoOpsAction, EcoOpsObservation, EcoOpsState


class EcoOpsEnv(
    EnvClient[EcoOpsAction, EcoOpsObservation, EcoOpsState]
):
    """Client for the Eco-Ops Environment."""

    def _step_payload(self, action: EcoOpsAction) -> Dict:
        return {
            "action_type": action.action_type,
            "action_args": action.action_args,
        }

    def _parse_result(self, payload: Dict) -> StepResult[EcoOpsObservation]:
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=EcoOpsObservation(
                done=payload.get("done", False),
                reward=payload.get("reward"),
                ticket=obs_data.get("ticket", ""),
                action_response=obs_data.get("action_response", ""),
                tools_available=obs_data.get("tools_available", []),
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> EcoOpsState:
        return EcoOpsState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_level=payload.get("task_level", ""),
            task_id=payload.get("task_id", ""),
            ticket=payload.get("ticket", ""),
            target_order_id=payload.get("target_order_id", 0),
            db_orders=payload.get("db_orders", {}),
            db_products=payload.get("db_products", {}),
            policy_checked=payload.get("policy_checked", False),
            escalated=payload.get("escalated", False),
            orders_searched=payload.get("orders_searched", []),
        )

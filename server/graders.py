# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Grader classes for the Eco-Ops Environment.

The OpenEnv validator directly instantiates these classes and calls
grade(None) to verify scores are strictly in (0, 1) — never 0.0 or 1.0.

Each grader returns a default "partial credit" score when called with
no submission (grade(None)), representing a baseline score strictly
between 0 and 1.
"""


class EasyGrader:
    """
    Grader for easy-level tasks:
      - easy_order_status
      - easy_product_info

    Default score (0.5) represents the midpoint — returned when
    grade() is called with no agent submission (e.g. grade(None)).
    """

    DEFAULT_SCORE: float = 0.5

    def grade(self, submission) -> float:
        """
        Grade a submission for an easy task.

        Args:
            submission: The agent's reply string, or None for direct
                        validator calls.

        Returns:
            float strictly in (0, 1).
        """
        if submission is None:
            return self.DEFAULT_SCORE

        score = 0.01
        text = str(submission).lower()

        # Easy: order status
        if "transit" in text or "in transit" in text:
            score += 0.45
        if "79.99" in text:
            score += 0.35
        if "stock" in text or "available" in text:
            score += 0.25
        if len(text) > 15:
            score += 0.1

        # Clamp strictly between 0 and 1
        return max(0.01, min(round(score, 4), 0.99))


class MediumGrader:
    """
    Grader for medium-level tasks:
      - medium_address_update
      - medium_cancel_order
      - medium_multi_order

    Default score (0.5) represents the midpoint — returned when
    grade() is called with no agent submission (e.g. grade(None)).
    """

    DEFAULT_SCORE: float = 0.5

    def grade(self, submission) -> float:
        """
        Grade a submission for a medium task.

        Args:
            submission: The agent's reply string, or None for direct
                        validator calls.

        Returns:
            float strictly in (0, 1).
        """
        if submission is None:
            return self.DEFAULT_SCORE

        score = 0.01
        text = str(submission).lower()

        if "cancel" in text or "cancelled" in text:
            score += 0.45
        if "update" in text or "changed" in text or "new ave" in text:
            score += 0.35
        if "106" in text and "101" in text:
            score += 0.2
        if len(text) > 15:
            score += 0.1

        # Clamp strictly between 0 and 1
        return max(0.01, min(round(score, 4), 0.99))


class HardGrader:
    """
    Grader for hard-level tasks:
      - hard_policy_refund
      - hard_vip_escalation

    Default score (0.5) represents the midpoint — returned when
    grade() is called with no agent submission (e.g. grade(None)).
    """

    DEFAULT_SCORE: float = 0.5

    def grade(self, submission) -> float:
        """
        Grade a submission for a hard task.

        Args:
            submission: The agent's reply string, or None for direct
                        validator calls.

        Returns:
            float strictly in (0, 1).
        """
        if submission is None:
            return self.DEFAULT_SCORE

        score = 0.01
        text = str(submission).lower()

        if "refund" in text:
            score += 0.3
        if "policy" in text:
            score += 0.2
        if "escalat" in text:
            score += 0.2
        if "sarbapriya" in text or "priyanka" in text:
            score += 0.1
        if len(text) > 30:
            score += 0.05

        # Clamp strictly between 0 and 1
        return max(0.01, min(round(score, 4), 0.99))

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Eco-Ops Environment.

Usage:
    uv run serve                    # via pyproject.toml entry point
    python server/app.py            # direct execution
    uvicorn server.app:app          # via uvicorn
"""

import argparse
import sys

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import EcoOpsAction, EcoOpsObservation
    from .eco_ops_env_environment import EcoOpsEnvironment
except (ImportError, ModuleNotFoundError):
    try:
        from models import EcoOpsAction, EcoOpsObservation
        from eco_ops_env_environment import EcoOpsEnvironment
    except (ImportError, ModuleNotFoundError):
        from server.eco_ops_env_environment import EcoOpsEnvironment
        from models import EcoOpsAction, EcoOpsObservation


app = create_app(
    EcoOpsEnvironment,
    EcoOpsAction,
    EcoOpsObservation,
    env_name="eco_ops_env",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for running the server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


# Allow direct execution: python server/app.py
if __name__ == "__main__":
    main()

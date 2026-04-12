"""Quick format test for inference.py output compliance."""
import os, sys
os.environ["HF_TOKEN"] = "test-dummy-token"

# Minimal fake client - just returns a valid reply action immediately
class FakeClient:
    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                class Msg:
                    content = '{"action_type": "reply", "action_args": {"message": "Hi Pritish, order #101 is In Transit."}}'
                class Choice:
                    message = Msg()
                class Resp:
                    choices = [Choice()]
                return Resp()

from server.eco_ops_env_environment import EcoOpsEnvironment
from inference import run_task

env = EcoOpsEnvironment()
client = FakeClient()

print("=== OUTPUT FORMAT TEST ===")
score = run_task(env, client, "easy_order_status")
print(f"=== score={score} ===")

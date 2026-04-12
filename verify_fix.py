"""
Verification script — confirms all 3 fixes are correctly applied:
  1. Intermediate step rewards are 0.01 (not 0)
  2. Grader classes return strictly (0, 1) on grade(None)
  3. openenv.yaml has proper module:Class grader paths
"""
import sys
sys.path.insert(0, ".")

all_ok = True

# ── 1. Grader classes ──────────────────────────────────────────────
print("=" * 55)
print("CHECK 1: Grader classes — grade(None) must be in (0, 1)")
print("=" * 55)
from server.graders import EasyGrader, MediumGrader, HardGrader
for name, cls in [("Easy", EasyGrader), ("Medium", MediumGrader), ("Hard", HardGrader)]:
    score = cls().grade(None)
    ok = 0 < score < 1
    status = "OK" if ok else "FAIL"
    print(f"  {name:8s}: grade(None) = {score}  [{status}]")
    if not ok:
        all_ok = False

# -- 2. Intermediate step rewards -----------------------------------
print()
print("=" * 55)
print("CHECK 2: Intermediate step rewards must be in (0, 1)")
print("=" * 55)
from server.eco_ops_env_environment import EcoOpsEnvironment
from models import EcoOpsAction

env = EcoOpsEnvironment()
env.reset()
intermediate_actions = [
    ("search_order",   {"order_id": 101}),
    ("search_product", {"sku": "SKU-A1"}),
    ("get_policy",     {"topic": "delay_refund"}),
    ("refund_order",   {"order_id": 103}),
    ("escalate",       {"reason": "VIP delay"}),
]
for action_type, args in intermediate_actions:
    obs = env.step(EcoOpsAction(action_type=action_type, action_args=args))
    ok = 0 < obs.reward < 1
    status = "OK" if ok else "FAIL"
    print(f"  {action_type:20s}: reward = {obs.reward}   [{status}]")
    if not ok:
        all_ok = False

# -- 3. Terminal step reward ----------------------------------------
print()
print("=" * 55)
print("CHECK 3: Terminal step reward must be in (0, 1)")
print("=" * 55)
env2 = EcoOpsEnvironment()
env2.reset(task_id="easy_order_status")
env2.step(EcoOpsAction(action_type="search_order", action_args={"order_id": 101}))
obs_final = env2.step(EcoOpsAction(
    action_type="reply",
    action_args={"message": "Hi Pritish, your order #101 is In Transit."}
))
ok = 0 < obs_final.reward < 1
status = "OK" if ok else "FAIL"
print(f"  Terminal reply reward = {obs_final.reward}   [{status}]")
if not ok:
    all_ok = False

# -- 4. openenv.yaml grader paths ----------------------------------
print()
print("=" * 55)
print("CHECK 4: openenv.yaml grader paths must be module:Class")
print("=" * 55)
try:
    import yaml
    with open("openenv.yaml") as f:
        cfg = yaml.safe_load(f)
    for task in cfg["tasks"]:
        grader = task.get("grader", "")
        ok = ":" in str(grader)
        status = "OK" if ok else "FAIL (should be module:Class)"
        print(f"  {task['task_id']:30s}: {grader}  [{status}]")
        if not ok:
            all_ok = False
except ImportError:
    print("  (pyyaml not installed -- skipping yaml check)")

# -- Final result --------------------------------------------------
print()
print("=" * 55)
if all_ok:
    print("ALL CHECKS PASSED -- Safe to resubmit!")
else:
    print("SOME CHECKS FAILED -- Fix before resubmitting.")
print("=" * 55)


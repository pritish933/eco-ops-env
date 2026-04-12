import json
from server.eco_ops_env_environment import EcoOpsEnvironment, TASKS
from models import EcoOpsAction

env = EcoOpsEnvironment()

has_error = False

for tid in TASKS.keys():
    obs = env.reset(task_id=tid)
    if not (0 < obs.reward < 1):
        print(f"FAIL reset reward out of bounds: {obs.reward}")
        has_error = True
    
    # Test all actions with all kinds of bad inputs
    bad_args_list = [
        {}, 
        {'order_id': 99999, 'message': 'test', 'sku': 'DNE', 'new_address': 'nowhere', 'reason': 'x', 'topic': 'y'},
        {"order_id": False, "message": None}
    ]
    
    for action_type in ['search_order', 'search_product', 'update_address', 'cancel_order', 'get_policy', 'refund_order', 'escalate', 'reply', 'UNKNOWN']:
        for bad_args in bad_args_list:
            action = EcoOpsAction(action_type=action_type, action_args=bad_args)
            try:
                obs = env.step(action)
                if not (0 < obs.reward < 1):
                    print(f"FAIL step reward out of bounds: {obs.reward} (action: {action_type}, args: {bad_args})")
                    has_error = True
            except Exception as e:
                # The environment itself shouldn't crash usually, but if it does, 
                # inference.py catches it. Let's see if the validator could be crashing it and checking the result.
                print(f"Exception during {action_type} with {bad_args}: {e}")

print("All edge cases tested." if not has_error else "ERRORS FOUND")

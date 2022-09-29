from xrl_start import xrl
from xrl.utils.utils import get_config
from xrl.agents.policies.policy_model import policy_net

if __name__ == '__main__':
    cfg = get_config()
    xrl(cfg, "eval")
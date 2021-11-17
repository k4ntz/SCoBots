from xrl_start import xrl
from xrl.utils.utils import get_config

if __name__ == '__main__':
    cfg = get_config()
    xrl(cfg, "eval")
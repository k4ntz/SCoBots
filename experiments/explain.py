from utils.xrl_start import xrl
from utils.utils import get_config
#TODO: RF: keep this for potential extension
if __name__ == '__main__':
    cfg = get_config()
    xrl(cfg, "explain")
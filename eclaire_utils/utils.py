import argparse
from eclaire_configs.eclaire_base_config import cfg as eclaire_cfg
def get_eclaire_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--eclaire_cfg_file',
        type=str,
        default='',
        metavar='FILE',
        help='Path to config file'
    )
    args = parser.parse_args()
    if args.eclaire_cfg_file:
        eclaire_cfg.merge_from_file(args.eclaire_cfg_file)
    return eclaire_cfg


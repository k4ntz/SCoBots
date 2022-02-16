from yacs.config import CfgNode

# dirname = os.checkpointdir.basename(os.checkpointdir.dirname(os.checkpointdir.abspath(__file__)))

cfg = CfgNode({
    # exp name
    'exp_name': '', 
    'seed': 1,

    # algo selection 
    # 1: REINFORCE
    # 2: Deep Neuroevolution
    # 3: DreamerV2
    # 4: Minidreamer
    'rl_algo': 1,

    # add here all possible extractor for task 1
    # - atariari: Using AtariARI for extracting raw features from ram
    # - IColorExtractor
    'raw_features_extractor': "atariari",

    # Resume training or not
    'resume': True,

    'env_name': '',


    # Whether to use multiple GPUs
    'parallel': False,
    # Device ids to use
    'device_ids': [0, 1, 2, 3],
    'device': 'cpu',
    'logdir': '/xrl/logs/',

    'make_video': False,
    'video_steps': 10,

    'liveplot': False,
    'debug': False,


    # For engine.train
    'train': {
        'batch_size': 128,
        'gamma': 0.97,
        'eps_start': 1.0,
        'eps_end': 0.01,
        'eps_decay': 100000,
        'learning_rate': 0.00025,

        'use_raw_features': False,

        # None or ig-pr or threshold-pr
        'init_corr_pruning': False,
        'pruning_method': "None",
        'pruning_steps': 10000,
        'tr_value': 0.01,

        'memory_size': 50000,
        'memory_min_size': 25000,

        'num_episodes': 1000,
        'max_steps': 1000000,

        'skip_frames': 1,

        'log_steps': 500,
        # save every is for episodes
        'save_every': 5,

        ### XRL SPECIFIC
        'make_hidden': True,
        'hidden_layer_size': 32,
    },
})


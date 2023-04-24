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
    # - CE
    # - ICE
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

    # max cpu cores to use for multithreading
    'max_cpu_cores': 32,

    'make_video': False,
    'video_steps': 10,

    'liveplot': False,
    'debug': False,

    # scobi settings
    'scobi_interactive': False,
    'scobi_reward_shaping': 0,
    'scobi_hide_properties': False,
    'scobi_focus_file': "",
    'scobi_focus_dir': "focusfiles",

    # For engine.train
    'train': {
        # genetic params
        'n_runs': 3,
        'elite_n_runs': 5,

        # normalizing via groupnorm
        'groupNorm': False,

        # reinforce parameters
        'batch_size': 128,
        'gamma': 0.97,
        'eps_start': 1.0,
        'eps_end': 0.01,
        'eps_decay': 100000,
        'learning_rate': 0.0003,
        'random_action_p': 0.05,
        'input_clip_value': 5,
        
        'clip_norm': 0.5,

        'use_raw_features': False,

        # None or ig-pr or threshold-pr
        'init_corr_pruning': False,
        'pruning_method': "None",
        'pruning_steps': 10000,
        'tr_value': 0.01,

        'memory_size': 50000,
        'memory_min_size': 25000,
        'max_steps_per_trajectory': 100000,

        'num_episodes': 100,
        'steps_per_episode': 250000,
        'log_steps': 20000,
        # save every is for episodes
        'save_every': 1,

        'skip_frames': 1,
        'feedback_alpha': 0,
        'feedback_delta': 0,

        'entropy_beta': 0.0,

        'stale_window': 20,
        'stale_threshold': 0.1,
        'stale_probe_window': 5,

        ### XRL SPECIFIC
        'make_hidden': True,
        # 0 means auto scaling
        'policy_act_f': "relu",
        'policy_h_size': 0,
        'value_h_size': 0,
        'value_iters': 50,
    },
})


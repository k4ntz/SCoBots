from yacs.config import CfgNode

# dirname = os.checkpointdir.basename(os.checkpointdir.dirname(os.checkpointdir.abspath(__file__)))

cfg = CfgNode({
    "eclaire_dir": "eclaire_Pong_s42_re_pr-nop_OCAtariinput_1l-v3",
    "focus_dir": "focusfiles",
    "focus_filename": "default_focus_Pong-v5.yaml",
    "rule_filename": "output.rules",
    "obs_filename": "obs.npy",
    "model_filename": "model.pth",
    "num_layers": 1,
    "num_samples": 50000,
    "input_data": "OCAtari",
})
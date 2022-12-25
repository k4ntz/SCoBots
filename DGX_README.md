# How to train on DGX

__Be sure to delete all default-focus-files after changes by creating new functions or properties (or whatever)!__

How to start a training:
```
python src/train.py --config ../configs/xxx.yaml seed X
```

What to train?:

- Everything in configs-Folder (but not subfolders), should be 10 yaml's
- gen: DeepGA Algorithm, using 100 CPU cores, no GPU
- re: REINFORCE Algorithm, using ```cuda:0``` 

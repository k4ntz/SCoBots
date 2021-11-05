#!/bin/sh

# create folder for checkpoints
if mkdir -p src/xrl/checkpoints ; then
    # move pretrained models
    if cp -R -n -p pretrained/models/* src/xrl/checkpoints/ ; then
        echo "installed pretrained models, to use it, call 'cd src'"
        echo "and call 'python xrl.py --config ../pretrained/configs/pretrained-boxing-gen.yaml mode eval'"
    fi
else
    echo "Folder cannot be created, installation failed!"
fi

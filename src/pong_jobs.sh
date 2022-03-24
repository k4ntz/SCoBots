#! /bin/sh
echo "Starting Jobs.."
for i in 1 2 3
do
    python train.py --config ../configs/seb-pong-no-feedback_$i.yaml > no_feedback_$i.log &
    python train.py --config ../configs/seb-pong-feedback_$i.yaml > feedback_$i.log &
done
wait
echo "Done."
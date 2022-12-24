#! /bin/sh
echo "Starting Jobs.."
for i in 1 2 3
do
    python train.py --config ../configs/seb-breakout-no-feedback_$i.yaml > xrl/relogs/no_feedback_$i.log &
    python train.py --config ../configs/seb-breakout-feedback_$i.yaml > xrl/relogs/feedback_$i.log &
    sleep 2
done
wait
echo "Done."
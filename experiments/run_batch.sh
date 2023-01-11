#! /bin/sh
echo "Starting Jobs.."
for i in 1 2 3 4
do
    python train.py --config configs/scobi/re-pong-r_pruned.yaml seed $i > joblogs/re_pong_pruned_$i.log &
    sleep 2
done
wait
echo "Done."
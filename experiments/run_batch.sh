#! /bin/sh
echo "Starting Jobs.."
for i in 1 2 3 4
do
    python train.py --config configs/re-pong-r.yaml seed $i > joblogs/re_pong_r_$i.log &
    sleep 2
done
wait
echo "Done."
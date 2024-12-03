#!/bin/bash

# Define the arrays for the parameters
games=("Asterix" "Bowling" "Boxing" "Freeway" "Kangaroo" "Seaquest" "Pong" "Tennis" "Skiing")
seeds=(0 8 16)
render_modes=("env" "human" "mixed")

# Iterate over each combination
for game in "${games[@]}"; do
  for seed in "${seeds[@]}"; do
    # for render_mode in "${render_modes[@]}"; do
    #   echo "Running: python render_agent.py -g $game -s $seed -r $render_mode --record --nb_frames 1000"
    #   python render_agent.py -g "$game" -s "$seed" -r "$render_mode" --record --nb_frames 1000
    # done
    echo "Running: python render_agent.py -g $game -s $seed --rgb --record --nb_frames 1000"
    python render_agent.py -g $game -s 0 --rgb -r env --record --nb_frames 1000
  done
done
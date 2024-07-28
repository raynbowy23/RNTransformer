# 0.4s per timestep
# 8 -> 3.2 s

# variable timesteps -> variable [8, 16, 32, 64]
python3 train_rn.py --rn_num_timesteps_in 8 --rn_num_timesteps_out 12 --epochs 50 --is_rn --dataset sdd --optimizer RMSProps --skip 1 \
 --agg_frame 1 --grid 6 --uid 2 --is_rn_preprocessed
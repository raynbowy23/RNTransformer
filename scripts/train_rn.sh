# 0.4s per timestep
# 8 -> 3.2 s

# variable timesteps -> variable [8, 16, 32, 64]
python3 train_rn.py --rn_num_timesteps_in 8 --rn_num_timesteps_out 12 --epochs 10 --is_rn --dataset eth --optimizer RMSProps --skip 1 \
 --agg_frame 1 --grid 8 --uid 202 --is_rn #--is_rn_preprocessed
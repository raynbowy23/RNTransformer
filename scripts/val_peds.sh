# python3 validation.py --num_timesteps_in 8 --num_timesteps_out 12 --epochs 100 --dataset sdd --bs 32 --lr_sh_rate 150 --optimizer SGD --use_lrschd --model_name social_stgcnn \
#     --skip 10 --agg_frame 10 --grid 6 --uid 39 --is_rn --sdd_loc 'nexus' --is_preprocessed
python3 validation.py --num_timesteps_in 8 --num_timesteps_out 12 --epochs 250 --dataset eth --bs 128 --lr_sh_rate 150 --optimizer SGD --use_lrschd --model_name social_stgcnn \
    --skip 1 --agg_frame 1 --grid 6 --uid 44 --is_rn --is_preprocessed
# python3 validation.py --num_timesteps_in 8 --num_timesteps_out 12 --epochs 100 --dataset eth --bs 128 --lr_sh_rate 150 --optimizer SGD --use_lrschd --model_name trajectory_model \
#     --skip 1 --agg_frame 1 --grid 6 --uid 41 --is_rn --is_preprocessed
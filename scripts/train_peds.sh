### 2.4s input, [4.8, 9.6, 12.0]  output. 0.4s per frame
### in 8 -> 3.2s input
### out 40 --> 16s output

### Data Frequency (The inD Dataset: A Drone Dataset of Naturalistic Road User Trajectories at German Intersections)
### SDD 25 Hz --> 25 frames on 1 seconds --> 1 frame on 0.04 seconds --> 10 frames on 0.4 seconds
### Eth 2.5 Hz --> 2.5 frames on 1 seconds --> 1 frame on 0.4 seconds

## Short term
# python3 run.py --num_timesteps_in 8 --num_timesteps_out 12 --epochs 60 --dataset sdd --sdd_loc nexus \
#     --optimizer SGD --temporal conv --bs 1 --model_name trajectory_model --is_rn # --is_preprocessed # --is_rn # --is_preprocessed
# python3 run.py --num_timesteps_in 8 --num_timesteps_out 12 --epochs 250 --dataset sdd --sdd_loc nexus --optimizer SGD --bs 32 --lr_sh_rate 150 --use_lrschd --model_name trajectory_model --agg_frame 10 --is_preprocessed  
# python3 run.py --num_timesteps_in 8 --num_timesteps_out 12 --epochs 60 --dataset sdd --sdd_loc nexus --optimizer SGD --temporal conv --bs 8 --model_name social_stgcnn --fusion conv --is_preprocessed  

### Trajectory Model Local
# python3 run.py --num_timesteps_in 8 --num_timesteps_out 12 --epochs 250 --dataset sdd --optimizer SGD --agg_frame 10 --skip 10 --model_name trajectory_model --is_normalize --bs 16 --is_rn --lr_sh_rate 150 --use_lrschd --is_preprocessed
# python3 run.py --num_timesteps_in 8 --num_timesteps_out 12 --epochs 60 --dataset eth --optimizer SGD --agg_frame 1 --skip 1 --model_name trajectory_model --bs 128 --lr_sh_rate 150 --use_lrschd --is_preprocessed --is_rn

# Train S-STGCNN on eth
# python3 run.py --num_timesteps_in 8 --num_timesteps_out 12 --epochs 250 --dataset eth --bs 128 --lr_sh_rate 150 --optimizer SGD --use_lrschd --model_name social_stgcnn --skip 1 --agg_frame 1 --grid 6 --uid 5 --is_preprocessed --is_rn
python3 run.py --num_timesteps_in 8 --num_timesteps_out 12 --epochs 250 --dataset eth --bs 32 --lr_sh_rate 150 --optimizer SGD --use_lrschd --model_name social_stgcnn \
    --skip 1 --agg_frame 1 --grid 6 --uid 42 --is_rn --is_preprocessed

# Train S-Implicit on eth
# python3 run.py --num_timesteps_in 8 --num_timesteps_out 12 --epochs 100 --dataset eth --bs 128 --optimizer SGD --lr_sh_rate 150 --use_lrschd --model_name social_implicit --skip 1 --agg_frame 1 --is_preprocessed --is_rn

# python3 run.py --num_timesteps_in 8 --num_timesteps_out 12 --epochs 250 --dataset eth \  # --dataset sdd --sdd_loc nexus 
#     --optimizer SGD --bs 128 --lr_sh_rate 150 --use_lrschd --model_name social_stgcnn # --is_preprocessed  

# python3 run.py --num_timesteps_in 8 --num_timesteps_out 36 --epochs 100 --dataset sdd --sdd_loc nexus \
#     --optimizer SGD --skip 12 --temporal conv # --is_preprocessed
## RN
# python3 run.py --num_timesteps_in 8 --num_timesteps_out 36 --horizon 2 --epochs 30 --dataset sdd --sdd_loc bookstore --optimizer SGD --skip 12 \
#     --temporal conv --fusion conv --rn_num_timesteps_in 8 --rn_num_timesteps_out 12 --is_preprocessed --is_rn # --is_rn_preprocessed 

# python3 run.py --num_timesteps_in 8 --num_timesteps_out 36 --horizon 4 --epochs 30 --dataset sdd --sdd_loc bookstore --optimizer SGD --skip 12 \
#     --temporal conv --fusion conv --rn_num_timesteps_in 8 --rn_num_timesteps_out 12 --is_preprocessed --is_rn # --is_rn_preprocessed 

# python3 run.py --num_timesteps_in 8 --num_timesteps_out 36 --horizon 8 --epochs 30 --dataset sdd --sdd_loc bookstore --optimizer SGD --skip 12 \
#     --temporal conv --fusion conv --rn_num_timesteps_in 8 --rn_num_timesteps_out 12 --is_preprocessed --is_rn # --is_rn_preprocessed 

## RN Fusion at LSTM
# python3 run.py --num_timesteps_in 6 --num_timesteps_out 12 --horizon 4 --temporal lstm --fusion lstm --epochs 20 --dataset sdd  \
#     --optimizer Adam --skip 12 --is_rn --is_preprocessed # --is_rn_preprocessed 


## Short term LSTM
# python3 run.py --num_timesteps_in 6 --num_timesteps_out 12 --epochs 20 --dataset sdd --optimizer Adam --skip 12 --is_preprocessed --temporal lstm
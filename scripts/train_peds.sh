### 2.4s input, [4.8, 9.6, 12.0]  output. 0.4s per frame
### in 8 -> 3.2s input
### out 40 --> 16s output

### Data Frequency (The inD Dataset: A Drone Dataset of Naturalistic Road User Trajectories at German Intersections)
### SDD 25 Hz --> 25 frames on 1 seconds --> 1 frame on 0.04 seconds --> 10 frames on 0.4 seconds
### Eth 2.5 Hz --> 2.5 frames on 1 seconds --> 1 frame on 0.4 seconds

# Train S-STGCNN on eth
# python3 run.py --num_timesteps_in 8 --num_timesteps_out 12 --epochs 250 --dataset eth --bs 128 --lr_sh_rate 150 --optimizer SGD --use_lrschd --model_name social_stgcnn --skip 1 --agg_frame 1 --grid 6 --uid 5 --is_preprocessed --is_rn
echo "train eth" & python3 run.py --num_timesteps_in 8 --num_timesteps_out 12 --epochs 150 --dataset eth --bs 128 --lr_sh_rate 150 --optimizer SGD --use_lrschd --model_name social_stgcnn \
    --skip 1 --agg_frame 1 --grid 6 --uid 70 --is_rn --is_preprocessed

echo "train hotel" & python3 run.py --num_timesteps_in 8 --num_timesteps_out 12 --epochs 150 --dataset hotel --bs 128 --lr_sh_rate 150 --optimizer SGD --use_lrschd --model_name social_stgcnn \
    --skip 1 --agg_frame 1 --grid 6 --uid 70 --is_rn #--is_preprocessed

echo "tarin univ" & python3 run.py --num_timesteps_in 8 --num_timesteps_out 12 --epochs 150 --dataset univ --bs 128 --lr_sh_rate 150 --optimizer SGD --use_lrschd --model_name social_stgcnn \
    --skip 1 --agg_frame 1 --grid 6 --uid 70 --is_rn #--is_preprocessed

echo "train zara1" & python3 run.py --num_timesteps_in 8 --num_timesteps_out 12 --epochs 150 --dataset zara1 --bs 128 --lr_sh_rate 150 --optimizer SGD --use_lrschd --model_name social_stgcnn \
    --skip 1 --agg_frame 1 --grid 6 --uid 70 --is_rn #--is_preprocessed

echo "train zara2" & python3 run.py --num_timesteps_in 8 --num_timesteps_out 12 --epochs 150 --dataset zara2 --bs 128 --lr_sh_rate 150 --optimizer SGD --use_lrschd --model_name social_stgcnn \
    --skip 1 --agg_frame 1 --grid 6 --uid 70 --is_rn #--is_preprocessed
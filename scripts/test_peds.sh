### S-STGCNN
# python3 test.py --num_timesteps_in 8 --num_timesteps_out 12 --dataset eth --model_name social_stgcnn --skip 1 --agg_frame 1 --uid 42 --grid 6 --is_visualize
# python3 test.py --num_timesteps_in 8 --num_timesteps_out 12 --dataset eth --model_name social_stgcnn --skip 1 --agg_frame 1 --uid 41 --grid 6 --pretrained_epoch 200 --is_visualize
# python3 test.py --num_timesteps_in 8 --num_timesteps_out 12 --dataset eth --model_name social_stgcnn --skip 1 --agg_frame 1 --uid 43 --grid 6 --pretrained_epoch 40 --is_rn 
python3 test.py --num_timesteps_in 8 --num_timesteps_out 12 --dataset eth --model_name social_stgcnn --skip 1 --agg_frame 1 --uid 43 --grid 6 --pretrained_epoch 130 --is_rn 
python3 test.py --num_timesteps_in 8 --num_timesteps_out 12 --dataset eth --model_name social_stgcnn --skip 1 --agg_frame 1 --uid 43 --grid 6 --pretrained_epoch 140 --is_rn 
python3 test.py --num_timesteps_in 8 --num_timesteps_out 12 --dataset eth --model_name social_stgcnn --skip 1 --agg_frame 1 --uid 43 --grid 6 --pretrained_epoch 150 --is_rn 
python3 test.py --num_timesteps_in 8 --num_timesteps_out 12 --dataset eth --model_name social_stgcnn --skip 1 --agg_frame 1 --uid 43 --grid 6 --pretrained_epoch 160 --is_rn 
# python3 test.py --num_timesteps_in 8 --num_timesteps_out 12 --dataset eth --model_name social_stgcnn --skip 1 --agg_frame 1 --uid 41 --grid 6 --pretrained_epoch 250 --is_visualize

### S-Implicit
# python3 test.py --num_timesteps_in 8 --num_timesteps_out 12 --epochs 300 --dataset eth --model_name social_implicit --skip 1 --agg_frame 1 --uid 119 --grid 4 --pretrained_epoch 90 --is_rn --is_visualize

# python3 test.py --num_timesteps_in 8 --num_timesteps_out 12 --dataset eth --skip 1 --model_name social_stgcnn --agg_frame 1 --uid 112 --grid 6 --is_rn --is_visualize
# python3 test.py --num_timesteps_in 8 --num_timesteps_out 12 --dataset eth --skip 1 --model_name social_implicit --skip 1 --agg_frame 1 --is_visualize --is_rn --pretrained_epoch 20 #--is_rn #--is_rn #--pretrained_epoch 50
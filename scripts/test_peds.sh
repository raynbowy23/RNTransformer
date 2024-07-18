### S-STGCNN

python3 test.py --num_timesteps_in 8 --num_timesteps_out 12 --dataset eth --model_name social_stgcnn --skip 1 --agg_frame 1 --uid 69 --grid 6 #--is_rn --is_preprocessed #--is_visualize
python3 test.py --num_timesteps_in 8 --num_timesteps_out 12 --dataset hotel --model_name social_stgcnn --skip 1 --agg_frame 1 --uid 69 --grid 6 #--is_preprocessed
python3 test.py --num_timesteps_in 8 --num_timesteps_out 12 --dataset univ --model_name social_stgcnn --skip 1 --agg_frame 1 --uid 69 --grid 6 #--pretrained_epoch 150 --is_preprocessed
python3 test.py --num_timesteps_in 8 --num_timesteps_out 12 --dataset zara1 --model_name social_stgcnn --skip 1 --agg_frame 1 --uid 69 --grid 6 #--pretrained_epoch 150 --is_preprocessed
python3 test.py --num_timesteps_in 8 --num_timesteps_out 12 --dataset zara2 --model_name social_stgcnn --skip 1 --agg_frame 1 --uid 69 --grid 6 #--pretrained_epoch 150 --is_preprocessed

# python3 test.py --num_timesteps_in 8 --num_timesteps_out 12 --dataset eth --model_name social_implicit --skip 1 --agg_frame 1 --uid 46 --grid 6 --pretrained_epoch 70 --is_preprocessed
# python3 test.py --num_timesteps_in 8 --num_timesteps_out 12 --dataset hotel --model_name social_implicit --skip 1 --agg_frame 1 --uid 46 --grid 6 --pretrained_epoch 70 --is_preprocessed
# python3 test.py --num_timesteps_in 8 --num_timesteps_out 12 --dataset univ --model_name social_implicit --skip 1 --agg_frame 1 --uid 46 --grid 6 --pretrained_epoch 70 --is_preprocessed
# python3 test.py --num_timesteps_in 8 --num_timesteps_out 12 --dataset zara1 --model_name social_implicit --skip 1 --agg_frame 1 --uid 46 --grid 6 --pretrained_epoch 70 --is_preprocessed
# python3 test.py --num_timesteps_in 8 --num_timesteps_out 12 --dataset zara2 --model_name social_implicit --skip 1 --agg_frame 1 --uid 46 --grid 6 --pretrained_epoch 70 --is_preprocessed

# init=85
# max=150
# for ((i=init;i<=$max;i+=5))
# do 
#     python3 test.py --num_timesteps_in 8 --num_timesteps_out 12 --dataset eth --model_name social_implicit --skip 1 --agg_frame 1 --uid 69 --grid 6 --pretrained_epoch $i #--is_preprocessed #--is_rn
#     # python3 test.py --num_timesteps_in 8 --num_timesteps_out 12 --dataset sdd --model_name social_stgcnn --skip 10 --agg_frame 10 --uid 65 --grid 6 --pretrained_epoch $i --is_rn --is_preprocessed
# done
python src/run.py --argoverse --future_frame_num 30 \
  --do_train --data_dir train/data/ --output_dir output_train_6+point \
  --hidden_size 128 --train_batch_size 32 --sub_graph_batch_size 4096 --use_map \
  --core_num 16 --use_centerline --distributed_training 4 --reuse_temp_file --temp_file_dir output \
  --other_params \
    semantic_lane direction l1_loss \
    goals_2D enhance_global_graph subdivide lazy_points new laneGCN point_sub_graph \
    stage_one stage_one_dynamic=0.95 laneGCN-4 point_level point_level-4 \
    point_level-4-3 complete_traj complete_traj-3 \

chmod +x $(pwd)/zhiyuan/train_infer.py
chmod +x $(pwd)/zhiyuan/infer.py
chmod +x $(pwd)/zhiyuan/multi_run_infer.py

# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "d" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "dq" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "dqs" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "dqsa" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "ds" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "da" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "dsa" -rn bm25_scu dpr mss-dpr mss contriever

# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "d" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "dq" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "dqs" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "dqsa" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "ds" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "da" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "dsa" -rn bm25_scu dpr mss-dpr mss contriever

# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "d" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "dq" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "dqs" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "dqsa" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "ds" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "da" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "dsa" -rn bm25_scu dpr mss-dpr mss contriever

# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "d" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "dq" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "dqs" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "dqsa" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "ds" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "da" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_normalize --exp_filter_mode "dsa" -rn bm25_scu dpr mss-dpr mss contriever

# python $(pwd)/zhiyuan/dsprk_train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "trivia" --training_sample 320

# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "ds" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "da" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "dsa" -rn bm25_scu dpr mss-dpr mss contriever

# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "ds" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "da" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "dsa" -rn bm25_scu dpr mss-dpr mss contriever

# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 5 --exp_mode 3 --exp_filter_mode "dsa" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_filter_mode "d" -rn bm25_scu dpr mss-dpr mss contriever

# python $(pwd)/zhiyuan/train_infer.py --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "trivia" --training_sample 320 --exp_filter_mode "d"
# python $(pwd)/zhiyuan/train_infer.py --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "trivia" --training_sample 320 --exp_filter_mode "q"
# python $(pwd)/zhiyuan/train_infer.py --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "trivia" --training_sample 320 --exp_filter_mode "s"
# python $(pwd)/zhiyuan/train_infer.py --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "trivia" --training_sample 320 --exp_filter_mode "a"
# python $(pwd)/zhiyuan/train_infer.py --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "trivia" --training_sample 320 --exp_filter_mode "sa"

# python $(pwd)/zhiyuan/train_infer.py --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "nq" --training_sample 320 --exp_filter_mode "d"
# python $(pwd)/zhiyuan/train_infer.py --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "nq" --training_sample 320 --exp_filter_mode "q"
# python $(pwd)/zhiyuan/train_infer.py --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "nq" --training_sample 320 --exp_filter_mode "s"
# python $(pwd)/zhiyuan/train_infer.py --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "nq" --training_sample 320 --exp_filter_mode "a"
# python $(pwd)/zhiyuan/train_infer.py --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "nq" --training_sample 320 --exp_filter_mode "sa"

# mode 3 320 train, best, only filter mode d
# nq 15
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 15 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.1
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 15 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.3
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 15 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.5
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 15 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.7
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 15 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.9
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 15 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 1.1
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 15 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 1.3
# webq 20
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.1
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.3
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.5
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.7
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.9
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 1.1
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 1.3
# # trivia 20
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.1
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.3
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.5
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.7
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.9
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 1.1
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 1.3
# # squad1 10
# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.1
# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.3
# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.5
# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.7
# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.9
# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 1.1
# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 1.3

# #nq k=15, margin=0.7
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 15 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.7 --enable_scheduler
# #webq k=20, margin=0.0
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.0 --enable_scheduler
# #squad1 k=10, margin=0.0
# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.0 --enable_scheduler
# #trivia k=20, margin=1.1
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 1.1 --enable_scheduler

#nq k=15, margin=0.7
python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 15 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.7 --training_sample 80
python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 15 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.7 --training_sample 160
#webq k=20, margin=0.0
python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.0 --training_sample 80
python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.0 --training_sample 160
# #squad1 k=10, margin=0.0
# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.0 --enable_scheduler --training_sample 640
# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.0 --enable_scheduler --training_sample 1280
# #trivia k=20, margin=1.1
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 1.1 --enable_scheduler --training_sample 640
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 1.1 --enable_scheduler --training_sample 1280

#squad1 k=10, margin=0.0
python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.0 --training_sample 80
python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 0.0 --training_sample 160
#trivia k=20, margin=1.1
python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 1.1 --training_sample 80
python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 0 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --exp_margin 1.1 --training_sample 160


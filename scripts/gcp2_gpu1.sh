chmod +x $(pwd)/zhiyuan/train_infer.py
chmod +x $(pwd)/zhiyuan/infer.py
chmod +x $(pwd)/zhiyuan/multi_run_infer.py

# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "d" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "dq" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "dqs" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "dqsa" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "ds" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "da" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "dsa" -rn bm25_scu dpr mss-dpr mss contriever

# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "d" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "dq" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "dqs" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "dqsa" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "ds" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "da" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "dsa" -rn bm25_scu dpr mss-dpr mss contriever

# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "d" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "dq" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "dqs" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "dqsa" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "ds" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "da" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "dsa" -rn bm25_scu dpr mss-dpr mss contriever

# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "d" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "dq" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "dqs" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "dqsa" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "ds" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "da" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 10 --exp_mode 3 --exp_normalize --exp_filter_mode "dsa" -rn bm25_scu dpr mss-dpr mss contriever

# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "ds" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "da" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "dsa" -rn bm25_scu dpr mss-dpr mss contriever

# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "ds" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "da" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "dsa" -rn bm25_scu dpr mss-dpr mss contriever

# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 15 --exp_mode 3 --exp_filter_mode "dqs" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_filter_mode "da" -rn bm25_scu dpr mss-dpr mss contriever

# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "webq" --training_sample 320 --exp_filter_mode "d"
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "webq" --training_sample 320 --exp_filter_mode "q"
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "webq" --training_sample 320 --exp_filter_mode "s"
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "webq" --training_sample 320 --exp_filter_mode "a"
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "webq" --training_sample 320 --exp_filter_mode "sa"

# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "squad1" --training_sample 320 --exp_filter_mode "d"
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "squad1" --training_sample 320 --exp_filter_mode "q"
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "squad1" --training_sample 320 --exp_filter_mode "s"
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "squad1" --training_sample 320 --exp_filter_mode "a"
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "squad1" --training_sample 320 --exp_filter_mode "sa"

# mode 3 320 train, best, only filter mode d
# nq d
# webq d 
# trivia d 
# squad1 d

# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "nq" --training_sample 320 --exp_margin 0.1
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "nq" --training_sample 320 --exp_margin 0.3
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "nq" --training_sample 320 --exp_margin 0.5
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "nq" --training_sample 320 --exp_margin 0.7
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "nq" --training_sample 320 --exp_margin 0.9
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "nq" --training_sample 320 --exp_margin 1.1
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "nq" --training_sample 320 --exp_margin 1.3

# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "webq" --training_sample 320 --exp_margin 0.1
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "webq" --training_sample 320 --exp_margin 0.3
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "webq" --training_sample 320 --exp_margin 0.5
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "webq" --training_sample 320 --exp_margin 0.7
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "webq" --training_sample 320 --exp_margin 0.9
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "webq" --training_sample 320 --exp_margin 1.1
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "webq" --training_sample 320 --exp_margin 1.3

# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "trivia" --training_sample 320 --exp_margin 0.1
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "trivia" --training_sample 320 --exp_margin 0.3
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "trivia" --training_sample 320 --exp_margin 0.5
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "trivia" --training_sample 320 --exp_margin 0.7
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "trivia" --training_sample 320 --exp_margin 0.9
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "trivia" --training_sample 320 --exp_margin 1.1
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "trivia" --training_sample 320 --exp_margin 1.3

# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "squad1" --training_sample 320 --exp_margin 0.1
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "squad1" --training_sample 320 --exp_margin 0.3
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "squad1" --training_sample 320 --exp_margin 0.5
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "squad1" --training_sample 320 --exp_margin 0.7
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "squad1" --training_sample 320 --exp_margin 0.9
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "squad1" --training_sample 320 --exp_margin 1.1
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "squad1" --training_sample 320 --exp_margin 1.3

# # nq margin=0.0
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "nq" --training_sample 320 --exp_margin 0.0 --enable_scheduler
# # webq margin=0.9
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "webq" --training_sample 320 --exp_margin 0.9 --enable_scheduler
# # squad1 margin=0.5
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "squad1" --training_sample 320 --exp_margin 0.5 --enable_scheduler
# # trivia margin=0.3
# python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "trivia" --training_sample 320 --exp_margin 0.3 --enable_scheduler

# nq margin=0.0
python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "nq" --training_sample 80 --exp_margin 0.0
python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "nq" --training_sample 160 --exp_margin 0.0
# webq margin=0.9
python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "webq" --training_sample 80 --exp_margin 0.9
python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "webq" --training_sample 160 --exp_margin 0.9
# squad1 margin=0.5
python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "squad1" --training_sample 80 --exp_margin 0.5
python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "squad1" --training_sample 160 --exp_margin 0.5
# trivia margin=0.3
python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "trivia" --training_sample 80 --exp_margin 0.3
python $(pwd)/zhiyuan/train_infer.py --device_idx 1 --fp16train --docspec --docspec_layer_num 1 --exp_mode 4 --exp_head_num 2 --exp_normalize -rn bm25_scu dpr mss-dpr mss contriever --dataset_name "trivia" --training_sample 160 --exp_margin 0.3
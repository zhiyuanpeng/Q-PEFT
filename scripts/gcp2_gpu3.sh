chmod +x $(pwd)/zhiyuan/train_infer.py
chmod +x $(pwd)/zhiyuan/infer.py
chmod +x $(pwd)/zhiyuan/multi_run_infer.py

# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 3 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "d" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 3 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "dq" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 3 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "dqs" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 3 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "dqsa" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 3 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "ds" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 3 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "da" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name trivia --device_idx 3 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "dsa" -rn bm25_scu dpr mss-dpr mss contriever

# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 3 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "d" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 3 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "dq" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 3 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "dqs" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 3 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "dqsa" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 3 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "ds" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 3 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "da" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name nq --device_idx 3 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "dsa" -rn bm25_scu dpr mss-dpr mss contriever

# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 3 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "d" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 3 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "dq" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 3 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "dqs" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 3 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "dqsa" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 3 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "ds" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 3 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "da" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name webq --device_idx 3 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "dsa" -rn bm25_scu dpr mss-dpr mss contriever

# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 3 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "d" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 3 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "dq" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 3 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "dqs" -rn bm25_scu dpr mss-dpr mss contriever
# python $(pwd)/zhiyuan/train_infer.py --dataset_name squad1 --device_idx 3 --fp16train --docspec --docspec_layer_num 1 --exp_k 20 --exp_mode 3 --exp_normalize --exp_filter_mode "dqsa" -rn bm25_scu dpr mss-dpr mss contriever



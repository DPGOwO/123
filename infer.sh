# sleep 10m
CUDA_VISIBLE_DEVICES=0 python infer.py --world_size 8 --rank 0 &
CUDA_VISIBLE_DEVICES=1 python infer.py --world_size 8 --rank 1 &
CUDA_VISIBLE_DEVICES=2 python infer.py --world_size 8 --rank 2 &
CUDA_VISIBLE_DEVICES=3 python infer.py --world_size 8 --rank 3 &
CUDA_VISIBLE_DEVICES=4 python infer.py --world_size 8 --rank 4 &
CUDA_VISIBLE_DEVICES=5 python infer.py --world_size 8 --rank 5 &
CUDA_VISIBLE_DEVICES=6 python infer.py --world_size 8 --rank 6 &
CUDA_VISIBLE_DEVICES=7 python infer.py --world_size 8 --rank 7 &
wait


# CUDA_VISIBLE_DEVICES=4 python infer.py --world_size 4 --rank 0 &
# CUDA_VISIBLE_DEVICES=5 python infer.py --world_size 4 --rank 1 &
# CUDA_VISIBLE_DEVICES=6 python infer.py --world_size 4 --rank 2 &
# CUDA_VISIBLE_DEVICES=7 python infer.py --world_size 4 --rank 3 &
# wait
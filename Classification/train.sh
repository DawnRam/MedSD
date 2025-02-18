CUDA_VISIBLE_DEVICES='0' python train.py --root_path ../../../Data/ISIC \
                                         --exp baseline_sampled \
                                         --sample_ratio 1 \
                                         --output_dir ../../../Log/Classification \
                                         --batch_size 64 \
                                         --base_lr 1e-4 \
                                         --deterministic 1 \
                                         --seed 1337 \
                                         --gpu 0

CUDA_VISIBLE_DEVICES='3' python train.py --root_path ~/CC/Data/ISIC \
                                         --exp baseline_ratio \
                                         --sample_ratio 0.05 \
                                         --output_dir ~/CC/Log/Classification \
                                         --batch_size 64 \
                                         --base_lr 1e-4 \
                                         --deterministic 1 \
                                         --seed 1337 \
                                         --gpu 3

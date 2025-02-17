import torch
from diffusers import UNet2DModel, DDIMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from torchvision import transforms
from accelerate import Accelerator
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from dataset import CheXpertDataset
import argparse
from tqdm.auto import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/cyang/Code/MedSD/Data/ISIC', help='dataset root dir')
parser.add_argument('--csv_file_train', type=str, default='/home/cyang/Code/MedSD/Classification/RAC-MT/data/skin/training_sampled.csv', help='training set csv file')
parser.add_argument('--csv_file_val', type=str, default='/home/cyang/Code/MedSD/Classification/RAC-MT/data/skin/validation.csv', help='validation set csv file')
parser.add_argument('--csv_file_test', type=str, default='/home/cyang/Code/MedSD/Classification/RAC-MT/data/skin/testing.csv', help='testing set csv file')
parser.add_argument('--image_size', type=int, default=256, help='图像大小')
parser.add_argument('--train_batch_size', type=int, default=4, help='训练批次大小')
parser.add_argument('--eval_batch_size', type=int, default=16, help='评估批次大小')
parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='梯度累积步数')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
parser.add_argument('--lr_warmup_steps', type=int, default=500, help='学习率预热步数')
parser.add_argument('--save_image_epochs', type=int, default=10, help='保存图像的轮数间隔')
parser.add_argument('--save_model_epochs', type=int, default=30, help='保存模型的轮数间隔')
parser.add_argument('--mixed_precision', type=str, default='fp16', help='混合精度训练类型')
parser.add_argument('--output_dir', type=str, default='ddpm-isic-skin', help='输出目录')
parser.add_argument('--classifier_free_prob', type=float, default=0.1, help='分类器自由引导概率')
parser.add_argument('--gpu_id', type=str, default='0', help='GPU设备ID')
parser.add_argument(
    "--scheduler_type",
    type=str,
    default="ddpm",
    choices=["ddpm", "ddim"],
    help="The type of scheduler to use (ddpm or ddim)"
)
parser.add_argument(
    "--inference_steps",
    type=int,
    default=1000,
    help="Number of inference steps for generation (can be lower for ddim)"
)

args = parser.parse_args()

def train(args):
    # 设置GPU设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
    
    # 创建日志文件
    log_file = os.path.join(args.output_dir, "training_log.txt")
    
    # 使用args替代config
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # 加载数据集
    dataset = CheXpertDataset(
        root_dir=args.root_path,
        csv_file=args.csv_file_train,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )
    train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    # 初始化模型并移至GPU
    model = UNet2DModel(
        sample_size=args.image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ).to(device)

    # 初始化 noise scheduler
    if args.scheduler_type == "ddpm":
        from diffusers import DDPMScheduler
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
    else:  # ddim
        from diffusers import DDIMScheduler
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            clip_sample=False,
            set_alpha_to_one=False
        )

    # 初始化优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 准备训练
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    global_step = 0
    # 开始训练
    for epoch in range(args.num_epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        epoch_loss = 0.0
        
        for step, batch in enumerate(progress_bar):
            clean_images = batch[2]
            class_labels = batch[3]
            
            # 采样 noise
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # 采样随机时间步
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), 
                device=clean_images.device
            ).long()

            # 添加 noise
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # 随机丢弃条件
                if torch.rand(1).item() < args.classifier_free_prob:
                    class_labels_batch = torch.ones_like(class_labels) * -1
                else:
                    class_labels_batch = class_labels
                
                # 预测noise
                noise_pred = model(
                    noisy_images, 
                    timesteps, 
                    class_labels=class_labels_batch,
                ).sample
                
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            progress_bar.set_postfix(MSE=loss.detach().item())
            epoch_loss += loss.detach().item()
            global_step += 1

        # 计算并记录每个epoch的平均损失
        avg_loss = epoch_loss / len(train_dataloader)
        log_message = f"Epoch {epoch}: Average Loss = {avg_loss:.6f}\n"
        
        if accelerator.is_main_process:
            with open(log_file, "a") as f:
                f.write(log_message)
            print(log_message)

        # 每隔指定轮数生成测试样本
        if epoch % args.save_image_epochs == 0:
            if accelerator.is_main_process:
                model.eval()
                num_samples_per_class = 50
                
                with torch.no_grad():
                    for class_label in range(2):
                        sample = torch.randn(
                            (num_samples_per_class, 3, args.image_size, args.image_size),
                            device=device
                        )
                        
                        noise_scheduler.set_timesteps(args.inference_steps)
                        
                        for t in tqdm(noise_scheduler.timesteps, desc=f"Generating class {class_label}"):
                            timesteps_tensor = torch.tensor([t] * num_samples_per_class, device=device)
                            class_labels = torch.tensor([class_label] * num_samples_per_class, device=device)
                            
                            model_output = model(
                                sample, 
                                timesteps_tensor,
                                class_labels=class_labels
                            ).sample
                            
                            # 根据采样器类型调用不同的step方法
                            if args.scheduler_type == "ddim":
                                sample = noise_scheduler.step(
                                    model_output,
                                    t,
                                    sample,
                                    eta=0.0
                                ).prev_sample
                            else:  # ddpm
                                sample = noise_scheduler.step(
                                    model_output,
                                    t,
                                    sample
                                ).prev_sample
                        
                        # 将生成的图像转换为PIL图像并保存
                        sample = (sample / 2 + 0.5).clamp(0, 1)
                        sample = sample.cpu().permute(0, 2, 3, 1).numpy()
                        
                        for idx in range(num_samples_per_class):
                            image = Image.fromarray((sample[idx] * 255).astype(np.uint8))
                            image.save(
                                os.path.join(
                                    args.output_dir,
                                    "samples",
                                    f"epoch_{epoch}_class_{class_label}_sample_{idx}.png"
                                )
                            )
                model.train()

        # 保存模型检查点
        if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                pipeline = DDPMPipeline(
                    unet=accelerator.unwrap_model(model),
                    scheduler=noise_scheduler,
                )
                pipeline.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{epoch}"))

if __name__ == "__main__":
    train(args)

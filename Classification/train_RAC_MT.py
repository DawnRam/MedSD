import os
import sys
import shutil
import argparse
import logging
import time
import random 
import math
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from sklearn import preprocessing

from networks.models import DenseNet121
from utils import losses, ramps
from utils.metrics import compute_AUCs
from utils.metric_logger import MetricLogger
from dataloaders import dataset
from dataloaders.dataset import TwoStreamBatchSampler
from utils.util import get_timestamp
from validation import epochVal, epochVal_metrics_test, show_confusion_matrix
from wideresnet import WNet

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/cyang/CC/Code/MedSD/Data/ISIC', help='dataset root dir')
parser.add_argument('--csv_file_train', type=str, default='/data/cyang/CC/Code/MedSD/Classification/RAC-MT/data/skin/training.csv', help='training set csv file')
parser.add_argument('--csv_file_val', type=str, default='/data/cyang/CC/Code/MedSD/Classification/RAC-MT/data/skin/validation.csv', help='validation set csv file')
parser.add_argument('--csv_file_test', type=str, default='/data/cyang/CC/Code/MedSD/Classification/RAC-MT/data/skin/testing.csv', help='testing set csv file')
parser.add_argument('--exp', type=str, default='baseline_sampled', help='model_name')
parser.add_argument('--epochs', type=int, default=180, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size per gpu')
parser.add_argument('--drop_rate', type=float, default=0.2, help='dropout rate')
parser.add_argument('--base_lr', type=float, default=1e-4, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--sample_ratio', type=float, default=0.05, help='ratio of training data to sample (0-1)')

args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

if torch.cuda.is_available():
    device = "cuda"

def create_model():
    # Network definition
    net = DenseNet121(out_size=dataset.N_CLASSES, mode='U-Zeros', drop_rate=args.drop_rate)
    if len(args.gpu.split(',')) > 1:
        net = torch.nn.DataParallel(net)
    return net.cuda()

model = create_model()
optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
# 添加学习率调整策略
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.1, patience=10, verbose=True
)

# 简化数据加载器,只加载有标签数据
train_dataset = dataset.CheXpertDataset(
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

# 在创建train_dataloader之前添加采样
num_total = len(train_dataset)
num_samples = int(num_total * args.sample_ratio)
indices = random.sample(range(num_total), num_samples)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    sampler=torch.utils.data.SubsetRandomSampler(indices),  # 使用SubsetRandomSampler替代shuffle
    num_workers=0,
    pin_memory=True
)

val_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                      csv_file=args.csv_file_val,
                                      transform=transforms.Compose([
                                          transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ]))

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

test_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                      csv_file=args.csv_file_test,
                                      transform=transforms.Compose([
                                          transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ]))

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

loss_fn = losses.cross_entropy_loss()

if __name__ == "__main__":
    ## make logging file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        os.makedirs(snapshot_path + './checkpoint')
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    writer = SummaryWriter(snapshot_path+'/log')

    best_val_auroc = 0  # 记录最佳验证集性能
    
    # 简化训练循环
    for epoch in range(args.epochs):
        model.train()
        meters_loss = MetricLogger(delimiter="  ")
        
        for i, (_, _, image_batch, label_batch) in enumerate(train_dataloader):
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            activations, outputs = model(image_batch)
            loss = loss_fn(outputs, label_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            meters_loss.update(loss=loss)
            
            if i % 80 == 0:
                logging.info(
                    "\nEpoch: {}, iteration: {}/{}, loss: {:.6f}, lr: {}"
                    .format(epoch, i, len(train_dataloader), 
                           meters_loss.loss.avg, optimizer.param_groups[0]['lr'])
                )
        
        # 验证
        AUROCs, Accus, Senss, Specs, F1 = epochVal_metrics_test(model, val_dataloader)
        AUROC_avg = np.array(AUROCs).mean()
        logging.info("\nVAL AUROC: {:.6f}".format(AUROC_avg))
        
        # 更新学习率
        scheduler.step(AUROC_avg)
        
        # 如果当前模型性能最好，更新best_val_auroc
        if AUROC_avg > best_val_auroc:
            best_val_auroc = AUROC_avg
            # 保存最佳模型
            save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_auroc': best_val_auroc
            }, save_mode_path)
            logging.info("保存最佳模型，验证集AUROC: {:.6f}".format(best_val_auroc))
        
        # 测试
        AUROCs, Accus, Senss, Specs, F1 = epochVal_metrics_test(model, test_dataloader)
        AUROC_avg = np.array(AUROCs).mean()
        logging.info("\nTEST AUROC: {:.6f}".format(AUROC_avg))

    # 保存最终模型
    save_mode_path = os.path.join(snapshot_path, 'final_model.pth')
    torch.save({
        'epoch': args.epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, save_mode_path)
    logging.info("保存最终模型到 {}".format(save_mode_path))
    writer.close()

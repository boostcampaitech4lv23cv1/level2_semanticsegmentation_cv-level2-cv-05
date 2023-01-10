import os
import glob
import re
import random
import multiprocessing
import argparse
import warnings
import numpy as np
import torch
import wandb
from datetime import datetime, timedelta
from importlib import import_module
from pathlib import Path
from dataset import CustomDataset, category_names
from loss import create_criterion
from optimizer import create_optimizer
from scheduler import get_scheduler
from utils import add_hist, label_accuracy_score


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def save_model(model, saved_dir, file_name='efficient_unet_best_model.pt'):
    os.makedirs(saved_dir, exist_ok=True)
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)


def increment_path(path, exist_ok=False):
    """ Automatically increment path,
        i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def validation(epoch, model, data_loader, criterion, device, use_wandb=None):
    print(f'Start validation #{epoch}')
    model.eval()

    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0

        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(data_loader):

            images = torch.stack(images)
            masks = torch.stack(masks).long()

            images, masks = images.to(device), masks.to(device)

            # device 할당
            model = model.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1

            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()

            hist = add_hist(hist, masks, outputs, n_class=n_class)

        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        IoU_by_class = [{classes: round(IoU, 4)} for IoU, classes in zip(IoU, category_names)]

        avrg_loss = total_loss / cnt
        print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                mIoU: {round(mIoU, 4)}')
        print(f'IoU by class : {IoU_by_class}')

        if use_wandb is not None:
            wandb.log({
                "val/acc": acc, 
                "val/acc_cls": acc_cls, 
                "val/mIoU": mIoU, 
                "val/fwavacc": fwavacc,
                "val/avg_loss": avrg_loss,
                "val/Background": IoU_by_class[0]["Backgroud"],
                "val/General_Trash": IoU_by_class[1]["General trash"],
                "val/Paper": IoU_by_class[2]["Paper"],
                "val/Paper_pack": IoU_by_class[3]["Paper pack"],
                "val/Metal": IoU_by_class[4]["Metal"],
                "val/Glass": IoU_by_class[5]["Glass"],
                "val/Plastic": IoU_by_class[6]["Plastic"],
                "val/Styrofoam": IoU_by_class[7]["Styrofoam"],
                "val/Plastic_bag": IoU_by_class[8]["Plastic bag"],
                "val/Battery": IoU_by_class[9]["Battery"],
                "val/Clothing": IoU_by_class[10]["Clothing"],
            })

    return avrg_loss, mIoU


def collate_fn(batch):
    return tuple(zip(*batch))


def train(args):

    seed_everything(args.seed)

    # GPU 사용 가능 여부에 따라 device 정보 저장
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = multiprocessing.cpu_count() // 2

    n_class = 11
    best_mIoU = float('-inf')

    name = args.name
    saved_dir = increment_path(os.path.join('./saved', name))

    dataset_path = args.data_dir
    train_json = args.train_json
    val_json = args.train_json

    log_interval = args.log_interval
    val_every = args.val_every
    num_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay

    # Model 정의
    model_module = getattr(import_module("model"), 'get_smp_model')
    model = model_module(args.model, args.encoder, args.encoder_weights)


    # Loss function 정의
    criterion = create_criterion(args.criterion)

    # Optimizer 정의
    optimizer = create_optimizer(args.optimizer, params=model.parameters(),
                                 lr=lr, weight_decay=weight_decay)
    # Scheduler 정의
    scheduler = get_scheduler(args.scheduler, optimizer)

    # Augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)
    train_transform = transform_module()
    val_transform_module = getattr(import_module("dataset"),
                                   'BaseAugmentation')
    val_transform = val_transform_module()

    # Dataset
    train_dataset = CustomDataset(dataset_path, train_json, mode='train',
                                  transform=train_transform)
    val_dataset = CustomDataset(dataset_path, val_json, mode='val',
                                transform=val_transform)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               drop_last=True,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             collate_fn=collate_fn)

    if args.use_amp is not None:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()

        if args.use_wandb is not None:
            wandb.log({
                'epoch': epoch + 1,
                'lr': optimizer.param_groups[0]['lr'],
            })

        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(train_loader):
            images = torch.stack(images)
            masks = torch.stack(masks).long()

            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)

            # device 할당
            model = model.to(device)

            optimizer.zero_grad()

            if args.use_amp is not None:
                with torch.cuda.amp.autocast():
                    # inference
                    outputs = model(images)
                    # loss 계산
                    loss = criterion(outputs, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()

            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)

            # step 주기에 따른 loss 출력
            if (step + 1) % log_interval == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], \
                        Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')
                if args.use_wandb is not None:
                    wandb.log({                        
                        'train/acc': acc,
                        'train/acc_cls': acc_cls,
                        'train/mIoU': mIoU,
                        'train/fwavacc': fwavacc,
                        'train/loss': loss,
                    })

        # validation 주기에 따른 mIoU 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            _, val_mIoU = validation(epoch + 1, model, val_loader, criterion, device, args.use_wandb)
            if val_mIoU > best_mIoU:
                print(f"Best mIoU {round(val_mIoU, 4)} at epoch: {epoch + 1}")
                best_mIoU = val_mIoU
                fname = f'epoch{epoch + 1}_mIoU{round(val_mIoU, 4)}.pt'
                save_model(model, saved_dir, file_name=fname)
                print(f"Save model in {saved_dir}")

                wandb.log({'Best_mIoU': best_mIoU})

            if epoch + 1 == num_epochs:
                save_model(model, saved_dir, file_name='latest.pt')

        if scheduler is not None:
            scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=21, help='random seed (default: 21)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 70)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 8)')
    parser.add_argument('--model', type=str, default='Unet', help='model type (default: Unet)')
    parser.add_argument('--encoder', type=str, default='efficientnet-b0', help='model encoder (default: efficientnet-b0')
    parser.add_argument('--encoder_weights', type=str, default="imagenet")
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--scheduler', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=25, help='how many batches to wait before logging training status')
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--name', default='exp', help='model save at {name}')
    parser.add_argument('--data_dir', type=str, default='../../data')
    parser.add_argument('--train_json', type=str, default='train.json')
    parser.add_argument('--val_json', type=str, default='val.json')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)

    args = parser.parse_args()
    print(args)

    warnings.filterwarnings(action='ignore')
    torch.cuda.empty_cache()

    now = (datetime.now().replace(microsecond=0) + timedelta(hours=9)).strftime("%m-%d %H:%M")

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f'{args.name}_{now}',
        tags=[args.model, args.encoder, args.optimizer, args.criterion],
    )

    train(args)

    wandb.finish()

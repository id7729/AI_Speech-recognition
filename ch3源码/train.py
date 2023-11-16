"""
训练代码
"""
import os
from argparse import ArgumentParser
import torch.optim as optim
import torch
import random
import numpy as np
import torch.nn as nn
from model import Model
from cfg_parse import cfg
from dataset_se import MelLoader
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter
from collections import Counter

logger = SummaryWriter('./log') # 训练 log存放地址

# seed init: Ensure Reproducible Result     随机数种子，使用相同的种子，生成的随机数也会相同
seed = 123
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


# 每次验证的时候，均衡的从evaluate set 中抽取
def balance_sample(dataset_):
    assert isinstance(dataset_, MelLoader)
    items = dataset_.items
    random.shuffle(items)
    sample_indices_dict = {str(k): [] for k in range(6)}
    for idx, (_, lab) in enumerate(items):
        if len(sample_indices_dict[lab]) < 20:
            sample_indices_dict[lab].append(idx)
    sample_indices = []
    [sample_indices.extend(it) for it in list(sample_indices_dict.values())]
    return sample_indices


# 计算输出类别的比例，用于调整 Weighted CE
def ratio_calc(lst):
    num = len(lst)
    lst_dict = Counter(lst)
    ret = []
    for lab in range(6):
        if lab not in lst_dict:
           ret.append(0)
        else:
            ret.append(lst_dict[lab]/num)
    ret = ['%.3f' % it for it in ret]
    return ret


ce_fw = open('./log/ce_weigh.log', 'a+')


# 从evaluate set中均衡抽取一个subset, 计算各类别的比例
def evaluate(model_, valset, crit):
    model_.eval()
    subset_indices = balance_sample(valset)
    subset = Subset(valset, subset_indices)
    val_loader = DataLoader(subset, batch_size=1, shuffle=True)
    sum_loss = 0.
    with torch.no_grad():
        y_actual_list, y_pred_list = [], []
        for batch in val_loader:
            inputs, lab = batch
            inputs, lab = inputs.to(cfg['device']), lab.to(cfg['device'])
            pred = model_(inputs)
            loss = crit(pred, lab)
            sum_loss += loss.item()
            y_actual_list.append(lab.item())
            y_pred_list.append(pred.squeeze().argmax().item())
        ce_weight_log = 'y ratios:'+':'.join(ratio_calc(y_actual_list))+'\n' + \
                        'y_pred ratios:'+':'.join(ratio_calc(y_pred_list))+'\n'
        print(ce_weight_log)
        ce_fw.write(ce_weight_log)
        ce_fw.flush()
    model_.train()
    return sum_loss/len(val_loader)


# 保存模型
def save_checkpoint(model_, epoch_, optm, checkpoint_path):
    save_dict = {
        'epoch': epoch_,
        'model_state_dict': model_.state_dict(),
        'optimizer_state_dict': optm.state_dict(),
    }
    torch.save(save_dict, checkpoint_path)


# 训练
def train():
    parser = ArgumentParser(description='Model Train')
    parser.add_argument(
        '--train_meta_path', # train.csv
        type=str,
        help='train meta csv'
    )
    parser.add_argument(
        '--eval_meta_path', # eval.csv
        type=str,
        help='eval meta csv'
    )
    parser.add_argument(    # 可以从某个checkpoint恢复训练
        '--c',  # checkpoint path
        default=None,
        type=str,
        help='train from scratch if it is none, or resume training from checkpoint'
    )
    args = parser.parse_args()
    model = Model(cfg)
    # 根据第一次训练计算的weighted cross-entropy, 调整的weighted ce
    weights = [0.8867, 1.1350, 0.9683, 0.9632, 0.97508, 1.1316]     # 设置的原则，越好识别weight越小，越难识别weight越大
    t_weights = torch.FloatTensor(weights).to(cfg['device'])
    criterion = nn.CrossEntropyLoss(weight=t_weights)

    opt = optim.Adam(model.parameters(), lr=cfg['lr'])

    trainset = MelLoader(args.train_meta_path, cfg)
    train_loader = DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)

    evalset = MelLoader(args.eval_meta_path, cfg)

    start_epoch = 0
    iteration = 0

    if args.c:
        checkpoint = torch.load(args.c)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        iteration = start_epoch*len(train_loader)
        print('Resume training from %s' % args.c)
    else:
        print('trainig from scratch!')

    model = model.to(cfg['device'])
    model.train() # drop=0, train_flag = True

    # 主循环
    for epoch in range(start_epoch, cfg['epoch']):
        print('='*33, 'Start Epoch %d, Total： %d iters' % (epoch, len(trainset)/cfg['batch_size']), '='*33)
        sum_loss = 0.
        for batch in train_loader:
            inputs, lab = batch
            inputs, lab = inputs.to(cfg['device']), lab.to(cfg['device'])
            opt.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, lab)
            sum_loss + loss.item()
            loss.backward()
            opt.step()

            logger.add_scalar('Loss/Train', loss, iteration)

            if not iteration % cfg['verbose_step']:
                eval_loss = evaluate(model, evalset, criterion)
                logger.add_scalar('Loss/Eval', eval_loss, iteration)
                print('Train Loss: %.4f, Eval Loss: %.4f' % (sum_loss / cfg['verbose_step'], eval_loss))

            if not iteration % cfg['save_step']:
                model_path = 'model_%d_%d.pth' % (epoch, iteration)
                save_checkpoint(model, epoch, opt, os.path.join('model_save', model_path))

            iteration += 1
            logger.flush()
            print('Epoch: [%d/%d], step: %d Train Loss: %.4f' % (epoch, cfg['epoch'], iteration, loss.item()))

    logger.close()


if __name__ == '__main__':
    train()

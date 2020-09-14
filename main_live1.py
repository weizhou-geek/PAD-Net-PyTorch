import os
import time
import argparse
import torch
import numpy as np
import torch.optim as optim
import torch.optim.lr_scheduler as LS
from torch.autograd import Variable
from torchvision import models
import scipy.io as scio
from scipy import stats

import utils
from datasets.data_live1 import get_dataset
from model.model_c1 import ImgComNet

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# Training settings
parser = argparse.ArgumentParser(description='Quality Prediction')
parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--total_epochs', type=int, default=300)
parser.add_argument('--total_iterations', type=int, default=1000000)
parser.add_argument('--batch_size', '-b', type=int, default=16, help="Batch size")
parser.add_argument('--lr', type=float, default=1e-4, metavar=' LR', help='learning rate (default: 0.0001)')

parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=4)
parser.add_argument('--number_gpus', '-ng', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--no_cuda', action='store_true')

parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')

parser.add_argument('--inference', action='store_true')
parser.add_argument('--skip_training', default=False, action='store_true')
parser.add_argument('--skip_validation', action='store_true')

parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N', help="Log every n batches")

main_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(main_dir)
args = parser.parse_args()


if args.inference:
    args.skip_validation = True
    args.skip_training = True
    args.total_epochs = 1
    args.inference_dir = "{}/inference".format(args.save)


kwargs = {'num_workers': args.number_workers}
if not args.skip_training:
    train_set = get_dataset(is_training=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, **kwargs)

test_set = get_dataset(is_training=False)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, **kwargs)

res_net = models.resnet18(pretrained=False)
model = ImgComNet(N=128, M=192, model=res_net).cuda()

g_analysis_params = list(map(id, model.g_analysis.parameters()))
g_synthesis_params = list(map(id, model.g_synthesis.parameters()))
resnet_params = list(map(id, model.resnet.parameters()))
base_params = filter(lambda p: id(p) not in g_analysis_params + g_synthesis_params + resnet_params, model.parameters())
optimizer = optim.Adam([
    {'params': base_params},
    {'params': model.g_analysis.parameters(), 'lr': 1e-5},
    {'params': model.g_synthesis.parameters(), 'lr': 1e-5},
    {'params': model.resnet.parameters(), 'lr': args.lr * 0.5}], lr=args.lr)

scheduler = LS.MultiStepLR(optimizer, milestones=[50, 100, 150, 200], gamma=0.25)
scheduler.last_epoch = args.start_epoch


def train(epoch, iteration):
    model.train()
    scheduler.step()
    end = time.time()
    log = [0 for _ in range(1)]
    for batch_idx, batch in enumerate(train_loader):
        datal, datar, label, _ = batch
        datal = Variable(datal.cuda())
        datar = Variable(datar.cuda())
        label = Variable(label.cuda())
        optimizer.zero_grad()
        _, _, batch_info = model(datal, datar, label, requires_loss=True)
        batch_info.backward()
        optimizer.step()

        log = [log[i] + batch_info.item() * len(datal) for i in range(1)]
        iteration += 1

    log = [log[i] / len(train_loader.dataset) for i in range(1)]
    epoch_time = time.time() - end
    print('Train Epoch: {}, Loss: {:.6f}'.format(epoch, log[0]))
    print('LogTime: {:.4f}s'.format(epoch_time))
    return log



def crop_patch(datal, datar, label):
    col_stride = 192
    row_stride = 104
    patch_size = 256
    imageL= torch.zeros(6, 3, 256, 256)
    imageR = torch.zeros(6, 3, 256, 256)
    label_list = torch.zeros(6, 1)
    for i in range(3):
        for j in range(2):
            idx = i * 2 + j
            imageL[idx, :, :, :] = datal[:, :, j * row_stride:j * row_stride + patch_size, i * col_stride:i * col_stride + patch_size]
            imageR[idx, :, :, :] = datar[:, :, j * row_stride:j * row_stride + patch_size, i * col_stride:i * col_stride + patch_size]
            label_list[idx] = label[0]
    return imageL, imageR, label_list


def eval():
    model.eval()
    log = [0 for _ in range(1)]
    score_list=[]
    label_list=[]
    name_list=[]

    for batch_idx, batch in enumerate(test_loader):
        datal, datar, label, imgname = batch
        datal, datar, label = crop_patch(datal, datar, label)
        datal = Variable(datal.cuda())
        datar = Variable(datar.cuda())
        label = Variable(label.cuda())

        score, label = model(datal, datar, label, requires_loss=False)

        score = score.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        score = np.mean(score)
        label = np.mean(label)
        res = (score - label)*(score - label)
        score_list.append(score)
        label_list.append(label)
        name_list.append(imgname[0])
        ## release memory
        torch.cuda.empty_cache()
        log[0] += res

    log = [log[i] / len(test_loader) for i in range(1)]
    print('Average LOSS: %.2f' % (log[0]))
    score_list = np.reshape(np.asarray(score_list), (-1,))
    label_list = np.reshape(np.asarray(label_list), (-1,))
    name_list = np.reshape(np.asarray(name_list), (-1,))
    scio.savemat('data_live1.mat', {'score': score_list, 'label': label_list, 'name': name_list})
    srocc = stats.spearmanr(label_list, score_list)[0]
    plcc = stats.pearsonr(label_list, score_list)[0]
    rmse = np.sqrt(((label_list - score_list) ** 2).mean())
    print('SROCC: %.4f\n' % (srocc))
    return srocc, plcc, rmse


if not args.skip_training:
    if args.resume:
        utils.load_model(model, args.resume)
        print('Train Load pre-trained model!')
    best_srocc = 0
    best_plcc = 0
    for epoch in range(args.start_epoch, args.total_epochs+1):
        iteration = (epoch-1) * len(train_loader) + 1
        log = train(epoch, iteration)
        log2 = eval()
        plcc = log2[1]
        if plcc > best_plcc:
            best_plcc = plcc
            checkpoint = os.path.join(args.save, 'checkpoint')
            utils.save_model(model, checkpoint, epoch, is_epoch=True)
else:
    print('Test Load pre-trained model!')
    utils.load_model(model, args.resume)
    eval()

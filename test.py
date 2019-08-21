import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import torchvision
from crops_normal1 import CropSegmentation
from utils import AverageMeter, inter_and_union
from glob import glob
#lovasz_losses
#focal loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
os.environ["CUDA_VISIBLE_DEVICES"]='2'
class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, target, class_weight=None, type='softmax'):
        target = target.view(-1, 1).long()
        if type=='sigmoid':
            if class_weight is None:
                class_weight = [1]*2 #[0.5, 0.5]
            prob   = F.sigmoid(logit)
            prob   = prob.view(-1, 1)
            prob   = torch.cat((1-prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)
        elif  type=='softmax':
            B,C,H,W = logit.size()
            if class_weight is None:
                class_weight =[1]*C #[1/C]*C
            logit   = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob    = F.softmax(logit,1)
            select  = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)
        class_weight = torch.FloatTensor(class_weight).cuda().view(-1,1)
        class_weight = torch.gather(class_weight, 0, target)
        prob       = (prob*select).sum(1).view(-1,1)
        prob       = torch.clamp(prob,1e-8,1-1e-8)
        batch_loss = - class_weight *(torch.pow((1-prob), self.gamma))*prob.log()
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss
        return loss


parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', default=False,
                    help='training mode')
parser.add_argument('--gpu', type=int, default=2,
                    help='test time gpu device id')
parser.add_argument('--backbone', type=str, default='resnet101',
                    help='resnet101')
parser.add_argument('--dataset', type=str, default='pascal',
                    help='pascal or cityscapes')
parser.add_argument('--groups', type=int, default=None, 
                    help='num of groups for group normalization')
parser.add_argument('--epochs', type=int, default=30,
                    help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('--base_lr', type=float, default=0.00025,
                    help='base learning rate')
parser.add_argument('--last_mult', type=float, default=1.0,
                    help='learning rate multiplier for last layers')
parser.add_argument('--scratch', action='store_true', default=False,
                    help='train from scratch')
parser.add_argument('--freeze_bn', action='store_true', default=False,
                    help='freeze batch normalization parameters')
parser.add_argument('--weight_std', action='store_true', default=False,
                    help='weight standardization')
parser.add_argument('--beta', action='store_true', default=False,
                    help='resnet101 beta')
parser.add_argument('--crop_size', type=int, default=512,
                    help='image crop size')
parser.add_argument('--resume', type=str, default=None,
                    help='path to checkpoint to resume from')
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers')
args = parser.parse_args()

def main():

    maxIOU=0.0
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    model_fname1 = glob("model/*")
#     model_fname=[]
#     for name in model_fname1:
#         if 'resUnet34' in name: 
#             model_fname.append(name) 
#     model_fname=["data3/deeplabv3_crops_epoch19.pth"]
    print(model_fname1)
    
#     train_dataset = CropSegmentation(train=True, crop_size=args.crop_size)
    test_dataset = CropSegmentation(train=False, crop_size=args.crop_size)
    

    model=torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=4, aux_loss=True)    
#     model=ResNetUNet(4)
#     model=AttU_Net(img_ch=3,output_ch=4)
#     model=R2AttU_Net(img_ch=3,output_ch=4,t=2)
    model = model.cuda()
    model.eval()
    for name in model_fname1[:]:
        checkpoint = torch.load(name)
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
        model.load_state_dict(state_dict)
        inter_meter = AverageMeter()
        union_meter = AverageMeter()
        for i in range(len(test_dataset)):
            inputs, target = test_dataset[i]
            inputs = Variable(inputs.cuda())
            outputs = model(inputs.unsqueeze(0))
            _, pred = torch.max(outputs["out"], 1)
            pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
            mask = target.numpy().astype(np.uint8)
            print('eval: {0}/{1}'.format(i + 1, len(test_dataset)))

            inter, union = inter_and_union(pred, mask, len(test_dataset.CLASSES))
            inter_meter.update(inter)
            union_meter.update(union)
        sumiou=0.0
        iou = inter_meter.sum / (union_meter.sum + 1e-10)
        for i, val in enumerate(iou):
            print('IoU {0}: {1:.2f}'.format(test_dataset.CLASSES[i], val * 100))
            if i>0:
                sumiou=sumiou+val
                        
        print('Mean IoU: {0:.2f}'.format(sumiou/3 * 100))
        if maxIOU<sumiou * 100:
            maxIOU=sumiou * 100
            MAXNAME=name
            print(maxIOU, MAXNAME)
    print(maxIOU, MAXNAME)
              
                    


if __name__ == "__main__":
    main()
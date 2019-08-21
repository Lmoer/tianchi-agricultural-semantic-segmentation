import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from PIL import Image
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
import torchvision
from crops_dataset import CropSegmentation
from utils import AverageMeter, inter_and_union
#lovasz_losses
#focal loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# os.environ["CUDA_VISIBLE_DEVICES"]='0,2'


parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', default=False,
                    help='training mode')
parser.add_argument('--exp', type=str, required=True,
                    help='name of experiment')
parser.add_argument('--gpu', type=int, default=2,
                    help='test time gpu device id')
parser.add_argument('--backbone', type=str, default='resnet101',
                    help='resnet101')
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
    model_fname = '../data/model/deeplabv3_{0}_epoch%d.pth'.format('crops')
    
    train_dataset = CropSegmentation(train=True, crop_size=args.crop_size)
#     test_dataset = CropSegmentation(train=False, crop_size=args.crop_size)
        
    

    model=torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=4, aux_loss=True)
#     model=ResNetUNet(4)
#     model=AttU_Net(img_ch=3,output_ch=4)
    if args.train:
        weight=np.ones(4)
        weight[2]=5
        weight[3]=5
        w=torch.FloatTensor(weight).cuda()
        criterion = nn.CrossEntropyLoss()#ignore_index=255 weight=w
        model = nn.DataParallel(model).cuda()
        
        for param in model.parameters():
            param.requires_grad = True
        
#         epoch = 60
        optimizer = optim.SGD(model.parameters(),lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
        
#         optimizer = optim.SGD(model.parameters(),lr = config.lr,momentum=0.9,weight_decay=1e-4) 
#         scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max = (args.epochs // 9) + 1)

        
        
        dataset_loader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=args.batch_size, shuffle=args.train,
                        pin_memory=True, num_workers=args.workers)
        
        
        max_iter = args.epochs * len(dataset_loader)
        losses = AverageMeter()
        start_epoch = 0
        

        if args.resume:
            if os.path.isfile(args.resume):
                print('=> loading checkpoint {0}'.format(args.resume))
                checkpoint = torch.load(args.resume)
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print('=> loaded checkpoint {0} (epoch {1})'.format(args.resume, checkpoint['epoch']))
            
            else:
                print('=> no checkpoint found at {0}'.format(args.resume))
            

        for epoch in range(start_epoch, args.epochs):
#             scheduler.step(epoch)
            model.train()
            for i, (inputs, target) in enumerate(dataset_loader):

                inputs = Variable(inputs.cuda())
                target = Variable(target.cuda())
                outputs = model(inputs)
#                 loss1 = criterion1(outputs, target)
#                 out1 = F.softmax(outputs, dim=1)
#                 loss2 = L.lovasz_softmax(out1, target)
#                 out2 = F.softmax(outputs['aux'], dim=1)
#                 loss2 = L.lovasz_softmax(out2, target)
#                 loss = loss1 + 0.4*loss2
                loss1 = criterion(outputs['out'], target)
                loss2 = criterion(outputs['aux'], target)
                loss = loss1 + 0.2*loss2
                if np.isnan(loss.item()) or np.isinf(loss.item()):
                    pdb.set_trace()
                
                losses.update(loss.item(), args.batch_size)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                print('epoch: {0}\t'
                      'iter: {1}/{2}\t'
                      'loss: {loss.val:.4f} ({loss.ema:.4f})'.format(
                      epoch + 1, i + 1, len(dataset_loader),loss=losses))

            if epoch % 1 == 0:
                torch.save({
                  'epoch': epoch + 1,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  }, model_fname % (epoch + 1))
                    


if __name__ == "__main__":
    main()
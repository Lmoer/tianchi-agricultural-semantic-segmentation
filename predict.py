
import os
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import torchvision
from glob import glob
import cv2
import os
from PIL import Image
import pdb
from crops_dataset import CropSegmentation
from utils import AverageMeter, inter_and_union

Image.MAX_IMAGE_PIXELS = None
os.environ["CUDA_VISIBLE_DEVICES"]='0'
print("choice best model...")
maxIOU=0.0
assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
model_fname1 = glob("../data/model/*")
print(model_fname1)
    
test_dataset = CropSegmentation(train=False, crop_size=512)
    

model=torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=4, aux_loss=True)    
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


print("load model.....")
model=torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=4, aux_loss=True)


model = model.cuda()
model.eval()
checkpoint = torch.load(MAXNAME)
state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
model.load_state_dict(state_dict)
print("load model success")

print("predict image3.....")
img_path=glob('../data/test/image3' + '/*.png')
for i in range(len(img_path)):
    img = cv2.imread(img_path[i])
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     if img.shape[0]!=512 or img.shape[1]!=512:
#         img1 = cv2.resize(img1,(512,512),interpolation=cv2.INTER_CUBIC)
    data_transforms = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    inputs = data_transforms(img1)

    inputs = Variable(inputs.cuda())
    outputs = model(inputs.unsqueeze(0))
    _, pred = torch.max(outputs["out"], 1)
    pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
#     if img.shape[0]!=512 or img.shape[1]!=512:
#         pred = cv2.resize(pred,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_NEAREST)
    print(i,len(img_path),img.shape, pred.shape)
    im_name=img_path[i].split('/')[-1]
    mask_path='../data/test/mask3'
    mask_name=os.path.join(mask_path,im_name)
    cv2.imwrite(mask_name,pred)
print("predict image3 done")

print("predict image4.....")
img_path=glob('../data/test/image4' + '/*.png')
for i in range(len(img_path)):
    img = cv2.imread(img_path[i])
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     if img.shape[0]!=512 or img.shape[1]!=512:
#         img1 = cv2.resize(img1,(512,512),interpolation=cv2.INTER_CUBIC)
    data_transforms = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    inputs = data_transforms(img1)

    inputs = Variable(inputs.cuda())
    outputs = model(inputs.unsqueeze(0))
    _, pred = torch.max(outputs["out"], 1)
    pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
#     if img.shape[0]!=512 or img.shape[1]!=512:
#         pred = cv2.resize(pred,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_NEAREST)
    print(i,len(img_path),img.shape, pred.shape)
    im_name=img_path[i].split('/')[-1]
    mask_path='../data/test/mask4'
    mask_name=os.path.join(mask_path,im_name)
    cv2.imwrite(mask_name,pred)
print("predict image4 done")    
    
    
base_grid=512
#拼接测试集img3: 19903 * 37241 img4: 28832 * 25936
print("generate mask3.....")
mask_path=glob('../data/test/mask3' + '/*.png')
image_3_predict=np.zeros((19903,37241)).astype(int)
image_4_predict=np.zeros((28832,25936)).astype(int)
for i in range(len(mask_path)):
    row=mask_path[i].split('/')[-1].split('.')[0].split('_')[1]
    col=mask_path[i].split('/')[-1].split('.')[0].split('_')[2]
    sliceimg=cv2.imread(mask_path[i])
#     image_4_predict[int(row)*base_grid:min((int(row)+1)*base_grid,28832),int(col)*base_grid:min((int(col)+1)*base_grid,25936)]=sliceimg[:,:,0]
    image_3_predict[int(row)*base_grid:min((int(row)+1)*base_grid,19903),int(col)*base_grid:min((int(col)+1)*base_grid,37241)]=sliceimg[:,:,0]

cv2.imwrite('../submit/image_3_predict.png',image_3_predict)
print("generate mask3 done")

print("generate mask4.....")
mask_path=glob('../data/test/mask4' + '/*.png')
image_3_predict=np.zeros((19903,37241)).astype(int)
image_4_predict=np.zeros((28832,25936)).astype(int)
for i in range(len(mask_path)):
    row=mask_path[i].split('/')[-1].split('.')[0].split('_')[1]
    col=mask_path[i].split('/')[-1].split('.')[0].split('_')[2]
    sliceimg=cv2.imread(mask_path[i])
    image_4_predict[int(row)*base_grid:min((int(row)+1)*base_grid,28832),int(col)*base_grid:min((int(col)+1)*base_grid,25936)]=sliceimg[:,:,0]
#     image_3_predict[int(row)*base_grid:min((int(row)+1)*base_grid,19903),int(col)*base_grid:min((int(col)+1)*base_grid,37241)]=sliceimg[:,:,0]

cv2.imwrite('../submit/image_4_predict.png',image_4_predict)
print("generate mask4 done.")
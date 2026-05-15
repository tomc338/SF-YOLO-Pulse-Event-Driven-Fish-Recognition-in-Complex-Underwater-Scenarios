import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time,os
import torch
import cv2
import os
from torch.nn import functional as F

thresh = '0.05'
aimsize = 256

datasetpath = r"C:\Users\Administrator\Desktop\data1\data\zj5class0"
datapath = r"C:\Users\Administrator\Desktop\data1\data\zj5class0" # 转换后数据存路径，转换前创建
root_path = r"C:\Users\Administrator\Desktop\data1\data\zj4" # 转换前训练集的根目录

if not os.path.exists(datasetpath):
    os.makedirs(datasetpath)




thresh = float(thresh)

from torchvision import transforms, datasets as ds
import torchvision as tv
from torch.utils.data import DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#检验是否有GPU
print(device)#打印GPU
f = open(os.path.join(datasetpath, 'trainlabel.txt'), 'w')#写入模式打开文件


#定义了一个名为get_file_path的函数，该函数用于递归地获取指定根路径下的所有文件的路径，并将文件路径存储在file_list中，将目录路径存储在dir_list中。
def get_file_path(root_path,file_list,dir_list):
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        dir_file_path = os.path.join(root_path,dir_file)
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            get_file_path(dir_file_path,file_list,dir_list)
        else:
            file_list.append(dir_file_path)



#创建了一个空的file_list和dir_list列表。
#调用之前定义的get_file_path函数，传递了root_path、file_list和dir_list作为参数，这样可以获取root_path路径下的所有文件和目录的路径，并将它们存储在file_list和dir_list中。
#定义了两个函数traceX和traceY，这两个函数可以根据传入的参数t来返回tracex和tracey列表中对应位置的值。
file_list = []
dir_list = []
get_file_path(root_path,file_list,dir_list)
tracex = [1,0,2,1,0,2,1,1,2]
tracey = [0,2,1,0,1,2,0,1,1]

#print(file_list)
#print(dir_list)


def traceX(t):
    x = tracex[t]
    return x


def traceY(t):
    y = tracey[t]
    return y


count = 0
start = time.time()

countda = 0
shapeCount = 0

dircount = 0

maxlength = 0
maxwidth = 0
minlength = 99999
minwidth = 99999

uppulse = list([])
downpulse = list([])
pulse = list([])
timelist = list([])
error_sample = 0
labeldir = {}
event_rate = 0


#打开名为'trainlabel_dir.txt'的文件，并逐行读取文件内容。
#对于每一行，使用split()方法将其分割成两部分，分别赋值给变量n和l。这假设每一行包含两个由空格分隔的部分。
#将n作为键，将l转换为整数后作为值，添加到名为labeldir的字典中。
for line in open('trainlabel_dir.txt'):
    n,l = line.split()
    labeldir[n] = int(l)
    print(labeldir[n])

for root,dirs,files in os.walk(root_path,True):  # 使用os.walk()函数遍历指定路径root_path下的所有子目录和文件。
    print(dirs)
    # 将目录名称映射到整数标签
    for i, d in enumerate(dirs):
        labeldir[d] = i

    # 现在labeldir字典的键是目录名称的字符串，值是整数标签
    # 例如：{'class0': 0, 'class1': 1, 'class2': 2, ...}

    # 现在您可以使用labeldir字典来获取目录对应的整数标签
    for d in dirs:
        dircount = labeldir[d]
        print(f"目录 {d} 对应的整数标签是 {dircount}")
    for dir in dirs:


        subcount = 0
        dircount = labeldir[dir]  # 对于每个子目录，根据其名称获取相应的类别标签（通过labeldir字典）。
        mkpath=datapath+"class"+str(dircount)
        dirname = "class"+str(dircount)
        if not os.path.exists(mkpath):
            os.makedirs(mkpath, exist_ok=True) #创建目录
        fulldir = root_path + r"\\" + dir
        print(fulldir)

        for root1,dirs1,files1 in os.walk(fulldir,True):
            print(files1)
            for file in files1:
                time0 = time.time()
                imagetest = cv2.imread(fulldir + r"\\" +file)
                print(imagetest)
                imagetest = imagetest / 255.0
                imagetest = np.transpose(imagetest, (2, 0, 1))
                imagetorch = torch.from_numpy(imagetest)
                imagetorch = imagetorch.unsqueeze(0)
                a = imagetorch.shape[2]
                b = imagetorch.shape[3]
                rootname,subfilename = os.path.split(file)
                filename,_ = subfilename.split(".")
                if (maxlength < a):
                    maxlength = a
                if (maxwidth < b):
                    maxwidth = b
                if (minlength > a):
                    minlength = a
                if (minwidth > b):
                    minwidth = b
                if (a>b):
                    image = F.interpolate(imagetorch, size = (aimsize,(int)(b*aimsize/a)), mode = 'nearest')
                    b = (int)(b*aimsize/a)
                    a = aimsize
                else:
                    image = F.interpolate(imagetorch, size = ((int)(a*aimsize/b),aimsize), mode = 'nearest')
                    a = (int)(a*aimsize/b)
                    b = aimsize


                image = image[0].to(device)
                imageInfo ,_ = torch.max(image,dim=0)
                imageShape0 = (imageInfo.size()[0], imageInfo.size()[1])

                imageInfo1 = torch.ones(imageShape0[0],imageShape0[1],dtype = torch.uint8,device=device)
                imageShape = (imageInfo.size()[0]+4,imageInfo.size()[1]+4)

                lastImage = torch.zeros(imageShape,device=device)
                newImage = torch.zeros(imageShape,device=device)
                torch1 = torch.ones([imageShape[0],imageShape[1]],dtype = torch.uint8,device=device)
                torch0 = torch.zeros_like(torch1,device=device)
                imageStorep = torch.ones([0,3],dtype = torch.uint8,device=device)
                imageStoren = torch.ones([0,3],dtype = torch.uint8,device=device)
                elen = 0
                for t in range(9):
                    newImage = torch.zeros(imageShape,device=device)
                    x = traceX(t)
                    y = traceY(t)
                    x0 =traceX(t-1)
                    y0 = traceY(t-1)
                    newImage[x:x+imageShape0[0],y:y+imageShape0[1]] = imageInfo
                    if t != 0:
                        diffImage = newImage - lastImage
                        diffImageInfo1 = torch.where(diffImage > thresh, torch1 , torch0)
                        diffImageInfo2 = torch.where(diffImage < -thresh, torch1 , torch0)
                        diffImageInfo1[x:x0,:] = 0
                        diffImageInfo1[x0:x,:] = 0
                        diffImageInfo1[x+imageShape0[0]:x0+imageShape0[0],:] = 0
                        diffImageInfo1[x0+imageShape0[0]:x+imageShape0[0],:] = 0
                        diffImageInfo1[:,y:y0] = 0
                        diffImageInfo1[:,y0:y] = 0
                        diffImageInfo1[:,y+imageShape0[1]:y0+imageShape0[1]] = 0
                        diffImageInfo1[:,y0+imageShape0[1]:y+imageShape0[1]] = 0

                        diffImageInfo2[x:x0,:] = 0
                        diffImageInfo2[x0:x,:] = 0
                        diffImageInfo2[x+imageShape0[0]:x0+imageShape0[0],:] = 0
                        diffImageInfo2[x0+imageShape0[0]:x+imageShape0[0],:] = 0
                        diffImageInfo2[:,y:y0] = 0
                        diffImageInfo2[:,y0:y] = 0
                        diffImageInfo2[:,y+imageShape0[1]:y0+imageShape0[1]] = 0
                        diffImageInfo2[:,y0+imageShape0[1]:y+imageShape0[1]] = 0

                        imageStore1 = torch.nonzero(diffImageInfo1,as_tuple=False)
                        x = imageStore1[:,0]
                        y = imageStore1[:,1]
                        good_xy = (x < 250) & (x > 6) & (y < 250) & (y > 6)
                        imageStore1 = imageStore1[good_xy,:]
                        timeStore1 = torch.zeros([imageStore1.size()[0],1],device=device).fill_(t)
                        imageStore1 = torch.cat((imageStore1,timeStore1),1)
                        imageStorep = torch.cat((imageStorep,imageStore1),0)

                        imageStore2 = torch.nonzero(diffImageInfo2,as_tuple=False)
                        x = imageStore2[:,0]
                        y = imageStore2[:,1]
                        good_xy = (x < 250) & (x > 6) & (y < 250) & (y > 6)
                        imageStore2 = imageStore2[good_xy,:]
                        timeStore2 = torch.zeros([imageStore2.size()[0],1],device=device).fill_(t)
                        imageStore2 = torch.cat((imageStore2,timeStore2),1)
                        imageStoren = torch.cat((imageStoren,imageStore2),0)

                        elen+= (imageStore1.size()[0]+imageStore2.size()[0])

                    lastImage = newImage
                subcount += 1
                if(elen<1000):#0.2%
                    error_sample += 1
                    continue
                else:
                    event_rate += elen
                    f.write('class'+str(dircount)+'/class'+str(dircount)+'_'+str(subcount)+'.npz'+' '+str(dircount)+' '+str(a)+' '+str(b))
                    f.write('\n')
                    imageStorep = imageStorep.cpu().numpy().astype(np.uint8)
                    imageStoren = imageStoren.cpu().numpy().astype(np.uint8)
                    count += 1
                    documentName = mkpath + r"/"+dirname + "_" + str(subcount)
                    np.savez_compressed(documentName,pos = imageStorep, neg = imageStoren)

                if (count % 1000 == 0):
                    print(count)
                    print(time.time()-start)
                    print(error_sample)
                    timelist.append(time.time()-start)
                    print('event_rate = ', event_rate/1000/(224*224*8*2)*100,'%')
                    event_rate = 0


print(count)
print("maxlength = ",maxlength)
print("minlength = ",minlength)
print("maxwidth = ",maxwidth)
print("minwidth = ",minwidth)
end = time.time()
print("time=",end-start)

print(timelist)
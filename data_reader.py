import torch
import torch.utils.data as data
from torchvision import transforms


def reader(txt):
    fh = open(txt)  
    c=0  
    imgs=[]  
    class_names=[]  
    for line in  fh.readlines():  
        if c==0:  
            class_names=[n.strip() for n in line.rstrip().split('   ')]  
        else:  
            cls = line.split()   
            fn = cls.pop(0)
            imgs.append((fn, tuple([float(v) for v in cls])))  
        c=c+1
    return class_names,imgs
  
class myImageFloder(data.Dataset):  
    def __init__(self, root, label, transform = None, target_transform=None, loader=default_loader):  
        self.classes1,self.imgs1 = reader(label)
        self.classes2,self.imgs2 = reader(root)
  
    def __getitem__(self, index):  
        fn1, label1 = self.imgs1[index]
        fn2, label2 = self.imgs2[index]
        return torch.Tensor(label1),torch.Tensor(label2)  
  
    def __len__(self):  
        return len(self.imgs1)  

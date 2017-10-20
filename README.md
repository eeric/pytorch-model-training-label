# pytorch-model-training-label
Please read with 'Raw' mode.

1.Loading dataset with label

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
imgs:label,e.g.:[1,0,0,1], class_names:attribute，such as sex.

face features data by reader as well.

2.simple model design
cmodel=nn.Linear(100, 2)    #nn.Sequential(nn.Linear(100, 2)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.classify=cmodel
    def forward(self, x):
        x=self.classify(x)
        return x,
3.training model
#load training dataset
train_data_loader = torch.utils.data.DataLoader(  \
         ImageFloder(root = "./fea.txt", label = "./label.txt"), batch_size= 2, shuffle= False, num_workers= 4)
root: feature data, label: label data, ImageFloder clsss following：
class ImageFloder(data.Dataset):  
    def __init__(self, root, label）：
	self.classes1,self.imgs1 = reader(label)
        self.classes2,self.imgs2 = reader(root)
    def __getitem__(self, index):  
        fn1, label1 = self.imgs1[index]
        fn2, label2 = self.imgs2[index]
	return torch.Tensor(label1),torch.Tensor(label2)
    def __len__(self):  
        return len(self.imgs1)
        
See details of this project

4.Usage
python train

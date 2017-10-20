import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
import data_reader as reader

#fully connection layer
cmodel=nn.Linear(3, 2) #nn.Sequential(nn.Linear(3, 2))
#criterion
criterion = torch.nn.CrossEntropyLoss()

#Loading dataset
train_data_loader = torch.utils.data.DataLoader(  \
         reader.myImageFloder(root = "./fea.txt", label = "./label.txt"), \
         batch_size= 2, shuffle= False, num_workers= 4)
test_data_loader = torch.utils.data.DataLoader(  \
         reader.myImageFloder(root = "./fea.txt", label = "./label.txt"),\
         batch_size= 2, shuffle= False, num_workers= 4)

#simple model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.classify=cmodel
    def forward(self, x):
        x=self.classify(x)
        return x

model = Net()
model = model.cuda()

print model

def train_batch(optimizer, batch, label):
        loss = 0    
        model.train()
        model.zero_grad()
        input = Variable(batch)
        output = model(input)
        lab = Variable(label.long()) #target type：LongTensor
        loss = criterion(output, lab)
        loss.backward()
        optimizer.step()
        return loss

def train_epoch(optimizer,epoch):
		loss_all = 0
    c = 0
		for i,data in enumerate( train_data_loader):
        batch = data[1]
        label = data[0].view(-1) #label: 1D tensor
        c += 1
		    loss_all += train_batch(optimizer, batch.cuda(), label.cuda())
        print('Train Epoch: {} \tLoss:[{:.6f}]'.format(epoch, (loss_all/c).data[0]))
#test			
def test():
		model.eval()
		correct = 0
		total = 0
		for i, data in enumerate(test_data_loader):
        batch = data[1]
        label = data[0].view(-1)
		    batch = batch.cuda()
		    output = model(Variable(batch))
		    pred = output.data.max(1)[1]
	 	    correct += pred.cpu().eq(label.long()).sum()
	 	    total += label.size(0) 	
	 	print "Accuracy :", float(correct) / total

def train(optimizer = None, epoches = 10):	
		for i in range(epoches):
			print "Epoch: ", i
			train_epoch(optimizer,i)
			test()
		print "Finished training."

#training
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
train(optimizer, epoches = 10)



import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transform
import torch.nn.functional as F
from sklearn.model_selection import KFold
from operator import itemgetter 

import matplotlib.pyplot as plt
import numpy as np
import math

from utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  

print(torch.cuda.is_available())

batch_size = 8

img_transform = transform.Compose([transform.ToTensor(), transform.Normalize((0.5,),(0.5,))]) 
#
#train_set = torchvision.datasets.MNIST(root = '../../data', train= True, transform= img_transform, download= True)
#test_set = torchvision.datasets.MNIST(root = '../../data', train= False, transform= img_transform, download= True)
## https://pytorch.org/vision/stable/datasets.html#mnist
# https://pytorch.org/docs/stable/data.html

#print(train_set)

#train_set= get_imgs('./dataset/train/')
#imgs = get_imgs('./dataset/train_cleaned/')

#img, a = train_set[0]
#print(a)
#print(img.shape)
#print(aaaa)
train_set , test_set,labels_test = build('./dataset/train/', './dataset/train_cleaned/' ,'./dataset/test/','cnn')

#train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
#test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder,self).__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size=3,padding='same'),
        nn.ReLU(),
        nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3,padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2,return_indices = True)
    )

    self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3,padding='same'),
        nn.ReLU(),
        nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3,padding='same'),
        nn.ReLU(),
        #nn.MaxPool2d(kernel_size=2, stride=2,return_indices = True)
    )

    self.conv3 = nn.Sequential(
        nn.Conv2d(in_channels = 64, out_channels = 256, kernel_size=3,padding='same'),
        nn.ReLU(),
        nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=3,padding='same'),
        nn.ReLU(),
        nn.Conv2d(in_channels = 256, out_channels = 1024, kernel_size=3,padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2,return_indices = True)
    )

    self.conv4 = nn.Sequential(
        nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size=3,padding='same'),
        nn.ReLU(),
        nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size=3,padding='same'),
        nn.ReLU(),
        nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size=3,padding='same'),
        nn.ReLU(),
        #nn.MaxPool2d(kernel_size=2, stride=2,return_indices = True)
    )

    self.conv5 = nn.Sequential(
        nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size=3,padding='same'),
        nn.ReLU(),
        nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size=3,padding='same'),
        nn.ReLU(),
        nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size=3,padding='same'),
    )
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2,return_indices = True)
    self.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(7*7*1024,20),
        nn.Softmax(),
    )
    self.dp = nn.Dropout(0.4)

  def forward(self, image):
    indexes = []
    #print(image.shape)
    out,index = self.conv1(image)
    #print(out.shape)
    indexes.append(index)
    id = out
    out = F.relu(self.conv2(out)+id)
    out,index = self.pool(out)
    out = self.dp(out)
    #print(out.shape)    
    indexes.append(index)
    out,index = self.conv3(out)
    #print(out.shape)
    indexes.append(index)
    #print(out.shape)
    id = out
    out = F.relu(self.conv4(out)+id)
    out,index = self.pool(out)
    indexes.append(index)
    #print(out.shape)
    id = out
    out = F.relu(self.conv5(out)+id)
    out,index = self.pool(out)
    #print(out.shape)
    indexes.append(index)
    z = self.fc(out)
    return z,indexes 


class Decoder(nn.Module):
  def __init__(self):
    super(Decoder,self).__init__()  
    self.fc = nn.Sequential(
        nn.Linear(20,7*7*1024),
        nn.ReLU()
    )
    
    self.conv5 = nn.Sequential(
        nn.ConvTranspose2d(in_channels = 1024, out_channels = 1024, kernel_size=3,padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels = 1024, out_channels = 1024, kernel_size=3,padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels = 1024, out_channels = 1024, kernel_size=3,padding=1),
        nn.ReLU(),
    )
    self.conv4 = nn.Sequential(
        nn.ConvTranspose2d(in_channels = 1024, out_channels = 1024, kernel_size=3,padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels = 1024, out_channels = 1024, kernel_size=3,padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels = 1024, out_channels = 1024, kernel_size=3,padding=1),
        nn.ReLU(),
    )
    self.conv3 = nn.Sequential(
        nn.ConvTranspose2d(in_channels = 1024, out_channels = 256, kernel_size=3,padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels = 256, out_channels = 256, kernel_size=3,padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels = 256, out_channels = 64, kernel_size=3,padding=1),
        nn.ReLU(),
    )
    self.conv2 = nn.Sequential(
        nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size=3,padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size=3,padding=1),
        nn.ReLU(),
    )
    self.conv1 = nn.Sequential(
        nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size=3,padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels = 64, out_channels = 1, kernel_size=3,padding=1),
        nn.Tanh(),
    )
    self.dp = nn.Dropout(0.4)
    self.pool= nn.MaxUnpool2d(kernel_size=2, stride=2)

  def forward(self, z,indexes):
    #print(z)
    out = self.fc(z)
    #print(out.shape)
    out = out.view(out.size(0), 1024,7, 7 )
    # print(out.shape)
    
    out = self.pool(out,indexes[4])
    id = out
    out = F.relu(self.conv5(out) +id)
    #out = F.relu(self.conv5(out))
    out = self.dp(out)
    
    out = self.pool(out,indexes[3])
    id = out
    out = F.relu(self.conv4(out) +id)
    
    out = self.pool(out,indexes[2])
    out = self.conv3(out)

    out = self.pool(out,indexes[1])
    id = out
    out = F.relu(self.conv2(out) +id)
    out = self.dp(out)
    
    out = self.pool(out,indexes[0])
    out = self.conv1(out)
    # print(out.shape)
    return out
 
class Autoncoder(nn.Module):
  def __init__(self):
    super(Autoncoder,self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()

  def forward(self, image):
    z,indexes = self.encoder(image)
    out =  self.decoder(z,indexes)
    return out



def train(model, train_loader, Epochs, loss_fn):
  train_loss_avg = []
  for epoch in range(Epochs):
    train_loss_avg.append(0)
    num_batches = 0
  
    for noisy, clean in train_loader:
       # img = img + torch.randn(img.size()) * 0.01 + 0.1
        #noisy = noisy.view(noisy.size(0),-1)
        #clean = clean.view(clean.size(0),-1)
        noisy = noisy.to(device)
        #print(noisy.shape)
        #print(noisy)
        img_recon = model(noisy)
        loss = loss_fn(img_recon, clean)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss_avg[-1] += loss.item()
        num_batches += 1
        
    train_loss_avg[-1] /= num_batches
    print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, Epochs, train_loss_avg[-1]))
  return train_loss_avg

#si no se quiere utilizar el K-fold cross-validation, comente desde aqui


model=0
loss_temp = 10000000
iter = 0
final_train=0
kf = KFold(n_splits=12,shuffle = True,random_state =2)
for train_split, test_split in kf.split(train_set):
  learning_rate = 0.0001
  autoencoder = Autoncoder()
  autoencoder.to(device)
  loss = nn.MSELoss()
  
  train_split = itemgetter(*train_split)(train_set)
  test_split = itemgetter(*test_split)(train_set)
  
  optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)
  train_loader = torch.utils.data.DataLoader(dataset=train_split, batch_size=batch_size, shuffle=True)
  loss_values = train(autoencoder, train_loader,40, loss)
  with torch.no_grad():
    test_split = [i[0]for i in test_split]
    test_split = torch.stack(test_split).to(device)
    new_images = autoencoder(test_split)
    loss_fn = loss(new_images,test_split)
    if(loss_temp>loss_fn):
      #model = autoencoder
      loss_temp=loss_fn
      final_train = train_split
  iter+=1 
  print("iter ",iter," error:",loss_values)
print(loss_temp)

train_loader = torch.utils.data.DataLoader(dataset=final_train, batch_size=batch_size, shuffle=True)
#Hasta aqui




def Show(out, title = ''):
  print(title)
  out = out.permute(1,0,2,3)
  grilla = torchvision.utils.make_grid(out,10,5)
  plt.imshow(transform.ToPILImage()(grilla), 'jet')
  plt.show()

def Show_Weight(out):
  grilla = torchvision.utils.make_grid(out)
  plt.imshow(transform.ToPILImage()(grilla), 'jet')
  plt.show()

# y descomentar aqui 
#train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True) 
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
learning_rate = 0.0001
autoencoder = Autoncoder()
autoencoder.to(device)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)
loss_values = train(autoencoder, train_loader,40, loss)




fig = plt.figure()
plt.plot(loss_values)
plt.xlabel('Epochs')
plt.ylabel('Reconstruction error')
plt.show()




with torch.no_grad():
  i = 0
  for j,batch in enumerate(test_loader):
    for image in batch:
      print(i)
      image = image.unsqueeze(0)
      new_image = autoencoder(image).cpu()
      label = labels_test[i].split('\\')[1]
      plt.axis('off')
      plt.imshow(new_image[0][0], cmap='gray' , aspect='auto')
      plt.savefig('./results_cnn2/test' + label, bbox_inches='tight')
      i += 1

  print("imagen leida")
  #fig, ax = plt.subplots(figsize=(10, 10))
  #Show_Weight(image1[1:10])
  plt.axis('off')


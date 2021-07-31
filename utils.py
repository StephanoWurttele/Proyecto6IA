import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np 
import math 
from PIL import Image
import glob
from sklearn.model_selection import train_test_split

# Dimensions: 540 x algo
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  

def get_imgs(directory):
    images = glob.glob(directory+'*')
    return images 


def process_image(filename):
    img = Image.open(filename)
    img = img.resize((256,128),Image.ANTIALIAS)
    w,h = img.size
    img = torch.tensor(img.getdata()).float()
    img = img.reshape((w,h)).unsqueeze(0)   #[1][w][h]
    std,mean = torch.std_mean(img)
    transform_norm = transforms.Compose([
        transforms.Normalize(mean, std),
    ])
    img2 = transform_norm(img)
    ##img2 = torch.tensor(img2).float()
    return (img2)

def process_image_CNN(filename):
    img = Image.open(filename)
    #img = img.resize((224,128),Image.ANTIALIAS)
    img = img.resize((224,224),Image.ANTIALIAS)
    w,h = img.size
    img = torch.tensor(img.getdata()).float().to(device)
    img = img.reshape((w,h)).unsqueeze(0)   #[1][w][h]
    std,mean = torch.std_mean(img)
    transform_norm = transforms.Compose([
        transforms.Normalize(mean, std),
    ])
    img2 = transform_norm(img)
    #print(img2.shape )
    ##img2 = torch.tensor(img2).float()
    return (img2)



def build_set(filenames,type='mlp'):
    st = []
    for file in filenames:
        if(type=='mlp'):
            st.append(process_image(file))
        else:
            st.append(process_image_CNN(file))
    #st = torch.tensor(st)
    #print(st)
    return st

def build_train(train_set, clean_set):
    response = []
    for i in range(len(train_set)):
        response.append([train_set[i],clean_set[i]])
    return response

def build_test(test_set):
    response = []
    for i in range(len(test_set)):
        response.append([test_set[i],[]])
    return response

def build(noisy, clean, test,type='mlp'):
    imgs_blurred = build_set(get_imgs(noisy),type)
    imgs = build_set(get_imgs(clean),type)
    imgs_test = build_set(get_imgs(test),type)
    return [build_train(imgs_blurred, imgs), build_test(imgs_test)]



#dataset = build('./dataset/train/', './dataset/train_cleaned/')
#print(len(dataset))
#print(len(dataset[0]))
#print(dataset[0][0])
#print(dataset[0][1])
#imgs_blurred= get_imgs('./dataset/train/')
#imgs = get_imgs('./dataset/train_cleaned/')
#
#xd = process_image(imgs[0])
#xd2 = process_image(imgs_blurred[0])
#print(len(imgs_blurred))
#print(len(imgs))
#print(len(xd))
#print(len(xd2))
#print(xd2)
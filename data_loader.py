from os import listdir
from torch.nn import functional as F
import cv2
import torch
import numpy as np
import os
import random
import scipy.io as scio
import h5py
from sklearn.model_selection import train_test_split


def addZeroPadding(X, margin=2):
    """
    add zero padding to the image
    """
    newX = np.zeros((
      X.shape[0] + 2 * margin,
      X.shape[1] + 2 * margin,
      X.shape[2]
            ))
    newX[margin:X.shape[0]+margin, margin:X.shape[1]+margin, :] = X
    return newX

def createImgCube(X ,gt ,pos:list ,windowSize=25):
    """
    create Cube from pos list
    return imagecube gt nextPos
    """
    margin = (windowSize-1)//2
    zeroPaddingX = addZeroPadding(X, margin=margin)
    dataPatches = np.zeros((pos.__len__(), windowSize, windowSize, X.shape[2]))
    if( pos[-1][1]+1 != X.shape[1] ):
        nextPos = (pos[-1][0] ,pos[-1][1]+1)
    elif( pos[-1][0]+1 != X.shape[0] ):
        nextPos = (pos[-1][0]+1 ,0)
    else:
        nextPos = (0,0)
    return np.array([zeroPaddingX[i:i+windowSize, j:j+windowSize, :] for i,j in pos ]),\
    np.array([gt[i,j] for i,j in pos]) ,\
    nextPos

def createPos(shape:tuple, pos:tuple, num:int):
    """
    creatre pos list after the given pos
    """
    if (pos[0]+1)*(pos[1]+1)+num >shape[0]*shape[1]:
        num = shape[0]*shape[1]-( (pos[0])*shape[1] + pos[1] )
    return [(pos[0]+(pos[1]+i)//shape[1] , (pos[1]+i)%shape[1] ) for i in range(num) ]

def createPosWithoutZero(hsi, gt):
    """
    creatre pos list without zero labels
    """
    mask = gt > 0
    return [(i,j) for i , row  in enumerate(mask) for j , row_element in enumerate(row) if row_element]

def splitTrainTestSet(X, gt, testRatio, randomState=111):
    """
    random split data set
    """
    X_train, X_test, gt_train, gt_test = train_test_split(X, gt, test_size=testRatio, random_state=randomState, stratify=gt)
    return X_train, X_test, gt_train, gt_test

def createImgPatch(lidar, pos:list, windowSize=25):
    """
    return lidar Img patches
    """
    margin = (windowSize-1)//2
    zeroPaddingLidar = np.zeros((
      lidar.shape[0] + 2 * margin,
      lidar.shape[1] + 2 * margin
            ))
    zeroPaddingLidar[margin:lidar.shape[0]+margin, margin:lidar.shape[1]+margin] = lidar
    return np.array([zeroPaddingLidar[i:i+windowSize, j:j+windowSize] for i,j in pos ])

def minmax_normalize(array):
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)

def data_aug(train_hsiCube, train_patches, train_labels):
    Xh = []
    Xl = []
    y = []
    for i in range(train_hsiCube.shape[0]):
        Xh.append(train_hsiCube[i])
        Xl.append(train_patches[i])

        noise = np.random.normal(0.0, 0.02, size=train_hsiCube[0].shape)
        noise2 = np.random.normal(0.0, 0.02, size=train_patches[0].shape)
        Xh.append(np.flip(train_hsiCube[i] + noise, axis=1))
        Xl.append(np.flip(train_patches[i] + noise2, axis=1))

        k = np.random.randint(4)
        Xh.append(np.rot90(train_hsiCube[i], k=k))
        Xl.append(np.rot90(train_patches[i], k=k))

        y.append(train_labels[i])
        y.append(train_labels[i])
        y.append(train_labels[i])

    train_labels = np.asarray(y, dtype=np.int8)
    train_hsiCube = np.asarray(Xh, dtype=np.float32)
    train_patches = np.asarray(Xl, dtype=np.float32)
    train_hsiCube = torch.from_numpy(train_hsiCube.transpose(0, 3, 1, 2)).float()
    train_patches = torch.from_numpy(train_patches.transpose(0, 3, 1, 2)).float()
    return train_hsiCube, train_patches, train_labels

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, hsi, sar, labels):
        self.len = labels.shape[0]
        self.hsi = torch.FloatTensor(hsi)
        self.sar = torch.FloatTensor(sar)
        self.labels = torch.LongTensor(labels - 1)
    def __getitem__(self, index):
        return self.hsi[index], self.sar[index], self.labels[index]
    def __len__(self):
        return self.len


def build_datasets(root, dataset, patch_size, batch_size,test_ratio):

    data_hsi = scio.loadmat(root + dataset + '/data_hsi.mat')['data']
    data_sar = scio.loadmat(root + dataset + '/data_sar.mat')['data']
    data_traingt = scio.loadmat(root + dataset + '/mask_train.mat')['mask_train']

    data_hsi = minmax_normalize(data_hsi)
    data_sar = minmax_normalize(data_sar)
    print(data_hsi.shape)

    # training / testing set for 2D-CNN
    train_hsiCube, train_labels ,_ = createImgCube(data_hsi, data_traingt, createPosWithoutZero(data_hsi, data_traingt), windowSize=patch_size)
    train_patches, _ ,_ = createImgCube(data_sar, data_traingt, createPosWithoutZero(data_sar, data_traingt), windowSize=patch_size)

    train_hsiCube, train_patches, train_labels = data_aug(train_hsiCube, train_patches, train_labels)
    X_train, X_test, gt_train, gt_test = splitTrainTestSet(train_hsiCube, train_labels, test_ratio, randomState=128)
    X_train_2, X_test_2, _, _ = splitTrainTestSet(train_patches, train_labels, test_ratio, randomState=128)

    print (X_train.shape)
    print (X_test.shape)
    print("Creating dataloader")
    trainset = TensorDataset(X_train, X_train_2, gt_train)
    testset = TensorDataset(X_test, X_test_2, gt_test)
    train_loader = torch.utils.data.DataLoader(dataset= trainset, batch_size= batch_size, shuffle= True, num_workers = 0)
    test_loader = torch.utils.data.DataLoader(dataset= testset, batch_size= batch_size, shuffle= False, num_workers = 0)

    return train_loader, test_loader
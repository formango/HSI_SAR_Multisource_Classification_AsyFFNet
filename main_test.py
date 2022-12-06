import time
import os
import torch.optim
from models.ClassifierNet import Net, Bottleneck
import args_parser
from time import *
import scipy.io as scio
import numpy as np

args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print (args)

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

def minmax_normalize(array):
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)

def main():
    end = time()
    if args.dataset == 'Berlin':
      args.hsi_bands = 244
      args.sar_bands = 4
      args.num_class = 8
    elif args.dataset == 'Augsburg':
      args.hsi_bands = 180
      args.sar_bands = 4
      args.num_class = 7
    elif args.dataset == 'HHK':
      args.hsi_bands = 166
      args.sar_bands = 3
      args.num_class = 5

    data_hsi = scio.loadmat(args.root + args.dataset + '/data_hsi.mat')['data']
    data_sar = scio.loadmat(args.root + args.dataset + '/data_sar.mat')['data']
    data_gt = scio.loadmat(args.root + args.dataset + '/mask_test.mat')['mask_test']
        
    height, width, c = data_hsi.shape
    data_hsi = minmax_normalize (data_hsi)
    data_sar = minmax_normalize(data_sar)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net(args.hsi_bands, args.sar_bands, args.hidden_size, Bottleneck, args.num_parallel, args.num_reslayer,
                args.num_class, args.bn_threshold).eval().to(device)
    model.load_state_dict(torch.load('./checkpoints/model.pth'))

    margin = (args.patch_size-1) // 2
    data_hsi = addZeroPadding(data_hsi, margin)
    data_sar = addZeroPadding(data_sar, margin)
    data_gt = np.pad(data_gt, ((margin, margin), (margin, margin)), 'constant', constant_values=(0, 0))


    idx, idy = np.where(data_gt != 0)
    labelss = np.array([0])

    batch = 200
    num = 10
    total_batch = int(len(idx)/batch +1)
    print ('Total batch number is :', total_batch)

    for j in range(int((len(idx) - (len(idx) % batch)) / batch + 1)):
        if int(100*j // total_batch) == num:
            print('... ... ',  int(num ), '% batch handling ... ...')
            num = num + 10
        if batch * (j + 1) > len(idx):
            num_cat = len(idx) - batch * j
        else:
            num_cat = batch

        tmphsi = np.array([data_hsi[idx[j*batch + i] - margin:idx[j*batch + i] +
                                                       margin + 1, idy[j * batch + i] - margin:idy[j*batch + i] + margin + 1, :] for i in range(num_cat)])
        tmpsar = np.array([data_sar[idx[j*batch + i] - margin:idx[j*batch + i] +
                                                        margin + 1,idy[j*batch + i] - margin:idy[j*batch + i] + margin + 1,:] for i in range(num_cat)])
        tmphsi = torch.FloatTensor(tmphsi.transpose(0, 3, 1, 2)).to(device)
        tmpsar = torch.FloatTensor(tmpsar.transpose(0, 3, 1, 2)).to(device)

        prediction, _ = model(tmphsi, tmpsar)
        labelss = np.hstack([labelss, np.argmax(prediction.detach().cpu().numpy(), axis=1)])
    print('... ... ', int(100), '% batch handling ... ...')
    labelss = np.delete(labelss, [0])
    new_map = np.zeros((height, width))
    for i in range(len(idx)):
        new_map[idx[i] - margin, idy[i] - margin] = labelss[i] + 1
    scio.savemat(args.result_path + '/result.mat', {'output': new_map})
    print('Finish!!!')
    end2 = time()
    minutes = int((end2 - end) / 60)
    seconds = int((end2 - end) - minutes * 60)
    print("运行时间：", minutes, "分", seconds, "秒")

if __name__ == '__main__':
    main()

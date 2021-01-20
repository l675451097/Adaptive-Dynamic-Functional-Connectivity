
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from Dataset_73 import dataset, datasettest
import random
from model import *
from train import Trainer
import torch.utils.data as Data
import scipy.io as io
from torch.autograd import Variable
from sklearn import svm

train_num = 72
test_num = 1
index = 10
epochs = 600 #400


def test_svm(X, y, x_test):
    SVM = svm.SVC(kernel = 'linear') 
    SVM.fit(X, y)
    predicted =  SVM.predict(x_test)
    y_predict = SVM.predict(X)
    return predicted, sum(y_predict == y)

def test(n):
    loader = io.loadmat('./feature/feature{}.mat'.format(n))
    train_X = loader['trainData']
    train_Y = loader['trainLabel'].squeeze()    
    test_X = loader['testData']
    predict_Y, _ = test_svm(train_X, train_Y, test_X)
    return predict_Y

def best_epoch(epoch_list):
    epoch_list = np.array(epoch_list)
    epoch_list = epoch_list[epoch_list[:, 1]>1.1, :]
    epoch_list = epoch_list[epoch_list[:, 1]<3, :]
    return epoch_list[np.argmin(epoch_list[:, 1]), 0]
    
    


def main(n):
    print(n)
    parser = argparse.ArgumentParser(description = 'VAE MNIST Example')
    parser.add_argument('--batch_size', type = int, default = 72, metavar = 'N',
                        help = 'input batch size for training (default: 128)')
    parser.add_argument('--learning_rate', type = float, default = 0.005, metavar = 'N',
                        help = 'learning rate for training(default: 0.001)')
    parser.add_argument('--test_every', type = int, default = 5, metavar = 'N',
                        help = 'test every 10 epochs while training')
    parser.add_argument('--learning_rate_decay', type = float, default = 0.99, metavar='N',
                        help = 'learning rate decay 1 for every epoch)')
    parser.add_argument('--weight_decay', type = float, default = 5e-4, metavar = 'N',
                        help = 'weight decay for training(default: 5e-4)')
    parser.add_argument('--epochs', type = int, default = epochs, metavar = 'N',
                        help = 'number of epochs to train (default: 41)')
    parser.add_argument('--checkpoint_dir', type = str, default = './model_state/', metavar = 'N',
                        help = 'a dir for saving model.state.dict()')
    parser.add_argument('--no-cuda', action = 'store_true', default = False,
                        help = 'enables CUDA training')
    parser.add_argument('--to_train', action = 'store_true', default = True,
                        help = 'whether train or not')
    parser.add_argument('--seed', type = int, default = 21, metavar = 'S',
                        help = 'random seed (default: 1)')
    parser.add_argument('--log-interval', type = int, default = 10, metavar = 'N',
                        help = 'how many batches to wait before logging training status')
    args = parser.parse_args()
    #args.cuda = not args.no_cuda and torch.cuda.is_available() 
    args.cuda = False
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    setup_seed(5) 
    data1 = dataset(n)
    data2 = dataset(n)
    test_data = datasettest(n) #测试集
    model_1 = model1()
    model_2 = model2()
    class_D = classD()
    if args.cuda:
        model_1, model_2, class_D = model_1.cuda(), model_2.cuda(), class_D.cuda()  
    trainloader = Data.DataLoader(dataset = data1,
                                  batch_size = args.batch_size,
                                  shuffle = True)
    trainloaderT = Data.DataLoader(dataset = data2,
                                  batch_size = train_num,
                                  shuffle = False)
    
    testloader = Data.DataLoader(dataset = test_data,
                                  batch_size = test_num,
                                  shuffle = False)

    trainer = Trainer(trainloader, model_1, model_2, class_D, args, testloader, trainloaderT) 
    model_1, model_2, class_D, train_XX, test_XX, train_YY, epoch_list = trainer.train()
    print(best_epoch(epoch_list)) 
    torch.save(model_1.state_dict(), args.checkpoint_dir + 'checkpoint_1_{}.pth'.format(n))
    torch.save(model_2.state_dict(), args.checkpoint_dir + 'checkpoint_2_{}.pth'.format(n))
    torch.save(class_D.state_dict(), args.checkpoint_dir + 'checkpoint_3_{}.pth'.format(n))

    
    #io.savemat('./Data/featureFUSEss.mat', {'trainData': np.array(trainZ), 'trainLabel': np.array(trainLabel)})
    io.savemat('./feature/feature{}.mat'.format(n), {'trainData': train_XX, 'trainLabel': train_YY, 'testData': test_XX})

if __name__ == "__main__":
    main(index)
    xx = test(index)    
    print(xx)

 


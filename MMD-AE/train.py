# -*- coding: utf-8 -*-
from torch.autograd import Variable
from torch import optim
import torch
#from tensorboardX import SummaryWriter
import shutil
#from tqdm import tqdm
import numpy as np
from torchvision.utils import save_image
import torch.nn as nn
from sklearn import svm
import scipy.io as io
epoch_list = []

def test_svm(X, x_test, y):
    SVM = svm.SVC(kernel = 'linear') 
    SVM.fit(X, y)
    predicted =  SVM.predict(x_test)
    y_predict = SVM.predict(X)
    return predicted, sum(y_predict == y)

def compute_kernel(x, y):
    bandlist = [0.25, 0.5, 1, 2, 4]
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = [torch.exp(-(tiled_x - tiled_y).pow(2).sum(2)/float(bandwidth)) for bandwidth in bandlist]
    return kernel_input  #[.., .., .., .., ..] every .. x_size*y_size

def compute_mmd(x, y):

    x_kernel = sum(compute_kernel(x, x))      
    y_kernel = sum(compute_kernel(y, y))
    xy_kernel = sum(compute_kernel(x, y))
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

def forward_parallel(net, input, ngpu = 2):
    if ngpu > 1:
        return nn.parallel.data_parallel(net, input, range(ngpu))
    else:
        return net(input)


class Trainer:
    def __init__(self, trainloader, model1, model2, classD, args, testloader, trainloaderT):
        self.trainloader = trainloader
        self.model1 = model1
        self.model2 = model2
        self.classD = classD
        self.args = args
        self.testloader = testloader
        self.trainloaderT = trainloaderT
    def train(self):
        trainloader = self.trainloader
        model1 = self.model1
        model2 = self.model2
        testloader = self.testloader
        args = self.args
        classD = self.classD
        trainloaderT = self.trainloaderT
        optimizer1 = optim.Adam(model1.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))              
        optimizer2 = optim.Adam(model2.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
        optimizerC = optim.Adam(classD.parameters(), lr=args.learning_rate*0.1, betas=(0.0, 0.9))

        scheduler1 = optim.lr_scheduler.ExponentialLR(optimizer1, gamma=0.99) 
        scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer2, gamma=0.99)
        schedulerC = optim.lr_scheduler.ExponentialLR(optimizerC, gamma=0.99)
        
    
        ae_criterion = nn.MSELoss(size_average = False)    #均方差
        crossEntropy = nn.CrossEntropyLoss(size_average = False)   #交叉熵
        xxxx = torch.zeros(600, 2)

        for epoch in range(args.epochs):
            for i, (data, targets) in enumerate(trainloader):
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                optimizerC.zero_grad()
                data, targets = Variable(data), Variable(targets)
                if args.cuda:
                    data = data.cuda()
                    targets = targets.cuda()
                    
                encoding1, fake1 = model1(data[:, :292])   
                ae_loss1 = ae_criterion(fake1, data[:, :292])      #292个特征重构的和原始的差异
                
                encoding2, fake2 = model2(data[:, 292:357])                      
                ae_loss2 = ae_criterion(fake2, data[:, 292:357])   #65 个特征重构的和原始的差异

                
                z = torch.cat((encoding1, encoding2), 1)   #中间变量8和16拼接          
                f = torch.cat((z, data[:, 357:]), 1)       #拼接后的中间变量和45和65再拼接
                a = classD(f)        #感知机输出a
                acc = a.cpu().data.numpy().argmax(1) == targets.cpu().data.numpy()      #感知机训练集上的准确率
                class_loss = crossEntropy(a, targets.squeeze())       #感知机的交叉熵损失
                trueSample = Variable(torch.randn(z.size()[0], z.size()[1]))       #标准正态分布采样
                if args.cuda:
                    trueSample = trueSample.cuda()
                mmd_loss = compute_mmd(z, trueSample)    #计算拼接后的中间变量(8+16)分布和正态分布的差异
                l = ae_loss1 + ae_loss2 + 0.5*class_loss + 1.5*73*mmd_loss     #将三部分loss加权相加
                l.backward()    #计算各模型参数的梯度
                optimizer1.step()    #参数更新
                optimizer2.step()
                optimizerC.step()
                
                 
                if epoch % 100 == 0:  #每100个epoch学习率衰减一次
                    scheduler1.step()
                    scheduler2.step()
                    schedulerC.step()
                

            
                model1.eval()   #此时模型是测试模式，参数不更新，也就是说每个训练中都可以在model.eval()之后看看在测试集上的结果，但是测试及结果不会影响参数更新
                model2.eval()
                classD.eval()
                
                for i, (data, targets) in enumerate(trainloaderT):
                    data, targets = Variable(data), Variable(targets)
                    if args.cuda:
                        data = data.cuda()
                        targets = targets.cuda()
                    encoding1, fake1 = model1(data[:, :292])                
                    encoding2, fake2 = model2(data[:, 292:357])   
                

                    z = torch.cat((encoding1, encoding2), 1)
                    f = torch.cat((z, data[:, 357:]), 1)        #生成给svm的训练集
                
                for i, (data_test, targets_test) in enumerate(testloader):
                    data_test, targets_test = Variable(data_test), Variable(targets_test)
                    if args.cuda:
                        data_test = data_test.cuda()
                        targets_test = targets_test.cuda()
                    encoding1, fake1 = model1(data_test[:, :292])                 
                    encoding2, fake2 = model2(data_test[:, 292:357])
                

                    z_test = torch.cat((encoding1, encoding2), 1)
                    f_test = torch.cat((z_test, data_test[:, 357:]), 1)   #生成给svm的测试集
                    a_test = classD(f_test)     #测试集的感知机结果
                    class_losst = crossEntropy(a_test, targets_test)
                    predict, acc_svm = test_svm(f.cpu().data.numpy(), f_test.cpu().data.numpy(), targets.cpu().data.numpy())  #测试在svm上的结果

                    
                model1.train()  
                model2.train()
                classD.train()
            if predict == targets_test.cpu().data.numpy():
            #if epoch >=300 and epoch <= 500 and predict == targets_test.cpu().data.numpy():
                print(epoch, class_losst.cpu().data.numpy(), a_test.cpu().data.numpy().argmax(), predict,  sum(acc), acc_svm, class_loss.cpu().data.numpy())
                ##
                xxxx[epoch, 0]=class_loss.cpu()
                xxxx[epoch, 1]=l
                ##
            if epoch > 340 and epoch < 400:
                epoch_list.append([epoch, class_loss, acc_svm])
     
        ##
        result11 = xxxx.detach().numpy()
        io.savemat('save.mat',{'result11':result11})
        ##
        print('final_epoch', class_losst.cpu().data.numpy(), a_test.cpu().data.numpy().argmax(), predict,  sum(acc), acc_svm, class_loss.cpu().data.numpy())       
        return model1, model2, classD, f.cpu().data.numpy(), f_test.cpu().data.numpy(), targets.cpu().data.numpy(), epoch_list
            


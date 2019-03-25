from models import G
from models import D
import torch as t
import numpy as np
import time
import tqdm
import argparse
from data import Eventbased,transform
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils import adjust_learning_rate
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--imgroot', required=False,default='./data/urban/images/', help='path to img dataset')
parser.add_argument('--eventpicroot', required=False,default='./data/urban//recovery/', help='path to event-pic dataset')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--inputChannelSize', type=int, default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int, default=1, help='size of the output channels')
parser.add_argument('--ngf', type=int, default=64,help='size of first channel of generator')
parser.add_argument('--ndc', type=int, default=4,help='size input channel of discriminator')
parser.add_argument('--ndf', type=int, default=64,help='size first channel of discriminator')
parser.add_argument('--load_G_path', default='./checkpoints/netG_epoch_399.pth',help='the path of Generator')
parser.add_argument('--load_D_path', default='./checkpoints/netD_epoch_0.pth',help='the path of Discriminator')
parser.add_argument('--use_gpu', type=int, default=1,help='size first channel of discriminator')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--wd', type=float, default=0.0000, help='weight decay in D')
parser.add_argument('--niter', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--d_every', type=int, default=1, help='d_every times to train d')
parser.add_argument('--g_every', type=int, default=1, help='g_every times to train g')
parser.add_argument('--lambdaIMG', type=float, default=0.1, help='lambdaIMG')
parser.add_argument('--annealStart', type=int, default=0, help='annealing learning rate start to')
parser.add_argument('--annealEvery', type=int, default=400, help='epoch to reaching at learning rate of 0')
opt = parser.parse_args()

def train():
    net_G = G(opt.inputChannelSize, opt.outputChannelSize, opt.ngf)
    net_D = D(opt.ndc, opt.ndf)
    if opt.load_G_path:
        net_G.load(opt.load_G_path)
    if opt.load_D_path:
        net_D.load(opt.load_D_path)
    if opt.use_gpu:
        net_G.cuda()
        net_D.cuda()

    train_dataset=Eventbased(opt.imgroot,opt.eventpicroot,test=False,transforms=transform)
    trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    criterionBCE = nn.BCELoss()
    criterionCAE = nn.L1Loss()
    lrD=opt.lrD
    lrG=opt.lrG
    optimizerD = optim.Adam(net_D.parameters(), lr=lrD, betas=(opt.beta1, 0.999), weight_decay=opt.wd)
    optimizerG = optim.Adam(net_G.parameters(), lr=lrG, betas=(opt.beta1, 0.999), weight_decay=0.0)

    for iteration in tqdm.tqdm(range(opt.niter)):
        if iteration > opt.annealStart:
            adjust_learning_rate(optimizerD, opt.lrD, opt.annealEvery)
            adjust_learning_rate(optimizerG, opt.lrG, opt.annealEvery)
        for ii,(graypic,eventpic,true_label) in enumerate(trainloader):
            fake_label=t.ones(1)
            G_loss=0
            D_loss=0
            if opt.use_gpu:
                eventpic=eventpic.cuda()
                graypic=graypic.cuda()
                true_label=true_label.cuda()
                fake_label=fake_label.cuda()
            eventpic = Variable(eventpic)

            if (ii+1) % opt.d_every==0:
                optimizerD.zero_grad()
                true_catDinput=t.cat([graypic,eventpic],1)
                true_output=net_D(true_catDinput)
                error_d_real=criterionBCE(true_output,true_label)
                error_d_real.backward()

                fake_gray=net_G(eventpic).detach()
                fake_catDinput=t.cat([fake_gray,eventpic],1)
                fake_output=net_D(fake_catDinput)
                error_d_fake=criterionBCE(fake_output,fake_label)
                D_loss=error_d_fake.item()+error_d_real.item()
                error_d_fake.backward()

                optimizerD.step()

            if (ii+1)%opt.g_every==0:
                optimizerG.zero_grad()
                fake_gray=net_G(eventpic)
                fake_catDinput=t.cat([fake_gray,eventpic],1)
                fake_output=net_D(fake_catDinput)
                error_g=criterionBCE(fake_output,true_label)
                L_img=criterionCAE(graypic,fake_gray)
                L_img=opt.lambdaIMG*L_img
                L_loss=L_img+error_g
                G_loss=L_loss.item()
                L_loss.backward()
                optimizerG.step()
            print('Iteration=%d,Loop=%d,G_Loss= %f,D_loss=%f'%(iteration,ii,G_loss,D_loss))
        net_G.save('./checkpoints/netG_epoch_%d.pth' % iteration)
        net_D.save('./checkpoints/netD_epoch_%d.pth' % iteration)

def test():

    generate_path='./data/urban/generate/'
    net_G = G(opt.inputChannelSize, opt.outputChannelSize, opt.ngf)
    if opt.load_G_path:
        net_G.load(opt.load_G_path)
        print('Net_G has been loaded!')
    else:
        print('Without models!')
        return
    if opt.use_gpu:
        net_G.cuda()

    test_dataset = Eventbased(opt.imgroot, opt.eventpicroot, test=True, transforms=transform)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    print('Start test!\n')
    for testpic,name in tqdm.tqdm(testloader):
        if opt.use_gpu:
            testpic=testpic.cuda()
        outputpic=net_G(testpic)
        outputpic=outputpic[0,:,:,:]
        outputpic=outputpic.detach().cpu().numpy()
        outputpic=np.transpose(outputpic,(1,2,0))
        #print(outputpic*255)
        cv2.imwrite(generate_path+name[0],outputpic*255,[int(cv2.IMWRITE_PNG_COMPRESSION),0])










if __name__=='__main__':
    test()

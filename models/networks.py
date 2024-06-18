import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
import torchvision.models as models
from torchvision.models.googlenet import GoogLeNet_Weights

###############################################################################
# Functions
###############################################################################


def weight_init_googlenet(key, module, weights=None):

    if key == "LSTM":
        for name, param in module.named_parameters():
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.xavier_normal_(param)
    elif weights is None:
        init.constant_(module.bias.data, 0.0)
        if key == "XYZ":
            init.normal_(module.weight.data, 0.0, 0.5)
        elif key == "LSTM":
            init.xavier_normal_(module.weight.data)
        else:
            init.normal_(module.weight.data, 0.0, 0.01)
    else:
        # print(key, weights[(key+"_1").encode()].shape, module.bias.size())
        module.bias.data[...] = torch.from_numpy(weights[(key+"_1").encode()])
        module.weight.data[...] = torch.from_numpy(weights[(key+"_0").encode()])
    return module

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    # 기준: 학습률을 선형으로 감소시킴.
    # 사용 상황: 학습률을 처음에는 높게 시작해 점진적으로 감소시키고 싶을 때.

    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    # 기준: 특정 간격(step_size)마다 학습률을 감소시킴.
    # 사용 상황: 학습이 진행됨에 따라 주기적으로 학습률을 감소시키고 싶을 때.
    elif opt.lr_policy == 'plateau':
    # 기준: 검증 손실이 개선되지 않을 때 학습률을 감소시킴.
    # 사용 상황: 손실이 더 이상 감소하지 않을 때 학습률을 줄여 모델의 성능을 개선하고 싶을 때.
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_network(input_nc, lstm_hidden_size, model, init_from=None, isTest=False, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())

    if model == 'posenet':
        netG = PoseNet(input_nc, weights=None, isTest=isTest, gpu_ids=gpu_ids)
        googlenet = models.googlenet(pretrained=True)
        netG.load_state_dict(googlenet.state_dict(), strict=False)
    elif model == 'poselstm':
        netG = PoseLSTM(input_nc, lstm_hidden_size, weights=init_from, isTest=isTest, gpu_ids=gpu_ids)
    elif model == 'resnet50':
        netG = ResNet50Pose(input_nc, isTest=isTest, gpu_ids=gpu_ids)
    elif model == 'resnet101':
        netG = ResNet101Pose(input_nc, isTest=isTest, gpu_ids=gpu_ids)
    elif model == 'vgg16':
        netG = VGG16Pose(input_nc, isTest=isTest, gpu_ids=gpu_ids)
    elif model == 'vgg19':
        netG = VGG19Pose(input_nc, isTest=isTest, gpu_ids=gpu_ids)
    elif model == 'efficientnetB0':
        netG = EfficientNetB0Pose(input_nc, isTest=isTest, gpu_ids=gpu_ids)
    elif model == 'mobilenetV2':
        netG = MobileNetV2Pose(input_nc, isTest=isTest, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Model name [%s] is not recognized' % model)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    return netG

##############################################################################
# Classes
##############################################################################
# """기존 PoseNet"""
# # defines the regression heads for googlenet
# class RegressionHead(nn.Module):
#     def __init__(self, lossID, weights=None, lstm_hidden_size=None,output_dim=6):
#         super(RegressionHead, self).__init__()
#         self.has_lstm = lstm_hidden_size != None
#         dropout_rate = 0.5 if lossID == "loss3" else 0.7
#         nc_loss = {"loss1": 512, "loss2": 528}
#         nc_cls = [1024, 2048] if lstm_hidden_size is None else [lstm_hidden_size*4, lstm_hidden_size*4]

#         self.dropout = nn.Dropout(p=dropout_rate)
#         if lossID != "loss3":
#             self.projection = nn.Sequential(*[nn.AvgPool2d(kernel_size=5, stride=3),
#                                               weight_init_googlenet(lossID+"/conv", nn.Conv2d(nc_loss[lossID], 128, kernel_size=1), weights),
#                                               nn.ReLU(inplace=True)])
#             self.cls_fc_pose = nn.Sequential(*[weight_init_googlenet(lossID+"/fc", nn.Linear(2048, 1024), weights),
#                                                nn.ReLU(inplace=True)])
#             self.cls_fc_xy = weight_init_googlenet("XYZ", nn.Linear(nc_cls[0], 3))
#             self.cls_fc_wpqr = weight_init_googlenet("WPQR", nn.Linear(nc_cls[0], 4))
#             if lstm_hidden_size is not None:
#                 self.lstm_pose_lr = weight_init_googlenet("LSTM", nn.LSTM(input_size=32, hidden_size=lstm_hidden_size, bidirectional=True, batch_first=True))
#                 self.lstm_pose_ud = weight_init_googlenet("LSTM", nn.LSTM(input_size=32, hidden_size=lstm_hidden_size, bidirectional=True, batch_first=True))
#         else:
#             self.projection = nn.AvgPool2d(kernel_size=7, stride=1)
#             self.cls_fc_pose = nn.Sequential(*[weight_init_googlenet("pose", nn.Linear(1024, 2048)),
#                                                nn.ReLU(inplace=True)])
#             self.cls_fc_xy = weight_init_googlenet("XYZ", nn.Linear(nc_cls[1], 3))
#             self.cls_fc_wpqr = weight_init_googlenet("WPQR", nn.Linear(nc_cls[1], 4))

#             if lstm_hidden_size is not None:
#                 self.lstm_pose_lr = weight_init_googlenet("LSTM", nn.LSTM(input_size=64, hidden_size=lstm_hidden_size, bidirectional=True, batch_first=True))
#                 self.lstm_pose_ud = weight_init_googlenet("LSTM", nn.LSTM(input_size=32, hidden_size=lstm_hidden_size, bidirectional=True, batch_first=True))

#     def forward(self, input):
#         output = self.projection(input)
#         output = self.cls_fc_pose(output.view(output.size(0), -1))
#         if self.has_lstm:
#             output = output.view(output.size(0),32, -1)
#             _, (hidden_state_lr, _) = self.lstm_pose_lr(output.permute(0,1,2))
#             _, (hidden_state_ud, _) = self.lstm_pose_ud(output.permute(0,2,1))
#             output = torch.cat((hidden_state_lr[0,:,:],
#                                 hidden_state_lr[1,:,:],
#                                 hidden_state_ud[0,:,:],
#                                 hidden_state_ud[1,:,:]), 1)
#         output = self.dropout(output)
#         output_xy = self.cls_fc_xy(output)
#         output_wpqr = self.cls_fc_wpqr(output)
#         output_wpqr = F.normalize(output_wpqr, p=2, dim=1)
#         return [output_xy, output_wpqr]
class RegressionHead(nn.Module):
    def __init__(self, lossID, weights=None, lstm_hidden_size=None, output_dim=6): # default to 6 for compatibility
        super(RegressionHead, self).__init__()
        self.has_lstm = lstm_hidden_size is not None
        dropout_rate = 0.5 if lossID == "loss3" else 0.7
        nc_loss = {"loss1": 512, "loss2": 528}
        nc_cls = [1024, 2048] if lstm_hidden_size is None else [lstm_hidden_size*4, lstm_hidden_size*4]

        self.dropout = nn.Dropout(p=dropout_rate)
        if lossID != "loss3":
            self.projection = nn.Sequential(*[nn.AvgPool2d(kernel_size=5, stride=3),
                                              weight_init_googlenet(lossID+"/conv", nn.Conv2d(nc_loss[lossID], 128, kernel_size=1), weights),
                                              nn.ReLU(inplace=True)])
            self.cls_fc_pose = nn.Sequential(*[weight_init_googlenet(lossID+"/fc", nn.Linear(2048, 1024), weights),
                                               nn.ReLU(inplace=True)])
            self.cls_fc_xy = weight_init_googlenet("XYZ", nn.Linear(nc_cls[0], 3))
            self.cls_fc_wpqr = weight_init_googlenet("WPQR", nn.Linear(nc_cls[0], output_dim-3)) # use output_dim-3
            if lstm_hidden_size is not None:
                self.lstm_pose_lr = weight_init_googlenet("LSTM", nn.LSTM(input_size=32, hidden_size=lstm_hidden_size, bidirectional=True, batch_first=True))
                self.lstm_pose_ud = weight_init_googlenet("LSTM", nn.LSTM(input_size=32, hidden_size=lstm_hidden_size, bidirectional=True, batch_first=True))
        else:
            self.projection = nn.AvgPool2d(kernel_size=7, stride=1)
            self.cls_fc_pose = nn.Sequential(*[weight_init_googlenet("pose", nn.Linear(1024, 2048)),
                                               nn.ReLU(inplace=True)])
            self.cls_fc_xy = weight_init_googlenet("XYZ", nn.Linear(nc_cls[1], 3))
            self.cls_fc_wpqr = weight_init_googlenet("WPQR", nn.Linear(nc_cls[1], output_dim-3)) # use output_dim-3

            if lstm_hidden_size is not None:
                self.lstm_pose_lr = weight_init_googlenet("LSTM", nn.LSTM(input_size=64, hidden_size=lstm_hidden_size, bidirectional=True, batch_first=True))
                self.lstm_pose_ud = weight_init_googlenet("LSTM", nn.LSTM(input_size=32, hidden_size=lstm_hidden_size, bidirectional=True, batch_first=True))

    def forward(self, input):
        output = self.projection(input)
        output = self.cls_fc_pose(output.view(output.size(0), -1))
        if self.has_lstm:
            output = output.view(output.size(0), 32, -1)
            _, (hidden_state_lr, _) = self.lstm_pose_lr(output)
            _, (hidden_state_ud, _) = self.lstm_pose_ud(output.permute(0, 2, 1))
            output = torch.cat((hidden_state_lr[0], hidden_state_lr[1], hidden_state_ud[0], hidden_state_ud[1]), 1)
        output = self.dropout(output)
        output_xy = self.cls_fc_xy(output)
        output_wpqr = self.cls_fc_wpqr(output)
        output_wpqr = F.normalize(output_wpqr, p=2, dim=1)
        
        # print(f"output_xy: {output_xy}")
        # print(f"output_wpqr: {output_wpqr}")

        return [output_xy, output_wpqr]

# define inception block for GoogleNet
class InceptionBlock(nn.Module):
    def __init__(self, incp, input_nc, x1_nc, x3_reduce_nc, x3_nc, x5_reduce_nc,
                 x5_nc, proj_nc, weights=None, gpu_ids=[]):
        super(InceptionBlock, self).__init__()
        self.gpu_ids = gpu_ids
        # first
        self.branch_x1 = nn.Sequential(*[
            weight_init_googlenet("inception_"+incp+"/1x1", nn.Conv2d(input_nc, x1_nc, kernel_size=1), weights),
            nn.ReLU(inplace=True)])

        self.branch_x3 = nn.Sequential(*[
            weight_init_googlenet("inception_"+incp+"/3x3_reduce", nn.Conv2d(input_nc, x3_reduce_nc, kernel_size=1), weights),
            nn.ReLU(inplace=True),
            weight_init_googlenet("inception_"+incp+"/3x3", nn.Conv2d(x3_reduce_nc, x3_nc, kernel_size=3, padding=1), weights),
            nn.ReLU(inplace=True)])

        self.branch_x5 = nn.Sequential(*[
            weight_init_googlenet("inception_"+incp+"/5x5_reduce", nn.Conv2d(input_nc, x5_reduce_nc, kernel_size=1), weights),
            nn.ReLU(inplace=True),
            weight_init_googlenet("inception_"+incp+"/5x5", nn.Conv2d(x5_reduce_nc, x5_nc, kernel_size=5, padding=2), weights),
            nn.ReLU(inplace=True)])

        self.branch_proj = nn.Sequential(*[
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            weight_init_googlenet("inception_"+incp+"/pool_proj", nn.Conv2d(input_nc, proj_nc, kernel_size=1), weights),
            nn.ReLU(inplace=True)])

        if incp in ["3b", "4e"]:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.pool = None

    def forward(self, input):
        outputs = [self.branch_x1(input), self.branch_x3(input),
                   self.branch_x5(input), self.branch_proj(input)]
        # print([[o.size()] for o in outputs])
        output = torch.cat(outputs, 1)
        if self.pool is not None:
            return self.pool(output)
        return output

class PoseNet(nn.Module):
    def __init__(self, input_nc, weights=None, isTest=False,  gpu_ids=[]):
        super(PoseNet, self).__init__()
        self.gpu_ids = gpu_ids
        self.isTest = isTest
        # self.before_inception = nn.Sequential(*[
        #     weight_init_googlenet("conv1/7x7_s2", nn.Conv2d(input_nc, 64, kernel_size=7, stride=2, padding=3), weights),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #     nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
        #     weight_init_googlenet("conv2/3x3_reduce", nn.Conv2d(64, 64, kernel_size=1), weights),
        #     nn.ReLU(inplace=True),
        #     weight_init_googlenet("conv2/3x3", nn.Conv2d(64, 192, kernel_size=3, padding=1), weights),
        #     nn.ReLU(inplace=True),
        #     nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #     ])
        googlenet = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
        self.before_inception = nn.Sequential(*list(googlenet.children())[:5])
        self.inception_3a = InceptionBlock("3a", 192, 64, 96, 128, 16, 32, 32, weights, gpu_ids)
        self.inception_3b = InceptionBlock("3b", 256, 128, 128, 192, 32, 96, 64, weights, gpu_ids)
        self.inception_4a = InceptionBlock("4a", 480, 192, 96, 208, 16, 48, 64, weights, gpu_ids)
        self.inception_4b = InceptionBlock("4b", 512, 160, 112, 224, 24, 64, 64, weights, gpu_ids)
        self.inception_4c = InceptionBlock("4c", 512, 128, 128, 256, 24, 64, 64, weights, gpu_ids)
        self.inception_4d = InceptionBlock("4d", 512, 112, 144, 288, 32, 64, 64, weights, gpu_ids)
        self.inception_4e = InceptionBlock("4e", 528, 256, 160, 320, 32, 128, 128, weights, gpu_ids)
        self.inception_5a = InceptionBlock("5a", 832, 256, 160, 320, 32, 128, 128, weights, gpu_ids)
        self.inception_5b = InceptionBlock("5b", 832, 384, 192, 384, 48, 128, 128, weights, gpu_ids)

        self.cls1_fc = RegressionHead(lossID="loss1", weights=weights)
        self.cls2_fc = RegressionHead(lossID="loss2", weights=weights)
        self.cls3_fc = RegressionHead(lossID="loss3", weights=weights, output_dim=6) # 최종 출력 6차원으로 변경

        self.model = nn.Sequential(*[self.inception_3a, self.inception_3b,
                                   self.inception_4a, self.inception_4b,
                                   self.inception_4c, self.inception_4d,
                                   self.inception_4e, self.inception_5a,
                                   self.inception_5b, self.cls1_fc,
                                   self.cls2_fc, self.cls3_fc
                                   ])
        if self.isTest:
            self.model.eval() # ensure Dropout is deactivated during test

    def forward(self, input):
        output_bf = self.before_inception(input)
        output_3a = self.inception_3a(output_bf)
        output_3b = self.inception_3b(output_3a)
        output_4a = self.inception_4a(output_3b)
        output_4b = self.inception_4b(output_4a)
        output_4c = self.inception_4c(output_4b)
        output_4d = self.inception_4d(output_4c)
        output_4e = self.inception_4e(output_4d)
        output_5a = self.inception_5a(output_4e)
        output_5b = self.inception_5b(output_5a)
    
        if not self.isTest:
            output = self.cls1_fc(output_4a) + self.cls2_fc(output_4d) + self.cls3_fc(output_5b)
        else:
            output = self.cls3_fc(output_5b)

        # # 출력 확인
        # print(f"Output 4a: {output_4a.size()}")
        # print(f"Output 4d: {output_4d.size()}")
        # print(f"Output 5b: {output_5b.size()}")
        # print(f"Final output: {output}")

        return output

        # output_bf = self.before_inception(input)
        # output_3a = self.inception_3a(output_bf)
        # output_3b = self.inception_3b(output_3a)
        # output_4a = self.inception_4a(output_3b)
        # output_4b = self.inception_4b(output_4a)
        # output_4c = self.inception_4c(output_4b)
        # output_4d = self.inception_4d(output_4c)
        # output_4e = self.inception_4e(output_4d)
        # output_5a = self.inception_5a(output_4e)
        # output_5b = self.inception_5b(output_5a)
        
        # print(f"Output 4a: {output_4a.size()}")
        # print(f"Output 4d: {output_4d.size()}")
        # print(f"Output 5b: {output_5b.size()}")
        

        # if not self.isTest:
        #     return self.cls1_fc(output_4a) + self.cls2_fc(output_4d) +  self.cls3_fc(output_5b)
        # return self.cls3_fc(output_5b)
    
    
    
"""기존 PoseNet"""


########### ResNet50 posenet ##############
class ResNet50Pose(nn.Module):
    def __init__(self, input_nc, isTest=False, gpu_ids=[]):
        super(ResNet50Pose, self).__init__()
        self.gpu_ids = gpu_ids
        self.isTest = isTest
        resnet = models.resnet50(pretrained=True)
        self.initial = nn.Sequential(*list(resnet.children())[:4])  # Initial layers
        self.layer1 = nn.Sequential(*list(resnet.layer1.children()))  # ResNet Block 1
        self.layer2 = nn.Sequential(*list(resnet.layer2.children()))  # ResNet Block 2
        self.layer3 = nn.Sequential(*list(resnet.layer3.children()))  # ResNet Block 3
        self.layer4 = nn.Sequential(*list(resnet.layer4.children()))  # ResNet Block 4

        self.fc_pos1 = nn.Linear(2048, 3)
        self.fc_ori1 = nn.Linear(2048, 3)
        self.fc_pos2 = nn.Linear(2048, 3)
        self.fc_ori2 = nn.Linear(2048, 3)
        self.fc_pos3 = nn.Linear(2048, 3)
        self.fc_ori3 = nn.Linear(2048, 3)

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        pos1 = self.fc_pos1(x)
        ori1 = self.fc_ori1(x)
        pos2 = self.fc_pos2(x)
        ori2 = self.fc_ori2(x)
        pos3 = self.fc_pos3(x)
        ori3 = self.fc_ori3(x)
        return [pos1, ori1, pos2, ori2, pos3, ori3]

########### ResNet50 posenet ##############

########### ResNet101 posenet ##############
class ResNet101Pose(nn.Module):
    def __init__(self, input_nc, isTest=False, gpu_ids=[]):
        super(ResNet101Pose, self).__init__()
        self.gpu_ids = gpu_ids
        self.isTest = isTest
        resnet = models.resnet101(pretrained=True)
        self.initial = nn.Sequential(*list(resnet.children())[:4])  # Initial layers
        self.layer1 = nn.Sequential(*list(resnet.layer1.children()))  # ResNet Block 1
        self.layer2 = nn.Sequential(*list(resnet.layer2.children()))  # ResNet Block 2
        self.layer3 = nn.Sequential(*list(resnet.layer3.children()))  # ResNet Block 3
        self.layer4 = nn.Sequential(*list(resnet.layer4.children()))  # ResNet Block 4

        self.fc_pos1 = nn.Linear(2048, 3)
        self.fc_ori1 = nn.Linear(2048, 3)
        self.fc_pos2 = nn.Linear(2048, 3)
        self.fc_ori2 = nn.Linear(2048, 3)
        self.fc_pos3 = nn.Linear(2048, 3)
        self.fc_ori3 = nn.Linear(2048, 3)

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        pos1 = self.fc_pos1(x)
        ori1 = self.fc_ori1(x)
        pos2 = self.fc_pos2(x)
        ori2 = self.fc_ori2(x)
        pos3 = self.fc_pos3(x)
        ori3 = self.fc_ori3(x)
        return [pos1, ori1, pos2, ori2, pos3, ori3]

########### ResNet101 posenet ##############

########### VGG16 posenet ##############
class VGG16Pose(nn.Module):
    def __init__(self, input_nc, isTest=False, gpu_ids=[]):
        super(VGG16Pose, self).__init__()
        self.gpu_ids = gpu_ids
        self.isTest = isTest
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout()
        )
        self.fc_pos1 = nn.Linear(4096, 3)
        self.fc_ori1 = nn.Linear(4096, 3)
        self.fc_pos2 = nn.Linear(4096, 3)
        self.fc_ori2 = nn.Linear(4096, 3)
        self.fc_pos3 = nn.Linear(4096, 3)
        self.fc_ori3 = nn.Linear(4096, 3)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        pos1 = self.fc_pos1(x)
        ori1 = self.fc_ori1(x)
        pos2 = self.fc_pos2(x)
        ori2 = self.fc_ori2(x)
        pos3 = self.fc_pos3(x)
        ori3 = self.fc_ori3(x)
        return [pos1, ori1, pos2, ori2, pos3, ori3]

########### VGG19 posenet ##############
class VGG19Pose(nn.Module):
    def __init__(self, input_nc, isTest=False, gpu_ids=[]):
        super(VGG19Pose, self).__init__()
        self.gpu_ids = gpu_ids
        self.isTest = isTest
        vgg = models.vgg19(pretrained=True)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout()
        )
        self.fc_pos1 = nn.Linear(4096, 3)
        self.fc_ori1 = nn.Linear(4096, 3)
        self.fc_pos2 = nn.Linear(4096, 3)
        self.fc_ori2 = nn.Linear(4096, 3)
        self.fc_pos3 = nn.Linear(4096, 3)
        self.fc_ori3 = nn.Linear(4096, 3)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        pos1 = self.fc_pos1(x)
        ori1 = self.fc_ori1(x)
        pos2 = self.fc_pos2(x)
        ori2 = self.fc_ori2(x)
        pos3 = self.fc_pos3(x)
        ori3 = self.fc_ori3(x)
        return [pos1, ori1, pos2, ori2, pos3, ori3]

########### Efficient B0 posenet ##############
class EfficientNetB0Pose(nn.Module):
    def __init__(self, input_nc, isTest=False, gpu_ids=[]):
        super(EfficientNetB0Pose, self).__init__()
        self.gpu_ids = gpu_ids
        self.isTest = isTest
        efficientnet = models.efficientnet_b0(pretrained=True)
        self.features = efficientnet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(True),
            nn.Dropout()
        )
        self.fc_pos1 = nn.Linear(512, 3)
        self.fc_ori1 = nn.Linear(512, 3)
        self.fc_pos2 = nn.Linear(512, 3)
        self.fc_ori2 = nn.Linear(512, 3)
        self.fc_pos3 = nn.Linear(512, 3)
        self.fc_ori3 = nn.Linear(512, 3)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        pos1 = self.fc_pos1(x)
        ori1 = self.fc_ori1(x)
        pos2 = self.fc_pos2(x)
        ori2 = self.fc_ori2(x)
        pos3 = self.fc_pos3(x)
        ori3 = self.fc_ori3(x)
        return [pos1, ori1, pos2, ori2, pos3, ori3]

########## MobileV2 posenet ##############
class MobileNetV2Pose(nn.Module):

    def __init__(self, input_nc, isTest=False, gpu_ids=[]):
        super(MobileNetV2Pose, self).__init__()
        self.gpu_ids = gpu_ids
        self.isTest = isTest
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(True),
            nn.Dropout()
        )
        self.fc_pos1 = nn.Linear(512, 3)
        self.fc_ori1 = nn.Linear(512, 3)
        self.fc_pos2 = nn.Linear(512, 3)
        self.fc_ori2 = nn.Linear(512, 3)
        self.fc_pos3 = nn.Linear(512, 3)
        self.fc_ori3 = nn.Linear(512, 3)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        pos1 = self.fc_pos1(x)
        ori1 = self.fc_ori1(x)
        pos2 = self.fc_pos2(x)
        ori2 = self.fc_ori2(x)
        pos3 = self.fc_pos3(x)
        ori3 = self.fc_ori3(x)
        return [pos1, ori1, pos2, ori2, pos3, ori3]


### 안 씀 ###
class PoseLSTM(ResNet50Pose):
    def __init__(self, input_nc, lstm_hidden_size, weights=None, isTest=False, gpu_ids=[]):
        super(PoseLSTM, self).__init__(input_nc, isTest, gpu_ids)
        self.lstm_pose_lr = nn.LSTM(input_size=2048, hidden_size=lstm_hidden_size, bidirectional=True, batch_first=True)
        self.lstm_pose_ud = nn.LSTM(input_size=2048, hidden_size=lstm_hidden_size, bidirectional=True, batch_first=True)
        self.fc_pos = nn.Linear(lstm_hidden_size * 4, 3)
        self.fc_ori = nn.Linear(lstm_hidden_size * 4, 4)
    
    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), 32, -1)
        _, (hidden_state_lr, _) = self.lstm_pose_lr(x)
        _, (hidden_state_ud, _) = self.lstm_pose_ud(x)
        x = torch.cat((hidden_state_lr[0], hidden_state_lr[1], hidden_state_ud[0], hidden_state_ud[1]), dim=1)
        pos = self.fc_pos(x)
        ori = self.fc_ori(x)
        ori = F.normalize(ori, p=2, dim=1)
        return [pos, ori]

class PoseLSTM(PoseNet):
    def __init__(self, input_nc, lstm_hidden_size, weights=None, isTest=False,  gpu_ids=[]):
            super(PoseLSTM, self).__init__(input_nc, weights, isTest, gpu_ids)
            self.cls1_fc = RegressionHead(lossID="loss1", weights=weights, lstm_hidden_size=lstm_hidden_size)
            self.cls2_fc = RegressionHead(lossID="loss2", weights=weights, lstm_hidden_size=lstm_hidden_size)
            self.cls3_fc = RegressionHead(lossID="loss3", weights=weights, lstm_hidden_size=lstm_hidden_size)

            self.model = nn.Sequential(*[self.inception_3a, self.inception_3b,
                                       self.inception_4a, self.inception_4b,
                                       self.inception_4c, self.inception_4d,
                                       self.inception_4e, self.inception_5a,
                                       self.inception_5b, self.cls1_fc,
                                       self.cls2_fc, self.cls3_fc
                                       ])
            if self.isTest:
                self.model.eval() # ensure Dropout is deactivated during test

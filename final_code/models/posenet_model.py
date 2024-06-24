import numpy as np
import torch
import torch.nn.functional as F
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import pickle
import numpy
from scipy.spatial.transform import Rotation as R  # 누락된 부분 추가
import pdb
import torch
import torch.nn.functional as F
import numpy as np

class GeometricLoss(torch.nn.Module):
    def __init__(self):
        super(GeometricLoss, self).__init__()

    def forward(self, pos_pred, ori_pred, pos_target, ori_target):
        pos_loss = torch.mean(torch.sqrt(torch.sum((pos_pred - pos_target)**2, dim=1)))
        ori_loss = torch.mean(torch.sqrt(torch.sum((ori_pred - ori_target)**2, dim=1)))
        return pos_loss + ori_loss

# class QuaternionLoss(torch.nn.Module):
#     def __init__(self):
#         super(QuaternionLoss, self).__init__()

#     def forward(self, pred, target):
#         return torch.mean(1 - torch.sum(pred * target, dim=1)**2)

class UncertaintyAwareLoss(torch.nn.Module):
    def __init__(self):
        super(UncertaintyAwareLoss, self).__init__()

    def forward(self, pos_pred, ori_pred, pos_target, ori_target, sigma):
        pos_loss = torch.sum((pos_pred - pos_target)**2) / (2 * sigma[0]**2) + torch.log(sigma[0])
        ori_loss = torch.sum((ori_pred - ori_target)**2) / (2 * sigma[1]**2) + torch.log(sigma[1])
        return pos_loss + ori_loss
def eul2rot(heading, pitch, roll):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(heading), -np.sin(heading), 0],
                    [np.sin(heading), np.cos(heading), 0],
                    [0, 0, 1]])
    
    R = np.dot(R_z, np.dot(R_y, R_x))
    
    return R

def rotation_matrix_to_euler_angles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    
    return np.array([z, y, x])

def transform_to_sim_coordinate(x_team1, y_team1, z_team1, heading_team1, pitch_team1, roll_team1, T):
    team1_position = np.vstack((x_team1, y_team1, z_team1, np.ones_like(x_team1)))
    sim_position = np.dot(T, team1_position)
    sim_positions = sim_position[:3, :].T

    sim_x = sim_positions[:, 0]
    sim_y = sim_positions[:, 1]
    sim_z = sim_positions[:, 2]

    R_team1 = np.array([eul2rot(h, p, r) for h, p, r in zip(heading_team1, pitch_team1, roll_team1)])
    R_sim = np.array([T[:3, :3] @ R for R in R_team1])
    sim_euler_angles = np.array([rotation_matrix_to_euler_angles(R) for R in R_sim])

    sim_heading = sim_euler_angles[:, 0]
    sim_pitch = sim_euler_angles[:, 1]
    sim_roll = sim_euler_angles[:, 2]

    return sim_x, sim_y, sim_z, np.rad2deg(sim_heading), np.rad2deg(sim_pitch), np.rad2deg(sim_roll)

def print_network(network):
        num_params = sum(p.numel() for p in network.parameters())
        print(network)
        print(f'Total number of parameters: {num_params}')

class PoseNetModel(BaseModel):
    def name(self):
        return 'PoseNetModel'
    """
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # 네트워크 정의 전에 초기화
        self.netG = networks.define_network(opt.input_nc, None, opt.model, isTest=not self.isTrain, gpu_ids=self.gpu_ids)
        
        # define tensors
        self.input_A = torch.empty(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize, dtype=torch.float32, device=self.gpu_ids[0])
        self.input_B = torch.empty(opt.batchSize, opt.output_nc, dtype=torch.float32, device=self.gpu_ids[0])

        # load/define networks
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)

        self.mean_image = np.load(os.path.join(opt.dataroot , 'mean_image.npy'))

        # if len(self.gpu_ids) > 1:
        #     self.netG = torch.nn.DataParallel(self.netG, device_ids=self.gpu_ids).cuda(self.gpu_ids[0])
        # else:
        #     self.netG = self.netG.cuda(self.gpu_ids[0])
        self.netG = self.netG.cuda(self.gpu_ids[0])

        
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # # define loss functions
            # self.criterion = torch.nn.MSELoss()
            
             # Define loss functions
            if opt.loss == 'mse': # 원본 PoseNet Loss
                self.criterion = torch.nn.MSELoss()
            elif opt.loss == 'geometric': # Geometric Loss
                self.criterion = GeometricLoss()
            # elif opt.loss == 'quaternion': # Quaternion Loss
            #     self.criterion = QuaternionLoss()
            elif opt.loss == 'uncertainty': # Bayesian PoseNet
                self.criterion = UncertaintyAwareLoss()
            elif opt.loss == 'mae': # mean absolute error
                self.criterion = torch.nn.L1Loss()    
            else:
                raise ValueError("Loss function [%s] not recognized." % opt.loss)

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, eps=1,
                                                weight_decay=0.0625,
                                                betas=(self.opt.adambeta1, self.opt.adambeta2))
            self.optimizers.append(self.optimizer_G)
            # for optimizer in self.optimizers:
            #     self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        print_network(self.netG)
        print('-----------------------------------------------')
    """
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.netG = networks.define_network(opt.input_nc, None, opt.model, isTest=not self.isTrain, gpu_ids=self.gpu_ids)
        
        self.input_A = torch.empty(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize, dtype=torch.float32, device=self.gpu_ids[0])
        self.input_B = torch.empty(opt.batchSize, opt.output_nc, dtype=torch.float32, device=self.gpu_ids[0])

        self.mean_image = np.load(os.path.join(opt.dataroot , 'mean_image.npy'))

        self.netG = self.netG.cuda(self.gpu_ids[0])

        if self.isTrain:
            self.old_lr = opt.lr

            if opt.loss == 'mse':
                self.criterion = torch.nn.MSELoss()
            elif opt.loss == 'geometric':
                self.criterion = GeometricLoss()
            elif opt.loss == 'uncertainty':
                self.criterion = UncertaintyAwareLoss()
            elif opt.loss == 'mae':
                self.criterion = torch.nn.L1Loss()    
            else:
                raise ValueError("Loss function [%s] not recognized." % opt.loss)

            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, eps=1,
                                                weight_decay=0.0625,
                                                betas=(self.opt.adambeta1, self.opt.adambeta2))
            self.optimizers.append(self.optimizer_G)

        print('---------- Networks initialized -------------')
        print_network(self.netG)
        print('-----------------------------------------------')
    
    
    # def set_input(self, input):
    #     #pdb.set_trace()
    #     input_A = input['A']
    #     input_B = input['B']
    #     self.image_paths = input['A_paths']
    #     ##
    #     self.input_A.resize_(input_A.size()).copy_(input_A)
    #     # converted_B = self.convert_to_euler(input_B)
        
    #     # print(f"Original Input B shape: {input_B.shape}")
    #     # print(f"Converted Input B shape: {converted_B.shape}")
        
    #     # self.input_B.resize_(converted_B.size()).copy_(converted_B)
    #     self.input_B.resize_(input_B.size()).copy_(input_B)
        
    def set_input(self, input):
        input_A = input['A']
        input_B = input['B']
        self.image_paths = input['A_paths']
        
        # 변환 행렬
        transform_matrix = np.array([
            [-0.10158499, -0.0204423,  0.0192721,  -7.92273845],
            [-0.12908179, -0.08904014, 0.04370393,  1.5485332 ],
            [ 0.24091402,  0.02519702, -0.11671822, 3.03941279],
            [ 0.0,         0.0,         0.0,        1.0       ]
        ])
        
        # 좌표 변환
        x_team1 = input_B[:, 0].cpu().numpy()
        y_team1 = input_B[:, 1].cpu().numpy()
        z_team1 = input_B[:, 2].cpu().numpy()
        heading_team1 = input_B[:, 3].cpu().numpy()
        pitch_team1 = input_B[:, 4].cpu().numpy()
        roll_team1 = input_B[:, 5].cpu().numpy()

        sim_x, sim_y, sim_z, sim_heading, sim_pitch, sim_roll = transform_to_sim_coordinate(x_team1, y_team1, z_team1, heading_team1, pitch_team1, roll_team1, transform_matrix)

        transformed_B = torch.tensor(np.vstack((sim_x, sim_y, sim_z, sim_heading, sim_pitch, sim_roll)).T, dtype=torch.float32, device=self.gpu_ids[0])

        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(transformed_B.size()).copy_(transformed_B)
    
    """이거 됐었음"""
    # def set_input(self, input):
    #     # pdb.set_trace()
    #     input_A = input['A']
    #     input_B = input['B']
    #     self.image_paths = input['A_paths']
    
    # # Transform the coordinates in input_B using the transformation matrix T
    #     T = np.array([
    #         [-0.05622614, -0.96979975, -0.21520289, -0.70764561],
    #         [ 0.99302464, -0.06977679,  0.05551802, -4.98129455],
    #         [-0.06948705, -0.21255867,  0.96937524,  2.97621823],
    #         [ 0.0,         0.0,         0.0,         1.0]
    #     ])  
    
    #     # Create a copy of input_B to hold transformed values
    #     transformed_B = input_B.clone()
    
    #     for i in range(input_B.shape[0]):
    #         x, y, z, heading, pitch, roll = input_B[i]
        
    #         # Create a homogeneous coordinate vector
    #         point = np.array([x, y, z, 1.0])
        
    #         # Apply the transformation matrix
    #         transformed_point = T.dot(point)
        
    #         # Extract the transformed coordinates
    #         transformed_x, transformed_y, transformed_z = transformed_point[:3]
        
    #         # Update the transformed_B tensor
    #         transformed_B[i, 0] = transformed_x
    #         transformed_B[i, 1] = transformed_y
    #         transformed_B[i, 2] = transformed_z
    #         transformed_B[i, 3] = heading  # Assuming heading, pitch, roll do not change
    #         transformed_B[i, 4] = pitch
    #         transformed_B[i, 5] = roll
    
    #     self.input_A.resize_(input_A.size()).copy_(input_A)
    
    #     # Use the transformed coordinates for input_B
    #     self.input_B.resize_(transformed_B.size()).copy_(transformed_B)
    """이거 됐었음"""
    def convert_to_euler(self, input_B):
        positions = input_B[:, :3]
        quaternions = input_B[:, 3:]
        rotations = R.from_quat(quaternions.cpu().numpy())
        eulers = torch.tensor(rotations.as_euler('xyz', degrees=True)).to(input_B.device)
        return torch.cat((positions, eulers), dim=1)

    def forward(self):
        self.pred_B = self.netG(self.input_A)

    # no backprop gradients
    def test(self):
        self.forward()

    # get image paths
    def get_image_paths(self):
        return self.image_paths
    
    # def backward(self):
    #     self.loss_G = 0
    #     self.loss_pos = 0
    #     self.loss_ori = 0
    #     loss_weights = [0.3, 0.3, 1]
    #     self.opt.beta= 100
    #     pos_pred = self.pred_B[0]
    #     ori_pred = self.pred_B[1]
    #     pos_target = self.input_B[:, 0:3]
    #     ori_target = self.input_B[:, 3:]

    #     if isinstance(self.criterion, UncertaintyAwareLoss):
    #         sigma = torch.tensor([100, 100], device=self.input_B.device)
    #         loss = self.criterion(pos_pred, ori_pred, pos_target, ori_target, sigma)
    #         pos_loss = torch.sum((pos_pred - pos_target)**2) / (2 * sigma[0]**2) + torch.log(sigma[0])
    #         ori_loss = torch.sum((ori_pred - ori_target)**2) / (2 * sigma[1]**2) + torch.log(sigma[1])
    #     elif isinstance(self.criterion, GeometricLoss):
    #         loss = self.criterion(pos_pred, ori_pred, pos_target, ori_target)
    #         pos_loss = torch.mean(torch.sqrt(torch.sum((pos_pred - pos_target)**2, dim=1)))
    #         ori_loss = torch.mean(torch.sqrt(torch.sum((ori_pred - ori_target)**2, dim=1)))
    #     else:
    #         pos_loss = self.criterion(pos_pred, pos_target)
    #         ori_loss = self.criterion(ori_pred, ori_target)
    #         loss = pos_loss + ori_loss * self.opt.beta

    #     self.loss_G += loss
    #     self.loss_pos += pos_loss.item()
    #     self.loss_ori += ori_loss.item() * self.opt.beta
    #     self.loss_G.backward()
    def backward(self):
        self.loss_G = 0
        self.loss_pos = 0
        self.loss_ori = 0
        loss_weights = [0.3, 0.3, 1]
        self.opt.beta=100
        for l, w in enumerate(loss_weights):
            pos_pred = self.pred_B[2*l]
            ori_pred = self.pred_B[2*l+1]
            pos_target = self.input_B[:, 0:3]
            ori_target = self.input_B[:, 3:] * np.pi / 180.0

            if isinstance(self.criterion, UncertaintyAwareLoss):
                # sigma 값을 0.1로 설정
                sigma = torch.tensor([100, 100], device=self.input_B.device)
                loss = self.criterion(pos_pred, ori_pred, pos_target, ori_target, sigma)
                pos_loss = torch.sum((pos_pred - pos_target)**2) / (2 * sigma[0]**2) + torch.log(sigma[0])
                ori_loss = torch.sum((ori_pred - ori_target)**2) / (2 * sigma[1]**2) + torch.log(sigma[1])
            elif isinstance(self.criterion, GeometricLoss):
                loss = self.criterion(pos_pred, ori_pred, pos_target, ori_target)
                pos_loss = torch.mean(torch.sqrt(torch.sum((pos_pred - pos_target)**2, dim=1)))
                ori_loss = torch.mean(torch.sqrt(torch.sum((ori_pred - ori_target)**2, dim=1)))
            else:
                pos_loss = self.criterion(pos_pred, pos_target)
                ori_loss = self.criterion(ori_pred, ori_target)
                #pdb.set_trace()
                loss = pos_loss + ori_loss * self.opt.beta

            self.loss_G += loss * w
            self.loss_pos += pos_loss.item() * w
            self.loss_ori += ori_loss.item() * w * self.opt.beta
        self.loss_G.backward()
    
    def optimize_parameters(self):
        self.forward()
        #pdb.set_trace()
        self.optimizer_G.zero_grad()
        self.backward()
        self.optimizer_G.step()

    def get_current_errors(self):
        if self.opt.isTrain:
            return OrderedDict([('pos_err', self.loss_pos),
                                ('ori_err', self.loss_ori),
                                ])

        pos_err = torch.dist(self.pred_B[0], self.input_B[:, 0:3])
        ori_gt = F.normalize(self.input_B[:, 3:], p=2, dim=1)
        abs_distance = torch.abs((ori_gt.mul(self.pred_B[1])).sum())
        ori_err = 2*180/numpy.pi* torch.acos(abs_distance)
        return [pos_err.item(), ori_err.item()]
    
    # # 아래가 바꾼거
    # def get_current_errors(self):
    #     pos_pred = self.pred_B[0]
    #     ori_pred = self.pred_B[1]
    #     pos_target = self.input_B[:, 0:3]
    #     ori_target = self.input_B[:, 3:]
    #     pos_err = torch.mean((pos_pred - pos_target) ** 2)
    #     ori_err = torch.mean((ori_pred - ori_target) ** 2)
    #     return pos_err, ori_err

    def get_current_pose(self):
        return numpy.concatenate((self.pred_B[0].data[0].cpu().numpy(),
                                  self.pred_B[1].data[0].cpu().numpy()))

    def get_current_visuals(self):
        input_A = util.tensor2im(self.input_A.data)
        # pred_B = util.tensor2im(self.pred_B.data)
        # input_B = util.tensor2im(self.input_B.data)
        return OrderedDict([('input_A', input_A)])
    
    def state_dict(self):
        return self.netG.state_dict()

    # def save(self, label):
    #     self.save_network(self.netG, 'G', label, self.gpu_ids)
    def save(self, label):
        torch.save(self.state_dict(), f'{label}_netG.pth')
class ResNet50PoseModel(PoseNetModel):
    def name(self):
        return 'ResNet50PoseModel'
class GoogleNetPoseModel(PoseNetModel):
    def name(self):
        return 'GoogleNetPoseModel'

class ResNet101PoseModel(PoseNetModel):
    def name(self):
        return 'ResNet101PoseModel'

class VGG16PoseModel(PoseNetModel):
    def name(self):
        return 'VGG16PoseModel'
class VGG19PoseModel(PoseNetModel):
    def name(self):
        return 'VGG19PoseModel'

class EfficientNetB0PoseModel(PoseNetModel):
    def name(self):
        return 'EfficientNetB0PoseModel'

class MobileNetV2PoseModel(PoseNetModel):
    def name(self):
        return 'MobileNetV2PoseModel'
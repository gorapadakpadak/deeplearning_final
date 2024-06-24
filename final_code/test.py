# import time
# import os
# from options.test_options import TestOptions
# from data.data_loader import CreateDataLoader
# from models.models import create_model
# from util.visualizer import Visualizer
# from util import html
# import numpy

# opt = TestOptions().parse()
# opt.nThreads = 1   # test code only supports nThreads = 1
# opt.batchSize = 1  # test code only supports batchSize = 1
# opt.serial_batches = True  # no shuffle
# opt.no_flip = True  # no flip

# data_loader = CreateDataLoader(opt)
# dataset = data_loader.load_data()

# results_dir = os.path.join(opt.results_dir, opt.name)
# if not os.path.exists(results_dir):
#     os.makedirs(results_dir)

# besterror  = [0, float('inf'), float('inf')] # nepoch, medX, medQ
# if opt.model == 'posenet':
#     testepochs = numpy.arange(450, 500+1, 5)
# else:
#     testepochs = numpy.arange(450, 1200+1, 5)

# testfile = open(os.path.join(results_dir, 'test_median.txt'), 'a')
# testfile.write('epoch medX  medQ\n')
# testfile.write('==================\n')

# model = create_model(opt)
# visualizer = Visualizer(opt)

# # 직접 가중치 파일 경로를 지정하여 로드
# weight_path = '/mnt/8TB_1/chaewon/sim/poselstm-pytorch/checkpoints/real_kc_mobilenetV2_final.pth'
# if not os.path.isfile(weight_path):
#     raise FileNotFoundError(f'Weight file not found: {weight_path}')
# model.load_network(model.netG, 'G', weight_path)

# visualizer.change_log_path('mobile_mse')

# # test
# err = []
# print("Testing with weights from mobile_mse.pth")
# for i, data in enumerate(dataset):
#     model.set_input(data)
#     model.test()
#     img_path = model.get_image_paths()[0]
#     print('\t%04d/%04d: process image... %s' % (i, len(dataset), img_path), end='\r')
#     image_path = img_path.split('/')[-2] + '/' + img_path.split('/')[-1]
#     pose = model.get_current_pose()
#     visualizer.save_estimated_pose(image_path, pose)
#     err_p, err_o = model.get_current_errors()
#     err.append([err_p, err_o])

# median_pos = numpy.median(err, axis=0)
# if median_pos[0] < besterror[1]:
#     besterror = ['mobile_mse', median_pos[0], median_pos[1]]
# print()
# print("\tmedian wrt pos.: {0:.2f}m {1:.2f}°".format(median_pos[0], median_pos[1]))
# testfile.write("{0:<5} {1:.2f}m {2:.2f}°\n".format('mobile_mse',
#                                                  median_pos[0],
#                                                  median_pos[1]))
# testfile.flush()
# print("{0:<5} {1:.2f}m {2:.2f}°\n".format(*besterror))
# testfile.write('-----------------\n')
# testfile.write("{0:<5} {1:.2f}m {2:.2f}°\n".format(*besterror))
# testfile.write('==================\n')
# testfile.close()

import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import numpy as np
import torch

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.model = 'mobilenetV2'  # mobilenetV2 모델을 명시적으로 설정

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

results_dir = os.path.join(opt.results_dir, opt.name)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

besterror  = [0, float('inf'), float('inf')] # nepoch, medX, medQ

testfile = open(os.path.join(results_dir, 'test_median.txt'), 'a')
testfile.write('epoch medX  medQ\n')
testfile.write('==================\n')

model = create_model(opt)
visualizer = Visualizer(opt)

# 직접 가중치 파일 경로를 지정하여 로드
weight_path = '/mnt/8TB_1/chaewon/sim/poselstm-pytorch/checkpoints/real_kc_mobilenetV2_final.pth'
if not os.path.isfile(weight_path):
    raise FileNotFoundError(f'Weight file not found: {weight_path}')

# 모델 가중치 로드
state_dict = torch.load(weight_path)
model.netG.load_state_dict(state_dict)

visualizer.change_log_path('mobile_mse')

# test
err = []
predictions = []

print("Testing with weights from mobile_mse.pth")
for i, data in enumerate(dataset):
    model.set_input(data)
    model.test()
    img_path = model.get_image_paths()[0]
    print('\t%04d/%04d: process image... %s' % (i, len(dataset), img_path), end='\r')
    image_path = img_path.split('/')[-2] + '/' + img_path.split('/')[-1]
    pose = model.get_current_pose().tolist()
    visualizer.save_estimated_pose(image_path, pose)
    err_p, err_o = model.get_current_errors()
    err.append([err_p, err_o])
    # Save predictions
    predictions.append({
        'image_path': image_path,
        'predicted_pose': pose,
        'ground_truth': model.input_B.cpu().numpy().tolist()  # Assuming target_pose is defined in the model
    })

median_pos = np.median(err, axis=0)
if median_pos[0] < besterror[1]:
    besterror = ['mobile_mse', median_pos[0], median_pos[1]]
print()
print("\tmedian wrt pos.: {0:.2f}m {1:.2f}°".format(median_pos[0], median_pos[1]))
testfile.write("{0:<5} {1:.2f}m {2:.2f}°\n".format('mobile_mse',
                                                 median_pos[0],
                                                 median_pos[1]))
testfile.flush()
print("{0:<5} {1:.2f}m {2:.2f}°\n".format(*besterror))
testfile.write('-----------------\n')
testfile.write("{0:<5} {1:.2f}m {2:.2f}°\n".format(*besterror))
testfile.write('==================\n')
testfile.close()

# Save predictions to a file
predictions_file = os.path.join(results_dir, 'predictions.json')
import json
with open(predictions_file, 'w') as f:
    json.dump(predictions, f, indent=4)

print(f"Predictions saved to {predictions_file}")

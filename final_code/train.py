# import time
# from options.train_options import TrainOptions
# from data.data_loader import CreateDataLoader
# from models.models import create_model
# from util.visualizer import Visualizer
# import matplotlib.pyplot as plt
# import torch
# import numpy as np
# import random
# import os

# # 옵션 파싱
# opt = TrainOptions().parse()

# # 재현성을 위한 시드 설정
# torch.manual_seed(opt.seed)
# np.random.seed(opt.seed)
# random.seed(opt.seed)
# torch.backends.cudnn.deterministic = True

# # 모델 학습 및 평가 함수
# def train_and_evaluate(opt, model_name, num_epochs):
#     opt.model = model_name
#     model = create_model(opt)
#     data_loader = CreateDataLoader(opt)
#     dataset = data_loader.load_data()
#     dataset_size = len(data_loader)
#     visualizer = Visualizer(opt)
#     total_steps = 0

#     epoch_loss = []
#     epoch_pos_err = []
#     epoch_ori_err = []

#     for epoch in range(num_epochs):
#         epoch_start_time = time.time()
#         epoch_iter = 0

#         for i, data in enumerate(dataset):
#             iter_start_time = time.time()
#             visualizer.reset()
#             total_steps += opt.batchSize
#             epoch_iter += opt.batchSize
            
#             model.set_input(data)
#             model.optimize_parameters()

#             if total_steps % opt.print_freq == 0:
#                 errors = model.get_current_errors()
#                 t = (time.time() - iter_start_time) / opt.batchSize
#                 visualizer.print_current_errors(epoch, epoch_iter, errors, t)
#                 if opt.display_id > 0:
#                     visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

#         if epoch % opt.save_epoch_freq == 0:
#             print('saving the model at the end of epoch %d, iters %d' %
#                   (epoch, total_steps))
#             model.save('latest')
#             model.save(epoch)

#         errors = model.get_current_errors()
#         epoch_loss.append(model.loss_G.item())  # 손실 값 기록
#         epoch_pos_err.append(errors['pos_err'])
#         epoch_ori_err.append(errors['ori_err'])

#         print('End of epoch %d / %d \t Time Taken: %d sec' %
#               (epoch, num_epochs, time.time() - epoch_start_time))
#         model.update_learning_rate()

#     # 모델 저장
#     model_save_path = os.path.join(opt.checkpoints_dir, f'greatcourt_{model_name}_final.pth')
#     torch.save(model.state_dict(), model_save_path)
#     print(f'Model {model_name} saved to {model_save_path}')

#     return epoch_loss, epoch_pos_err, epoch_ori_err

# # 결과를 시각화하고 저장하는 함수
# def plot_results(results, save_path):
#     fig, axs = plt.subplots(3, 1, figsize=(10, 15))
#     for key in results:
#         epochs = range(1, len(results[key]['loss']) + 1)
#         axs[0].plot(epochs, results[key]['loss'], label=key)
#         axs[1].plot(epochs, results[key]['pos_err'], label=key)
#         axs[2].plot(epochs, results[key]['ori_err'], label=key)

#     axs[0].set_title('Loss vs. Epochs')
#     axs[0].set_xlabel('Epochs')
#     axs[0].set_ylabel('Loss')
#     axs[0].legend()

#     axs[1].set_title('Position Error vs. Epochs')
#     axs[1].set_xlabel('Epochs')
#     axs[1].set_ylabel('Position Error (meters)')
#     axs[1].legend()

#     axs[2].set_title('Orientation Error vs. Epochs')
#     axs[2].set_xlabel('Epochs')
#     axs[2].set_ylabel('Orientation Error (degrees)')
#     axs[2].legend()

#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()

# # 메인 스크립트
# if __name__ == '__main__':
#     models_to_compare = ['posenet', 'resnet50', 'resnet101', 'vgg16', 'vgg19', 'efficientnetB0', 'mobilenetV2']
#     results = {}
#     num_epochs_initial = 50  # 초기 학습 에포크
#     num_epochs_finetune = 100  # 파인튜닝 에포크

#     # Step 1: 모든 backbone 모델을 초기 학습
#     for model_name in models_to_compare:
#         print(f"Training {model_name}")
#         opt.model = model_name
#         loss, pos_err, ori_err = train_and_evaluate(opt, model_name, num_epochs_initial)
#         results[model_name] = {'loss': loss, 'pos_err': pos_err, 'ori_err': ori_err}

#     # 초기 학습 결과 그래프 저장
#     plot_results(results, 'gr_initial_training_results.png')

#     # Step 2: 상위 4개 모델 선택 및 파인튜닝
#     top_4_models = sorted(results.keys(), key=lambda x: results[x]['loss'][-1])[:4]

#     for model_name in top_4_models:
#         print(f"Fine-tuning {model_name}")
#         opt.model = model_name
#         loss, pos_err, ori_err = train_and_evaluate(opt, model_name, num_epochs_finetune)

#         # 결과를 추가하여 갱신
#         results[model_name]['loss'].extend(loss)
#         results[model_name]['pos_err'].extend(pos_err)
#         results[model_name]['ori_err'].extend(ori_err)

#     # 최종 결과 그래프 저장
#     plot_results(results, 'gr_final_training_results.png')

import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import os
import torchsummary
from fvcore.nn import FlopCountAnalysis
def free_gpu_memory():
    torch.cuda.empty_cache()


# 옵션 파싱
opt = TrainOptions().parse()

# 재현성을 위한 시드 설정
torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)
torch.backends.cudnn.deterministic = True

# # 모델 학습 및 평가 함수
# def train_and_evaluate(opt, model_name, num_epochs):
#     opt.model = model_name
#     model = create_model(opt)
#     data_loader = CreateDataLoader(opt)
#     dataset = data_loader.load_data()
#     dataset_size = len(data_loader)
#     visualizer = Visualizer(opt)
#     total_steps = 0

#     epoch_loss = []
#     epoch_pos_err = []
#     epoch_ori_err = []

#     for epoch in range(num_epochs):
#         epoch_start_time = time.time()
#         epoch_iter = 0

#         for i, data in enumerate(dataset):
#             iter_start_time = time.time()
#             visualizer.reset()
#             total_steps += opt.batchSize
#             epoch_iter += opt.batchSize
            
#             model.set_input(data)
#             model.optimize_parameters()

#             if total_steps % opt.print_freq == 0:
#                 errors = model.get_current_errors()
#                 t = (time.time() - iter_start_time) / opt.batchSize
#                 visualizer.print_current_errors(epoch, epoch_iter, errors, t)
#                 if opt.display_id > 0:
#                     visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

#         if epoch % opt.save_epoch_freq == 0:
#             print('saving the model at the end of epoch %d, iters %d' %
#                   (epoch, total_steps))
#             model.save('latest')
#             model.save(epoch)

#         errors = model.get_current_errors()
#         epoch_loss.append(model.loss_G.item())  # 손실 값 기록
#         epoch_pos_err.append(errors['pos_err'])
#         epoch_ori_err.append(errors['ori_err'])

#         print('End of epoch %d / %d \t Time Taken: %d sec' %
#               (epoch, num_epochs, time.time() - epoch_start_time))
#         model.update_learning_rate()

#     # 모델 저장
#     model_save_path = os.path.join(opt.checkpoints_dir, f'kc_{model_name}_final.pth')
#     torch.save(model.state_dict(), model_save_path)
#     print(f'Model {model_name} saved to {model_save_path}')

#     return epoch_loss, epoch_pos_err, epoch_ori_err
# 학습 곡선을 시각화하는 함수
def plot_learning_curve(train_losses, val_losses=None, train_pos_err=None, val_pos_err=None, train_ori_err=None, val_ori_err=None, save_path='learning_curve.png'):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(18, 10))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    if train_pos_err:
        plt.subplot(2, 2, 2)
        plt.plot(epochs, train_pos_err, 'b-', label='Training Position Error')
        if val_pos_err:
            plt.plot(epochs, val_pos_err, 'r-', label='Validation Position Error')
        plt.title('Training and Validation Position Error')
        plt.xlabel('Epochs')
        plt.ylabel('Position Error (meters)')
        plt.legend()

    if train_ori_err:
        plt.subplot(2, 2, 3)
        plt.plot(epochs, train_ori_err, 'b-', label='Training Orientation Error')
        if val_ori_err:
            plt.plot(epochs, val_ori_err, 'r-', label='Validation Orientation Error')
        plt.title('Training and Validation Orientation Error')
        plt.xlabel('Epochs')
        plt.ylabel('Orientation Error (degrees)')
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # 모델 학습 및 평가 함수
def train_and_evaluate(opt, model_name, num_epochs):
    opt.model = model_name
    model = create_model(opt)
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print("데이터크기가 말이야~~~~~~~~~~~~~₩",dataset_size)
    visualizer = Visualizer(opt)
    total_steps = 0

    epoch_loss = []
    epoch_pos_err = []
    epoch_ori_err = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        errors = model.get_current_errors()
        epoch_loss.append(model.loss_G.item())  # 손실 값 기록
        epoch_pos_err.append(errors['pos_err'])
        epoch_ori_err.append(errors['ori_err'])

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, num_epochs, time.time() - epoch_start_time))
        model.update_learning_rate()

    # 모델 저장
    model_save_path = os.path.join(opt.checkpoints_dir, f'real_kc_{model_name}_final.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f'Model {model_name} saved to {model_save_path}')

    return epoch_loss, epoch_pos_err, epoch_ori_err
# def calculate_model_complexity(model, input_size=(3, 224, 224)):
#     """
#     모델의 FLOPs와 파라미터 수를 계산하는 함수
#     """
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     input_tensor = torch.randn(1, *input_size).to(device)
    
#     # FLOPs 계산
#     flops = FlopCountAnalysis(model, input_tensor).total()
    
#     # 파라미터 수 계산
#     summary = torchsummary.summary(model, input_size, verbose=0)
    
#     return flops, summary.total_params

# 이동 평균 함수 정의
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# 결과를 시각화하고 저장하는 함수
def plot_results(results, save_path, top_4_models=None):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    if top_4_models:
        results = {key: results[key] for key in top_4_models}
    
    for key in results:
        epochs = range(1, len(results[key]['loss']) + 1)
        
        window_size = 5  # 이동 평균의 윈도우 크기
        smoothed_loss = moving_average(results[key]['loss'], window_size)
        smoothed_pos_err = moving_average(results[key]['pos_err'], window_size)
        smoothed_ori_err = moving_average(results[key]['ori_err'], window_size)

        axs[0].plot(range(1, len(smoothed_loss) + 1), smoothed_loss, label=key)
        axs[1].plot(range(1, len(smoothed_pos_err) + 1), smoothed_pos_err, label=key)
        axs[2].plot(range(1, len(smoothed_ori_err) + 1), smoothed_ori_err, label=key)

    axs[0].set_title('Loss vs. Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].set_title('Position Error vs. Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Position Error (meters)')
    axs[1].legend()

    axs[2].set_title('Orientation Error vs. Epochs')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Orientation Error (degrees)')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 메인 스크립트
if __name__ == '__main__':
    models_to_compare = ['mobilenetV2']
    results = {}
    num_epochs_initial = 100  # 초기 학습 에포크
    # num_epochs_finetune = 100  # 파인튜닝 에포크

    # Step 1: 모든 backbone 모델을 초기 학습
    for model_name in models_to_compare:
        print(f"Training {model_name}")
        opt.model = model_name
        if opt.model=='resnet101':
            opt.batchSize = 16  # 배치 크기를 줄이기 (기존 값에서 절반으로 줄이기)
        loss, pos_err, ori_err = train_and_evaluate(opt, model_name, num_epochs_initial)
        results[model_name] = {'loss': loss, 'pos_err': pos_err, 'ori_err': ori_err}
    # 모델별 FLOPs와 파라미터 수 계산
    # complexity_results = {}
    # for model_name in models_to_compare:
    #     opt.model = model_name
    #     model = create_model(opt)
    #     flops, params = calculate_model_complexity(model)
    #     complexity_results[model_name] = {'flops': flops, 'params': params}

# 결과 출력
    # for model_name, complexity in complexity_results.items():
    #     print(f"{model_name} - FLOPs: {complexity['flops']:.2e}, Params: {complexity['params']:.2e}")

    # 초기 학습 결과 그래프 저장
    plot_results(results, 'kc_fianl_training_results.png')
    # 학습 곡선 시각화
    for model_name in models_to_compare:
        plot_learning_curve(results[model_name]['loss'], train_pos_err=results[model_name]['pos_err'], train_ori_err=results[model_name]['ori_err'], save_path=f'kc_{model_name}_learning_curve.png')
    
    # for model_name in top_3_models:
    #     print(f"Fine-tuning {model_name}")
    #     opt.model = model_name
    #     loss, pos_err, ori_err = train_and_evaluate(opt, model_name, num_epochs_finetune)

    #     # 결과를 추가하여 갱신
    #     results[model_name]['loss'].extend(loss)
    #     results[model_name]['pos_err'].extend(pos_err)
    #     results[model_name]['ori_err'].extend(ori_err)

    # # 최종 결과 그래프 저장 (상위 3개 모델만)
    
    
# 학습 및 평가 후 메모리 해제
free_gpu_memory()
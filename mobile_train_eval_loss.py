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

# 옵션 파싱
opt = TrainOptions().parse()

# 재현성을 위한 시드 설정
torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)
torch.backends.cudnn.deterministic = True

# 모델 학습 및 평가 함수
def train_and_evaluate(opt, num_epochs):
    model = create_model(opt)
    
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
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
        epoch_loss.append(float(model.loss_G))  # 손실 값 기록
        epoch_pos_err.append(float(errors['pos_err']))
        epoch_ori_err.append(float(errors['ori_err']))

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, num_epochs, time.time() - epoch_start_time))
        model.update_learning_rate()

    # 모델 저장
    model_save_path = os.path.join(opt.checkpoints_dir, f'{opt.loss}_mobilenetV2_final.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f'Model mobilenetV2 with {opt.loss} loss saved to {model_save_path}')

    return epoch_loss, epoch_pos_err, epoch_ori_err
# 이동 평균 함수 정의
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
# 결과를 시각화하고 저장하는 함수
def plot_results_smoothed(results, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for key in results:
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
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
        plt.savefig(f'{save_dir}/{key}_results.png')
        plt.close()

# 결과를 시각화하고 저장하는 함수
def plot_results(results, save_path):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    for key in results:
        epochs = range(1, len(results[key]['loss']) + 1)
        axs[0].plot(epochs, results[key]['loss'], label=key)
        axs[1].plot(epochs, results[key]['pos_err'], label=key)
        axs[2].plot(epochs, results[key]['ori_err'], label=key)

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
    opt.model = 'mobilenetV2'
    losses_to_compare = ['geometric', 'mae']
    results = {}
    num_epochs_initial = 100  # 초기 학습 에포크

    # for loss_function in losses_to_compare:
    #     print(f"Training mobilenetV2 with {loss_function} loss")
    #     opt.loss = loss_function
    #     loss, pos_err, ori_err = train_and_evaluate(opt, num_epochs_initial)
    #     results[loss_function] = {'loss': loss, 'pos_err': pos_err, 'ori_err': ori_err}
    opt.loss = 'mse'
    loss, pos_err, ori_err = train_and_evaluate(opt, num_epochs_initial)
    results['mse'] = {'loss': loss, 'pos_err': pos_err, 'ori_err': ori_err}
    # 결과 그래프 저장
    # plot_results(results, 'mobilenetV2_loss_comparison_results_mse_b100_lr0005.png')
    plot_results_smoothed(results, 'mobilenetV2_loss_comparison_results_smoothedddd')

# 결과를 시각화하고 저장하는 함수
def plot_results_separately(results, save_dir):
    # 디렉토리가 존재하지 않으면 생성
    os.makedirs(save_dir, exist_ok=True)
    
    for key in results:
        epochs = range(1, len(results[key]['loss']) + 1)
        
        # 손실 함수별 그래프 저장
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        
        axs[0].plot(epochs, results[key]['loss'], label=f'{key} loss')
        axs[0].set_title(f'{key.capitalize()} Loss vs. Epochs')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].legend()
        
        axs[1].plot(epochs, results[key]['pos_err'], label=f'{key} position error')
        axs[1].set_title(f'{key.capitalize()} Position Error vs. Epochs')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Position Error (meters)')
        axs[1].legend()
        
        axs[2].plot(epochs, results[key]['ori_err'], label=f'{key} orientation error')
        axs[2].set_title(f'{key.capitalize()} Orientation Error vs. Epochs')
        axs[2].set_xlabel('Epochs')
        axs[2].set_ylabel('Orientation Error (degrees)')
        axs[2].legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{key}_results.png')
        plt.close()

# 메인 스크립트
if __name__ == '__main__':
    opt.model = 'mobilenetV2'
    losses_to_compare = ['mse', 'geometric', 'mae']  # 손실 함수 목록
    results = {}
    num_epochs_initial = 100  # 초기 학습 에포크

    for loss_function in losses_to_compare:
        print(f"Training mobilenetV2 with {loss_function} loss")
        opt.loss = loss_function
        loss, pos_err, ori_err = train_and_evaluate(opt, num_epochs_initial)
        results[loss_function] = {'loss': loss, 'pos_err': pos_err, 'ori_err': ori_err}

    # 결과 그래프 저장
    plot_results_smoothed(results, f'mobilenetV2_loss_comparison_{loss_function}_results_000005_100')

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
# def train_and_evaluate(opt, num_epochs):
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
#     model_save_path = os.path.join(opt.checkpoints_dir, f'{opt.loss}_mobilenetV2_final.pth')
#     torch.save(model.state_dict(), model_save_path)
#     print(f'Model mobilenetV2 with {opt.loss} loss saved to {model_save_path}')

#     return epoch_loss, epoch_pos_err, epoch_ori_err

# # 결과를 시각화하고 저장하는 함수
# def plot_results(results, save_path):
#     fig, axs = plt.subplots(3, 1, figsize=(10, 15))
#     for key in results:
#         epochs = range(1, len(results[key]['loss']) + 1)
#         axs[0].plot(epochs, [l.cpu().item() for l in results[key]['loss']], label=key)
#         axs[1].plot(epochs, [p.cpu().item() for p in results[key]['pos_err']], label=key)
#         axs[2].plot(epochs, [o.cpu().item() for o in results[key]['ori_err']], label=key)

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
#     opt.model = 'mobilenetV2'
#     losses_to_compare = ['mse', 'geometric','uncertainty', 'mae']
#     results = {}
#     num_epochs_initial = 50  # 초기 학습 에포크

#     for loss_function in losses_to_compare:
#         print(f"Training mobilenetV2 with {loss_function} loss")
#         opt.loss = loss_function
#         loss, pos_err, ori_err = train_and_evaluate(opt, num_epochs_initial)
#         results[loss_function] = {'loss': loss, 'pos_err': pos_err, 'ori_err': ori_err}

#     # 결과 그래프 저장
#     plot_results(results, 'mobilenetV2_loss_comparison_results.png')

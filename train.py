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
def train_and_evaluate(opt, model_name, num_epochs):
    opt.model = model_name
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
        epoch_loss.append(model.loss_G.item())  # 손실 값 기록
        epoch_pos_err.append(errors['pos_err'])
        epoch_ori_err.append(errors['ori_err'])

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, num_epochs, time.time() - epoch_start_time))
        model.update_learning_rate()

    # 모델 저장
    model_save_path = os.path.join(opt.checkpoints_dir, f'{model_name}_final.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f'Model {model_name} saved to {model_save_path}')

    return epoch_loss, epoch_pos_err, epoch_ori_err

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
    models_to_compare = ['posenet', 'resnet50', 'resnet101', 'vgg16', 'vgg19', 'efficientnetB0', 'mobilenetV2']
    results = {}
    num_epochs_initial = 50  # 초기 학습 에포크
    num_epochs_finetune = 100  # 파인튜닝 에포크

    # Step 1: 모든 backbone 모델을 초기 학습
    for model_name in models_to_compare:
        print(f"Training {model_name}")
        opt.model = model_name
        loss, pos_err, ori_err = train_and_evaluate(opt, model_name, num_epochs_initial)
        results[model_name] = {'loss': loss, 'pos_err': pos_err, 'ori_err': ori_err}

    # 초기 학습 결과 그래프 저장
    plot_results(results, 'initial_training_results.png')

    # Step 2: 상위 4개 모델 선택 및 파인튜닝
    top_4_models = sorted(results.keys(), key=lambda x: results[x]['loss'][-1])[:4]

    for model_name in top_4_models:
        print(f"Fine-tuning {model_name}")
        opt.model = model_name
        loss, pos_err, ori_err = train_and_evaluate(opt, model_name, num_epochs_finetune)

        # 결과를 추가하여 갱신
        results[model_name]['loss'].extend(loss)
        results[model_name]['pos_err'].extend(pos_err)
        results[model_name]['ori_err'].extend(ori_err)

    # 최종 결과 그래프 저장
    plot_results(results, 'final_training_results.png')

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        """교수님 데이터셋 for test"""
        # self.parser.add_argument('--csv_file', type=str, default='/mnt/8TB_1/chaewon/sim/poselstm-pytorch/datasets/color/rene4_cpu_int_ext_rpy.csv', help='path to the CSV file containing the dataset')
        # self.parser.add_argument('--image_folder', type=str, default='/mnt/8TB_1/chaewon/sim/poselstm-pytorch/datasets/color', help='path to the folder containing the images')
        # self.parser.add_argument('--mean_image_path', type=str, default='/mnt/8TB_1/chaewon/sim/poselstm-pytorch/datasets/color/mean_image.npy', help='path to the mean image file')
        """우리 데이터셋 for train"""
        self.parser.add_argument('--csv_file', type=str, default='/mnt/8TB_1/chaewon/sim/poselstm-pytorch/datasets/deepl_final_6_18_v5.csv', help='path to the CSV file containing the dataset')
        self.parser.add_argument('--image_folder', type=str, default='/mnt/8TB_1/chaewon/sim/poselstm-pytorch/datasets/d/align_data', help='path to the folder containing the images')
        self.parser.add_argument('--mean_image_path', type=str, default='/mnt/8TB_1/chaewon/sim/poselstm-pytorch/datasets/d/align_data/mean_image.npy', help='path to the mean image file')
       
       
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=500, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--adambeta1', type=float, default=0.9, help='first momentum term of adam')
        self.parser.add_argument('--adambeta2', type=float, default=0.999, help='second momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--use_html', action='store_true', help='save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument('--init_weights', type=str, default='pretrained_models/places-googlenet.pickle', help='initiliaze network from, e.g., pretrained_models/places-googlenet.pickle')

        self.parser.add_argument('--loss', type=str, default='mse', help='loss function to use: mse | geometric | uncertainty ')
        
        self.isTrain = True

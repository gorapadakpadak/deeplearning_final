from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--csv_file', type=str, default='/mnt/8TB_1/chaewon/sim/poselstm-pytorch/datasets/color/rene4_cpu_int_ext_rpy.csv', help='path to the CSV file containing the dataset')
        self.parser.add_argument('--image_folder', type=str, default='/mnt/8TB_1/chaewon/sim/poselstm-pytorch/datasets/color', help='path to the folder containing the images')
        self.parser.add_argument('--mean_image_path', type=str, default='/mnt/8TB_1/chaewon/sim/poselstm-pytorch/datasets/color/mean_image.npy', help='path to the mean image file')
        self.isTrain = False

# import torch.utils.data
# from data.base_data_loader import BaseDataLoader


# def CreateDataset(opt):
#     dataset = None
#     if opt.dataset_mode == 'unaligned_posenet':
#         from data.unaligned_posenet_dataset import UnalignedPoseNetDataset
#         dataset = UnalignedPoseNetDataset()
#     else:
#         raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

#     print("dataset [%s] was created" % (dataset.name()))
#     dataset.initialize(opt)
#     return dataset


# class CustomDatasetDataLoader(BaseDataLoader):
#     def name(self):
#         return 'CustomDatasetDataLoader'

#     def initialize(self, opt):
#         BaseDataLoader.initialize(self, opt)
#         self.dataset = CreateDataset(opt)

#         def init_fn(worker_id):
#             torch.manual_seed(opt.seed)

#         self.dataloader = torch.utils.data.DataLoader(
#             self.dataset,
#             batch_size=opt.batchSize,
#             shuffle=not opt.serial_batches,
#             num_workers=int(opt.nThreads),
#             worker_init_fn=init_fn)

#     def load_data(self):
#         return self

#     def __len__(self):
#         return min(len(self.dataset), self.opt.max_dataset_size)

#     def __iter__(self):
#         for i, data in enumerate(self.dataloader):
#             if i >= self.opt.max_dataset_size:
#                 break
#             yield data

#ours
import torch.utils.data
from data.base_data_loader import BaseDataLoader
from data.unaligned_posenet_dataset import get_default_transform

def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'unaligned_posenet':
        from data.unaligned_posenet_dataset import UnalignedPoseNetDataset
        print("csv 이거다잉~~~~~~~~~~~~~",opt.csv_file)
        dataset = UnalignedPoseNetDataset(
            csv_file=opt.csv_file,  # 추가된 옵션
            image_folder=opt.image_folder,  # 추가된 옵션
            mean_image_path=opt.mean_image_path,  # 추가된 옵션
            transform=get_default_transform(opt.mean_image_path)  # 추가된 옵션
        )
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset was created" )
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)

        def init_fn(worker_id):
            torch.manual_seed(opt.seed)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads),
            worker_init_fn=init_fn)

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data


# import torch.utils.data
# from data.base_data_loader import BaseDataLoader

# def CreateDataset(opt):
#     dataset = None
#     if opt.dataset_mode == 'unaligned_posenet':
#         from data.unaligned_posenet_dataset import UnalignedPoseNetDataset
#         # print("######################",opt.csv_file,"###############################")
#         dataset = UnalignedPoseNetDataset(opt.csv_file, opt.image_folder, mean_image_path=opt.mean_image_path)
#     else:
#         raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

#     print("dataset [%s] was created" % (dataset.__class__.__name__))
#     return dataset

# class CustomDatasetDataLoader(BaseDataLoader):
#     def name(self):
#         return 'CustomDatasetDataLoader'

#     def initialize(self, opt):
#         BaseDataLoader.initialize(self, opt)
#         self.dataset = CreateDataset(opt)

#         def init_fn(worker_id):
#             torch.manual_seed(opt.seed)

#         self.dataloader = torch.utils.data.DataLoader(
#             self.dataset,
#             batch_size=opt.batchSize,
#             shuffle=not opt.serial_batches,
#             num_workers=int(opt.nThreads),
#             worker_init_fn=init_fn)

#     def load_data(self):
#         return self

#     def __len__(self):
#         return min(len(self.dataset), self.opt.max_dataset_size)

#     def __iter__(self):
#         for i, data in enumerate(self.dataloader):
#             if i >= self.opt.max_dataset_size:
#                 break
#             yield data

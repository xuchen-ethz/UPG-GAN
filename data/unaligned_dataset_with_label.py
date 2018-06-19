import os.path
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, make_dataset_label
from PIL import Image
import PIL
import random
import numpy

class UnalignedDatasetWithLabel(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        # New: make dataset
        self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')
        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)
        # New: make dataset
        self.C_paths = make_dataset_label(self.dir_C)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        # New: make dataset
        self.C_paths = sorted(self.C_paths)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        # New: make dataset
        self.C_size = len(self.C_paths)
        if self.B_size != self.C_size and self.opt.isTrain:
            raise ValueError("B and C should be paired and have the same length!")

        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        index_A = index % self.A_size
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # New: Use paired B and C
        C_path = self.C_paths[index_B]

        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # New: Use gray image as condition
        C_arr = numpy.loadtxt(C_path, dtype=numpy.float32)
        if  self.opt.cond_nc == 1:
            C_value = C_arr
            C_arr = numpy.ndarray((1,1,1), dtype=numpy.float32)
            C_arr[0] = C_value
        C_arr = numpy.reshape(C_arr, (C_arr.shape[0]))

        # C_arr = numpy.reshape(C_arr, (C_arr.shape[0],1,1))
        # C_arr = numpy.repeat(C_arr, self.opt.fineSize, axis=1)
        # C_arr = numpy.repeat(C_arr, self.opt.fineSize, axis=2)
        A = self.transform(A_img)
        B = self.transform(B_img)
        C = torch.from_numpy(C_arr)
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B, 'C': C,
                'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDatasetWithLabel'

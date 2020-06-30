from torch.utils.data import Dataset
import numpy as np
import os
import struct
from dataset import SmallNORBDataset as SNDataset
import torch
from PIL import Image
import argparse
from torchvision import transforms
import torchvision.utils as vutils

class SmallNORB(Dataset):

    def __init__(self, root_dir, train=True, transform=None):
        """
        Args:
            norb_root (string): path to the directory with the six different uncompressed data files
            train (bool): set up the train dataset (else test) (default: True)
            transform (callable, optional): Optional transform to be applied to a sample
            """
        self.root_dir =  root_dir
        self.transform  = transform
        if train:
            # training data files
            self.file_dat = os.path.join(root_dir, 'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat')
            self.file_cat =  os.path.join(root_dir, 'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat')
            self.file_info =  os.path.join(root_dir, 'smallnorb-5x46789x9x18x6x2x96x96-training-info.mat')
        else:
            #testing data files
            self.file_dat = os.path.join(root_dir, 'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat')
            self.file_cat =  os.path.join(root_dir, 'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat')
            self.file_info =  os.path.join(root_dir, 'smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat')

        #self._read_header_dat
        #with open(self.file_dat, mode='rb') as f:
        #    self.header_dat = SNDataset._parse_small_NORB_header(f)
        #self.header_cat =
        #self.header_info =
        self._fill_data_structures()


    def __len__(self):
        return len(self.dat_data) // 2

    def _fill_data_structures(self):
        """
        Fill SmallNORBDataset data structures for a certain `dataset_split`.

        This means all images, category and additional information are loaded from binary
        files of the current split.

        Parameters
        ----------
        dataset_split: str
            Dataset split, can be either 'train' or 'test'

        Returns
        -------
        None

        """
        self.dat_data  = SNDataset._parse_NORB_dat_file(self.file_dat)
        self.cat_data  = SNDataset._parse_NORB_cat_file(self.file_cat)
        self.info_data = SNDataset._parse_NORB_info_file(self.file_info)
        #for i, small_norb_example in enumerate(self.data[dataset_split]):
        #    small_norb_example.lighting  = info_data[i][3]

    def _read_data(self, idx):
        '''reads the data file at a given index'''
        raise NotImplementedError

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if not 0 <= idx < self.__len__():
            raise IndexError

        image_lt  = self.dat_data[2*idx]
        image_rt  = self.dat_data[2*idx+1]
        category  = self.cat_data[idx]
        instance, elevation, azimuth, lighting  = self.info_data[idx]
        # returns a PIL image
        tf = self.transform if self.transform is not None else lambda x: x
        sample = {'image_lt': tf(Image.fromarray(image_lt, mode='L')),
                    'image_rt': tf(Image.fromarray(image_rt, mode='L')),
                    'category': category,
                    'intance': instance,
                    'elevation': elevation,
                    'azimuth': azimuth,
                    'lighting': lighting
                    }

        return sample





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path')

    args = parser.parse_args()

    tf = transforms.Compose([
        #transforms.Resize(28),
        transforms.ToTensor()
    ])
    dataset= SmallNORB(args.path, transform=tf)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100)

    for i, batch in enumerate(dataloader):
        lt, rt = batch['image_lt'], batch['image_rt']
        vutils.save_image(lt, filename='test.png')

        pass



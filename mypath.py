import os

from torch.utils.data.dataloader import DataLoader

DATASET_ROOT = os.getenv('DATASET_ROOT', './datasets')
                                        
                                                 


class Path(object):
    @staticmethod
    def getPath(dataset):
        if dataset == 'brats3d-acn':
            # Please replace with your local path to BraTS dataset
            path = '/path/to/your/BraTS_2018_Data_Training/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError

        return path

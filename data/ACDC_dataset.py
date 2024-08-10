from torch.utils.data import Dataset
import data.util_3D as Util
import os
import numpy as np
import scipy.io as sio
import json
import SimpleITK as sitk
from glob import glob

class ACDCDataset(Dataset):
    def __init__(self, dataroot, split='train'):
        self.split = split
        self.imageNum = []
        self.dataroot = dataroot

        patient_dirs = sorted(glob(os.path.join(dataroot, '*.')))

        self.data_len = len(patient_dirs)
        self.fineSize = [128, 128, 32]

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        patient_dir = self.imageNum[index]
        patient_files = sorted(glob(patient_dir, "*"))
        ED_gt, ED_im, ES_gt, ES_im = sorted([file_ for file_ in patient_files if "frame" in file_])
        dataA = ED_im
        dataA=sitk.ReadImage(dataA)
        dataA=sitk.GetArrayFromImage(dataA).astype(np.float32).transpose(2,1,0)
        # print(dataA.shape)
        dataB = ES_im
        dataB=sitk.ReadImage(dataB)
        dataB=sitk.GetArrayFromImage(dataB).astype(np.float32).transpose(2,1,0)
        label_dataA = ED_gt
        label_dataA=sitk.ReadImage(label_dataA)
        label_dataA=sitk.GetArrayFromImage(label_dataA).transpose(2,1,0)
        label_dataB = ES_gt
        label_dataB=sitk.ReadImage(label_dataB)
        label_dataB=sitk.GetArrayFromImage(label_dataB).transpose(2,1,0)


        dataA -= dataA.min()
        dataA /= dataA.std()
        dataA -= dataA.min()
        dataA /= dataA.max()

        dataB -= dataB.min()
        dataB /= dataB.std()
        dataB -= dataB.min()
        dataB /= dataB.max()

        nh, nw, nd = dataA.shape
        # print(dataA.shape,dataB.shape)

        sh = int((nh - self.fineSize[0]) / 2)
        sw = int((nw - self.fineSize[1]) / 2)
        dataA = dataA[sh:sh + self.fineSize[0], sw:sw + self.fineSize[1]]
        dataB = dataB[sh:sh + self.fineSize[0], sw:sw + self.fineSize[1]]
        label_dataA = label_dataA[sh:sh + self.fineSize[0], sw:sw + self.fineSize[1]]
        label_dataB = label_dataB[sh:sh + self.fineSize[0], sw:sw + self.fineSize[1]]

        if nd >= 32:
            sd = int((nd - self.fineSize[2]) / 2)
            dataA = dataA[..., sd:sd + self.fineSize[2]]
            dataB = dataB[..., sd:sd + self.fineSize[2]]
            label_dataA = label_dataA[..., sd:sd + self.fineSize[2]]
            label_dataB = label_dataB[..., sd:sd + self.fineSize[2]]
        else:
            sd = int((self.fineSize[2] - nd) / 2)
            dataA_ = np.zeros(self.fineSize)
            dataB_ = np.zeros(self.fineSize)
            dataA_[:, :, sd:sd + nd] = dataA
            dataB_[:, :, sd:sd + nd] = dataB
            label_dataA_ = np.zeros(self.fineSize)
            label_dataB_ = np.zeros(self.fineSize)
            label_dataA_[:, :, sd:sd + nd] = label_dataA
            label_dataB_[:, :, sd:sd + nd] = label_dataB
            dataA, dataB = dataA_, dataB_
            label_dataA, label_dataB = label_dataA_, label_dataB_
        [data, label] = Util.transform_augment([dataA, dataB], split=self.split, min_max=(-1, 1))
        
        return {'M': data, 'F': label, 'MS': label_dataA, 'FS': label_dataB, 'Index': index, 'patient_dir': patient_dir}

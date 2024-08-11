import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from math import *
import time
import numpy as np
import torch.nn.functional as F
import h5py  # Import h5py for saving to HDF5 files
import ants

if __name__ == "__main__":
    data_root = "/local2/amvepa91/FSDiffReg/database/ACDC/training"
    out_dir = "/local2/amvepa91/FSDiffReg/ACDC/training_group_reg"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/.json',
                        help='JSON file for configuration')
    parser.add_argument('-w', '--weights', type=str, default='',
                        help='weights file for validation')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    phase = 'test'
    dataset_opt = opt['datasets']['test']
    test_set = Data.create_dataset_3D(dataset_opt, phase)
    test_loader = Data.create_dataloader(test_set, dataset_opt, phase)
    print('Dataset Initialized')

    print("Begin Group-wise Registration.")

    print(len(test_loader))
    # Initialize empty lists to store ED and ES images
    ed_images = []
    es_images = []
    ed_masks = []
    es_masks = []
    patient_dirs = []
    for istep, test_data in enumerate(test_loader):
        idx_ += 1
        dataName = istep
        time1 = time.time()
        patient_dir = os.path.basename(test_data["patient_dir"][0])
        patient_dirs.append(patient_dir)
        print("processing ", patient_dir)

        ed_images.append(ants.from_numpy(test_data['ED'].cpu().numpy()))
        es_images.append(ants.from_numpy(test_data['ES'].cpu().numpy()))
        ed_masks.append(ants.from_numpy(test_data['ED_mask'].cpu().numpy()))
        es_masks.append(ants.from_numpy(test_data['ES_mask'].cpu().numpy()))

        group_ed_img = ants.average_images([ed_images])
        group_es_img = ants.average_images([es_images])

        for ed_img,es_img,ed_mask,es_mask,patient_dir in zip(ed_images,es_images,ed_masks,es_masks,patient_dirs):

            # Define the HDF5 file path for the current patient
            patient_h5_path = os.path.join(out_dir, f'{patient_dir}.h5')

            # Open an HDF5 file to store the results for this patient
            with h5py.File(patient_h5_path, 'w') as h5f:
                # Perform group-wise registration for ED phase

                # Register ED image to the group-wise ED template
                reg_ed = ants.registration(fixed=group_ed_img, moving=ed_img, type_of_transform='SyN')

                # Apply the transformation to the ED image and mask
                warped_ed_img = ants.apply_transforms(fixed=group_ed_img, moving=ed_img,
                                                      transformlist=reg_ed['fwdtransforms'])
                warped_ed_mask = ants.apply_transforms(fixed=group_ed_img, moving=ed_mask,
                                                       transformlist=reg_ed['fwdtransforms'],
                                                       interpolator='nearestNeighbor')

                # Save the registered ED image and mask to the HDF5 file
                ed_group = h5f.create_group('ED')
                ed_group.create_dataset('reg_image_ed', data=warped_ed_img.view())
                ed_group.create_dataset('reg_scribble_ed', data=warped_ed_mask.view())


                # Register ES image to the group-wise ES template
                reg_es = ants.registration(fixed=group_es_img, moving=es_img, type_of_transform='SyN')

                # Apply the transformation to the ES image and mask
                warped_es_img = ants.apply_transforms(fixed=group_es_img, moving=es_img,
                                                      transformlist=reg_es['fwdtransforms'])
                warped_es_mask = ants.apply_transforms(fixed=group_es_img, moving=es_mask,
                                                       transformlist=reg_es['fwdtransforms'],
                                                       interpolator='nearestNeighbor')

                # Save the registered ES image and mask to the HDF5 file
                es_group = h5f.create_group('ES')
                es_group.create_dataset('reg_image_es', data=warped_es_img.view())
                es_group.create_dataset('reg_scribble_es', data=warped_es_mask.view())

            print(f"{patient_dir} data saved to {patient_h5_path}")

        print("All patients processed and saved to individual HDF5 files.")
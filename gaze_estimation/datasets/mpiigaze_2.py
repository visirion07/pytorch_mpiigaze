from typing import Callable, Tuple
import numpy as np
import pathlib
import cv2
import h5py
import torch
from torch.utils.data import Dataset


class OnePersonDataset(Dataset):
    def __init__(self, person_id_str: str, dataset_path: pathlib.Path, image_path: str,
                 transform: Callable):
        self.transform = transform

        # In case of the MPIIGaze dataset, each image is so small that
        # reading image will become a bottleneck even with HDF5.
        # So, first load them all into memory.
        # with h5py.File(dataset_path, 'r') as f:
        #     images = f.get(f'{person_id_str}/image')[()]
        #     poses = f.get(f'{person_id_str}/pose')[()]
        #     gazes = f.get(f'{person_id_str}/gaze')[()]
        # assert len(images) == 3000
        # assert len(poses) == 3000
        # assert len(gazes) == 3000
        person = str(dataset_path) + "/" + person_id_str + "z.npy"
        person_from_face = str(dataset_path) + "/" + person_id_str + "b.npy"
        print(person)
        print(image_path)
        day_lev_dic = {}
        images = []
        poses = []
        gazes = []
        gazes_from_face = []
        npfile = np.load(person, allow_pickle=True)
        npfile_from_face = np.load(person_from_face, allow_pickle=True)
        print(npfile.shape)
        print(npfile_from_face.shape)
        print(npfile_from_face[0])
        self.images = np.array([])
        self.poses = np.array([])
        self.gazes = np.array([])
        return
        # for row in npfile_from_face:
            
        for row in npfile:
            # img_name = row[-3]
            # # image_paath = image_path + img_name
            # image_paath = image_path  +  img_name.split('l')[1] + "_patch.jpg"
            # # print(image_paath)
            # img = cv2.imread(image_paath, 0)
            # img = cv2.equalizeHist(img)
            # print(img.shape, np.count_nonzero(img))
            images.append(row[-3])
            poses.append(row[-2])
            gazes.append(row[-1])
        self.images = np.array(images)
        self.poses = np.array(poses)
        self.gazes = np.array(gazes)
        print("images shape", self.images.shape)
        print("poses shape", self.poses.shape)
        print("gazes shape", self.gazes.shape)


    def __getitem__(self, index: int
                    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        image = self.transform(self.images[index])
        pose = torch.from_numpy(self.poses[index])
        gaze = torch.from_numpy(self.gazes[index])
        return image, pose, gaze

    def __len__(self) -> int:
        return len(self.images)

from typing import Callable, Tuple
import numpy as np
import pathlib
import cv2
import h5py
import torch
from torch.utils.data import Dataset


class OnePersonDataset(Dataset):
  def __init__(self, person_id_str: str, dataset_path: pathlib.Path, image_path: str,transform: Callable):

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
    person_f = str(dataset_path) + "/" + person_id_str + "b.npy"
    person_g = str(dataset_path) + "/" + person_id_str + "g.npy" 
    person_fg = str(dataset_path) + "/" + person_id_str + "m.npy" 

    print(person)
    print(image_path)
    day_lev_dic = {}

    images = []
    poses = []
    gazes = []
    gazes_f = {}
    gazes_z = {}
    gazes_o = {}

    rs = {}
    rs_f = {}

    gaze_O = {}

    R = {} 

    add1  = []
    add2 = []

    npfile = np.load(person, allow_pickle=True)
    npfile_f = np.load(person_f, allow_pickle=True)
    npfile_g = np.load(person_g, allow_pickle=True)
    npfile_fg = np.load(person_fg, allow_pickle=True)

    print("HH", npfile[0][2].shape)
    print("HH", npfile_f[0][4])
    print("HH", npfile_g[0])
    print("HH", npfile_fg[0][0])


    self.images = np.array([])
    self.poses = np.array([])
    self.gazes = np.array([])
    self.add1 = np.array([])
    self.add2 = np.array([])

    for row in npfile_g:
      # print("GG", row[0].shape, row[1].shape)
      img_s = row[-1].split('l')[1].split('/')
      
      day_ = img_s[1]
      img_ = img_s[2]

      if(day_ not in gazes_o.keys()):
        gazes_o[day_] = {}
      gazes_o[day_][img_] = row[0]

      if(day_ not in rs.keys()):
        rs[day_]= {}
      rs[day_][img_] = row[1]

      if(day_ not in gaze_O.keys()):
        gaze_O[day_] = {}
      gaze_O[day_][img_] = -row[0] 

    

    for row in npfile_f:

      # print("GG", row[-1].shape, row[-2].shape, row[2].reshape(6).shape, row[4])

      img_s = row[4].split('l')[1].split('/')
      day_ = img_s[1]
      img_ = img_s[2]
      if(day_ not in gazes_f.keys()):
        gazes_f[day_] = {}
      gazes_f[day_][img_] = row[-1]

      if(day_ not in rs_f.keys()):
        rs_f[day_] = {}
      rs_f[day_][img_] = np.array(row[-2])

      if(day_ not in gazes_z.keys()):
        gazes_z[day_] = {}
      gazes_z[day_][img_] = row[2].reshape(6)



      if(day_ not in gaze_O.keys()):
        gaze_O[day_] = {}
      gaze_O[day_][img_] += row[-1]
      gaze_O[day_][img_] = np.multiply(rs[day_][img_], gaze_O[day_][img_])


    for day_ in rs.keys():
      for img_ in rs[day_].keys():
        rp = rs_f[day_][img_]
        rp_norm = np.linalg.norm(rp)
        if(abs(rp_norm) < 1e-6 ):
          print("GG", img_, day_)
        else:
          rp_inv = np.linalg.inv(rp)
          rpp = rp_norm * np.multiply(rs[day_][img_], rp_inv)

          if(day_ not in R.keys()):
            R[day_] = {}
          R[day_][img_] = rpp 

      
    for row in npfile:
      day_ =  row[1]
      img_ = row[0]
      # print("GG", row[2].shape, day_, img_)
      # img = cv2.cvtColor(row[2],cv2.COLOR_BGR2GRAY)
      img = cv2.equalizeHist(row[2])
      if((day_ not in R.keys()) or (img_ not in R[day_].keys())):
        continue
      images.append(img)
      gazes.append(row[-1])
      poses.append(row[-2])
      add1.append(gaze_O[day_][img_])
      add2.append(R[day_][img_])
    

    self.images = np.array(images)
    self.poses = np.array(poses)
    self.gazes = np.array(gazes)
    self.add1 = np.array(add1)
    self.add2 = np.array(add2)
    print("images shape", self.images.shape)
    print("poses shape", self.poses.shape)
    print("gazes shape", self.gazes.shape)
    print("gazesO shape", self.add1.shape)
    print("R shape", self.add2.shape)

    return


















      # for row in npfile_from_face:




      # for row in npfile:
      # img_name = row[-3]
      # # image_paath = image_path + img_name
      # image_paath = image_path  +  img_name.split('l')[1] + "_patch.jpg"
      # # print(image_paath)
      # img = cv2.imread(image_paath, 0)
      # img = cv2.equalizeHist(img)
      # print(img.shape, np.count_nonzero(img))
        # images.append(row[-3])
        # poses.append(row[-2])
        # gazes.append(row[-1])

    


    # return





  def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    image = self.transform(self.images[index])
    pose = torch.from_numpy(self.poses[index])
    gaze = torch.from_numpy(self.gazes[index])
    return image, pose, gaze

  def __len__(self) -> int:
    return len(self.images)

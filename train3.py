import os,sys
import cv2 as cv
import h5py
import numpy as np

import time
import cv2
import numpy as np
from os import path
from subprocess import call
import pickle
import torch


import warnings
warnings.filterwarnings("ignore")


import cv2
import numpy as np
import random
import threading
import pickle
import sys
import cv2
import numpy as np
import csv
import scipy.io as sio
import torch
sys.path.append("../src")
from losses import GazeAngularLoss


def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
    ret, rvec, tvec = cv2.solvePnP(face_model[:4], landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)

    ## further optimize
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(face_model[:4], landmarks, camera, distortion, rvec, tvec, True)

    return rvec, tvec

def normalizeData(img, face, hr, ht, cam):
    ## normalized camera parameters
    focal_norm = 960 # focal length of normalized camera
    distance_norm = 600 # normalized distance between eye and camera
    roiSize = (60, 36) # size of cropped eye image

    img_u = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3,1))
    
    hR = cv2.Rodrigues(hr)[0] # rotation matrix
    # print("HR SHAPE", hR.shape)
    # print("FACE SHAPE", face.shape)
    Fc = np.dot(hR, face) + ht # 3D positions of facial landmarks
    # print("FC SHAPE", Fc.shape)
    re = 0.5*(Fc[:,0] + Fc[:,1]).reshape((3,1)) # center of left eye
    le = 0.5*(Fc[:,2] + Fc[:,3]).reshape((3,1)) # center of right eye
    et = re
    distance = np.linalg.norm(et) # actual distance between eye and original camera
        
    z_scale = distance_norm/distance
    cam_norm = np.array([
        [focal_norm, 0, roiSize[0]/2],
        [0, focal_norm, roiSize[1]/2],
        [0, 0, 1.0],
    ])
    S = np.array([ # scaling matrix
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])
    
    hRx = hR[:,0]
    forward = (et/distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T # rotation matrix R
    ## normalize each eye
    
        
    return [re, R]
def process(img, camera_matrix, distortion, annots, curr_img, store_path, por_available=False, show=False):
# process(data, camera_matrix, distortion, annots[pic], curr_img, por_available=True, show=True)
        
        
        face = sio.loadmat("/content/MPIIGaze/6 points-based face model.mat")["model"]
        num_pts = face.shape[1]
        facePts = face.T.reshape(num_pts, 1, 3)
        img_u = cv2.undistort(img, camera_matrix, distortion)
     
        
        fx, _, cx, _, fy, cy, _, _, _ = camera_matrix.flatten()
        camera_parameters = np.asarray([fx, fy, cx, cy])
        # rvec, tvec = self.head_pose_estimator.fit_func(pts, camera_parameters)

        s = [int(x) for x in annots[:24]]
        landmarks = np.array([[s[0], s[1]], [s[6], s[7]], [s[12], s[13]], [s[18], s[19]]])
        landmarks = landmarks.astype(np.float32)
        landmarks = landmarks.reshape(4, 1, 2)

        hr, ht = estimateHeadPose(landmarks, facePts, camera_matrix, distortion)
        g_t = gt = np.array(annots[27:30]).reshape(3, 1)
        data = normalizeData(img, face, hr, ht, camera_matrix)
        
        returnv = [data[0], data[1], curr_img]




        

        return returnv

# directions = ['l', 'r', 'u', 'd']
# keys = {'u': 82,
#         'd': 84,
#         'l': 81,
#         'r': 83}






store_path = "/content/Processed/"
os.system("mkdir " + store_path)
path_original = "/content/MPIIGaze/Data/Original"
to_write = {}
output_path = "/content/Drive/MyDrive/MPIIGaze.h5"
def add(key, value):  # noqa
    if key not in to_write:
        to_write[key] = [value]
    else:
        to_write[key].append(value)
x1 = []
x2 = []
y = []
num_k = 0
path_original = "/content/MPIIGaze/Data/Original"
to_write = {}

x1 = []
x2 = []
y = []

for person in os.listdir(path_original):
    os.system("mkdir "+ store_path +person+"/" )
    num = int(person[1:])
    curr_person_path = os.path.join(path_original, person)
    intense_arr = []
    print("Processing person ", person)
    curr_path = os.path.join(curr_person_path, "Calibration")
    cameraCalib = sio.loadmat(os.path.join(curr_path, "Camera.mat"))
    camera_matrix = cameraCalib['cameraMatrix']
    distortion = cameraCalib['distCoeffs']
    final_input_dict = []
    for day in os.listdir(curr_person_path):
        if(day=="Calibration"):
            continue
        else:
            print("Processing Person and day", person + "/" + day)
            os.system("mkdir "+ store_path + person + "/"+day+"/")
            curr_path = os.path.join(curr_person_path, day)
            annotaion_file_path = os.path.join(curr_path, "annotation.txt")
            filea = open(annotaion_file_path, 'r')
            Lines = filea.readlines()
            annots = []
            for line in Lines:
                annots.append(np.array([float(x) for x in line.split(' ')]))
            for img in os.listdir(curr_path):

                if(img=="annotation.txt"):
                # print(len(annots))
                    continue
                else:
                    curr_img = path_original + person + "/"+day+"/" + img 
                    stpath = store_path + person + "/"+day+"/" + img 
                    pic = int(img.split('.')[0])-1
                    # print(curr_img, pic)
                    data = cv2.imread(os.path.join(curr_path, img), cv2.COLOR_BGR2RGB)
                    # print(data.shape)
                    ret_v = process(data, camera_matrix, distortion, annots[pic], curr_img, stpath, por_available=True, show=True)
                    final_input_dict.append(np.array([ret_v[0], ret_v[1], ret_v[2]]))
        

    final_ip = np.array(final_input_dict)
    intense_arr = np.array(intense_arr)
    np.save("/content/Drive/MyDrive/"+ person + "g", final_ip)   
    print("Saved output for person ", person)
    print("Mean in intensity ", np.mean(intense_arr))
    print("Variance in intensity", np.var(intense_arr))    
#             break1
#     break

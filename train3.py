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















import os,sys
import cv2 as cv
import h5py
import numpy as np
import scipy.io
import time
import cv2
import numpy as np
from os import path
from subprocess import call
import pickle
import torch


import warnings
warnings.filterwarnings("ignore")

from camera import cam_calibrate
from person_calibration import collect_data, fine_tune
from frame_processor import frame_processer
import cv2
import numpy as np
import random
import threading
import pickle
import sys

import torch
sys.path.append("../src")
from losses import GazeAngularLoss



def process(gaze_network, img, camera_matrix, distortion, annots, curr_img, store_path, por_available=False, show=False):
# process(data, camera_matrix, distortion, annots[pic], curr_img, por_available=True, show=True)
        g_t = None
        # data = {'image_a': [], 'gaze_a': [], 'head_a': [], 'R_gaze_a': [], 'R_head_a': []}
        
        face = scipy.io.loadmat("/content/MPIIGaze/6 points-based face model.mat")["model"]
        num_pts = face.shape[1]
        facePts = face.T.reshape(num_pts, 1, 3)

#         img = self.undistorter.apply(img)
        img = cv2.undistort(img, camera_matrix, distortion)
        img_u = img
        # if por_available:
        #     g_t = targets[frames_read]
        # frames_read += 1

        # detect face
        #face_location = face.detect(img,  scale=0.25, use_max='SIZE')

        # if (len(face_location) == 0):
        #   return []
        # use kalman filter to smooth bounding box position
        # assume work with complex numbers:
        # print("number of face location", len(face_location))
        # output_tracked = self.kalman_filters[0].update(face_location[0] + 1j * face_location[1])
        # face_location[0], face_location[1] = np.real(output_tracked), np.imag(output_tracked)
        # output_tracked = self.kalman_filters[1].update(face_location[2] + 1j * face_location[3])
        # face_location[2], face_location[3] = np.real(output_tracked), np.imag(output_tracked)

        # # detect facial points
        # pts = self.landmarks_detector.detect(face_location, img)
        # # run Kalman filter on landmarks to smooth them
        # for i in range(68):
        #     kalman_filters_landm_complex = self.kalman_filters_landm[i].update(pts[i, 0] + 1j * pts[i, 1])
        #     pts[i, 0], pts[i, 1] = np.real(kalman_filters_landm_complex), np.imag(kalman_filters_landm_complex)

        # compute head pose
        def vector_to_pitchyaw(vectors):
    # """Convert given gaze vectors to yaw (theta) and pitch (phi) angles."""
            n = vectors.shape[0]
            out = np.empty((n, 2))
            vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
            out[:, 0] = np.arcsin(vectors[:, 1])  # theta
            out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
            return out
        
        fx, _, cx, _, fy, cy, _, _, _ = camera_matrix.flatten()
        camera_parameters = np.asarray([fx, fy, cx, cy])
        # rvec, tvec = self.head_pose_estimator.fit_func(pts, camera_parameters)
        





        #head pose detection

        s = [int(x) for x in annots[:24]]
        landmarks = np.array([[s[0], s[1]], [s[6], s[7]], [s[12], s[13]], [s[18], s[19]]])
        landmarks = landmarks.astype(np.float32)
        landmarks = landmarks.reshape(4, 1, 2)

        rotm = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64) 
        facePtsr = np.matmul(facePts.reshape(-1,3), rotm)
#         ret, rvec, tvec = cv2.solvePnP(facePts[:4], landmarks, camera_matrix, distortion, flags=cv2.SOLVEPNP_EPNP)

#         ## further optimize
#         ret, rvec, tvec = cv2.solvePnP(facePts[:4], landmarks, camera_matrix, distortion, rvec, tvec, True)
        suc, rvec, tvec, inliers = cv2.solvePnPRansac(facePtsr[:4], landmarks, camera_matrix, distortion, flags = cv2.SOLVEPNP_EPNP)
        success, rvec, tvec = cv2.solvePnP(facePtsr[:4], landmarks, camera_matrix, distortion,
                                           rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        # rvec = annotations[30:33].reshape(3, 1)
        # tvec = annotations[33:36].reshape(3, 1)
        ######### GAZE PART #########

        # create normalized eye patch and gaze and head pose value,
        # if the ground truth point of regard is given
        head_pose = (rvec, tvec)
        # por = None
        # if por_available:
        #     por = np.zeros((3, 1))
        #     por[0] = g_t[0]
        #     por[1] = g_t[1]
        
        
        #gaze target
        g_t = gt = np.array(annots[27:30]).reshape(3, 1)






        entry = {
                'full_frame': img,
                '3d_gaze_target': g_t,
                'camera_parameters': camera_parameters,
                'full_frame_size': (img.shape[0], img.shape[1]),
                # 'face_bounding_box': (int(face_location[0]), int(face_location[1]),
                #                       int(face_location[2] - face_location[0]),
                #                       int(face_location[3] - face_location[1]))
                }

        

        #normalization process begins


        #setting crop sizes and focal lengths for camera normalization
        focal_norm = 1300
        distance_norm = 600
        roiSize = (256, 64)

        #getting rotation matrix and getting R, S matrixes

        hR = cv2.Rodrigues(rvec)[0]
        # Fc = np.dot(hR, face) + tvec
        
        # #setting the origin of the shot
        # g_o = et = 0.25*(Fc[:,0] + Fc[:,1]+Fc[:, 2]+Fc[:, 3]).reshape((3,1))


        Fc = np.dot(hR, facePtsr.T).T + tvec.T

        # print("FC SHAPE IS", Fc.shape)
        # et = 0.25*(Fc[:,0] + Fc[:,1]+Fc[:, 2]+Fc[:, 3]).reshape((3,1))
        # et = 0.5*(Fc[:,0] + Fc[:,1]).reshape((3,1))
        et = np.mean(Fc[1:3], axis =0)
        g_o = et = et.reshape(3,1)

        
#         g_o = et = 0.5*(Fc[:, 1] + Fc[:, 1]).reshape((3,1)) - et 
        distance = np.linalg.norm(et)
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


        #doing the actual transformations

        hRx = hR[:,0]
        forward = (et/distance).reshape(3)
        down = np.cross(forward, hRx)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        R = np.c_[right, down, forward].T

        # W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(camera_matrix)))
        # # print("Original image shape", img_u.shape)
        # img_warped = cv2.warpPerspective(img_u, W, roiSize)
        # # print("Warped image shape", img_warped.shape)
        # lamb = np.count_nonzero(img_warped)
        # if(lamb==0):
            # print("Hey ", curr_img)
        # print("Number of non_zero pixels in img_earped", lamb)
        # cv2.imwrite(curr_img + "_patch.jpg", img_warped)
        # print("saving patch")

        #correcting head pose
        # R = np.asmatrix(R)
        # head_mat = R * hR
        # n_h = np.array([np.arcsin(head_mat[1, 2]), np.arctan2(head_mat[0, 2], head_mat[2, 2])])
        # g = g_t - g_o
        # g /= np.linalg.norm(g)

        # n_g = R * g
        # n_g /= np.linalg.norm(n_g)
        # n_g = vector_to_pitchyaw(-n_g.T).flatten()


        # patch = img_warped
        # g_n = n_g
        # h_n = n_h
        # inverse_M = np.transpose(R)
        # gaze_cam_origin = g_o
        # gaze_cam_target = g_t

        # # [patch, h_n, g_n, inverse_M, gaze_cam_origin, gaze_cam_target] = normalize(entry, head_pose, curr_img)
        # # cv2.imshow('raw patch', patch)
        
        # def preprocess_image(image, store_path):
        #     ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        #     ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        #     # ycrcb[:, :, 0] = 50
        #     intensity = np.mean(ycrcb[:, :, 0])
        #     image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        #     cv2.imwrite(store_path, image)

        #     image = np.transpose(image, [2, 0, 1])  # CxHxW
        #     image = 2.0 * image / 255.0 - 1
        #     return image, intensity

        # # estimate the PoR using the gaze network
        # processed_patch, intensity = preprocess_image(patch, store_path)
        # processed_patch = processed_patch[np.newaxis, :, :, :]
        # # print(patch.shape)
        # # Functions to calculate relative rotation matrices for gaze dir. and head pose
        # def R_x(theta):
        #     sin_ = np.sin(theta)
        #     cos_ = np.cos(theta)
        #     return np.array([
        #         [1., 0., 0.],
        #         [0., cos_, -sin_],
        #         [0., sin_, cos_]
        #     ]).astype(np.float32)

        # def R_y(phi):
        #     sin_ = np.sin(phi)
        #     cos_ = np.cos(phi)
        #     return np.array([
        #         [cos_, 0., sin_],
        #         [0., 1., 0.],
        #         [-sin_, 0., cos_]
        #     ]).astype(np.float32)

        # def calculate_rotation_matrix(e):
        #     return np.matmul(R_y(e[1]), R_x(e[0]))

        # def pitchyaw_to_vector(pitchyaw):

        #     vector = np.zeros((3, 1))
        #     vector[0, 0] = np.cos(pitchyaw[0]) * np.sin(pitchyaw[1])
        #     vector[1, 0] = np.sin(pitchyaw[0])
        #     vector[2, 0] = np.cos(pitchyaw[0]) * np.cos(pitchyaw[1])
        #     return vector

        # # compute the ground truth POR if the
        # # ground truth is available
        # R_head_a = calculate_rotation_matrix(h_n)
        # R_gaze_a = np.zeros((1, 3, 3))
        # if type(g_n) is np.ndarray:
        #     R_gaze_a = calculate_rotation_matrix(g_n)

        #     # verify that g_n can be transformed back
        #     # to the screen's pixel location shown
        #     # during calibration
        #     gaze_n_vector = pitchyaw_to_vector(g_n)
        #     gaze_n_forward = -gaze_n_vector
        #     g_cam_forward = inverse_M * gaze_n_forward

            # compute the POR on z=0 plane
            #d = -gaze_cam_origin[2] / g_cam_forward[2]
            #por_cam_x = gaze_cam_origin[0] + d * g_cam_forward[0]
            #por_cam_y = gaze_cam_origin[1] + d * g_cam_forward[1]
            #por_cam_z = 0.0

            #x_pixel_gt, y_pixel_gt = mon.camera_to_monitor(por_cam_x, por_cam_y)
            # verified for correctness of calibration targets
        
        # if por_available:
        #     data['image_a'].append(processed_patch)
        #     data['gaze_a'].append(g_n)
        #     data['head_a'].append(h_n)
        #     data['R_gaze_a'].append(R_gaze_a)
        #     data['R_head_a'].append(R_head_a)

        # if show:

            # compute eye gaze and point of regard
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # output_dict = gaze_network(input_dict)
        # output_dict = dict([(k, v.cpu().detach().numpy()) for k, v in output_dict.items()])

        # Process output line by line
        # print(input_dict['name'])
        returnv = [
            curr_img, 
            R, 
            g_o
        ]

            # g_cnn = output.data.cpu().numpy()
            # g_cnn = g_cnn.reshape(3, 1)
            # g_cnn /= np.linalg.norm(g_cnn)

            # # compute the POR on z=0 plane
            # g_n_forward = -g_cnn
            # g_cam_forward = inverse_M * g_n_forward
            # g_cam_forward = g_cam_forward / np.linalg.norm(g_cam_forward)

            # d = -gaze_cam_origin[2] / g_cam_forward[2]
            # por_cam_x = gaze_cam_origin[0] + d * g_cam_forward[0]
            # por_cam_y = gaze_cam_origin[1] + d * g_cam_forward[1]
            # por_cam_z = 0.0

            # x_pixel_hat, y_pixel_hat = mon.camera_to_monitor(por_cam_x, por_cam_y)

            # output_tracked = self.kalman_filter_gaze[0].update(x_pixel_hat + 1j * y_pixel_hat)
            # x_pixel_hat, y_pixel_hat = np.ceil(np.real(output_tracked)), np.ceil(np.imag(output_tracked))

            # # show point of regard on screen
            # display = np.ones((mon.h_pixels, mon.w_pixels, 3), np.float32)
            # h, w, c = patch.shape
            # display[0:h, int(mon.w_pixels/2 - w/2):int(mon.w_pixels/2 + w/2), :] = 1.0 * patch / 255.0
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # if type(g_n) is np.ndarray:
            #     cv2.putText(display, '.', (, y_pixel_gt), font, 0.5, (0, 0, 0), 10, cv2.LINE_AA)
            # cv2.putText(display, '.', (int(x_pixel_hat), int(y_pixel_hat)), font, 0.5, (0, 0, 255), 10, cv2.LINE_AA)
            # cv2.namedWindow("por", cv2.WINDOW_NORMAL)
            # cv2.setWindowProperty("por", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # cv2.imshow('por', display)

            # # also show the face:
            # cv2.rectangle(img, (int(face_location[0]), int(face_location[1])),
            #               (int(face_location[2]), int(face_location[3])), (255, 0, 0), 2)
            # self.landmarks_detector.plot_markers(img, pts)
            # self.head_pose_estimator.drawPose(img, rvec, tvec, self.cam_calib['mtx'], np.zeros((1, 4)))
            # cv2.imshow('image', img)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     cv2.destroyAllWindows()
            #     cap.release()
            #     break

        # read the next frame

        return returnv

directions = ['l', 'r', 'u', 'd']
keys = {'u': 82,
        'd': 84,
        'l': 81,
        'r': 83}







ted_parameters_path = 'demo_weights/weights_ted.pth.tar'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sys.path.append("../src")
from models import DTED
gaze_network = DTED(
    growth_rate=32,
    z_dim_app=64,
    z_dim_gaze=2,
    z_dim_head=16,
    decoder_input_c=32,
    normalize_3d_codes=True,
    normalize_3d_codes_axis=1,
    backprop_gaze_to_encoder=False,
).to(device)

ted_weights = torch.load(ted_parameters_path)
if torch.cuda.device_count() == 1:
    if next(iter(ted_weights.keys())).startswith('module.'):
        ted_weights = dict([(k[7:], v) for k, v in ted_weights.items()])
  
gaze_network.load_state_dict(ted_weights)

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
    if(num not in [5, 8, 9]):
        continue
    else:
        print(person)
    #     continue
    curr_person_path = os.path.join(path_original, person)
    intense_arr = []
    print("Processing person ", person)
    curr_path = os.path.join(curr_person_path, "Calibration")
    cameraCalib = scipy.io.loadmat(os.path.join(curr_path, "Camera.mat"))
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
                    ret_v = process(gaze_network, data, camera_matrix, distortion, annots[pic], curr_img, stpath, por_available=True, show=True)
                    final_input_dict.append(np.array([ret_v[0], ret_v[1], ret_v[2]]))
                    # intense_arr.append(ret_v[8])
#                     continue
#                     x1.append(ret_v[2])
#                     x2.append(ret_v[3])   
#                     y.append(ret_v[0])
#                     imr = ret_v[-1]
#                     print(imr.shape)
                        # print(x1, x2, y)
#                     break1


            # processed_entry = data_normalization_entry(os.path.join(curr_path, img), camera_matrix, distortion, annots[pic], pic, day, person)
            # add('pixels', processed_entry['patch'])
            # add('labels', np.concatenate([
            #     processed_entry['normalized_gaze_direction'],
            #     processed_entry['normalized_head_pose'],
            # ]))
            # pic += 1
            # # print()
            # if(pic%50==1 and day=="day25" and person=="p08"): 
            #   cv.imwrite("patch"+str(pic)+"_"+ day + "_"+ person + ".jpg",processed_entry['patch'])
            #   print(os.path.join(curr_path, img), pic)
    final_ip = np.array(final_input_dict)
    intense_arr = np.array(intense_arr)
    np.save("/content/Drive/MyDrive/"+ person + "m", final_ip)   
    print("Saved output for person ", person)
    # print("Mean in intensity ", np.mean(intense_arr))
    # print("Variance in intensity", np.var(intense_arr))    
#             break1
#     break


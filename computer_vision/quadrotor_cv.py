import cv2 as cv
import numpy as np
import panda3d
import time
import math
from collections import deque
from scipy.spatial.transform import Rotation as R
from tabulate import tabulate

class computer_vision():
    def __init__(self, render, quad_model, quad_env, quad_sens, quad_pos, mydir, IMG_POS_DETER):
        self.IMG_POS_DETER = IMG_POS_DETER
        self.mydir = mydir
        self.quad_model = quad_model
        self.quad_env = quad_env
        self.quad_sens = quad_sens
        self.quad_pos = quad_pos
        self.image_pos = None
        self.vel_sens = deque(maxlen=100)
        self.vel_img = deque(maxlen=100)
        self.calibrated = False
        self.render = render  
        # Load the checkerboard actor
        self.render.checker = self.render.loader.loadModel(self.mydir + '/models/checkerboard.egg')
        self.render.checker.reparentTo(self.render.render)
        self.checker_scale = 0.5
        self.checker_sqr_size = 0.2046
        self.render.checker.setScale(self.checker_scale, self.checker_scale, 1)
        self.render.checker.setPos(3*self.checker_scale*self.checker_sqr_size+0.06, 2.5*self.checker_scale*self.checker_sqr_size+0.06, 0.001)
        self.render.taskMgr.add(self.calibrate, 'Camera Calibration')
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.render.taskMgr.add(self.pos_deter, 'Position Determination')   

        window_size = (self.render.win.getXSize(), self.render.win.getYSize())     
        self.render.buffer = self.render.win.makeTextureBuffer('Buffer', *window_size, None, True)
        self.render.cam_1 = self.render.makeCamera(self.render.buffer)
        self.render.cam_1.setName('cam_1')     
        self.render.cam_1.node().getLens().setFilmSize(36, 24)
        self.render.cam_1.node().getLens().setFocalLength(45)
        self.render.cam_1.reparentTo(self.render.render)
        self.render.cam_1.setPos(0, 0, 6.1)
        self.render.cam_1.setHpr(0, 270, 0)

        self.render.buffer2 = self.render.win.makeTextureBuffer('Buffer2', *window_size, None, True)
        self.render.cam_2 = self.render.makeCamera(self.render.buffer2)
        self.render.cam_2.setName('cam_2')     
        self.render.cam_2.node().getLens().setFilmSize(36, 24)
        self.render.cam_2.node().getLens().setFocalLength(45)
        self.render.cam_2.reparentTo(self.render.render)
        self.render.cam_2.setPos(0.1, 0, 6.1)
        self.render.cam_2.setHpr(0, 270, 0)

            
    def calibrate(self, task):
            if task.frame == 0:
                self.fast = cv.FastFeatureDetector_create()
                self.fast.setThreshold(20)
                self.criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 1, 0.0001) 
                self.nCornersCols = 9
                self.nCornersRows = 6
                self.objp = np.zeros((self.nCornersCols*self.nCornersRows, 3), np.float32)
                self.objp[:,:2] = (np.mgrid[0:self.nCornersCols, 0:self.nCornersRows].T.reshape(-1,2))*self.checker_scale*self.checker_sqr_size
                try: 
                    npzfile = np.load('./config/camera_calibration.npz')
                    self.mtx = npzfile[npzfile.files[0]]
                    self.dist = npzfile[npzfile.files[1]]
                    self.calibrated = True
                    print('Calibration File Loaded')
                    return task.done
                except:
                    print('Could Not Load Calibration File, Calibrating... ')
                    self.calibrated = False
                    self.quad_model.setPos(10,10,10)
                    self.render.cam_pos = []
                    self.objpoints = []
                    self.imgpoints = []
                    
            rand_pos = (np.random.random(3)-0.5)*5
            rand_pos[2] = np.random.random()*3+2
            cam_pos = tuple(rand_pos)
            self.render.cam.reparentTo(self.render.render)
            self.render.cam_1.reparentTo(self.render.render)
            self.render.cam.setPos(*cam_pos)
            self.render.cam.lookAt(self.render.checker)
            self.render.cam_1.setPos(*cam_pos)
            self.render.cam_1.lookAt(self.render.checker)
            ret, image = self.get_image(self.render.buffer)
            if ret:
                img = cv.cvtColor(image, cv.COLOR_RGBA2BGR)
                self.gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)                
                ret, corners = cv.findChessboardCorners(self.gray, (self.nCornersCols, self.nCornersRows), 
                                                        cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FILTER_QUADS+ cv.CALIB_CB_FAST_CHECK)
                if ret:
                    # corners = cv.cornerSubPix(self.gray,corners,(11,11),(-1,-1),self.criteria)
                    self.objpoints.append(self.objp)             
                    self.imgpoints.append(corners)
                    img = cv.drawChessboardCorners(img, (self.nCornersCols, self.nCornersRows), corners, ret)
                    cv.imshow('img',img)
    
            if len(self.objpoints) > 50:
                ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints, self.gray.shape[::-1], None, None)
                if ret:
                    h,  w = img.shape[:2]
                    newcameramtx, roi=cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
                    dist = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]) #Camera Perfeita (Simulada), logo não há distorção
                    dst = cv.undistort(img, mtx, dist, None, newcameramtx)    
                    cv.imshow('img', dst)
                    self.mtx = mtx
                    self.dist = dist
                    print('Calibration Complete')
                    self.calibrated = True
                    np.savez('./config/camera_calibration', mtx, dist)
                    print('Calibration File Saved')
                    self.render.cam.reparentTo(self.render.render)
                    self.render.cam.setPos(self.render.cam_neutral_pos)
                    self.render.cam_1.reparentTo(self.quad_model)
                    self.render.cam_1.setPos(0,0,0.01)
                    return task.done                
                else:
                    return task.cont
            else:
                return task.cont  
        
    def get_image(self, buffer):
        tex = buffer.getTexture()  
        img = tex.getRamImage()
        image = np.frombuffer(img, np.uint8)
        
        if len(image) > 0:
            image = np.reshape(image, (tex.getYSize(), tex.getXSize(), 4))
            image = cv.resize(image, (0,0), fx=0.7, fy=0.7)
            return True, image
        else:
            return False, None
   
    def tabulate_gen(self, real, image, accel, gps, gyro, triad):
        data = []
        header = ['---', 'State', 'Image State', 'Accelerometer State', 'GPS State', 'Gyro State', 'Triad State']
        data_name = ['x', 'y', 'z', 'q0', 'q1', 'q2', 'q3']

        for i in range(3):
            data.append((data_name[i], str(real[i]), str(image[i]), str(accel[i]), str(gps[i]), "0", "0"))
        
        for i in range(4):
            data.append((data_name[i+3], str(real[i+3]), str(image[i+3]), "0", "0", str(gyro[i]), str(triad[i])))
            
        return data, header
    
    def draw(self, img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 1)
        img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 1)
        img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 1)
        return img    
    
    
    def nothing(x):
            pass

    # #Criar janela para trackbar
    # cv.namedWindow("Trackbars")

    # #Criar trackbars
    # cv.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    # cv.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    # cv.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    # cv.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    # cv.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    # cv.createTrackbar("U - V", "Trackbars", 255, 255, nothing)


    

    def pos_deter(self, task):
        
        cX = None
        cY = None
        cX2 = None
        cY2 = None

        #Calculo de FPS
        
        #Setup de fonte
        font = cv.FONT_HERSHEY_PLAIN
        
        if task.frame % 10 == 0:           
            ret, image = self.get_image(self.render.buffer)
            ret, image2 = self.get_image(self.render.buffer2)
            if ret:
                #Converte frame para HSV
                hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
                hsv2 = cv.cvtColor(image2, cv.COLOR_BGR2HSV)
                # l_h = cv.getTrackbarPos("L - H", "Trackbars")
                # l_s = cv.getTrackbarPos("L - S", "Trackbars")
                # l_v = cv.getTrackbarPos("L - V", "Trackbars")
                # u_h = cv.getTrackbarPos("U - H", "Trackbars")
                # u_s = cv.getTrackbarPos("U - S", "Trackbars")
                # u_v = cv.getTrackbarPos("U - V", "Trackbars")

                #Detecção de cor através de HSV
                # lower = np.array([l_h, l_s, l_v])
                # upper = np.array([u_h, u_s, u_v])
                lower = np.array([0, 239, 222])
                upper = np.array([179, 255, 255])
                #Cria mascara para filtrar o objeto pela cor definida pelos limites
                mask = cv.inRange(hsv, lower, upper)
                mask2 = cv.inRange(hsv2, lower, upper)
                #Cria kernel
                kernel = np.ones((5,5), np.uint8)
                #Aplica processo de Abertura (Erosão seguido de Dilatação)
                opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations = 1)
                opening2 = cv.morphologyEx(mask2, cv.MORPH_OPEN, kernel, iterations = 1)
    
    
                _, cnts, _ = cv.findContours(opening.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
                _, cnts2, _ = cv.findContours(opening2.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

                # loop over the contours
                for c in cnts:
                    # compute the center of the contour
                    M = cv.moments(c)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    perimeter = cv.arcLength(c, True)
                    metric = (4*math.pi*M["m00"])/perimeter**2
                    print("Metric:",metric)
                    if metric > 0.7:
        
                        #draw the contour and center of the shape on the image
                        cv.drawContours(image, [c], -1, (255, 0, 0), 1)
                        cv.circle(image, (cX, cY), 1, (255, 0, 0), 1)
                
                for c in cnts2:
                    # compute the center of the contour
                    M = cv.moments(c)
                    cX2 = int(M["m10"] / M["m00"])
                    cY2 = int(M["m01"] / M["m00"])
                    
                    perimeter = cv.arcLength(c, True)
                    metric = (4*math.pi*M["m00"])/perimeter**2
                    print("Metric2:",metric)
                    if metric > 0.7:
        
                        #draw the contour and center of the shape on the image
                        cv.drawContours(image2, [c], -1, (255, 0, 0), 1)
                        cv.circle(image2, (cX2, cY2), 1, (255, 0, 0), 1)
    
    
                cv.putText(image," Center:"+str(cX)+','+str(cY), (10, 80), font, 1, (255,255,255), 1)
                cv.putText(image2," Center:"+str(cX2)+','+str(cY2), (10, 80), font, 1, (255,255,255), 1)
                
                cv.imshow('Camera 1',image)
                cv.imshow('Camera 2', image2)
                #cv.imshow('Drone Camera2',np.flipud(cv.cvtColor(image2, cv.COLOR_RGB2BGR)))
                key = cv.waitKey(1)     

        return task.cont
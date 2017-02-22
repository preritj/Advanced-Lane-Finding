import numpy as np
import cv2, os
import matplotlib.pyplot as plt
from glob import glob
import matplotlib.image as mpimg



class calibration :
    def __init__(self, camera_matrix=None, dist_coefs=None) :
        self.images = {}
        self.nb_corners = {}
        self.images_with_corners = {}
        self.dim = None
        self.obj_points = None
        self.img_points = None   
        # load existing calibration if known
        self.camera_matrix = camera_matrix
        self.dist_coefs = dist_coefs
        
    def add_image(self, filename, nx, ny) :
        image = mpimg.imread(filename)
        if image is None :
            print("Failed to read {}".format(filename))
        else :
            h,w = image.shape[:2]
            if not self.dim :
                self.dim = (w,h)
            else :
                if w!=self.dim[0] or h!=self.dim[1] :
                    print("Default image size is {}".format(self.dim))
                    print("whereas {} size is {}".format(filename, (w,h)))
                    print("Use with caution...")
            self.images[filename] = image
            self.nb_corners[filename] = (nx,ny)
            
    def plot_images(self, images) :
        ncols = 5
        nrows = int(np.ceil(len(images)/ncols))
        fig, axes = plt.subplots(nrows,ncols, figsize=(ncols*4, nrows*2.5))
        for filename,ax in zip(images,axes.flatten()) : 
            ax.imshow(images[filename])
            ax.set_title(os.path.basename(filename))
        for ax in axes.flatten() : 
            ax.axis('off') 
            
    def display_images(self) :
        images = self.images
        self.plot_images(images)

    def display_images_with_corners(self) :
        self.find_corners(Display=True)
        images = self.images_with_corners
        self.plot_images(images)
        
        
            
    def find_corners(self, Display=False) :
        self.obj_points = []
        self.img_points = []
        self.images_with_corners = {}
        for filename, img in self.images.items() :
            nx,ny = self.nb_corners[filename]
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            found, corners = cv2.findChessboardCorners(gray, (nx, ny))
            if found:
                objp = np.zeros((nx*ny,3), np.float32)
                objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
                self.obj_points.append(objp)
                self.img_points.append(corners)
                # Draw and display the corners
                if Display :
                    img_with_corners = np.copy(img)
                    cv2.drawChessboardCorners(img_with_corners, (nx, ny), corners, found)
                    self.images_with_corners[filename] = img_with_corners
            else :
                print("Corners not found in {}".format(filename))  
                
    def calc_distortion(self) :
        self.find_corners()
        # calculate camera distortion
        rms, self.camera_matrix, self.dist_coefs, rvecs, tvecs = cv2.calibrateCamera(self.obj_points, 
                                                                    self.img_points, self.dim, None, None)
        print("RMS:", rms)
        print("camera matrix:\n", self.camera_matrix)
        print("distortion coefficients: ", self.dist_coefs.ravel())
        
    def undistort_img(self, img) :
        assert (self.camera_matrix is not None) and (self.dist_coefs is not None), "Calibrate the camera first." 
        return cv2.undistort(img, self.camera_matrix, self.dist_coefs, None, self.camera_matrix)
        
    def display_undistorted_images(self,files=None) :
        dist = {}
        undist = {}
        if not files :
            files = self.images.keys()
        for filename in files :
            img = mpimg.imread(filename)
            dist[filename] = img
            undist[filename] = cv2.undistort(img, self.camera_matrix, self.dist_coefs, None, self.camera_matrix)
        ncols = 2
        nrows = len(dist)
        fig, axes = plt.subplots(nrows,2, figsize=(ncols*4, nrows*2.5))
        for filename,ax in zip(dist,axes) : 
            ax[0].imshow(dist[filename])
            ax[1].imshow(undist[filename])
            ax[0].set_title(os.path.basename(filename)+" (original)")
            ax[1].set_title(os.path.basename(filename)+" (undistorted)")
        for ax in axes.flatten() : 
            ax.axis('off')
        plt.show()

            
            
calib_DIR = "camera_cal"
calib_imgs = glob(os.path.join(calib_DIR,"calibration*.jpg"))
cal = calibration()
for calib_img in calib_imgs :
    nx, ny = 9,6
    cal.add_image(calib_img, nx,ny)
cal.calc_distortion()
        
        
if __name__ == "__main__": 
    cal.display_images_with_corners()
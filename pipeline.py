import numpy as np
import cv2, os
import matplotlib.pyplot as plt
from glob import glob
import matplotlib.image as mpimg
from calibrate import *
from lane_detection import *
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
img_dim = (1280, 720)

class lane_line :
    def __init__(self, n_iter=8) :
        # number of iterations to use for avraging/smoothing
        self.n_iter = n_iter
        # was the line detected in the last iteration?
        self.detected = False
        # counter
        self.counter = 0
        
        
        # x values of the current and last n fits of the line
        self.current_xfitted = None 
        self.recent_xfitted = [] 
        self.avg_xfitted = None
        self.yfitted = np.linspace(0, img_dim[1]-1, img_dim[1])
        
        # polynomial coefficients of the current and last n iterations
        self.current_fit = None
        self.recent_fits = []
        self.avg_fit = None
        self.res = None # residuals from fit
        
        self.current_pixels = None
        self.previous_pixels = []
        self.radius_of_curvature = None
        self.line_base_pos = None
        # x values for detected line pixels
        self.pixels_x = None
        # y values for detected line pixels
        self.pixels_y = None
        
    def add_line(self,x_pixels, y_pixels) :
        self.detected = True
        self.counter += 1
        self.pixels_x = x_pixels
        self.pixels_y = y_pixels
        self.curve_fit()
        self.calc_R(self.current_fit)
        self.calc_base_dist(self.current_fit)
        
    def curve_fit(self, poly_order=2):
        fit, self.res,_,_,_ = np.polyfit(self.pixels_y, self.pixels_x, poly_order, full=True)
        self.res = self.res/len(self.pixels_x)**1.2
        self.poly_order=poly_order
        self.current_fit = fit
        
        h = img_dim[1]
        y = self.yfitted
        x_fit = fit[poly_order]
        for i in range(poly_order) :
            x_fit += fit[i]*y**(poly_order-i)
        self.current_xfitted = x_fit
        if self.avg_xfitted is None :
            self.update()
            
    def calc_R(self, fit) :
        y=img_dim[1]
        self.radius_of_curvature = ((ym_per_pix**2 + xm_per_pix**2*(2*fit[0]*y + fit[1])**2)**1.5)/(2
                                    *xm_per_pix*ym_per_pix*fit[0])
    
    def calc_base_dist(self, fit) :
        y = img_dim[1]
        dist = -img_dim[0]/2
        for i in range(self.poly_order+1) :
            dist += fit[i]*y**(self.poly_order-i)
        self.line_base_pos = dist*xm_per_pix
            
    def update(self) :
        if len(self.recent_fits) >= self.n_iter : 
            self.recent_xfitted.pop(0)
            self.recent_fits.pop(0)
        
        #n_iter = len(self.recent_fits)+1
        #weights = np.array([(i+1)/n_iter for i in range(n_iter)])/sum([i+1 for i in range(n_iter)])
        self.recent_xfitted.append(self.current_xfitted)
        self.avg_xfitted = np.average(np.array(self.recent_xfitted), axis=0)
        #self.avg_xfitted = np.average(np.array(self.recent_xfitted), axis=0, weights=weights)
        self.recent_fits.append(self.current_fit)
        self.avg_fit = np.average(np.array(self.recent_fits), axis=0)
        #self.avg_fit = np.average(np.array(self.recent_fits), axis=0, weights=weights)
        self.calc_R(self.avg_fit)
        self.calc_base_dist(self.avg_fit)
        self.detected = False
        
    def radius_ratio(self, other_line) :
        delta_r = abs(self.radius_of_curvature-other_line.radius_of_curvature)
        min_r = min(abs(self.radius_of_curvature),abs(other_line.radius_of_curvature))
        return delta_r/min_r
    
    def check_diverging_curves(self, other_line):
        R1 = self.radius_of_curvature
        R2 = other_line.radius_of_curvature
        if max(abs(R1), abs(R2)) > 1500 :
            return False
        else :
            return (R1*R2<0) 
    
    def fit_ratio(self, other_line):
        fit1 = np.array(self.current_fit)
        fit2 = np.array(other_line.current_fit)
        delta_fit = fit1-fit2
        min_fit = np.minimum(np.absolute(fit1), np.absolute(fit2)) 
        return np.linalg.norm(delta_fit[:2]/min_fit[:2])*2000/(
            abs(self.radius_of_curvature) + abs(other_line.radius_of_curvature))
    
    def base_gap(self, other_line):
        return abs(self.line_base_pos-other_line.line_base_pos)
    
    def delta_xfitted(self):
            return np.linalg.norm(self.current_xfitted - self.avg_xfitted)
        
        
line = {'left':lane_line(), 'right':lane_line()}
fail = {'left':2, 'right':2}

def pipeline(img) :
    img = cal.undistort_img(img)
    (h,w) = img.shape[:2] 
    lane = lane_detection(img)
    imgA = np.zeros_like(img)
    imgB = np.zeros_like(img)
    imgC = np.zeros_like(img)
    main_img = np.zeros_like(img).astype(np.uint8)
    if line['left'].counter%10==0 or line['right'].counter%10==0 or \
        line['left'].counter<10 or line['right'].counter<10 or \
        fail['left']>=2 or fail['right']>=2 :
        #print("using window ")
        imgA,imgB,imgC = lane.sliding_window()
    else :
        fit = {}
        for side in ['left','right'] :
            #print("using target ", side, line[side].avg_fit )
            fit[side] = line[side].avg_fit
        imgA,imgB,imgC = lane.target_search(fit)
        
    fit = {'left':None, 'right':None}    
    sides = ['left','right']
    

    for side in sides :
        if not lane.found[side] :
            fail[side]+=1
            line[side].detected=False
        else :
            pixels_x, pixels_y = lane.good_pixels_x, lane.good_pixels_y
            line[side].add_line(pixels_x[side], pixels_y[side])
            line[side].detected=True
    #print(line['left'].fit_ratio(line['right']),line['left'].base_gap(line['right']))
    if line['left'].check_diverging_curves(line['right']) or line['left'].fit_ratio(line['right'])>10 \
            or (not 400*xm_per_pix<line['left'].base_gap(line['right'])<750*xm_per_pix) :
        #print(line[side].delta_xfitted())
        for side in sides :
            #print("delta", line[side].delta_xfitted())
            #if line[side].delta_xfitted() > 500  :
            if line[side].delta_xfitted() > 1000 or line[side].res > 55: 
                fail[side] += 1
            else :
                line[side].update()
    else :
        for side in sides :
            if line[side].res > 55  :
                fail[side] +=1
            elif line[side].detected : 
                fail[side]=0
                line[side].update()
            
    
    for side in sides :  
        fit[side] = line[side].avg_fit
        pts = np.array(np.vstack((line[side].avg_xfitted, line[side].yfitted)).T, dtype=np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(imgC,[pts],False,(255,255,0), thickness=5)
        
        pts = np.array(np.vstack((line[side].current_xfitted, line[side].yfitted)).T, dtype=np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(imgC,[pts],False,(0,255,255), thickness=2)
        line[side].calc_R(line[side].avg_fit)
        line[side].calc_base_dist(line[side].avg_fit)

    R_avg = (line['left'].radius_of_curvature + line['right'].radius_of_curvature)/2
    base_gap = (line['left'].base_gap(line['right']))
    center_pos = (line['left'].line_base_pos + line['right'].line_base_pos)/2

    main_img = lane.plot_lane(fit)
    img_A = cv2.resize(imgA,None, fx=0.32, fy=0.34, interpolation=cv2.INTER_AREA)
    hA,wA = img_A.shape[:2]
    img_B = cv2.resize(imgB,None, fx=0.32, fy=0.34, interpolation=cv2.INTER_AREA)
    hB,wB = img_B.shape[:2]
    img_C = cv2.resize(imgC,None, fx=0.32, fy=0.34, interpolation=cv2.INTER_AREA)
    text_A = np.zeros((hA/4, wA,3))
    h_text = text_A.shape[0] 
    text_B = np.zeros((h_text, wA,3))
    text_C = np.zeros((h_text, wA,3))
    for i in range(1,3) :
        text_A[:,:,i] = 255
        text_B[:,:,i] = 255
        text_C[:,:,i] = 255
    text_A = text_A.astype(np.uint8)
    text_B = text_B.astype(np.uint8)
    text_C = text_C.astype(np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(text_A,'Threshold',(10,h_text-20), font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(text_B,'Threshold (yellow)',(10,h_text-20), font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(text_C,'Best fit',(10,h_text-20), font,1,(0,0,0),3,cv2.LINE_AA)
    img_combined_right = np.vstack((text_A,img_A,text_B, img_B, text_C, img_C))
    main_text = np.zeros((3*h_text+3*hA-h,w,3)).astype(np.uint8)
    h_main_text, w_main_text = main_text.shape[:2]
    cv2.putText(main_text,'Radius of curvature : {:5.2f} m'.format(abs(R_avg)),
                (10,35), font, 1,(255,255,255),3,cv2.LINE_AA)
    shift = "left" if center_pos>0 else "right"
    cv2.putText(main_text,'Vehicle is {:6.2f} m {:5} of center'.format(abs(center_pos), shift),
                (10,80), font, 1,(255,255,255),3,cv2.LINE_AA)
    if line['left'].avg_fit[0]>0.0001 and  line['right'].avg_fit[0]>0.0001 :
        cv2.putText(main_text,'Right curve ahead',
                (10,135), font, 1,(0,255,255),3,cv2.LINE_AA)
    elif line['left'].avg_fit[0]<-0.0001 and  line['right'].avg_fit[0]<-0.0001 :
        cv2.putText(main_text,'Left curve ahead',
                (10,135), font, 1,(0,255,255),3,cv2.LINE_AA)
    img_combined_left = np.vstack((main_img, main_text))
    return np.hstack((img_combined_left, img_combined_right))


#-------------------------------------------------------------------------------------------

if __name__ == "__main__": 
    line = {'left':lane_line(n_iter=7), 'right':lane_line(n_iter=7)}
    fail = {'left':3, 'right':3}

    output = 'output_video/project_video.mp4'
    clip = VideoFileClip("project_video.mp4")
    clip = clip.fl_image(pipeline) #NOTE: this function expects color images!!
    clip.write_videofile(output, audio=False)
import numpy as np
import cv2, os
import matplotlib.pyplot as plt
from glob import glob
import matplotlib.image as mpimg
from calibrate import *

class lane_detection :
    def __init__(self, image) :
        self.img = image
        
    def warper(self, img=None, src=None, dst=None, debug=False) :
        img = self.img if img is None else img
        h,w = img.shape[:2]
        self.src = src
        self.dst = dst
        if src is None : 
            self.src = np.float32([[220,700],[595,450],[685,450],[1060,700]])
        if dst is None :
            self.dst = np.float32([[w/4,h],[w/4,-100],[3*w/4,-100],[3*w/4,h]]) 
        M = cv2.getPerspectiveTransform(self.src, self.dst)
        warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_NEAREST)
        # use debug to draw lines 
        if debug :
            pts = np.int32(self.src)
            pts = pts.reshape((-1,1,2))
            annotated_img = np.copy(img)
            annotated_img = cv2.polylines(annotated_img,[pts],True,(255,0,0), thickness=5)
            warped = cv2.line(warped,(int(w/4),h),(int(w/4),-100),(255,0,0), thickness=5)
            warped = cv2.line(warped,(int(3*w/4),h),(int(3*w/4),-100),(255,0,0), thickness=5)
            return (annotated_img, warped)
        else :
            return warped
        
    def split_channels(self) :
        """
        returns a total of 7 channels : 
        4 edge channels : all color edges (including the signs), yellow edges (including the signs) 
        3 color channels : yellow and white (2 different thresholds are used for white) 
        """
        binary = {}  
        
        # thresholding parameters for various color channels and Sobel x-gradients
        h_thresh=(15, 35)
        s_thresh=(75, 255) #s_thresh=(30, 255)
        v_thresh=(175,255)
        vx_thresh = (20, 120)
        sx_thresh=(10, 100)

        img = np.copy(self.img)
        # Convert to HSV color space and separate the V channel
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
        h_channel = hsv[:,:,0]
        s_channel = hsv[:,:,1]
        v_channel = hsv[:,:,2]

        # Sobel x for v-channel
        sobelx_pos = cv2.Sobel(v_channel, cv2.CV_64F, 1, 0, ksize=3) # Take the derivative in x
        sobelx_neg = np.copy(sobelx_pos)
        sobelx_pos[sobelx_pos<=0] = 0
        sobelx_neg[sobelx_neg>0] = 0
        sobelx_neg = np.absolute(sobelx_neg)
        scaled_sobel_pos = np.uint8(255*sobelx_pos/np.max(sobelx_pos))
        scaled_sobel_neg = np.uint8(255*sobelx_neg/np.max(sobelx_neg))
        vxbinary_pos = np.zeros_like(v_channel)
        vxbinary_pos[(scaled_sobel_pos >= vx_thresh[0]) & (scaled_sobel_pos <= vx_thresh[1])] = 1
        binary['edge_pos'] = vxbinary_pos
        vxbinary_neg = np.zeros_like(v_channel)
        vxbinary_neg[(scaled_sobel_neg >= vx_thresh[0]) & (scaled_sobel_neg <= vx_thresh[1])] = 1
        binary['edge_neg'] = vxbinary_neg

        # Sobel x for s-channel
        sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0, ksize=3) # Take the derivative in x
        sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255*sobelx/np.max(sobelx))
        sxbinary_pos = np.zeros_like(s_channel)
        sxbinary_neg = np.zeros_like(s_channel)
        sxbinary_pos[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1]) 
                     & (scaled_sobel_pos >= vx_thresh[0]-10) & (scaled_sobel_pos <= vx_thresh[1])]=1
        sxbinary_neg[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1]) 
                     & (scaled_sobel_neg >= vx_thresh[0]-10) & (scaled_sobel_neg <= vx_thresh[1])]=1           
        binary['yellow_edge_pos'] = sxbinary_pos
        binary['yellow_edge_neg'] = sxbinary_neg

        # color thresholds for selecting white lines
        v_binary = np.zeros_like(v_channel)
        v_binary[(v_channel >= v_thresh[0]+s_channel+20) & (v_channel <= v_thresh[1])] = 1
        binary['white_tight'] = np.copy(v_binary)
        v_binary[v_channel >= v_thresh[0]+s_channel] = 1
        binary['white_loose'] = v_binary

        # color threshold for selecting yellow lines
        h_binary = np.zeros_like(h_channel)
        h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1]) & (s_channel >= s_thresh[0])] = 1
        binary['yellow'] = h_binary

        return binary
    
    def get_nonzero_pixels(self) :
        nonzero_x, nonzero_y = {},{}
        for channel in self.binary :
            nonzero = self.binary[channel].nonzero()
            nonzero_x[channel], nonzero_y[channel] = (np.array(nonzero[1]), np.array(nonzero[0]))
        return nonzero_x, nonzero_y

    
    def get_good_pixels(self, roi, min_pix=100, max_pix=10000, window_search=True, debug=False) :
        nb_pixels, x_mean, x_stdev = {},{},{}
        for channel in self.binary :
            nonzero_x, nonzero_y = self.nonzero_x[channel], self.nonzero_y[channel]
            # region of interest
            roi_ = roi[channel]
            count = np.sum(roi_)
            if count<min_pix or (count>max_pix and window_search) :
                continue
#             if count>20000 :
#                 continue
            nb_pixels[channel] = count
            x_mean[channel]    = np.mean(nonzero_x[roi_])
            x_stdev[channel]   = np.std(nonzero_x[roi_])
            if debug :
                print(channel, count, x_mean[channel], x_stdev[channel])
        
        if window_search :
            selected_channels = [c for c in x_stdev.keys() if x_stdev[c]<35]
        else :
            selected_channels = [c for c in x_stdev.keys()]
        # some consistency checks to select channels
        if 'edge_pos' in selected_channels :
            if ('edge_neg' not in x_stdev.keys()) :
                selected_channels.remove('edge_pos')
            elif (nb_pixels['edge_neg'] < nb_pixels['edge_pos']/3) or \
                    (abs(x_mean['edge_neg']-x_mean['edge_pos'])>50) :
                selected_channels.remove('edge_pos')
                if 'edge_neg' in selected_channels : selected_channels.remove('edge_neg')
            elif window_search and (x_mean['edge_pos'] > x_mean['edge_neg']) :
                selected_channels.remove('edge_pos')
                if 'edge_neg' in selected_channels : selected_channels.remove('edge_neg')
        if 'edge_neg' in selected_channels :
            if ('edge_pos' not in x_stdev.keys()) :
                selected_channels.remove('edge_neg')
            elif (nb_pixels['edge_pos'] < nb_pixels['edge_neg']/3) or \
                    (abs(x_mean['edge_neg']-x_mean['edge_pos'])>50) :
                if 'edge_pos' in selected_channels : selected_channels.remove('edge_pos')
                selected_channels.remove('edge_neg')
            elif window_search and (x_mean['edge_pos'] > x_mean['edge_neg']) :
                selected_channels.remove('edge_neg')
                if 'edge_pos' in selected_channels : selected_channels.remove('edge_pos')
        if 'yellow_edge_pos' in selected_channels :
            if ('yellow_edge_neg' not in x_stdev.keys()) :
                selected_channels.remove('yellow_edge_pos')
            elif (nb_pixels['yellow_edge_neg']< nb_pixels['yellow_edge_pos']/3) or \
                    (abs(x_mean['yellow_edge_neg']-x_mean['yellow_edge_pos'])>50) :
                if 'yellow_edge_neg' in selected_channels : selected_channels.remove('yellow_edge_neg')
                selected_channels.remove('yellow_edge_pos')
            elif ('yellow' in selected_channels) and (abs(x_mean['yellow']-x_mean['yellow_edge_pos'])>20):
                selected_channels.remove('yellow_edge_pos')
        if 'yellow_edge_neg' in selected_channels :
            if ('yellow_edge_pos' not in x_stdev.keys()) :
                selected_channels.remove('yellow_edge_neg')
            elif (nb_pixels['yellow_edge_pos']< nb_pixels['yellow_edge_neg']/3) or \
                    (abs(x_mean['yellow_edge_pos']-x_mean['yellow_edge_neg'])>50) : 
                selected_channels.remove('yellow_edge_neg')
                if 'yellow_edge_pos' in selected_channels : selected_channels.remove('yellow_edge_pos')
            elif ('yellow' in selected_channels) and (abs(x_mean['yellow']-x_mean['yellow_edge_neg'])>20):
                selected_channels.remove('yellow_edge_neg')
        if ('white_tight' in selected_channels) :
            if 'white_loose' in selected_channels : selected_channels.remove('white_loose')
            if nb_pixels['white_tight']>8000 or (window_search and nb_pixels['white_tight']>2000) :
                selected_channels.remove('white_tight')
        if 'white_loose' in selected_channels and (nb_pixels['white_loose']<100 or nb_pixels['white_loose']>5000):  
            selected_channels.remove('white_loose')
        if window_search and 'white_loose' in selected_channels and nb_pixels['white_loose']>500 : 
            selected_channels.remove('white_loose')
        if len(selected_channels)==1 and 'yellow' in selected_channels and nb_pixels['yellow']<300 :
            selected_channels.remove('yellow')
        
        if debug :
            print("selected " , selected_channels)
        
        # combine the selected channels
        comb_nonzero_x, comb_nonzero_y = [],[]
        for channel in selected_channels :
            nonzero_x, nonzero_y = self.nonzero_x[channel], self.nonzero_y[channel]
            roi_ = roi[channel]
            comb_nonzero_x.append(nonzero_x[roi_])
            comb_nonzero_y.append(nonzero_y[roi_])
        if comb_nonzero_x :
            return (True, np.concatenate(comb_nonzero_x), np.concatenate(comb_nonzero_y))
        else :
            return (False,None,None)
        
    def window_analysis(self, X1, Y1, X2, Y2) :
        found         = {'left':None, 'right':None}
        good_pixels_x = {'left':None, 'right':None}
        good_pixels_y = {'left':None, 'right':None}
        
        for side in ['left','right'] :
            # define region of interest
            roi={}
            for channel in self.binary :
                nonzero_x, nonzero_y = self.nonzero_x[channel], self.nonzero_y[channel]
                roi[channel] = ((nonzero_x>X1[side]) & (nonzero_x<X2[side]) & (nonzero_y>Y1) & (nonzero_y<=Y2))

            found[side], good_pixels_x[side], good_pixels_y[side] = self.get_good_pixels(roi) 

        if found['left'] or found['right'] :
            self.margin['left']  = 50 if (found['left'])  else 150
            self.margin['right'] = 50 if (found['right']) else 150
            self.x_current['left']  = np.mean(good_pixels_x['left'])  if found['left'] \
                                    else np.mean(good_pixels_x['right']) -min(600, self.lane_gap)
            self.x_current['right'] = np.mean(good_pixels_x['right']) if found['right'] \
                                    else np.mean(good_pixels_x['left'])  +min(600, self.lane_gap)
            for side in ['left','right'] : self.x_current[side] = np.int(self.x_current[side])
            if found['left'] or found['right'] :
                self.lane_gap = (self.lane_gap + self.x_last_found['right'] - self.x_last_found['left'])/2
        else :
            self.margin['left']=150 
            self.margin['right']=150
            
        return found, good_pixels_x, good_pixels_y
    
    
    def check_lanes(self, img_range, min_lane_gap=350, max_lane_gap=750) :
        if self.lane_gap < min_lane_gap : return False 
        if self.lane_gap > max_lane_gap : return False
        elif self.x_current['left']<img_range[0] or self.x_current['right']>img_range[1] : return False
        else : return True
    
    
    def sliding_window(self, nb_windows=15, visualize=True, debug=False) :
        
        # get channels and warp them
        self.binary = self.split_channels()
        self.binary = {k: self.warper(v) for k, v in self.binary.items()}
        
        # group A consists of all line edges and white color 
        group_A = np.dstack((self.binary['edge_pos'], self.binary['edge_neg'], self.binary['white_loose']))
        # group B consists of yellow edges and yellow color
        group_B = np.dstack((self.binary['yellow_edge_pos'], self.binary['yellow_edge_neg'], self.binary['yellow']))
        
        if visualize :
            out_img_A = np.copy(group_A)*255
            out_img_B = np.copy(group_B)*255
            out_img_C = np.zeros_like(out_img_A)
        
        # number of windows 
        nwindows = nb_windows
        
        h,w = group_A.shape[:2]
        self.dims = (w,h)
        window_height = np.int(h/nwindows)
        midpoint = np.int(w/2)
        # width of the windows +/- margin
        self.margin = {'left' : np.int(0.5*midpoint), 
                       'right': np.int(0.5*midpoint)}
        # center of current left and right windows
        self.x_current = {'left' : np.int(0.5*midpoint), 
                          'right': np.int(1.5*midpoint)}
        # center of left and right windows last found
        self.x_last_found = {'left' : np.int(0.5*midpoint), 
                             'right': np.int(1.5*midpoint)}
        # center of previous left and right windows
        x_prev = {'left' : None, 
                  'right': None}
        
        self.nonzero_x, self.nonzero_y = self.get_nonzero_pixels() 
        # good pixels
        self.good_pixels_x = {'left' : [], 'right' : []} 
        self.good_pixels_y = {'left' : [], 'right' : []} 
        
        self.lane_gap = 600
        momentum    = {'left' :0, 
                       'right':0}
        last_update = {'left' :0, 
                       'right':0}
        self.found  = {'left' :False, 
                       'right':False}
        
        # Step through the windows one by one
        for window in range(nwindows):
            
            # Identify window boundaries in x and y (and right and left)
            Y1 = h - (window+1)*window_height
            Y2 = h - window*window_height
            X1 = {side : self.x_current[side]-self.margin[side] for side in ['left','right']} 
            X2 = {side : self.x_current[side]+self.margin[side] for side in ['left','right']} 
            if debug :
                print("-----",window, X1, X2, Y1, Y2)
            
            found, good_pixels_x, good_pixels_y = self.window_analysis(X1,Y1,X2,Y2)
            if not self.check_lanes(min_lane_gap=350, img_range=(-50,w+50)) : 
                break
                
            # final window refinement with updated centers and margins
            Y1 = h - (window+1)*window_height
            Y2 = h - window*window_height
            X1 = {side : self.x_current[side]-self.margin[side] for side in ['left','right']} 
            X2 = {side : self.x_current[side]+self.margin[side] for side in ['left','right']} 
            if debug :
                print("-----",window, X1, X2, Y1, Y2)
            
            
            found, good_pixels_x, good_pixels_y = self.window_analysis(X1,Y1,X2,Y2)
            if not self.check_lanes(min_lane_gap=350, img_range=(-50,w+50)) : 
                break
            
#             if (not found['left']) and (not found['right']) :
#                 self.x_current = x_current
#                 self.lane_gap = lane_gap
            
            for i,side in enumerate(['left','right']) :
                # Add good pixels to list
                if found[side] :
                    self.good_pixels_x[side].append(good_pixels_x[side])
                    self.good_pixels_y[side].append(good_pixels_y[side])
                    self.x_last_found[side] = self.x_current[side]
                
                # Draw the windows on the visualization image
                if visualize :
                    cv2.rectangle(out_img_A,(X1[side],Y1) ,(X2[side],Y2) ,(0,255,i*255), 2)  
                    cv2.rectangle(out_img_B,(X1[side],Y1) ,(X2[side],Y2) ,(0,255,i*255), 2) 
                    # Draw good pixels 
                    out_img_C[good_pixels_y[side], good_pixels_x[side],i] = 255 
            
                # decide window centers based on previous windows 
                #momentum[side] = 0
                last_update[side] += 1
                if x_prev[side] :
                    momentum[side] = np.int(momentum[side]) 
                    if found[side] :
                        momentum[side] += np.int(0.5*(self.x_current[side] - x_prev[side])/(last_update[side]))
                if found[side] : 
                    x_prev[side] = self.x_current[side]
                    last_update[side] = 0
                self.x_current[side] += momentum[side]
        
        for side in ['left','right'] :
            if self.good_pixels_x[side] :
                self.found[side] = True
                self.good_pixels_x[side] = np.concatenate(self.good_pixels_x[side])
                self.good_pixels_y[side] = np.concatenate(self.good_pixels_y[side])
            else :
                self.good_pixels_x[side] = None
                self.good_pixels_y[side] = None
        if visualize :
            return out_img_A.astype(np.uint8), out_img_B.astype(np.uint8), out_img_C.astype(np.uint8)
    
    
    def curve_fit(self, poly_order=2):
        # fit data
        fit={'left':None, 'right':None}
        
        for side in ['left','right'] :
            if self.good_pixels_x[side] is not None :
                fit[side]  = np.polyfit(self.good_pixels_y[side], self.good_pixels_x[side], poly_order)
        self.fit = fit
        return poly_order, fit
    
    
    def radius_of_curvature(self, fit, y) :
        return ((1 + (2*fit[0]*y + fit[1])**2)**1.5) / np.absolute(2*fit[0])
    
    
    def plot_curve_fits(self) :
        poly_order, fit = self.curve_fit()
        h = self.dims[1]
        y = np.linspace(0, h-1, h)
        x_fit = {'left':None, 'right':None}
        for side in ['left','right'] :
            if fit[side] is not None :
                x_fit[side] = fit[side][poly_order]
                for i in range(poly_order) :
                    x_fit[side] += fit[side][i]*y**(poly_order-i)
        return x_fit,y
    
    def plot_lane(self, fit, poly_order=2) :
        #poly_order, fit = self.curve_fit()
        w,h = self.dims
        y = np.linspace(0, h-1, h)
        x_fit = {'left':None, 'right':None}
        for side in ['left','right'] :
            if fit[side] is not None :
                x_fit[side] = fit[side][poly_order]
                for i in range(poly_order) :
                    x_fit[side] += fit[side][i]*y**(poly_order-i)
                x_fit[side] = x_fit[side]
                    
        if (x_fit['left'] is not None) and (x_fit['right'] is not None) :
            lane_img = np.zeros((h,w+300,3))
            pts_x = np.hstack((x_fit['left'],x_fit['right'][::-1]))
            pts_y = np.hstack((y,y[::-1]))
            pts = np.vstack((pts_x, pts_y)).T

            cv2.fillPoly(lane_img, np.int32([pts]), (0,255, 0))

            # unwarp image
            Minv = cv2.getPerspectiveTransform(self.dst, self.src)
            lane_img = cv2.warpPerspective(lane_img, Minv, (self.img.shape[1], self.img.shape[0]))
            out_img = cv2.addWeighted(self.img, 1, np.uint8(lane_img), 0.3, 0)
            return out_img
        else :
            return self.img
        
    def target_search(self, fit, poly_order=2, margin=80, visualize=True) :
        # get channels and warp them
        self.binary = self.split_channels()
        self.binary = {k: self.warper(v) for k, v in self.binary.items()}
        
        # group A consists of all line edges and white color 
        group_A = np.dstack((self.binary['edge_pos'], self.binary['edge_neg'], self.binary['white_tight']))
        # group B consists of yellow edges and yellow color
        group_B = np.dstack((self.binary['yellow_edge_pos'], self.binary['yellow_edge_neg'], self.binary['yellow']))
        
        h,w = group_A.shape[:2]
        self.dims = (w,h)
        
        if visualize :
            out_img_A = np.copy(group_A)*255
            out_img_B = np.copy(group_B)*255
            out_img_C = np.zeros_like(out_img_A)
            margins_img = np.zeros_like(out_img_A)
            # search area for left and right lanes
            y = np.linspace(0, h-1, h)
            x_fit = {'left':[-margin,margin], 'right':[-margin,margin]}
            for side in ['left','right'] :
                for i in range(poly_order+1) :
                    x_fit[side][0] += fit[side][i]*y**(poly_order-i)
                    x_fit[side][1] += fit[side][i]*y**(poly_order-i)
                pts_x = np.hstack((x_fit[side][0],x_fit[side][1][::-1]))
                pts_y = np.hstack((y,y[::-1]))
                pts = np.vstack((pts_x, pts_y)).T
                cv2.fillPoly(margins_img, np.int32([pts]), (0,0, 255))
                
        
        self.found         = {'left':False,'right':False}
        self.good_pixels_x = {'left':None, 'right':None}
        self.good_pixels_y = {'left':None, 'right':None}
        
        self.nonzero_x, self.nonzero_y = self.get_nonzero_pixels()
        
        for i,side in enumerate(['left','right']) :
            # define region of interest
            roi={}
            for channel in self.binary :
                nonzero_x, nonzero_y = np.copy(self.nonzero_x[channel]), np.copy(self.nonzero_y[channel])
                roi[channel] = ((nonzero_x > (fit[side][0]*(nonzero_y**2) + fit[side][1]*nonzero_y 
                                + fit[side][2] - margin)) & (nonzero_x < (fit[side][0]*(nonzero_y**2) 
                                + fit[side][1]*nonzero_y + fit[side][2] + margin))) 

            self.found[side], self.good_pixels_x[side], self.good_pixels_y[side] = \
                    self.get_good_pixels(roi, window_search=False) 
                
            if visualize :
                if self.found[side] :
                    out_img_C[self.good_pixels_y[side], self.good_pixels_x[side],i] = 255        
            
        if visualize :
            #out_img_A = cv2.addWeighted(out_img_A, 1, margins_img, 0.1, 0)
            #out_img_B = cv2.addWeighted(out_img_B, 1, margins_img, 0.1, 0)
            out_img_C = cv2.addWeighted(out_img_C, 1, margins_img, 0.6, 0)
            return out_img_A.astype(np.uint8), out_img_B.astype(np.uint8), out_img_C.astype(np.uint8)
        
#-------------------------------------------------------------------------------------------
        
if __name__ == "__main__": 
    test_DIR = "test_images/"
    test_imgs = glob(os.path.join(test_DIR,"*.jpg"))[:2]
    nrows = len(test_imgs)
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols,3*nrows))
    for ax in axes.flatten() :
        ax.axis('off')
    for img,ax in zip(test_imgs, axes) :
        img = mpimg.imread(img)
        img = cal.undistort_img(img)
        lane = lane_detection(img)
        annotated_img, warped = lane.warper(debug=True)
        ax[0].imshow(annotated_img)
        ax[1].imshow(warped)
    axes[0,0].set_title('Original image')
    axes[0,1].set_title('Bird-eye view')
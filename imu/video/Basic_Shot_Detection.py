import os,sys
import cv2
import matplotlib.pyplot as plt
import math
from glob import glob
import numpy as np
from tqdm import tqdm_notebook as tqdm
import imu
from imu.io import writeTxt, arrToStr
from imu.video.html import saveShotDetection, html_shot
from skimage.measure import label

class Shot_detect():
    def __init__(self, path):
        super(Shot_detect, self).__init__()
        self.video = path
        self.fns = sorted(glob(self.video + '/*.jpg'))
        self.shape = cv2.imread(self.fns[0]).shape
        self.num_frame = len(self.fns)
        self.crop_indices = None
        self.hsv_low = None
        self.hsv_high = None
        self.hsv_low_tab = None
        self.hsv_high_tab = None
        self.feat = None
        self.shot_start = None
        self.shots = None

    def display_im(self, frame_id):
        im = cv2.imread(self.fns[frame_id])[:, :, ::-1]
        plt.imshow(im)
        plt.title('Frame %d' % (frame_id))
        plt.axis('off')
        plt.show()

    def normalize(self,list):
        list -= list.min()
        list /= list.max()
        return list

    def smoothen(list, momentum = 0.1):
        l = list.shape[0]
        newlist = np.zeros((l))
        newlist[0] = list[0]
        for i in range(1,l):
            diff = list[i] - newlist[i-1]
            newlist[i] = newlist[i-1] + momentum*diff
        return newlist

    def clip_peak(self, list, clipping_thresh=0.4, window_length=20, stride=10):
        i=0
        while (i+window_length)<self.num_frame:
            window = list[i:i+1+window_length]
            if window.max() >= clipping_thresh:
                window = 0#window.mean()
                list[i:i+1+window_length] = window
            i += stride
        return list

    def Gaussian_mask(self, c1, c2, v1, v2):
        '''
        Input: c1,c2 
        '''
        self.att = np.zeros(self.shape, dtype=float)
        for x in range(self.att.shape[0]):
            for y in range(self.att.shape[1]):
                a = math.exp(- ((x - c1) ** 2 / (360 * v1) + (x - c2) ** 2 / (640 * v2)))
                self.att[x, y, 0], self.att[x, y, 1], self.att[x, y, 2] = a, a, a
        self.att = self.att.round()

    def set_thresholds(self, thresh1=None, thresh2=None):
        '''
        Input: thresh1: Hand thresholds in HSV pairs, thresh 2: Shadow thresholds in HSV pairs 
        '''
        if thresh1 is not None:
            self.hsv_low = np.array(thresh1[0:3])
            self.hsv_high = np.array(thresh1[3:])
        if thresh2 is not None:
            self.hsv_low_tab = np.array(thresh2[0:3])
            self.hsv_high_tab = np.array(thresh2[3:])

    def set_crop_indices(self, crop_indices=None):
        self.crop_indices = crop_indices
        if self.crop_indices is None:
            self.crop_indices = [0, 360, 0, 640]

    def thresh_image(self, frame_id):
        '''
        Input: Frame_id 
        Output: Feature map containing hand and shadow regions segmented. 
        '''
        assert self.hsv_high is not None and self.hsv_high_tab is not None, 'Cannot segment image!! Please set HSV thresholds first.'
        im = cv2.imread(self.fns[frame_id])
        im_hsv = cv2.cvtColor(im[self.crop_indices[0]:self.crop_indices[1], self.crop_indices[2]:self.crop_indices[3], :], cv2.COLOR_BGR2HSV)
        out_hand = np.zeros(im.shape[0:2], dtype=int)
        out_tab = np.zeros(im.shape[0:2], dtype=int)

        # Feature extraction
        out_h = cv2.inRange(im_hsv, self.hsv_low, self.hsv_high)
        out_hand[self.crop_indices[0]:self.crop_indices[1], self.crop_indices[2]:self.crop_indices[3]] = out_h
        print(self.hsv_high_tab,self.hsv_low_tab)
        out_t = cv2.inRange(im_hsv, self.hsv_low_tab, self.hsv_high_tab)
        out_tab[self.crop_indices[0]:self.crop_indices[1], self.crop_indices[2]:self.crop_indices[3]] = out_t

        out = (out_hand == 255) + (out_tab==255)

        # Synthetics: Plot Resizing
        f = plt.figure()
        f.set_figwidth(30)
        f.set_figheight(20)

        # Subplot arrangements: From left: Image - Hand features - Table features - Table and Hand Features combined
        plt.title('Frame %d' % (frame_id))
        plt.subplot(141)
        plt.imshow(im[:, :, ::-1])
        plt.subplot(142)
        plt.imshow(out_hand)
        plt.subplot(143)
        plt.imshow(out_tab)
        plt.subplot(144)
        plt.imshow(out)
        plt.show()

    def extract_features(self, save_path=None):
        assert self.hsv_high is not None and self.hsv_high_tab is not None, 'Cannot extract features!! Please set HSV thresholds first.'
        assert self.crop_indices is not None, 'Cannot extract features!! Please set cropping indices first.'
        self.feat = np.zeros(self.num_frame)
        print('Please wait while all frames are being processed.')
        for i in tqdm(range(self.num_frame)):
            img = cv2.imread(self.fns[i])
            im_hsv = cv2.cvtColor(img[self.crop_indices[0]:self.crop_indices[1], self.crop_indices[2]:self.crop_indices[3]], cv2.COLOR_BGR2HSV)
            out_hand = np.zeros(img.shape[0:2], dtype=int)
            out_h = cv2.inRange(im_hsv, self.hsv_low, self.hsv_high)
            out_hand[self.crop_indices[0]:self.crop_indices[1], self.crop_indices[2]:self.crop_indices[3]] = out_h
            out_tab = np.zeros(img.shape[0:2], dtype=int)
            out_tab = cv2.inRange(im_hsv, self.hsv_low_tab, self.hsv_high_tab)
            out = (out_hand == 255)+ (out_tab == 255)
            self.feat[i] = out.sum()
        print('Done')
        if save_path is not None:
            self.save_lists(save_path)

    def detect_shots(self, th_low=0.2, th_high=1.0, th_num_frame=15, clipping_thresh=None, momentum=None, showPlot=False, plot_path=None):
        assert self.feat is not None, 'No features found!! Please extract features first.'
        if momentum is not None:
            self.feat = smoothen(feat, momentum)
            self.feat = self.normalize(self.feat)
        if clipping_thresh is not None:
            self.feat = self.clip_peak(self.feat, clipping_thresh)
            self.feat = self.normalize(self.feat)
        if momentum is None and clipping_thresh is None:
            self.feat = self.normalize(self.feat)
        
        cluster = label((self.feat >= th_low) * (self.feat <= th_high))
        ui, uc = np.unique(cluster, return_counts=True)
        uc[ui == 0] = 0

        cluster[np.in1d(cluster, ui[uc < th_num_frame])] = 0
        self.shot_start = [0]
        for i in np.unique(cluster[cluster > 0]):
            idx = np.where(cluster == i)[0]
            self.shot_start += [idx[0], idx[-1]]
        self.shot_start += [self.num_frame - 1]
        self.shot_start = np.array(self.shot_start)

        dsp_num = self.num_frame
        out2 = self.shot_start[self.shot_start < dsp_num]
        self.shots = out2
        print(out2)
        if showPlot:
            # Synthetics: Plot Resizing
            f = plt.figure()
            f.set_figwidth(30)
            f.set_figheight(30)
            plt.plot(self.feat[:dsp_num])
            plt.plot(out2, self.feat[out2], 'rx')
            if plot_path is not None:
                plt.savefig(plot_path)
            plt.show()
            plt.clf()

    def save_lists(self, feature_path, shot_path=None):
        np.save(feature_path, self.feat)
        if shot_path is not None:
            np.save(shot_path, self.shots)

    def load_features(self, feature_path):
        self.feat = np.load(feature_path)

    def generate(self):
        '''
        Generates a js file containing str shot_start and shot_selection and a html file for proofreading. 
        '''
        num = len(self.shot_start) // 2
        shot_label = np.vstack([np.ones([1, num]), np.zeros([1, num])]).T.reshape(-1)
        saveShotDetection(os.path.join(self.video, 'shot.js'), self.shot_start, shot_label)
        video_name = self.video[self.video[:-1].rfind('/') + 1:]
        out = html_shot(video_name,frame_name=video_name + '/%06d.jpg', file_result=video_name + '/shot.js', frame_fps=1)
        writeTxt(os.path.join(os.path.dirname(self.video), 'shot_detection_' + video_name + '.html'), out.getHtml())
        

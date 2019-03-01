class StereoImageHelper():
    def __init__(self):
        self.img1_l = []
        self.img1_r = []
        self.img2_l = []
        self.img2_r = []
    
    def push_back(self, im_left, im_right):
        self.img1_l = self.img2_l
        self.img1_r = self.img2_r
        self.img2_l = im_left
        self.img2_r = im_right

class StereoFeatureTracks():
    def __init__(self):
        self.stereo_obs_1_list = []
        self.stereo_obs_2_list = []
    
    def append(self, stereo_obs_1, stereo_obs_2):
        self.stereo_obs_1_list.append(stereo_obs_1)
        self.stereo_obs_2_list.append(stereo_obs_2)

class KITTIData(object):
    def __init__(self):
        self.train_sequences = []
        self.train_img_paths = []
        self.train_labels = []
        self.train_se3_precision = []

        self.val_sequence = ''
        self.val_tm_mat_path = ''  # Path to mat file containing the the trajectory (loaded by TrajectoryMetrics)
        self.val_img_paths = []
        self.val_labels = []

        self.test_sequence = ''
        self.test_tm_mat_path = ''
        self.test_img_paths = []
        self.test_labels = []

KITTI_SEQS_DICT = {'00': {'date': '2011_10_03',
                          'drive': '0027',
                          'frames': range(0, 4541)},
                   '01': {'date': '2011_10_03',
                          'drive': '0042',
                          'frames': range(0, 1101)},
                   '02': {'date': '2011_10_03',
                          'drive': '0034',
                          'frames': range(0, 4661)},
                   '04': {'date': '2011_09_30',
                          'drive': '0016',
                          'frames': range(0, 271)},
                   '05': {'date': '2011_09_30',
                          'drive': '0018',
                          'frames': range(0, 2761)},
                   '06': {'date': '2011_09_30',
                          'drive': '0020',
                          'frames': range(0, 1101)},
                   '07': {'date': '2011_09_30',
                          'drive': '0027',
                          'frames': range(0, 1101)},
                   '08': {'date': '2011_09_30',
                          'drive': '0028',
                          'frames': range(1100, 5171)},
                   '09': {'date': '2011_09_30',
                          'drive': '0033',
                          'frames': range(0, 1591)},
                   '10': {'date': '2011_09_30',
                          'drive': '0034',
                          'frames': range(0, 1201)}}
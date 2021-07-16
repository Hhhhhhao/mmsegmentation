import os 

from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class RaleighDataset(CustomDataset):
    CLASSES = (
        'background', 
        'low veg',
        'med veg',
        'high veg',
        'building',
        'water',
        'road',
        'bridge')
    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0]]

    def __init__(self, **kwargs):
        super(RaleighDataset, self).__init__(**kwargs)
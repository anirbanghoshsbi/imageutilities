#get all sub directories

import os

def get_all_subdirs(base_dir):

    base_dir = '/content/chapter05-fine_tuning/dataset/'
    subdirs = [os.path.join(base_dir, o) for o in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir,o))]
    return subdirs

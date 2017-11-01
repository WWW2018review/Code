# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 17:36:04 2015

@author: jr
"""

from gpa.gpa import GlobalPatternAnalysis
import pickle
import warnings
import copy as cp

warnings.filterwarnings("ignore")


# load data
data, duration, gt, class_gt = pickle.load(open('data/synth_behavior_data_3c_100_10k5m.pckl', 'rb'))
data_entity = cp.deepcopy(data)
duration = [[[float(ele) for ele in session] for session in entity] for entity in duration]
duration_entity = cp.deepcopy(duration)
data = [session for entity in data for session in entity]
duration = [session for entity in duration for session in entity]
gt = [session for entity in gt for session in entity]

# learn vector space
gpa = GlobalPatternAnalysis(data, duration, gt)

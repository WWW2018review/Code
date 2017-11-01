# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:52:56 2015

@author: jr
"""
import numpy as np


# Data Container is comprised of all necessary info related to the input data:
# - points: input data itself
# - assignments: final assignments (points -> super states)
# - gt: ground-truth (if it is known)
class DataContainer:
    def __init__(self, points, assignments, gt):
        self.points = points
        self.assignments = assignments
        self.gt = gt


# Model Container consists of all information related to the model:
# - beta: distribution over the super-states
# - psi: distribution over the sub-states of each super-state
# - theta: markov chain representation of each super-state
#       * row <-> from
#       * column <-> to
#       * last element of both: artificial boundary-state
# - key_reg: dictionary to translate internal sub-state representation to external one
# - L: maximum # super-states
# - L2: # unique sub-states
# - boundary: representation of the artificial boundary-state
class ModelContainer:
    def __init__(self, beta, pi, super_states, key_reg, l2, cfg, dmodel=None, didx=None, dtheta=None):
        self.beta = beta
        self.pi = pi
        self.super_states = super_states
        self.key_reg = key_reg
        self.L2 = l2
        self.cfg = cfg
        self.dmodel = dmodel
        self.didx = didx
        self.dtheta = dtheta


class Config:
    def __init__(self):

        self.gamma = 1.0  # for beta
        self.alpha = 1.0  # for pi
        self.sigma = 50.0  # for psi
        self.lam = 50.0  # for theta

        self.d_gamma = 1.0
        self.d_lam = 1.0
        self.d_base_prob = 1E-5

        self.rho = 10.0  # 80.0  # 1000 intra-transition bias

        self.L = 100  # 75
        self.numIt = 501
        self.boundary = -1
        self.recognition_threshold = 5e-4

        self.collapsed = False
        self.it_gmm = int(1e7)
        self.sub_data_size = 200000
        self.k_gmm = 40
        self.raster_thresh = 1E-2
        self.d_importance_thresh = 200

        # path where to store final model
        self.base_path = 'data/'
        self.log_path = 'data/'
        self.eval_path = ''
        self.eval_name = 'final_model_3c_100'
        self.assignments_name = 'final_assignments'

        self.extension = '.pckl'

        # for evaluation
        self.gt = []
        self.eval_intervals = np.array([(20, 30), (20, 40), (30, 50)])
        self.eval_ends = self.eval_intervals[:, 1]
        self.eval_beginnings = self.eval_intervals[:, 0]
        self.assignment_counter = dict()
        self.params_store = dict()
        self.active_counters = []
        self.elapsed_time = []
        self.base_prob = 1E-3

    def set_name(self, dname):
        self.log_path = self.base_path + dname + "/"

    def set_path(self, dname, aname, fname):
        self.set_name(dname)
        self.eval_path = self.log_path + '/' + aname + '_logs' + fname + '.csv'
        open(self.eval_path, 'w').close()

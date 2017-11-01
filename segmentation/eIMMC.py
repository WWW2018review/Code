# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.mixture import BayesianGaussianMixture as bgm
from sklearn.utils import shuffle
# from scipy.stats import invgamma as ig, t, norm
from tools.probprog import ProbProg
from segmentation.config import Config
from segmentation.config import DataContainer
from segmentation.config import ModelContainer
import copy as cp
import pickle
import copy
import warnings
import csv

warnings.filterwarnings("ignore")


class eIMMC:
    def __init__(self, cfg=None, debug=True):
        self.pp = ProbProg()
        self.cfg = Config() if cfg is None else cfg
        self.TM = []
        self.debug = debug
        self.transformed_data = []
        self.debugger = []

        self.data = []
        self.flat_data = []
        self.durations = []
        self.flat_durations = []
        self.nump = None
        self.key_reg = None
        self.L2 = None
        self.gt_display = None
        self.eval_mode = None
        self.duration_model_active = False

    # fit time-series data to IMMC
    # data: list of lists = set of time-series comprised of categorical values
    # key_reg: conversion from original categorical values to numerical values (used by the algo)
    # gt_display: ground-truth if known
    # eval_mode: mode that stores additional results for evaluation of IMMC
    def fit(self, data, durations=[], key_reg=[], gt_display=[], eval_mode=False):
        self.data = data
        self.flat_data = [o for s in self.data for o in s]
        self.durations = durations if len(durations) > 0 else [np.zeros(len(seq)) for seq in self.data]
        self.flat_durations = [o for s in self.durations for o in s]
        # nump: total number of obersvations (# elements in all time-series)
        self.nump = np.sum(len(seq) for seq in self.data)
        # check that nump is correct
        if len(gt_display) > 0: assert(len(gt_display) == self.nump)
        if not self.cfg.collapsed: self.__rasterize_durations()

        if len(key_reg) == 0:
            self.__build_key_reg()
        else:
            self.key_reg = key_reg

        # L2: cardinality of observation space
        self.L2 = len(set(self.flat_data)) if len(key_reg) == 0 else len(key_reg.keys())
        if self.debug: print(self.L2)
        if self.debug: print(self.nump)

        # base probability mass distributed among all states
        self.cfg.base_prob = 1E-1 / self.L2
        self.gt_display = gt_display
        self.eval_mode = eval_mode
        # sample initial parameters
        self.__init_params()
        if self.debug: print('Starting...')

        self.__perform_mcmc()
        return self.mc

    def load_model(self, mc: ModelContainer):
        self.duration_model_active = True
        self.cfg = mc.cfg
        self.beta = mc.beta  # [0]
        self.pi = mc.pi  # [0]
        self.key_reg = mc.key_reg
        self.L2 = mc.L2
        super_states = mc.super_states
        psi = []
        theta = []
        for super_state in super_states:
            psi.append(super_state[0])  # [0])
            theta.append(super_state[1])  # [0])
        self.psi = np.array(psi)
        self.theta = np.array(theta)

        self.d_model = mc.dmodel
        self.d_idx = mc.didx
        self.d_dim = len(self.d_idx)
        dsuper_states = mc.dtheta
        # dpsi = []
        dtheta = []
        for dsuper_state in dsuper_states:
            # dpsi.append(dsuper_state[0][0])
            dtheta.append(dsuper_state)  # [0])
        # self.dpsi = np.array(dpsi)
        self.dtheta = np.array(dtheta)

    def transform(self, X, Xd=[]):
        X_new = []
        self.content = [None, None, None, None]
        if not all(isinstance(el, list) for el in X):
            X = [X]
            Xd = [Xd]
        X = [list(map(self.key_reg.get, session)) for session in X]
        Xd = [self.d_model.predict_proba(np.array(session).reshape(-1, 1))[:, self.d_idx] for session in Xd]
        self.__compute_bw_msgs(X, Xd)
        for seq, msgs, durs in zip(X, self.msgs, Xd):
            z = np.array(self.__init_forward_bw(seq[0], msgs[0])).argmax()
            x_new = [z]
            self.content[1] = z
            for obs_id in range(1, len(seq)):
                self.content[0] = seq[obs_id]
                self.content[3] = durs[obs_id]
                prev = seq[obs_id - 1]
                f, inter_ratio = self.__forward_step(prev, msgs[obs_id].reshape(self.cfg.L, 1))
                z = f.argmax()
                x_new.append(z)
                self.content[1] = z
            X_new.append(cp.deepcopy(x_new))
        return X_new

    def __perform_mcmc(self):
        # loop of mcmc iterations
        for mcmc_iteration in range(self.cfg.numIt):
            # init (aux-)variables + compute backward BW
            self.__init_iteration()

            # bookkeeping
            iteration_assignments = []
            intra_step_count = 0
            s_idx = -1
            # loop over each time-series
            for seq, msgs, durs in zip(self.data, self.msgs, self.duration_vec):
                s_idx += 1
                temp_assignments = []
                seg = []
                ids = []
                assert(len(temp_assignments) == 0)
                assert(len(seg) == 0)
                assert(len(ids) == 0)
                # sample latent variable assignment of first element of time-series
                # CONTENT: [current observation, previous super state, current super state, current state duration]
                self.content[1] = self.pp.draw(self.__init_forward_bw(seq[0], msgs[0]))
                # update auxiliary variables
                self.m[self.content[1]] += 1
                self.d[self.content[1], seq[0]] += 1
                self.g[self.content[1], self.cfg.boundary, seq[0]] += 1
                temp_assignments.append(int(self.content[1]))

                for obs_id in range(1, len(seq)):
                    self.content[0] = seq[obs_id]
                    self.content[3] = durs[obs_id]
                    prev = seq[obs_id - 1]
                    # compute distribution over z at observed node obs_id
                    f, inter_ratio = self.__forward_step(prev, msgs[obs_id].reshape(self.cfg.L, 1))

                    # sample z
                    # print(np.sum(f))
                    self.content[2] = self.pp.draw(f)

                    # record all assignments
                    temp_assignments.append(int(self.content[2]))
                    self.__increment_aux()
                    if self.content[1] == self.content[2]:
                        # inter-transition
                        if np.random.binomial(1, inter_ratio):
                            seg = cp.deepcopy(self.__record_transition(prev, seg, ids))
                            ids = [obs_id]
                        # intra-transition
                        else:
                            self.g[self.content[2], prev, self.content[0]] += 1
                            intra_step_count += 1
                            seg.append(cp.deepcopy(self.content[0]))
                            ids.append(obs_id)
                            if not self.cfg.collapsed:
                                self.dg[self.content[2], prev, self.content[0], :] += self.content[3]
                    # inter-transition
                    else:
                        seg = cp.deepcopy(self.__record_transition(prev, seg, ids))
                        ids = [obs_id]
                    self.content[1] = copy.deepcopy(self.content[2])
                self.__record_eos(seg, ids)
                iteration_assignments.append(temp_assignments)
            self.__finalize_iteration(intra_step_count, mcmc_iteration, iteration_assignments)

    # Baum-Welch Backward over entire sequence...
    # @profile
    def __compute_bw_msgs(self, data=[], duration=[]):
        if len(data) == 0:
            data = self.data
            duration = self.duration_vec
        # list of msgs for backward BW
        # each list item: list of backward messages from one time-series
        self.msgs = []
        # iterate over each time-series to compute its backward messages
        # can be parallelized!
        for seq, seq_dur in zip(data, duration):
            msgs = [np.zeros((len(seq), 1)), self.__compute_intra([seq[-1], self.cfg.boundary])]
            for obs_id in np.arange(len(seq) - 2, 0, -1):
                c_obs = seq[obs_id:obs_id + 2]
                c_dur = seq_dur[obs_id + 1]
                c_msg = msgs[-1]
                c_msg = cp.deepcopy(self.__backward_step(c_msg, c_obs, c_dur))
                if np.nansum(c_msg) == 0:
                    c_msg = np.ones(self.cfg.L)
                c_msg /= np.nansum(c_msg)
                msgs.append(c_msg)
            prior = np.multiply(self.beta.reshape(self.cfg.L, 1), self.theta[:, self.cfg.boundary, seq[0]].reshape(
                self.cfg.L, 1))
            msg_new = np.multiply(prior, msgs[-1].reshape(self.cfg.L, 1), dtype=np.float128)
            msgs.append(msg_new)
            self.msgs.append(msgs[::-1])

    def __backward_step(self, msg, c_obs, c_dur):
        # probability for intra transitions
        intra = self.__compute_intra(c_obs, c_dur)

        # (x, y) in a sequence
        sos = np.multiply(self.beta.reshape(self.cfg.L, 1), self.theta[:, self.cfg.boundary, c_obs[1]].reshape(
            self.cfg.L, 1))
        sos = np.multiply(sos, msg.reshape(self.cfg.L, 1))
        inter = np.multiply(self.__compute_intra((c_obs[0], -1)), np.dot(self.pi, sos).reshape(self.cfg.L, 1))
        intra = np.multiply(intra, msg.reshape(self.cfg.L, 1))
        return inter + intra

    def __compute_intra(self, c_obs, c_dur=[]):
        theta = self.theta[:, c_obs[0], c_obs[1]].reshape(self.cfg.L, 1)
        if not self.cfg.collapsed and -1 not in c_obs and self.duration_model_active:
            dm_temp = np.multiply(self.dtheta[:, c_obs[0], c_obs[1], :], c_dur.reshape(1, self.d_dim))
            dm_temp = np.sum(dm_temp, axis=1).reshape(self.cfg.L, 1)
            theta = np.multiply(theta, dm_temp).reshape(self.cfg.L, 1)
        return np.multiply(self.psi[:, c_obs[0]].reshape(self.cfg.L, 1), theta)

    def __init_forward_bw(self, data, msg):
        ret = np.multiply(self.beta.reshape(self.cfg.L, 1), self.theta[:, self.cfg.boundary, data].reshape(
            self.cfg.L, 1))
        ret = np.multiply(ret, msg).reshape(self.cfg.L, 1)
        ret /= np.nansum(ret)
        return ret

    # Compute the distribution over z at node i of observed variables
    def __forward_step(self, prev, msgs):
        """
        :type prev: int
        :type msgs: vector Lx1
        """
        c_obs = self.content[0]
        p_obs = prev
        past = self.content[1]

        inter_intra_end = self.theta[past, p_obs, self.cfg.boundary]
        theta = self.theta[past, p_obs, c_obs]
        if not self.cfg.collapsed and self.duration_model_active:
            theta *= np.dot(self.dtheta[past, p_obs, c_obs, :], self.content[3])
        intra = theta * msgs[past] * self.cfg.rho

        inter = self.pi[past, :].reshape(self.cfg.L, 1)
        inter = np.multiply(inter, self.theta[:, self.cfg.boundary, c_obs].reshape(self.cfg.L, 1))
        inter = np.multiply(inter, msgs)
        # probability of transition to another su given current observation
        # combined probability
        distf = inter * inter_intra_end
        # probability that su transition (c -> c) is an inter-transition
        inter_ratio = cp.deepcopy(distf[past])
        distf[past] += intra
        inter_ratio /= distf[past]
        inter_ratio = inter_ratio.astype(np.float64)
        distf /= np.nansum(distf)
        return distf, inter_ratio

    def __init_params(self):
        """
        parameters of the model
        beta: prior probability of super states
        pi: transition probability of super states
        psi: prior probability of internal states
        theta: internal transition probability of super states
        """
        self.duration_model_active = False
        self.betaMass = np.ones(self.cfg.L) * self.cfg.gamma
        self.piMass = np.ones(self.cfg.L) * self.cfg.alpha
        self.psiMass = np.ones(self.L2) * self.cfg.sigma
        self.dSuper = np.ones(self.d_dim) * self.cfg.d_gamma

        # containers for parameters
        self.pi = np.zeros((self.cfg.L, self.cfg.L))
        self.psi = np.zeros((self.cfg.L, self.L2), dtype=np.float32)
        self.theta = np.zeros((self.cfg.L, self.L2, self.L2))
        self.dpsi = np.zeros((self.L2, self.d_dim))
        self.dtheta = np.zeros((self.cfg.L, self.L2, self.L2, self.d_dim))
        self.beta = np.random.dirichlet(self.betaMass)

        if not self.cfg.collapsed:
            self.a = np.zeros((self.L2, self.d_dim))
            for act in range(self.L2):
                idx = np.where(np.array(self.flat_data) == act)[0]
                self.a[act, :] += np.sum(self.flat_duration_vec[idx, :], axis=0)
                self.dpsi[act, :] = np.random.dirichlet(self.dSuper)

        # loop over super states
        for cl in range(self.cfg.L):
            # sample probability distribution for transition from super state cl
            self.pi[cl, :] = np.random.dirichlet(self.piMass * self.beta)
            # sample distribution of states within super state
            self.psi[cl, :] = np.random.dirichlet(self.psiMass)

            # sample probability distributions for transitions within super state
            self.theta[cl, :, :] = self.cfg.base_prob
            self.dtheta[cl, :, :, :] = self.cfg.d_base_prob
            sel_ids = np.where(self.psi[cl, :] > self.cfg.recognition_threshold)[0]
            for nID in sel_ids:
                self.theta[cl, nID, sel_ids] = np.random.dirichlet(self.cfg.lam * self.psi[cl, sel_ids])
                self.theta[cl, nID, :] /= np.sum(self.theta[cl, nID, :])
                if not self.cfg.collapsed:
                    for cID in sel_ids:
                        self.dtheta[cl, nID, cID, :] = np.random.dirichlet(self.cfg.d_lam * self.dpsi[nID, :])
            self.theta[cl, -1, sel_ids] = np.random.dirichlet(self.cfg.lam * self.psi[cl, sel_ids])
            self.theta[cl, -1, :] /= np.sum(self.theta[cl, -1, :])

    def __sample_params(self):
        self.duration_model_active = True
        # re-sample parameters as functions of their priors + evidence (of last iteration)
        self.beta = np.random.dirichlet(self.betaMass + self.m)
        # duration model
        if not self.cfg.collapsed:
            for act in range(self.L2):
                self.dpsi[act, :] = np.random.dirichlet(self.dSuper + self.a[act, :])

        for cl in range(self.cfg.L):
            self.pi[cl, :] = np.random.dirichlet((self.piMass * self.beta + self.t[cl, :]))
            self.psi[cl, :] = np.random.dirichlet(self.psiMass + self.d[cl, :])

            sel_ids = np.where(self.psi[cl, :] > self.cfg.recognition_threshold)[0]
            for nID in sel_ids:
                self.theta[cl, nID, sel_ids] = np.random.dirichlet(
                    self.cfg.lam * self.psi[cl, sel_ids] + self.g[cl, nID, sel_ids])
                self.theta[cl, nID, :] /= np.sum(self.theta[cl, nID, :])
                if not self.cfg.collapsed:
                    for cID in sel_ids:
                        self.dtheta[cl, nID, cID, :] = np.random.dirichlet(self.cfg.d_lam * self.dpsi[nID,
                                                                                            :] + self.dg[cl, nID,
                                                                                                         cID, :])
            self.theta[cl, -1, sel_ids] = np.random.dirichlet(
                self.psi[cl, sel_ids] * self.cfg.lam + self.g[cl, -1, sel_ids])
            self.theta[cl, -1, :] /= np.sum(self.theta[cl, -1, :])

    def __rasterize_durations(self):
        estimator = bgm(self.cfg.k_gmm, max_iter=self.cfg.it_gmm, covariance_type='diag')
        data = np.array(self.flat_durations).reshape(len(self.flat_durations), -1)
        if len(data) / self.cfg.sub_data_size >= 2:
            data = shuffle(data, n_samples=self.cfg.sub_data_size)
        self.d_model = estimator.fit(data)
        self.d_idx = np.where(self.d_model.weight_concentration_[0] > self.cfg.d_importance_thresh)[0]
        self.d_dim = len(self.d_idx)
        self.duration_vec = []
        self.flat_duration_vec = []
        for seq in self.durations:
            seq = np.array(seq).reshape(len(seq), -1)
            dvec_temp = self.d_model.predict_proba(seq)[:, self.d_idx].reshape(len(seq), self.d_dim)
            self.duration_vec.append(dvec_temp)
            if len(self.flat_duration_vec) == 0:
                self.flat_duration_vec = dvec_temp
            else:
                self.flat_duration_vec = np.vstack((self.flat_duration_vec, dvec_temp))
        if self.debug: print('latent duration space dimensions: %i' % self.d_dim)

    def __init_iteration(self):
        # variables for counting
        self.__init_vars()
        # 1) Baum-Welch Backward
        self.__compute_bw_msgs()
        self.__init_aux_vars()
        self.content = [None, self.cfg.boundary, 0.0, 0.0]

    # @profile
    def __init_vars(self):
        self.content_updates = []
        self.m = np.zeros(self.cfg.L)  # for beta
        self.t = np.zeros((self.cfg.L, self.cfg.L))  # for pi_twindle
        self.d = np.zeros((self.cfg.L, self.L2))  # for psi
        self.g = np.zeros((self.cfg.L, self.L2, self.L2), dtype=np.uint16)
        self.dg = np.zeros((self.cfg.L, self.L2, self.L2, self.d_dim))

    # @profile
    def __init_aux_vars(self):
        self.content = []
        self.segs = [[] for i in range(self.cfg.L)]
        self.ids = [[] for i in range(self.cfg.L)]
        self.trans = [[] for i in range(self.cfg.L)]

    def __increment_aux(self):
        self.m[self.content[2]] += 1
        self.d[self.content[2], self.content[0]] += 1

    # @profile
    def __record_eos(self, seg, ids):
        self.d[self.content[2], self.cfg.boundary] += 1
        self.segs[self.content[2]].append(seg)
        self.ids[self.content[2]].append(ids)
        self.g[self.content[2], self.content[0], self.cfg.boundary] += 1

    # @profile
    def __record_transition(self, prev, seg, ids):
        self.d[self.content[1], self.cfg.boundary] += 1
        self.g[self.content[1], prev, self.cfg.boundary] += 1
        self.g[self.content[2], self.cfg.boundary, self.content[0]] += 1
        self.t[self.content[1], self.content[2]] += 1
        self.segs[self.content[1]].append(cp.deepcopy(seg))
        self.ids[self.content[1]].append(cp.deepcopy(ids))
        self.trans[self.content[1]].append(cp.deepcopy(self.content[2]))
        return [self.content[0]]

    def __finalize_iteration(self, intra_step_count, mcmc_iteration, temp_assignments):
        temp_assignments = [a for aseq in temp_assignments for a in aseq]
        self.__sample_params()
        self.__tracking(mcmc_iteration, temp_assignments)
        if self.eval_mode:
            tracking = np.array(self.cfg.assignment_counter[0])
            assignments = np.array([obs.astype(int).argmax() for obs in tracking])
            with open(self.cfg.eval_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(assignments)
        else:
            self.__iteration_summary(mcmc_iteration, intra_step_count, temp_assignments)
        if not self.eval_mode and mcmc_iteration in self.cfg.eval_ends:
            tracks = self.cfg.eval_intervals[np.where(self.cfg.eval_intervals[:, 1] == mcmc_iteration)]
            track_end = np.max(tracks[:, 0])
            self.__save_lbs(tracks.astype(int))
            if mcmc_iteration == track_end:
                del self.cfg.assignment_counter[track_end]

    def __build_key_reg(self):
        # data into correct form
        self.key_reg = dict()
        set_data = list(set(self.flat_data))
        for i in range(len(set_data)):
            self.key_reg[set_data[i]] = i
        self.data = [list(map(self.key_reg.get, session)) for session in self.data]

    # save the following attributes:
    # points        - the observations (set of sequences)
    # counts        - counts of assigned observations per cluster
    # endStates     - end-states encoding cluster exits
    # clusters      - unnormalized transition matrices of the clusters
    # stateDist     - distribution over nodes in a cluster
    # assignments   - recorded assignments for each point in the observation (see points)
    # gt            - ground-truth of the assignments if available
    # key_reg       - reverse dictionary of observation-ids (used by the algorithm) to true labels
    # inter-trans   - record of transitions between clusters
    #
    # note that the inter-trans recordings are simplified here. transitions between a node to itself is recorded as an intra-transition
    # @profile
    # def __save_result(self, tracks):
    #     for eval_set in tracks:
    #         tracking = self.cfg.assignment_counter[eval_set[0]]
    #         assignments = np.array([np.bincount(obs.astype(int)).argmax() for obs in tracking])
    #         theta_count = np.zeros((self.cfg.L, self.L2, self.L2), dtype=np.float32)
    #         beta_count = np.zeros(self.cfg.L)
    #         psi_count = np.zeros((self.cfg.L, self.L2))
    #         pi_count = np.zeros((self.L + 1, self.L + 1))
    #
    #         assert (self.data[0] > self.cfg.boundary)
    #         print(len(assignments))
    #         print(len(self.data))
    #         assert (len(assignments) == len(self.data))
    #         prev_c = assignments[0]
    #         prev_p = self.data[0]
    #         theta_count[prev_c, self.cfg.boundary, prev_p] += 1
    #         beta_count[prev_c] += 1
    #         psi_count[prev_c, prev_p] += 1
    #         pi_count[self.cfg.boundary, prev_c] += 1
    #
    #         for pos in range(1, self.nump):
    #             cur_c = assignments[pos]
    #             cur_p = self.data[pos]
    #
    #             # transition between super states
    #             if prev_p == self.cfg.boundary or prev_c != cur_c:
    #                 if prev_p != self.cfg.boundary:
    #                     theta_count[prev_c, prev_p, self.cfg.boundary] += 1
    #                     pi_count[prev_c, cur_c] += 1
    #                 else:
    #                     pi_count[self.cfg.boundary, cur_c] += 1
    #                 beta_count[cur_c] += 1
    #                 psi_count[cur_c, cur_p] += 1
    #                 theta_count[cur_c, self.cfg.boundary, cur_p] += 1
    #             # end of a sequence
    #             elif cur_p == self.cfg.boundary:
    #                 theta_count[prev_c, prev_p, self.cfg.boundary] += 1
    #                 pi_count[prev_c, self.cfg.boundary] += 1
    #             else:
    #                 theta_count[cur_c, prev_p, cur_p] += 1
    #                 beta_count[cur_c] += 1
    #                 psi_count[cur_c, cur_p] += 1
    #
    #             prev_p = cp.deepcopy(cur_p)
    #             prev_c = cp.deepcopy(cur_c)
    #
    #         # pad counts to avoid dividing of zero
    #         theta_count += 1E-3
    #         psi_count += 1E-3
    #         pi_count += 1E-3
    #
    #         theta_final = np.array(
    #             [np.divide(tc_row, np.sum(tc_row, axis=1).reshape(self.L2, 1)) for tc_row in theta_count])
    #         beta_final = beta_count / float(np.sum(beta_count))
    #         psi_final = np.divide(psi_count, np.sum(psi_count, axis=1).reshape(self.cfg.L, 1))
    #         pi_final = np.divide(pi_count, np.sum(pi_count, axis=1).reshape(self.cfg.L + 1, 1))
    #
    #         data = DataContainer(self.data, assignments, self.cfg.gt)
    #         model = ModelContainer(beta_final, psi_final, theta_final, pi_final, self.key_reg, self.L2, self.cfg.boundary)
    #
    #         save_path = self.cfg.log_path + 'final_' + str(eval_set[0]) + '-' + str(eval_set[1]) + self.cfg.extension
    #         with open(save_path, 'w') as f:
    #             pickle.dump([data, model], f)
    #             # print('saved: final')

    def __save_lbs(self, tracks):
        for track in tracks:
            tracking = self.cfg.assignment_counter[track[0]]
            # identify final assignments
            # most frequent assignment per observation
            final_assignments = np.array([obs.argmax() for obs in tracking])
            # identify final active super states
            final_super_state_ids = list(set(final_assignments))
            cfg = cp.deepcopy(self.cfg)
            cfg.L = len(final_super_state_ids)
            # store data, assignments, gt
            dc = DataContainer(self.data, final_assignments, self.gt_display)
            # identify MAP of final model parameters
            tracking_params = self.cfg.params_store[track[0]]
            final_model_beta_mean = np.mean(np.array(tracking_params['beta']), axis=0)[final_super_state_ids]
            # final_model_beta_var = np.var(np.array(tracking_params['beta']), axis=0)[final_super_state_ids]
            final_model_pi_mean = np.mean(tracking_params['pi'], axis=0)[np.meshgrid(final_super_state_ids, final_super_state_ids)]
            # final_model_pi_var = np.var(np.array(tracking_params['pi']), axis=0)[np.meshgrid(final_super_state_ids, final_super_state_ids)]
            final_beta = final_model_beta_mean  # (final_model_beta_mean, final_model_beta_var)
            final_pi = final_model_pi_mean  # (final_model_pi_mean, final_model_pi_var)

            final_super_states = []
            final_dtheta = []
            # final_dsuper_states = []
            # for each parameter of a super state
            for idx in final_super_state_ids:
                psi = np.array(tracking_params['psi' + str(idx)])
                theta = np.array(tracking_params[str(idx)])
                psi_mean = np.mean(psi, axis=0)
                # psi_var = np.var(psi, axis=0)
                theta_mean =  np.mean(theta, axis=0)
                # theta_var = np.var(theta, axis=0)
                final_psi = psi_mean  # (psi_mean, psi_var)
                final_theta = theta_mean  # (theta_mean, theta_var)
                final_super_states.append((final_psi, final_theta))

                # dpsi = np.array(tracking_params['dpsi' + str(idx)])
                dtheta = np.array(tracking_params['d' + str(idx)])
                # dpsi_mean = np.mean(dpsi, axis=0)
                # dpsi_var = np.var(dpsi, axis=0)
                dtheta_mean = np.mean(dtheta, axis=0)
                # dtheta_var = np.var(dtheta, axis=0)
                # final_dpsi = (dpsi_mean, dpsi_var)
                final_dtheta.append(dtheta_mean)  # (dtheta_mean, dtheta_var))
                # final_dsuper_states.append((final_dpsi, final_dtheta))

            mc = ModelContainer(final_beta, final_pi, final_super_states, self.key_reg, self.L2, cfg, self.d_model, self.d_idx, final_dtheta)
            mc.cfg.assignment_counter = None
            mc.cfg.params_store = None
            mc.cfg.active_counters = None
            mc.cfg.elapsed_time = None
            save_path = self.cfg.log_path + self.cfg.eval_name + str(track[0]) + '-' + str(track[1]) + self.cfg.extension
            with open(save_path, 'wb') as f:
                pickle.dump(mc, f, protocol=4)
            save_path = self.cfg.log_path + self.cfg.assignments_name + str(track[0]) + '-' + str(track[1]) + self.cfg.extension
            with open(save_path, 'wb') as f:
                pickle.dump(dc, f, protocol=4)
        self.mc = mc

    def __tracking(self, mcmc_iteration, temp_assignments):
        assert (self.nump == len(temp_assignments))
        if mcmc_iteration in self.cfg.eval_beginnings:
            self.cfg.active_counters.append(mcmc_iteration)
            self.cfg.assignment_counter[mcmc_iteration] = np.zeros((self.nump, self.cfg.L))
            keys = ['beta', 'pi'] + ['psi' + str(i) for i in range(self.cfg.L)] + [str(k) for k in range(self.cfg.L)] + ['d' + str(l) for l in range(self.cfg.L)]
            d = dict()
            for k in keys:
                d[k] = []
            self.cfg.params_store[mcmc_iteration] = d
        for idc in self.cfg.active_counters:
            self.cfg.assignment_counter[idc][np.arange(self.nump), temp_assignments] += 1
            self.cfg.params_store[idc]['beta'].append(self.beta)
            self.cfg.params_store[idc]['pi'].append(self.pi)
            for idx in range(self.cfg.L):
                self.cfg.params_store[idc]['psi' + str(idx)].append(self.psi[idx, :])
                self.cfg.params_store[idc][str(idx)].append(self.theta[idx, :, :])
                self.cfg.params_store[idc]['d' + str(idx)].append(self.dtheta[idx, :, :])

    def __iteration_summary(self, mcmc_iteration, intra_step_count, temp_assignments):
        gt = cp.deepcopy(self.gt_display)
        gt += np.abs(np.min(gt))
        bgt = np.bincount(gt)
        print('iteration: %i' % mcmc_iteration)
        hist_num_groups = np.zeros(self.cfg.L)
        hist_num_groups[len(self.m[self.m > self.nump * 1E-2]) - 1] += 1
        valG = len(self.m[self.m > self.nump * 5E-4]) - 1
        selG = np.sort(self.m)[::-1]
        candidates = [obs_id for obs_id, n in enumerate(self.m) if n in selG[:valG + 1]]
        # print(candidates)
        print('Number of active clusters: %i' % len(candidates))
        res = sorted([int(itm) for itm in self.m[candidates] if itm > self.nump * 1E-4], reverse=True)
        print(res)
        if len(self.gt_display) > 0:
            print(set(self.gt_display))
            print('GT: %s' % bgt[bgt > 0])
            print('NMI: %f' % ami(self.gt_display, temp_assignments))
        print('residue: %i' % (self.nump - np.sum(res)))

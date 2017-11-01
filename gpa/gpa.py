from segmentation import config
from segmentation.eIMMC import eIMMC
from sklearn.model_selection import GridSearchCV
import copy as cp
import numpy as np
import pickle


class GlobalPatternAnalysis:
    # init LBS by inferring immc model from data
    # 1. FROM DATA: input X:data, Xd:state durations, Xgt:ground truth
    # 2. FROM FILE: input X:path
    def __init__(self, X, Xd=None, Xgt=None):
        self.MIX_BASED = 0
        self.TIME_BASED = 1
        self.COUNT_BASED = 2
        self.clf = None

        if isinstance(X, str):
            self.mc = pickle.load(open(X, 'rb'))
            self.immc = eIMMC()
            self.immc.load_model(self.mc)
        else:
            Xgt = [itm for seq in Xgt for itm in seq]

            # fit model to data
            self.immc = eIMMC()
            if len(Xd) == 0:
                self.immc.cfg.collapsed = True
            self.mc = self.immc.fit(X, durations=Xd, gt_display=Xgt)

        # # convert data according to identified super states
        # X_transformed = []
        # for session in X:
        #     x_t = [dc.assignments.pop(0) for i in session]
        #     X_transformed.append(x_t)

    # project data into vector space spanned by super states
    # modus 0: mixed, 1: time-based, 2: count-based
    def project(self, Xe, Xed=[], modus=0):
        if len(Xed) == 0: modus = self.COUNT_BASED
        X_project = []
        if not all(isinstance(el, list) for el in Xe):
            Xe = [Xe]
            Xed = [Xed]
        if not all(isinstance(el, list) for seq in Xe for el in seq):
            Xe = [Xe]
            Xed = [Xed]

        for X_entity, Xd_entity in zip(Xe, Xed):
            X_transformed = self.immc.transform(X_entity, Xd_entity)
            X_flat = [ele for seq in X_transformed for ele in seq]
            if modus < self.COUNT_BASED:
                x_ent = np.zeros(self.mc.cfg.L)
                Xd_flat = [ele for seq in Xd_entity for ele in seq]
                x_ent[:np.max(X_flat) + 1] = np.bincount(X_flat, Xd_flat)
                if modus == 1: X_project.append(x_ent)
            if modus % 2 == 0:
                xd_ent = np.zeros(self.mc.cfg.L)
                idx, bincounts = np.unique(X_flat, return_counts=True)
                xd_ent[idx] += bincounts
                if modus == 2: X_project.append(xd_ent)
            if modus == 0: X_project.append(x_ent + xd_ent)
        return X_project

    def fit(self, fct, Xp, y, params):
        self.clf = GridSearchCV(fct, params, cv=5, scoring='f1_weighted')
        self.clf.fit(Xp, y)

    def predict(self, Xp):
        return self.clf.predict(Xp)

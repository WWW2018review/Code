import numpy as np
import copy as cp
import pickle
# import operator


class DataHandler:
    def init(self, data, eval_mode):
        self.data = data
        self.eval_mode = eval_mode
        self.processed = []
        self.processed_train = []
        self.processed_test = []
        self.count = []
        self.key_reg = dict()
        self.rev_key_reg = dict()
        self.d = np.zeros(2)

        self.ENDSTATE = 'Z'
        self.nump = -1

    def process_list(self):
        action_set = set(x for l in self.data for x in l)
        self.numActions = len(action_set)
        print(self.numActions)
        self.__register_all(action_set)
        self.t_count = dict()
        self.s_count = dict()
        self.e_count = dict()
        # print(self.key_reg[self.ENDSTATE])

        for seq in self.data:
            self.d[0] = self.key_reg[self.ENDSTATE]
            self.count.append(len(seq))
            self.nump += len(seq) + 1
            for obs in seq:
                assert(self.key_reg[obs] < self.key_reg[self.ENDSTATE])
                self.d[1] = self.key_reg[obs]
                self.processed.append(cp.deepcopy(self.d))
                self.__record_trans(self.d)
                self.d[0] = self.d[1]
            self.d[1] = self.key_reg[self.ENDSTATE]
            self.processed.append(cp.deepcopy(self.d))
        self.processed = np.array(self.processed)

        self.count.reverse()
        self.rev_key_reg.update(reversed(i) for i in self.key_reg.items())

        return self.processed, self.count, self.rev_key_reg, self.nump, self.numActions + 1

    def process_string(self):
        action_set = set(self.data)
        self.numActions = len(action_set)
        print(self.numActions)
        self.__register_all(action_set)
        self.d[0] = self.key_reg[self.ENDSTATE]

        self.nump = len(self.data)
        print(self.nump)
        self.count.append(self.nump)
        for i in range(self.nump):
            action = self.data[i]
            # self.__register_action(action)
            self.d[1] = self.key_reg[action]
            self.processed.append(cp.deepcopy(self.d))
            self.d[0] = self.d[1]
        self.d[1] = self.key_reg[self.ENDSTATE]
        self.processed.append(cp.deepcopy(self.d))
        self.processed = np.array(self.processed)
        # self.count[0] -= 1
        self.count.reverse()
        self.rev_key_reg.update(reversed(i) for i in self.key_reg.items())

        return self.processed, self.count, self.rev_key_reg, self.nump, self.numActions + 1

    def process_pandas(self, eval_mode=False, test_ratio=0.1, data_fraction_rate=1.0):
        action_set = set(self.data.action)
        self.numActions = len(action_set)
        print(self.numActions)
        self.state_space = np.zeros(self.numActions + 1)
        self.state_space_test = np.zeros(self.numActions + 1)
        self.__register_all(action_set)
        self.d[0] = self.key_reg[self.ENDSTATE]

        session_ids = self.data.groupby(level=0).indices.keys()
        num_sessions = len(session_ids)
        selected_ids = np.random.choice(np.arange(num_sessions), size=np.ceil(num_sessions * data_fraction_rate)) if data_fraction_rate < 1.0 else np.arange(num_sessions)
        for i in selected_ids:  # range(len(session_ids)):
            session = self.data.ix[session_ids[i]].action
            numSI = len(session)  # + 1
            session_list = []
            for i_id in range(numSI):
                action = session.iloc[i_id]
                # self.__register_action(action)

                self.d[1] = self.key_reg[action]
                assert(not(self.d[0] == self.d[1] == self.ENDSTATE))
                session_list.append(cp.deepcopy(self.d))
                self.d[0] = self.d[1]
            self.d[1] = self.key_reg[self.ENDSTATE]
            session_list.append(cp.deepcopy(self.d))
            assert(not(self.d[0] == self.d[1] == self.ENDSTATE))
            if not eval_mode or not np.random.binomial(1, test_ratio):
                self.processed_train += cp.deepcopy(session_list)
                self.nump += numSI + 1
                self.count.append(numSI)
                self.__count_states(session_list)
            else:
                self.processed_test += cp.deepcopy(session_list)
                self.__count_states(session_list, test=True)

            self.d[0] = self.d[1]

        self.processed = np.array(self.processed_train)
        self.nump += 1
        print(self.processed.shape[0])
        print(self.nump)
        self.__show_state_dist()
        # self.count[0] -= 1
        self.count.reverse()
        self.rev_key_reg.update(reversed(i) for i in self.key_reg.items())
        # assert(np.max(set([np.max(aa) for aa in self.processed])) >= self.numActions)
        if eval_mode:
            with open('../data/files/train_test.pickle', 'wb') as f:
                pickle.dump([self.processed_train, self.processed_test, self.key_reg], f)

        return self.processed, self.count, self.rev_key_reg, self.nump, self.numActions + 1

    def __register_action(self, action):
        if (len(self.key_reg) == 0):
            self.ticket = 0
            self.key_reg[action] = self.numActions
        if not(action in self.key_reg):
            self.key_reg[action] = self.ticket
            self.ticket += 1

    def __register_all(self, actions):
        entries = zip(actions, np.arange(self.numActions))
        self.key_reg = dict(entries)
        self.key_reg[self.ENDSTATE] = self.numActions

    def __record_trans(self, d):
        d = tuple(d)
        if d in self.t_count:
            self.t_count[d] += 1
        else:
            self.t_count[d] = 1

        if d[0] in self.s_count:
            self.s_count[d[0]] += 1
        else:
            self.s_count[d[0]] = 1

        if d[1] in self.e_count:
            self.e_count[d[1]] += 1
        else:
            self.e_count[d[1]] = 1

    def __count_states(self, data, test=False):
        for itm in data:
            self.state_space[int(itm[1])] += 1
            if test:
                self.state_space_test[int(itm[1])] += 1

    def __show_state_dist(self):
        hist, bin_edges = np.histogram(self.state_space, bins=range(1000))
        import matplotlib.pyplot as plt
        plt.bar(bin_edges[:-1], hist, width=1)
        plt.xlim(min(bin_edges), max(bin_edges))
        plt.show()
        exit(-1)

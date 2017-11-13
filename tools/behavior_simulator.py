import pickle
import numpy as np
import copy as cp
from numpy import random as draw


# ASSUMPTIONS:
# 1. human behavior more diverse than bots (hnum vs bnum; variant of super states prior; etc)
# 2. human behavior is also less straight forward -> more noise, etc (noise-level; size of super states; etc)
# TBD
class BehaviorGenerator:
    def __init__(self):
        self.data = []
        self.duration = []
        self.gt = []
        self.state_space_cardinality = 0
        self.cardinalities = []
        self.spaces = []
        self.entities = []
        self.super_states = []
        self.global_state_prior = []
        self.type_super_state_priors = []
        self.etype = None

        self.duration_priors = ()
        self.duration_parameters = ()
        self.type_state_durations = []
        self.type_super_duration_shift = [[], []]
        self.type_state_duration_priors = []
        self.type_entity_duration_shift = ()

        self.BERNOULLI = 0
        self.BETA = 1
        self.CHISQRT = 2
        self.CHOICE = 3
        self.DIRICHLET = 4
        self.GAMMA = 5
        self.LOGNORMAL = 6
        self.NORMAL = 7
        self.NORMAL_UNIT = 8
        self.PARETO = 9
        self.UNIFORM = 10
        self.WEIBULL = 11

        # CONFIGURATION

        # CONFIG: general settings
        # total number of super states
        # super_state_space_cardinality = 10
        # total number of super states used by humans/bots
        human_super_state_space_card = 0
        bot_super_state_space_card = 0
        shared_super_state_space_card = 10
        self.cardinalities = [human_super_state_space_card, bot_super_state_space_card, shared_super_state_space_card]
        # total number of unique states shared by the super states
        self.state_space_cardinality = 30
        # for super state distribution prior (Dir)
        self.hyperparameter_beta = 5
        # for state distribution prior (Dir)
        self.hyperparameter_gamma = 3
        # for subordinate state priors (state priors conditioned on a super state) (Dir)
        self.hyperparameter_lambda = 1
        # minimal cardinality of each super state
        self.min_states = 5
        # maximum cardinality of each super state
        self.max_states = 20
        # mean value of state cardinality per super state (Normal)
        self.avg_cardinality_super = (self.max_states - self.min_states) / 2.0
        # variance in state cardinalities (Normal)
        self.var_cardinality_super = 5
        # hyperparameter: super state session length sample (super_state_card * gamma(len_param))
        self.len_param = 2
        # minimum super state session length
        self.min_len = 4

        self.behavior_noise = [0.1, 0.0]

        # CONFIG: duration parameters
        # prior of human/bot duration behavior
        self.duration_priors = [self.LOGNORMAL, self.LOGNORMAL]  # (self.LOGNORMAL, self.WEIBULL)
        # corresponding hyperparameters
        self.duration_parameters = [[2.0, 1.0], [1.5, 1.0]]
        # hyperparameters for entity duration shift behavior
        dur_params = [2.0, 1.0]
        dur_noise = [(0.5, 10), (0.3, 10)]
        self.type_entity_duration_shift = (dur_params, dur_noise)
        self.max_len_dur = 2400

        # CONFIG: parameters concerning entity creation
        # ratio of bots (bots / humans)
        self.num_types = 2
        self.type_ratios = [.5, .5]  # [.4, .3, .3]
        self.atomic_types = [0, 1]
        self.mixed_types = [2]
        self.mixing_cands = [[0, 1]]
        self.mixing_ratios = [[.5, .5]]
        self.hyperparameter_diversity = (6., 4.)  #2.)  # Gamma
        self.hyperparameter_session_len = (6, 10)  #
        self.type_session_len_noise = (.1, .05)
        self.max_len_session = 15

        # general noise in final durations
        self.dur_noise = 0.1

    def sample(self, n, size):
        self.sample_behaviormodel()
        self.sample_durationmodel()
        self.sample_entities(n=n)
        self.sample_data(size=size)

    def sample_behaviormodel(self):
        global_super_state_prior = self.__draw(self.DIRICHLET, [self.hyperparameter_beta] * self.cardinalities[2], 1)[0].tolist()
        global_super_state_prior = global_super_state_prior + self.__draw(self.DIRICHLET, [self.hyperparameter_beta] * self.cardinalities[0], 1)[0].tolist()
        global_super_state_prior = global_super_state_prior + self.__draw(self.DIRICHLET, [self.hyperparameter_beta] * self.cardinalities[1], 1)[0].tolist()
        #  distribution over activities/observations (global)
        self.global_state_prior = self.__draw(self.DIRICHLET, [self.hyperparameter_gamma] * self.state_space_cardinality, 1)[0]

        # cardinality of state space (each super state)
        super_state_spaces = np.round(self.__draw(self.NORMAL, self.avg_cardinality_super, self.var_cardinality_super, np.sum(self.cardinalities)))
        super_state_spaces = np.maximum(self.min_states, np.minimum(self.max_states, super_state_spaces))

        # generate each super state entity
        for super_state_id in range(np.sum(self.cardinalities)):
            shared_super_state = super_state_id < self.cardinalities[2]
            # cardinality
            state_space_card = int(super_state_spaces[super_state_id])
            # set of active states
            state_space = self.__draw(self.CHOICE, self.state_space_cardinality, self.global_state_prior, state_space_card)

            # distribution over states (local)
            pmass_acts = self.global_state_prior[state_space]
            pmass_acts /= np.sum(pmass_acts)

            if shared_super_state:
                hum_states_prior, hum_state_transitions = self.__build_super_state(state_space, pmass_acts, state_space_card)
                bot_states_prior, bot_state_transitions = self.__build_super_state(state_space, pmass_acts, state_space_card)
                self.super_states.append(({
                    'cardinality': state_space_card,
                    'super_state_ids': state_space,
                    'states_prior': hum_states_prior,
                    'state_transitions': hum_state_transitions
                }, {
                    'cardinality': state_space_card,
                    'super_state_ids': state_space,
                    'states_prior': bot_states_prior,
                    'state_transitions': bot_state_transitions
                }))
            else:
                states_prior, state_transitions = self.__build_super_state(state_space, pmass_acts, state_space_card)
                # store generated super state
                """
                super states:
                    num active states
                    ids of active states
                    initial states
                    state distribution
                    state transitions
                """
                self.super_states.append({
                    'cardinality': state_space_card,
                    'super_state_ids': state_space,
                    'states_prior': states_prior,
                    'state_transitions': state_transitions
                })

        # sample cyborg-/bot-/human super state space
        # sample super state spaces (human/bot/cyborg)
        offset_ids = [2, 0]
        for atype in self.atomic_types:
            offset = np.sum(self.cardinalities[offset_ids[i]] for i in range(atype + 1))
            type_space = np.concatenate((np.arange(self.cardinalities[2]), np.arange(offset, offset + self.cardinalities[atype])))
            self.spaces.append(cp.deepcopy(type_space))
            # sample super state prior
            tprior = np.zeros(np.sum(self.cardinalities))
            tprior[type_space] = self.__draw(self.DIRICHLET, (1 + np.array(global_super_state_prior)[type_space]).tolist(), 1)[0]
            tprior = (tprior + 1E-7) / np.sum(tprior + 1E-7)
            # store super state space and prior conditioned on type (human/bot/cyborg)
            self.type_super_state_priors.append(tprior)

    # currently reasonable supported duration distributions: normal, lognormal, uniform, weibull
    def sample_durationmodel(self):
        # state duration priors conditioned on entity type
        for atype in self.atomic_types:
            type_duration_prior = self.__draw(self.duration_priors[atype], *self.duration_parameters[atype], self.state_space_cardinality + 1)
            self.type_state_duration_priors.append(cp.deepcopy(type_duration_prior))

            # avg state durations conditioned on entity type, state transition
            tdur = np.array([np.multiply(self.type_state_duration_priors[atype], self.__draw(self.NORMAL_UNIT, self.state_space_cardinality + 1) * 2) for i in range(self.state_space_cardinality)])
            self.type_state_durations.append(cp.deepcopy(tdur))

            # average state duration SHIFT conditioned on the entity type, super state
            for super_state in self.super_states:
                self.type_super_duration_shift[atype].append(np.abs(self.__draw(self.NORMAL, 1, 0.5, self.state_space_cardinality + 1)))

    def sample_entities(self, n=10000):
        for entity in range(n):
            # (0) human or bot (1)
            entity_type = self.__draw(self.CHOICE, range(self.num_types), self.type_ratios, 1)[0]

            if entity_type in self.atomic_types:
                num_super, super_states, super_states_dist, super_state_transitions, avg_session_len, dur_shift, dur_noise_level = self.__build_entity([entity_type])
            else:
                num_super, super_states, super_states_dist, super_state_transitions, avg_session_len, dur_shift, dur_noise_level = self.__build_entity(self.mixing_cands[self.mixed_types.index(entity_type)])

            """
            entities:
                entity type
                ...
            """
            self.entities.append({'entity_type': entity_type,
                                  'num_super': num_super,
                                  'super_state_ids': super_states,
                                  'super_state_dist': super_states_dist,
                                  'super_state_transitions': super_state_transitions,
                                  'avg_super_per_session': avg_session_len,
                                  'duration_shift': dur_shift,
                                  'duration_noise_level': dur_noise_level})

    def sample_data(self, size=200000):
        self.data = []
        self.duration = []
        self.gt = []
        self.etype = []
        avg_dpoints = size / len(self.entities)
        for entity in self.entities:
            t_ent = entity['entity_type']
            mixed = t_ent in self.mixed_types
            eidx = 0
            if mixed:
                candidates = self.mixing_cands[self.mixed_types.index(t_ent)]
                p_cands = self.mixing_ratios[self.mixed_types.index(t_ent)]
            self.etype.append(t_ent)

            # deviation of each entity of the avg amount of observed data points
            dev = self.__draw(self.NORMAL_UNIT, 1) * 2
            num_dpoints = np.round(dev * avg_dpoints)
            # data points of current entity
            count_points = 0

            # current super state and its corresponding state
            current_status = [-1, -1]
            data_entity = []
            duration_entity = []
            gt_entity = []
            session_data = []
            session_duration = []
            session_gt = []
            while count_points < num_dpoints or current_status[1] >= 0 or len(session_data) == 0:
                # if previous super state has ended or non was started yet
                # sample next super state
                if current_status[1] < 0:
                    # if end of session reset session information and store previous session
                    if current_status[0] >= 0 and self.__draw(self.BERNOULLI, 1, 1.0 / entity['avg_super_per_session'][eidx]):
                        # previous session has ended
                        current_status[0] = -1
                        if len(session_data) > 0:
                            # store session details
                            print('length of session: %i' % len(session_data))
                            data_entity.append(cp.deepcopy(session_data))
                            gt_entity.append(cp.deepcopy(session_gt))
                            duration_entity.append(cp.deepcopy(session_duration))
                            # reset aux variables
                            session_data = []
                            session_duration = []
                            session_gt = []
                            if mixed:
                                eidx = self.__draw(self.CHOICE, candidates, p_cands, 1)[0]

                    # sample next super state
                    # new session = prop to super state prior dist
                    # active session = prop to super state transitions
                    if not current_status[0] < 0:
                        dist = entity['super_state_transitions'][eidx][entity['super_state_ids'][eidx].index(current_status[0]), :]
                    else:
                        dist = entity['super_state_dist'][eidx]
                    current_status[0] = int(self.__draw(self.CHOICE, entity['super_state_ids'][eidx], dist, 1)[0])
                    super_state = self.super_states[current_status[0]]
                    if current_status[0] < self.cardinalities[2]:
                        super_state = super_state[eidx]

                # sample next state
                dist = super_state['state_transitions'][current_status[1], :]
                dist = dist / np.sum(dist)
                next_state = self.__draw(self.CHOICE, range(self.state_space_cardinality + 1), dist, 1)[0]

                # sample duration
                if current_status[1] >= 0:
                    if self.__draw(self.BERNOULLI, 1, entity['duration_noise_level'][eidx]):
                        dur_state = self.__draw(self.UNIFORM, 1, self.max_len_dur)
                    else:
                        dur_state = self.type_state_durations[eidx][current_status[1], next_state]
                        dur_state = dur_state * self.type_super_duration_shift[eidx][current_status[0]][current_status[1]]
                        dur_state = dur_state * entity['duration_shift'][eidx] * self.__draw(self.NORMAL, 1, self.dur_noise, 1)
                    # RECORD DURATION
                    session_duration.append(dur_state)
                current_status[1] = cp.deepcopy(next_state)

                # check for end of super state
                if current_status[1] == self.state_space_cardinality:
                    current_status[1] = -1
                # RECORD SAMPLES
                # store super state, state and sample corresponding duration
                else:
                    session_data.append(cp.deepcopy(current_status[1]))
                    # gt: super state (+ sign, if human; - sign otherwise
                    session_gt.append((1 + current_status[0]) * (1 + (-2 * eidx)))
                # update amount of sampled observations of entity
                count_points += 1
            data_entity.append(cp.deepcopy(session_data))
            print('number of sessions of entity: %i' % len(data_entity))
            gt_entity.append(cp.deepcopy(session_gt))
            duration_entity.append(cp.deepcopy(session_duration))
            self.data.append(cp.deepcopy(data_entity))
            self.duration.append(cp.deepcopy(duration_entity))
            self.gt.append(cp.deepcopy(gt_entity))
        gMem = [session for entity in self.gt for session in entity]
        gt = [itm for seq in gMem for itm in seq]
        print(set(gt))
        gt += np.abs(np.min(gt))
        print(np.bincount(gt))

    # HELPER FUNCTIONS

    # sample from various distributions
    # input: distribution type + parameters (note: order according to documentation of functions)
    def __draw(self, *args):
        if args[0] == self.CHOICE:
            return sorted(draw.choice(args[1], p=args[2], replace=False, size=args[3]))
        return {
            # a, b, size
            self.BETA: draw.beta,
            # loc, scale, size
            self.NORMAL: draw.normal,
            # low, high, size
            self.UNIFORM: draw.uniform,
            # loc, scale, size
            self.GAMMA: draw.gamma,
            # loc, scale, size
            self.LOGNORMAL: draw.lognormal,
            # p, size
            self.BERNOULLI: draw.binomial,
            # p, size
            self.DIRICHLET: draw.dirichlet,
            # a, size
            self.PARETO: draw.pareto,
            # param, size
            self.WEIBULL: draw.weibull,
            # param, size
            self.CHISQRT: draw.chisquare,
            # dim, dim, dim, ...
            self.NORMAL_UNIT: draw.rand,
        }[args[0]](*args[1:])

    def __build_entity(self, entity_list):
        num_super = []
        super_states = []
        super_states_dist = []
        super_state_transitions = []
        avg_session_len = []
        dur_shift = []
        dur_noise_level = []
        for entity_type in entity_list:
            # number of used super states by entity
            num_super.append(int(np.maximum(np.minimum(self.cardinalities[entity_type], np.round(self.__draw(self.GAMMA, self.hyperparameter_diversity[entity_type]))), 1)))
            # ids of super states used
            p = self.type_super_state_priors[entity_type][self.spaces[entity_type]] / np.sum(self.type_super_state_priors[entity_type][self.spaces[entity_type]])
            super_states_temp = self.__draw(self.CHOICE, self.spaces[entity_type], p, num_super[-1])
            super_states.append([int(i) for i in super_states_temp])

            # distribution over super states
            p = self.type_super_state_priors[entity_type][super_states[-1]]
            p /= np.sum(p)
            super_states_dist.append(self.__draw(self.DIRICHLET, p + 1.0, 1)[0])

            # transition between super states
            super_state_transitions.append(self.__draw(self.DIRICHLET, super_states_dist[-1], num_super[-1]))

            # entity related general duration shift
            dur_shift.append(self.__draw(self.GAMMA, self.type_entity_duration_shift[0][entity_type], 1))
            dur_noise_level.append(self.__draw(self.BETA, *self.type_entity_duration_shift[1][entity_type], 1))

            # avg session length = avg number of super states per session
            if self.__draw(self.BERNOULLI, 1, self.type_session_len_noise[entity_type]):
                avg_session_len.append(self.__draw(self.UNIFORM, 1, self.max_len_session))
            else:
                avg_session_len.append(np.maximum(np.round(self.__draw(self.GAMMA, self.hyperparameter_session_len[entity_type], 1)), 1))
        return num_super, super_states, super_states_dist, super_state_transitions, avg_session_len, dur_shift, dur_noise_level

    def __build_super_state(self, state_space, pmass_acts, state_space_card):
        states_prior = np.zeros(self.state_space_cardinality)
        states_prior[state_space] = self.__draw(self.DIRICHLET, (pmass_acts + self.hyperparameter_lambda), 1)
        state_transitions = np.zeros((self.state_space_cardinality, self.state_space_cardinality))
        for state in state_space:
            state_transitions[state, state_space] = np.array(self.__draw(self.DIRICHLET, states_prior[state_space], 1))

        # number, id and dist of initial states
        num_init = int(np.ceil(self.__draw(self.BETA, 2, 5, 1) * state_space_card))

        initial_states = self.__draw(self.CHOICE, state_space, states_prior[state_space], num_init)
        initial_state_dist = self.__draw(self.DIRICHLET, states_prior[initial_states] + 1, 1)
        initial_dist = np.zeros(self.state_space_cardinality + 1)
        initial_dist[initial_states] = initial_state_dist[0]

        # adjust states prior
        sample = []
        for session in range(500):
            sample.append(self.__draw(self.CHOICE, state_space, initial_dist[state_space], 1)[0])
            for click in range(49):
                sample.append(self.__draw(self.CHOICE, state_space, state_transitions[sample[-1], state_space], 1)[0])
        bcount = np.bincount(sample, minlength=self.state_space_cardinality)
        states_prior = bcount / np.sum(bcount)

        # end state deployment
        super_state_length = np.maximum(self.min_len, np.round(state_space_card * self.__draw(self.GAMMA, self.len_param, 1)))
        end_state_dist = np.multiply(self.__draw(self.DIRICHLET, states_prior, 1), states_prior)
        end_state_dist = end_state_dist / (np.sum(end_state_dist) * super_state_length)

        # combine transitions with initial- and end-state distributions
        state_transitions = np.multiply(state_transitions, 1 - end_state_dist.reshape(self.state_space_cardinality, 1))
        state_transitions = np.hstack((state_transitions, end_state_dist.reshape(self.state_space_cardinality, 1)))
        state_transitions = np.vstack((state_transitions, initial_dist))
        return states_prior, state_transitions


# Types of data sets:
# 1. different ratios of overlapping super states (0.0, 0.25, 0.5, 0.75, 1.0)
# 2. different duration models per class
# 3. sybil and compromised accounts
bg = BehaviorGenerator()
bg.sample(n=1000, size=500000)
print(np.sum([len(session) for e in bg.data for session in e]))
with open('../data/small/synth_behavior_data_3c_100_1k500k2.pckl', 'wb') as f:
    pickle.dump([bg.data, bg.duration, bg.gt, bg.etype], f)

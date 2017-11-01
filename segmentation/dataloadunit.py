import pickle
import pandas as pd
# from tools.data_generator import BusinessProcessDatasetGenerator


class DataLoadUnit:
    def load_data(self, dID):
        self.basepath = '../data/files/'
        switcher = {
            0: self.__load_ratebeer,
            1: self.__load_fb,
            2: self.__load_moby,
            3: self.__nothing,
        }

        func = switcher.get(dID, self.__nothing)

        return func()

    def __load_ratebeer(self):
        with open(self.basepath + 'beeradvocate_sequence', 'rb') as f:
            return pickle.load(f)

    def __load_fb(self):
        path = self.basepath + 'prepared.csv'
        sessions = pd.read_csv(path, index_col=[0, 1])
        print("DATA LOADED")
        return sessions

    def __load_moby(self):
        with open(self.basepath + 'moby_dick_ch1-10', 'r') as myfile:
            return myfile.read()

    def __nothing(self):
        print('Generating data...')
        return []

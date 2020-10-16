import pickle
import numpy as np


class Recorder(object):
    def __init__(self, stuff_to_record):
        """
        :param stuff_to_record: a list of strings
        Those strings are keys for the dictionary of recorded things
        (could be reward, loss, action, parameters, gradients, evaluation metric, etc.)
        """
        self.tape = {}
        for entity_name in stuff_to_record:
            assert isinstance(entity_name, str)
            self.tape[entity_name] = []

    def append(self, key, value):
        self.tape[key].append(value)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.tape, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def init_from_pickle_file(cls, filename):
        with open(filename, 'rb') as f:
            loaded_tape = pickle.load(f)
        instance = cls(stuff_to_record=loaded_tape.keys())
        instance.tape.update(loaded_tape)

        return instance


class EpisodeRecorder(Recorder):
    """
    Object used to encapsulate all episode data
    """

    def __init__(self, stuff_to_record):
        for name in stuff_to_record:
            assert name in ['obs', 'action', 'reward', 'next_obs'], \
                f"{name} not in ['obs', 'action', 'reward', 'next_obs']"
        super(EpisodeRecorder, self).__init__(stuff_to_record)

    def add_step(self, obs, action, reward, next_obs):
        if 'obs' in self.tape.keys():
            self.tape['obs'].append(obs)
        if 'action' in self.tape.keys():
            self.tape['action'].append(action)
        if 'reward' in self.tape.keys():
            self.tape['reward'].append(reward)
        if 'next_obs' in self.tape.keys():
            self.tape['next_obs'].append(next_obs)

    def get_total_reward(self):
        return np.stack(self.tape['reward']).sum(axis=0)


class TrainingRecorder(Recorder):
    """
    Object used to encapsulate all training data that we are interested in
    Every variable in the recorder are saved in a pickle file as training progresses
    """

    def __init__(self, stuff_to_record):
        super(TrainingRecorder, self).__init__(stuff_to_record)

    def mean_over_subsets(self, key, subset_size):
        all_data = np.vstack(self.tape[key]).T
        assert all_data.shape[1] % subset_size == 0

        reshaped_data = np.reshape(all_data, newshape=(all_data.shape[0],
                                                       all_data.shape[1] // subset_size,
                                                       subset_size))

        means = np.mean(reshaped_data, axis=2, keepdims=False)
        stds = np.std(reshaped_data, axis=2, keepdims=False)

        return means, stds

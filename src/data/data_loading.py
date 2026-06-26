import pickle

def load_dataset(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_trial(data, subject_id, trial_id):
    return data[subject_id][trial_id]


def iter_trials(data):
    for subject_id, trials in data.items():
        for trial_id, trial in trials.items():
            yield subject_id, trial_id, trial
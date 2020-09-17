from sklearn import linear_model
import numpy as np
import json
from main_utils.utils import Timer, plot_conditional_idx_improvment
import sys
from tune import tune_with_pygor_from_file, tune_with_playground_from_file



# NEW  ------


class cs_compensation_model:

    def __init__(self, config_file):
        with open(config_file) as f:
            configs = json.load(f)

        self.cs_configs = configs["cs_compensation"]
        self.cs_gates = configs["cs_gates"]
        self.device_gates = configs["device_gates"]
        self.characterisation_range = configs["characterisation_range"]
        self. ml_model = configs["scikit_linear"]


def cs_compensator_data(config_file, jump, measure, check):
    with open(config_file) as f:
        configs = json.load(f)["cs_compensation"]

    cs_gates = configs["cs_gates"]
    device_gates = configs["device_gates"]
    characterisation_range = configs["characterisation_range"]
    ml_model = configs["scikit_linear"]

    chan_no = configs['chan_no']

    inv_timer = Timer()

    pygor_path = configs.get('path_to_pygor', None)
    if pygor_path is not None:
        sys.path.insert(0, pygor_path)
    import Pygor
    pygor = Pygor.Experiment(xmlip=configs.get('ip', None))

    gates = configs['gates']
    plunger_gates = configs['plunger_gates']


    def jump(params, plungers=False):
        # print(params)
        if plungers:
            labels = plunger_gates
        else:
            labels = gates
        pygor.setvals(labels, params)
        return params

    def measure():
        cvl = pygor.do0d()[chan_no][0]
        return cvl



    # -----------

def generate_cs_data(config_file):

    with open(config_file) as f:
        configs = json.load(f)

    pygor_path = configs.get('path_to_pygor', None)
    if pygor_path is not None:
        sys.path.insert(0, pygor_path)
    import Pygor
    pygor = Pygor.Experiment(xmlip=configs.get('ip', None))

    cs_gates = configs["gates"]
    device_gates = configs["device_gates"]
    characterisation_range = configs["characterisation_range"]
    characterisation_points = configs['characterisation_points']

    measurements = np.linspace(characterisation_range[0], characterisation_range[1], characterisation_points)


    for gate in device_gates:

        pygor.setvals(device_gates, [0] * len(device_gates))
        print("set: ", device_gates, [0] * len(device_gates))

        for measurement in measurements:

            save_name = str(gate) + "_" + str(measurement) + "_tuning.pkl"
            file_path = configs["save_dir"] + save_name

            print("setting: ", gate, measurement)
            pygor.setval(gate, measurement)

            try:
                results, sampler = tune_with_playground_from_file(config_file)
                sampler.t.add(characterised_gate=[gate], gate_voltage=[measurement])
                sampler.t.save(track=sampler.t["track"], file_pth=file_path)
                print("run pygor.")

            except TypeError:
                break





"""
Right now this is a very slow way of doing it but I am just seeing if it works. 
"""

"""
class compensator:

     def __init__(self, gate_comps):

        # List of gate_comp objects
        self.gate_comps = gate_comps

        #self.comp_models = np.array(self.gate_comps)


    def __call__(self, device_gate_vols):
        # n-d array of gate voltages : device_gate_vols
        # n-d array of functions : self.function

        # get scale matrix output of each in self.function
        output = [gate(value) for gate, value in zip(self.gate_comps, device_gate_vols)]

        # multiply each matrix in the output (recursively)

        # return
"""




def comp_matrix(gates, values):
    if len(gates) != len(values):
        raise ValueError("Different number of gate compensators and gate values")

    output = [gate(value) for gate, value in zip(gates, values)]




    out = np.eye(gates[0].n)

    for gate, value in zip(gates, values):
        mat = gate(value)
        out = out @ mat

    return out



class single_gate_comp:

    def __init__(self, x, y):
        """
        var x: list of gate voltage values for which transformations have been made
        var y: array of transform element values for a single gate-gate pair
        """

        self.model = linear_model.BayesianRidge()

        self.x = np.array(x).reshape(-1, 1)
        self.y = y

    def train(self):
        self.model.fit(self.x, self.y)

    def get_comp_value(self, gate_voltage):
        return self.model.predict([[gate_voltage]])

    def __call__(self, gate_voltage):
        return self.model.predict([[gate_voltage]])


class gate_comp:

    def __init__(self, cs_gates, x, single_gate_transforms):
        self.gate_models = []
        self.n = len(cs_gates)
        for cs_gate in cs_gates:
            self.gate_models.append(single_gate_comp(x, single_gate_transforms[:, cs_gate, cs_gate]))
        self.train()

    def train(self):
        for model in self.gate_models:
            model.train()

    def __call__(self, value):

        # This is going to be slow
        return np.linalg.inv(np.diag([model(value)[0] for model in self.gate_models]))


"""
cs_gates = [0, 1]

gate3 = gate_comp(cs_gates, x, np.array(transforms[0]))
gate4 = gate_comp(cs_gates, x, np.array(transforms[1]))
gate5 = gate_comp(cs_gates, x, np.array(transforms[2]))

gates = np.array([gate3, gate4, gate5])
"""

def comp_matrix(gates, values):
    if len(gates) != len(values):
        raise ValueError("Different number of gate compensators and gate values")

    output = [gate(value) for gate, value in zip(gates, values)]




    out = np.eye(gates[0].n)

    for gate, value in zip(gates, values):
        mat = gate(value)
        out = out @ mat

    return out

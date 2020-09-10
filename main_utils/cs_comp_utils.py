from sklearn import linear_model
import numpy as np

"""
Right now this is a very slow way of doing it but I am just seeing if it works. 
"""

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

        # This is going to be too slow
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

    out = np.eye(gates[0].n)

    for gate, value in zip(gates, values):
        mat = gate(value)
        out = out @ mat

    return out

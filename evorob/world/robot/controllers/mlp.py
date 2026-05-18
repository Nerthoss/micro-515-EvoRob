import numpy as np

from evorob.world.robot.controllers.base import Controller


class NeuralNetworkController(Controller):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 16,
    ):
        """
        A minimalistic Neural Network, using numpy.
        - One hidden layer with tanh activation
        - Output layer with tanh activation
        - No symmetry enforcement

        :param int input_size: Size of input vector
        :param int hidden_size: Size of hidden layer
        :param int output_size: Size of output vector
        """
        self.n_input = input_size
        self.n_output = output_size
        self.n_hidden = hidden_size
        self.n_con1 = input_size * hidden_size
        self.n_con2 = hidden_size * output_size
        self.lin = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.output = np.random.uniform(-1, 1, (output_size, hidden_size))
        self.n_params = self.get_num_params()

    def get_action(self, state):
        assert state.shape[-1] == self.n_input, (
            "State does not correspond with expected input size"
        )

        hid_l = np.tanh(state @ self.lin.T)
        output_l = np.tanh(hid_l @ self.output.T)
        return np.clip(output_l, -1.0, 1.0)

    def set_weights(self, weights):
        """
        Set weights of NN.

        :param np.ndarray weights: Vector of weights
        """
        assert len(weights) == self.n_con1 + self.n_con2, (
            f"Got {len(weights)} but expected {self.n_con1 + self.n_con2}"
        )
        weight_matrix1 = weights[: self.n_con1].reshape(self.lin.shape)
        weight_matrix2 = weights[-self.n_con2 :].reshape(self.output.shape)
        self.lin = weight_matrix1
        self.output = weight_matrix2

    def geno2pheno(self, genotype):
        """Alias for set_weights (genotype to phenotype mapping)."""
        self.set_weights(genotype)

    def get_num_params(self):
        """Return the total number of parameters in the network."""
        return self.n_con1 + self.n_con2

    def reset_controller(self, batch_size=1) -> None:
        pass

    class SymmetricNeuralNetworkController(Controller):
        """
        MLP with bilateral symmetry enforcement.

        The network only produces actions for 4 joints (one side of each leg pair).
        The other 4 are obtained by mirroring: hip action is negated (flips yaw
        direction), knee action is copied as-is.

        Default joint ordering assumed (matches FinalWorld defaults):
            0: FL hip,  1: FL knee
            2: FR hip,  3: FR knee
            4: BL hip,  5: BL knee
            6: BR hip,  7: BR knee

        Mirror pairs: (FL↔FR) and (BL↔BR).
        """

        # Which output indices are hips (negated when mirroring) vs knees (copied).
        # Half-output vector is [FL_hip, FL_knee, BL_hip, BL_knee] (indices 0-3).
        _HIP_INDICES = [0, 2]    # positions in the half-action that are hips
        _KNEE_INDICES = [1, 3]   # positions in the half-action that are knees

    def __init__(self, input_size: int, output_size: int = 8, hidden_size: int = 16):
        assert output_size == 8, "SymmetricController expects exactly 8 output joints"
        self.n_input = input_size
        self.n_output = output_size
        self.n_hidden = hidden_size
        self.n_half = output_size // 2          # 4: network only outputs one side
        self.n_con1 = input_size * hidden_size
        self.n_con2 = hidden_size * self.n_half  # smaller output layer
        self.lin = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.output = np.random.uniform(-1, 1, (self.n_half, hidden_size))
        self.n_params = self.get_num_params()

    def _mirror(self, half_action: np.ndarray) -> np.ndarray:
        """
        Diagonal pairing: FL↔BR and FR↔BL (cross-body symmetry).
        half_action = [FR_hip, FR_knee, FL_hip, FL_knee]

        Mirroring rule:
        BL_hip = -FR_hip,  BL_knee = FR_knee
        BR_hip = -FL_hip,  BR_knee = FL_knee

        Output order: [FL_hip, FL_knee, FR_hip, FR_knee, BL_hip, BL_knee, BR_hip, BR_knee]
        """
        FR_hip,  FR_knee  = half_action[0], half_action[1]
        FL_hip,  FL_knee  = half_action[2], half_action[3]

        full = np.empty(self.n_output)
        full[0] =  FL_hip    # FL_hip
        full[1] =  FL_knee   # FL_knee
        full[2] =  FR_hip    # FR_hip
        full[3] =  FR_knee   # FR_knee
        full[4] = -FR_hip    # BL_hip  = mirror of FR (negated)
        full[5] =  FR_knee   # BL_knee = mirror of FR (same)
        full[6] = -FL_hip    # BR_hip  = mirror of FL (negated)
        full[7] =  FL_knee   # BR_knee = mirror of FL (same)
        return full

    def get_action(self, state: np.ndarray) -> np.ndarray:
        assert state.shape[-1] == self.n_input
        hid = np.tanh(state @ self.lin.T)
        half_action = np.tanh(hid @ self.output.T)
        return np.clip(self._mirror(half_action), -1.0, 1.0)

    def set_weights(self, weights: np.ndarray):
        assert len(weights) == self.n_con1 + self.n_con2, (
            f"Expected {self.n_con1 + self.n_con2} params, got {len(weights)}"
        )
        self.lin = weights[: self.n_con1].reshape(self.lin.shape)
        self.output = weights[self.n_con1 :].reshape(self.output.shape)

    def geno2pheno(self, genotype: np.ndarray):
        self.set_weights(genotype)

    def get_num_params(self) -> int:
        return self.n_con1 + self.n_con2

    def reset_controller(self, batch_size=1):
        pass

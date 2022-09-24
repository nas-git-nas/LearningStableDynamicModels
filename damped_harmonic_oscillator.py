import numpy as np

class DampedHarmonicOscillator():
    def __init__(self):
        # system parameters
        self.n = 2 # number of dimensions
        self.mass = 1
        self.spring_const = 0.5
        self.friction_coeff = 0.1

        # system boundaries
        self.x_min = -1
        self.x_max = 1
        self.dx_min = -0.1
        self.dx_max = 0.1

    def generate_batch(self, b):
        """
        Description: generate one batch of samples X and its derivative f_X
        In b: batch size
        Out X: sample data (b x n)
        Out f_X: derivative of X (b x n)
        """
        X = self.generate_data(b)
        f_X = self.f_X(X)
        return X, f_X

    def generate_data(self, b):
        """
        Description: generate one batch of samples X
        In b: batch size
        Out X: sample data (b x n)
        """        
        X = np.random.rand(b, self.n)
        X[:,0] = X[:,0]*(self.x_max-self.x_min) + self.x_min
        X[:,1] = X[:,1]*(self.dx_max-self.dx_min) + self.dx_min
        return X

    def f_X(self, X):
        """
        Description: calc. derivative of damped harmonic oscillator
        In X: batch of sample data (b x n)
        Out f_X: derivative of X (b x n)
        """
        A = np.array([[0, 1],[-self.spring_const/self.mass, -self.friction_coeff/self.mass]])
        return X@A.T


def test_oscillator():
    # parameters
    batch_size = 3
    dho = DampedHarmonicOscillator()

    # generate batch
    X, f_X = dho.generate_batch(batch_size)
    print(f"X = {X}, \nf_X = {f_X}")

if __name__ == "__main__":
    test_oscillator()
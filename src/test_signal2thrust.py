import torch
import os
import numpy as np
import matplotlib.pyplot as plt


from args import Args
from system import  HolohoverSystem
from model_grey import HolohoverModelGrey
from learn import Learn


def main():
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"
    device = torch.device(dev) 

    # load params
    args = Args(model_type="Signal2Thrust")

    # get signal data
    sys = HolohoverSystem(args=args, dev=device)
    sys.loadData()  
    _, U, _ = sys.getData(u_map=False)
    print(f"U shape: {U.shape}")

    # get thrust data
    force = np.genfromtxt(os.path.join("experiment", args.series, "data_force.csv"), delimiter=",")
    force = torch.tensor(force)
    print(f"force shape: {force.shape}")

    # calc. thrust with signal
    model = HolohoverModelGrey(args=args, dev=device)
    thrust = model.signal2thrust(U)
    print(f"thrust shape: {thrust.shape}")

    # check if only one control input and thrust is not equal to zero
    non_zeros = torch.count_nonzero(U, dim=1)
    assert torch.sum(non_zeros > args.poly_expand_U) == 0, \
                f"nb. of elements with more than on input on: {torch.sum(non_zeros > args.poly_expand_U)}"
    non_zeros = torch.count_nonzero(thrust, dim=1)
    assert torch.sum(non_zeros > 1) == 0, \
                f"nb. of elements with more than on motor on: {torch.sum(non_zeros > 1)}"
    
    # calc. error between estimated thrust and measured force
    total_thrust = torch.sum(thrust, dim=1)
    force = torch.linalg.vector_norm(force[:,0:2], dim=1)
    assert total_thrust.shape == force.shape
    error = force - total_thrust
    print(f"abs. error = {torch.mean(torch.abs(error))}, mse = {torch.mean(torch.pow(error, 2))}")

    # plot estimated thrust and measured force
    plt.plot(total_thrust, label="thrust")
    plt.plot(force, label="force")
    plt.title(f"Measured force vs. estimated thrust")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()

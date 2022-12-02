import torch
import os
import numpy as np


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
    _, U, _ = sys.getData(u_map=True)
    print(f" U shape: {U.shape}")

    # get thrust data
    force = np.genfromtxt(os.path.join("experiment", args.series, "data_force.csv"), delimiter=",")
    print(f" force shape: {force.shape}")

    # calc. thrust with signal
    # model = HolohoverModelGrey(args=args, dev=device)
    # thrust = model.signal2thrust(U)

    ### change umap !!!!!!!!!

    
if __name__ == "__main__":
    main()

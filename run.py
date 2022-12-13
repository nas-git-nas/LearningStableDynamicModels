import torch

from src.args import Args
from src.system import DHOSystem, CSTRSystem, HolohoverSystem
from src.model_black import DHOModelBlack, CSTRModelBlack, HolohoverModelBlack
from src.model_grey import HolohoverModelGrey
from src.learn import Learn
from src.plot import Plot
from src.simulation import Simulation


def main():
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"
    device = torch.device(dev)

    torch.manual_seed(0)

    args = Args(model_type="HolohoverGrey")

    # init. system
    if args.model_type == "DHO":
        sys = DHOSystem(args=args, dev=device)
    elif args.model_type == "CSTR":
        sys = CSTRSystem(args=args, dev=device)
    elif args.model_type == "HolohoverBlack" or args.model_type == "HolohoverGrey":
        sys = HolohoverSystem(args=args, dev=device)

    # init. equilibrium point
    if args.model_type == "DHO":
        ueq = torch.tensor([0]) 
        xeq = sys.equPoint(ueq, U_hat=False)
        ueq = sys.uMap(ueq)     
    elif args.model_type == "CSTR":
        ueq = torch.tensor([14.19])
        xeq = sys.equPoint(ueq, U_hat=False)
        ueq = sys.uMap(ueq)
    elif args.model_type == "HolohoverBlack" or args.model_type == "HolohoverGrey":
        ueq = torch.zeros(sys.M)
        xeq = torch.zeros(sys.D)
        ueq = sys.uMap(ueq)

    # init. model
    if args.model_type == "DHO":
        model = DHOModelBlack(args=args, dev=device, generator=sys, xref=xeq)
    elif args.model_type == "CSTR":       
        model = CSTRModelBlack(args=args, dev=device, generator=sys, xref=xeq)
    elif args.model_type == "HolohoverBlack":
        model = HolohoverModelBlack(args=args, dev=device, generator=sys, xref=xeq)
    elif args.model_type == "HolohoverGrey":
        model = HolohoverModelGrey(args=args, dev=device)

    # load model to continue learning process
    if args.load_model:
        model.load_state_dict(torch.load(args.model_path))


    # learn dynamics
    ld = Learn(args=args, dev=device, system=sys, model=model)
    ld.optimize()
    ld.saveModel()

    # plot results
    plot = Plot(dev=device, model=model, system=sys, learn=ld)
    plot.greyModel(ueq)

    # simulate system
    sim = Simulation(sys, model)
    Xreal_seq, Xlearn_seq = sim.simGrey()
    plot.simGrey(Xreal_seq, Xlearn_seq)
    

if __name__ == "__main__":
    main()
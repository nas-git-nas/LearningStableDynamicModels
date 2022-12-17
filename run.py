from datetime import datetime
import os
import shutil
import torch

from src.args import Args
from src.params import Params
from src.system import DHOSystem, CSTRSystem, HolohoverSystem
from src.model_black import DHOModelBlack, CSTRModelBlack, HolohoverModelBlack
from src.model_grey import HolohoverModelGrey, CorrectModelGrey
from src.learn import LearnGreyModel, LearnCorrection
from src.plot import Plot
from src.simulation import Simulation


def main():
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"
    device = torch.device(dev)

    torch.manual_seed(0) # todo: better solution

    args = Args(model_type="HolohoverGrey")
    params = Params(args=args)

    # save model
    t = datetime.now()
    dir_name = t.strftime("%Y%m%d") + "_" + t.strftime("%H%M")
    args.dir_path = os.path.join("models", args.model_type, dir_name)
    if os.path.exists(args.dir_path):
        shutil.rmtree(args.dir_path)
    os.mkdir(args.dir_path)

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
        model = HolohoverModelGrey(args=args, params=params, dev=device)
        cor_model = CorrectModelGrey(args=args, dev=dev)

    # load model to continue learning process
    if args.load_model:
        model.load_state_dict(torch.load(args.model_path))


    # learn dynamics
    ld = LearnGreyModel(args=args, dev=device, system=sys, model=model)
    ld.optimize()
    ld.saveModel()

    # learn model correction
    lc = LearnCorrection(args=args, dev=dev, system=sys, model=cor_model, base_model=model)
    lc.optimize()

    # plot results
    plot = Plot(args=args, dev=device, model=model, cor_model=cor_model, system=sys, learn=ld, learn_cor=lc)
    plot.greyModel(ueq)
    plot.corModel()

    # # simulate system
    # sim = Simulation(sys, model)
    # Xreal_seq, Xreal_integ_seq, Xlearn_seq = sim.simGrey()
    # plot.simGrey(Xreal_seq, Xreal_integ_seq, Xlearn_seq)

    args.save()
    params.save(model)
    

if __name__ == "__main__":
    main()
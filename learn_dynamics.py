import torch

from system import DHOSystem, CSTRSystem, HolohoverSystem
from model import DHOModel, CSTRModel, HolohoverModel
from learn import Learn
from plot import Plot


def main():
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"
    device = torch.device(dev) 

    # system type
    model_type = "Holohover"
    controlled_system = True
    lyapunov_correction = False 
    load_model = False

    # init. system
    if model_type == "DHO":
        sys = DHOSystem(dev=device, controlled_system=controlled_system)
    elif model_type == "CSTR":
        sys = CSTRSystem(dev=device, controlled_system=controlled_system)
    elif model_type == "Holohover":
        sys = HolohoverSystem(dev=device, controlled_system=controlled_system)

    # init. equilibrium point
    if model_type == "DHO":
        ueq = torch.tensor([0]) 
        xeq = sys.equPoint(ueq, U_hat=False)
        ueq = sys.uMap(ueq)     
    elif model_type == "CSTR":
        ueq = torch.tensor([14.19])
        xeq = sys.equPoint(ueq, U_hat=False)
        ueq = sys.uMap(ueq)
    elif model_type == "Holohover":
        ueq = torch.zeros(sys.M)
        xeq = torch.zeros(sys.D)
        ueq = sys.uMap(ueq)

    # init. model
    if model_type == "DHO":
        model = DHOModel(   controlled_system=controlled_system, 
                            lyapunov_correction=lyapunov_correction, 
                            generator=sys, dev=device, xref=xeq)
    elif model_type == "CSTR":       
        model = CSTRModel(  controlled_system=controlled_system, 
                            lyapunov_correction=lyapunov_correction, 
                            generator=sys, dev=device, xref=xeq)
    elif model_type == "Holohover":
        model = HolohoverModel( controlled_system=controlled_system, 
                                lyapunov_correction=lyapunov_correction, 
                                generator=sys, dev=device, xref=xeq)

    # load model to continue learning process
    if load_model:
        if model_type == "DHO":
            model_path = "models/DHO/20221103_0822/20221103_0822_model"
        elif model_type == "CSTR":       
            model_path = "models/CSTR/20221106_1040/20221106_1040_model"
        elif model_type == "Holohover":
            model_path = None
        model.load_state_dict(torch.load(model_path))


    # learn dynamics
    ld = Learn( system=sys, model=model, dev=device, model_type=model_type)
    ld.optimize()
    ld.saveModel()

    # plot results
    # plot = Plot(model=model, system=sys, dev=device, learn=ld)
    # plot.fakeModel(ueq)

    

if __name__ == "__main__":
    main()     
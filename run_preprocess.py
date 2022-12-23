import torch

from src.args import Args
from src.params import Params
from src.system import HolohoverSystem
from src.model_grey import HolohoverModelGrey
from src_preprocess.data import Data
from src_preprocess.plot_holohover import PlotHolohover
from src_preprocess.preprocess_holohover import PreprocessHolohover

def holohover(device):
    series_name = "holohover_20221208"
    crop_data = None
    crop_exp = 1

    args = Args(model_type="HolohoverGrey")
    params = Params(args=args)
    sys = HolohoverSystem(args=args, dev=device)
    model = HolohoverModelGrey(args=args, params=params, dev=device)

    data = Data(series_name=series_name, crop_data=crop_data, crop_exp=crop_exp)
    plot = PlotHolohover(data=data)
    pp = PreprocessHolohover(data=data, plot=plot, sys=sys, model=model)

    pp.cropData(plot=False)
    pp.intermolateU(plot=False)
    pp.firstOrderU(tau_up=params.tau_up, tau_dw=params.tau_dw, plot=False)
    pp.diffX(plot=False)
    pp.alignData(plot=False)

    # data.saveData()

def validation(device):
    series_name = "validation_20221208"
    crop_data = None
    crop_exp = 1

    args = Args(model_type="HolohoverGrey")
    model = HolohoverModelGrey(args=args, dev=device)

    data = Data(series_name=series_name, crop_data=crop_data, crop_exp=crop_exp)
    plot = PlotHolohover(data=data)
    pp = PlotHolohover(data=data, plot=plot, model=model)

    pp.cropData(plot=False)
    pp.intermolateU(plot=False)
    pp.firstOrderU(plot=False)
    pp.diffX(plot=False)
    pp.alignData(plot=False)

    # data.saveData()


def main():
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"
    device = torch.device(dev)

    holohover(device=device)

    # validation(device=device)

if __name__ == "__main__":
    main()
import torch

from src.args import Args
from src.params import Params
from src.system import HolohoverSystem
from src.model_grey import HolohoverModelGrey
from src_preprocess.data import Data
from src_preprocess.preprocess_loadcell import Loadcell
from src_preprocess.preprocess_holohover import PreprocessHolohover
from src_preprocess.plot_loadcell import PlotLoadcell
from src_preprocess.plot_holohover import PlotHolohover



def loadcell(device):
    series_name = "signal_20221206"
    crop_data = None
    crop_exp = None

    args = Args(model_type="HolohoverGrey")
    params = Params(args=args)
    sys = HolohoverSystem(args=args, dev=device)
    model = HolohoverModelGrey(args=args, params=params, dev=device)

    data = Data(series_name=series_name, crop_data=crop_data, crop_exp=crop_exp)
    plot = PlotLoadcell(data=data)
    pp = Loadcell(data=data, plot=plot, sys=sys, model=model)
    pp.cropData()
    pp.interpolateU(plot=False)
    pp.locSig(trigger_delay=0.5, plot=False)
    pp.calcNorm(plot=False)
    pp.calcMeanNorm(plot=False)
    pp.signal2thrust(plot=False, verb=False)
    pp.thrust2signal(plot=False, verb=False)
    pp.motorTransition(thr_y_final=0.95, plot=True, signal_space=False)

    # s2t = Loadcell(series="signal_20221206")

    # thrusts = s2t.getThrust(plot=False)
    # s2t.approxSignal2Thrust(thrusts, plot=False, print_coeff=True)
    # s2t.approxThrust2Signal(thrusts, plot=False, print_coeff=True)

    # # s2t.intermolateForce(plot=True)
    # # s2t.saveData()

    # trans_up, trans_dw = s2t.motorTransition(plot=False, signal_space=False)
    # s2t.plotTransTime(trans_up, trans_dw)

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

    loadcell(device)

    # holohover(device=device)

    # validation(device=device)

if __name__ == "__main__":
    main()
import numpy as np

from src.args import Args
from src.model_grey import HolohoverModelGrey
from src.system import HolohoverSystem
from src_preprocess.data import Data
from src_preprocess.plot_holohover import PlotHolohover

def holohover():
    series_name = "holohover_20221208"
    crop_data = None
    crop_exp = None

    data = Data(series_name=series_name, crop_data=crop_data, crop_exp=crop_exp)
    plot = PlotHolohover(exps=data.exps)
    model = HolohoverModelGrey()
    pp = PlotHolohover(data=data, plot=plot, model=model)

    pp.cropData(plot=False)
    pp.intermolateU(plot=False)
    pp.firstOrderU(plot=False)
    pp.diffPosition(plot=False)
    pp.alignData(plot=True)

    data.saveData()

def validation():
    series_name = "validation_20221208"
    crop_data = None
    crop_exp = 1

    data = Data(series_name=series_name, crop_data=crop_data, crop_exp=crop_exp)
    plot = PlotHolohover(exps=data.exps)
    model = HolohoverModelGrey()
    pp = PlotHolohover(data=data, plot=plot, model=model)

    pp.cropData(plot=False)
    pp.intermolateU(plot=False)
    pp.firstOrderU(plot=False)
    pp.diffPosition(plot=False)
    pp.alignData(plot=False)

    data.saveData()


def main():


    holohover()

    validation()
    
    




if __name__ == "__main__":
    main()
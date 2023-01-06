# Learning System Dynamics of Holohover
This repository is the implementation to prepare the data for processing and to learn the system dynamics of the Holohover vehicle. It is a semester project at STI IGM LA3, EPFL. The report is found under _SemesterProjectReport_NicolajSchmid.pdf_.
* Student: Nicolaj Schmid
* Supervisor: Roland Schwan
* Professor: Colin Jones

## System requirements
The code is run with Python 3.10.9 and all library requirements are found in the file _requirements_.

## Data
The data is contained in the _experiment_ folder. Only the csv files are included due to the large size of the ROS files (deb3 and mcap files):
* holohover_20221130: First holohover measurements with random noise added to the LQR controller and without idle thrust
* holohover_20221208: First holohover measurements with random noise added to the LQR controller and with idle thrust
* signal_20221121: First force sensor measurement without idle thrust
* signal_20221206: First force sensor measurement with idle thrust
* validation_20221208: Could be used for validation

## Settings
_args_ contains the arguments for the different models and _params_ the parameters of the white box model.

## Data preparation
The data preparation is done by executing _run_preprocess.py_. All source code is contained in _src_preprocess_ and the plots are saved in the folder _plots_.

## Models
To train and evaluate the system dynamics execute _run.py_. Choose the model type by setting the variable _model_type_ to one of the following possibilities:
* DHO: Learn stabalizable system dynamics of _damped harmonic oscillator_
* CSTR: Learn stabalizable system dynamics of _continuous stirred tank reactor_
* HolohoverBlack: Learn stabalizable system dynamics of Holohover
* HolohoverBlackSimple: Learn control input to acceleration mapping without ensuring the dynamics to be stable
* HolohoverGrey: Learn grey box model of Holohover

All source code is contained in the folder _src_ and the models and plots are saved in _models_.

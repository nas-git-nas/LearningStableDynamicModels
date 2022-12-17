import os
import json
import jsbeautifier
import numpy as np

class Params():
    def __init__(self, args) -> None:
        self.args = args
        params_path = os.path.join("params", "params_"+self.args.model_type+".json")

        with open(params_path) as f:
            args = json.load(f)

        if self.args.model_type == "HolohoverGrey" or self.args.model_type == "Signal2Thrust":
            # distance form center (0,0,0) in robot frame to motors
            self.motor_distance = args["model"]["motor_distance"]
            # offset angle of first motor pair (angle between motor 1 and motor 2)
            self.motor_angle_offset = args["model"]["motor_angle_offset"]
            # angle between center of motor pair and motors to the left and right
            self.motor_angel_delta = args["model"]["motor_angel_delta"]
            # center of mass
            self.center_of_mass = args["model"]["center_of_mass"]
            # mass of holohover
            self.mass = args["model"]["mass"] 
            # intitial inertia
            self.inertia = args["model"]["inertia"]
            # initial signal2thrust coeff., tensor (M, poly_expand_U)
            # for each motor [a1, a2, a3] where thrust = a1*u + a2*u^2 + a3*u^3
            self.signal2thrust = args["model"]["signal2thrust"]
            # initial thrust2signal coeff., tensor (M, poly_expand_U)
            # for each motor [a1, a2, a3], where u = a1*thrust + a2*thrust^2 + a3*thrust^3
            self.thrust2signal = args["model"]["thrust2signal"]

    def save(self, model):
        params = {}
        if self.args.model_type == "HolohoverGrey" or self.args.model_type == "Signal2Thrust":
            params["model"] = {  
                "init_center_of_mass": self.center_of_mass,
                "learned_center_of_mass": list(model.center_of_mass.detach().numpy().astype(float)),
                "init_mass": self.mass,
                "learned_mass": model.mass.detach().numpy().astype(float).item(),
                "init_inertia": self.inertia,
                "learned_inertia": model.inertia.detach().numpy().astype(float).item(),
            }

            params["model"]["init_signal2thrust"] = self.signal2thrust
            sig2thr_list = []
            for lin_fct in model.sig2thr_fcts:
                sig2thr_list.append(list(lin_fct.weight.detach().numpy().flatten().astype(float)))
            params["model"]["learned_signal2thrust"] = sig2thr_list
            params["model"]["thrust2signal"] = self.thrust2signal

            params["model"]["motor_distance"] = self.motor_distance
            params["model"]["motor_angle_offset"] = self.motor_angle_offset
            params["model"]["motor_angel_delta"] = self.motor_angel_delta
            params["model"]["init_motors_vec"] = model.initMotorVec(self).detach().numpy().tolist()
            params["model"]["learned_motors_vec"] = model.motors_vec.detach().numpy().tolist()
            params["model"]["init_motors_pos"] = model.initMotorPos(self).detach().numpy().tolist()
            params["model"]["learned_motors_pos"] = model.motors_pos.detach().numpy().tolist()
        
        # Serializing json
        options = jsbeautifier.default_options()
        options.indent_size = 4
        json_object = jsbeautifier.beautify(json.dumps(params), options)
        
        # Writing to sample.json
        with open(os.path.join(self.args.dir_path, "params_"+self.args.model_type+".json"), "w") as outfile:
            outfile.write(json_object)


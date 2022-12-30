import os
import json

class Args():
    def __init__(self, model_type) -> None:
        args_path = os.path.join("args", "args_"+model_type+".json")

        with open(args_path) as f:
            args = json.load(f)

        self.model_type = model_type
        self.dir_path = None

        self.load_data = args["data"]["load_data"]
        self.series = args["data"]["series"]

        if model_type == "DHO" or model_type == "CSTR" or model_type == "HolohoverBlack":
            self.black_model = True
            self.controlled_system = args["black_model"]["controlled_system"]
            self.lyapunov_correction = args["black_model"]["lyapunov_correction"]
            self.u_map = args["black_model"]["u_map"]
        else:
            self.black_model = False
            self.u_map = False

        if model_type == "HolohoverGrey" or model_type == "Signal2Thrust":
            self.grey_model = True
            self.learn_center_of_mass = args["grey_model"]["learn_center_of_mass"]
            self.learn_mass = args["grey_model"]["learn_mass"]
            self.learn_inertia = args["grey_model"]["learn_inertia"]
            self.learn_signal2thrust = args["grey_model"]["learn_signal2thrust"]
            self.learn_motors_vec = args["grey_model"]["learn_motors_vec"]
            self.learn_motors_pos = args["grey_model"]["learn_motors_pos"]
            self.poly_expand_U = args["grey_model"]["poly_expand_U"]
            self.lr_center_of_mass = args["grey_model"]["lr_center_of_mass"]
            self.lr_inertia = args["grey_model"]["lr_inertia"]
            self.lr_mass = args["grey_model"]["lr_mass"]
            self.lr_signal2thrust = args["grey_model"]["lr_signal2thrust"]
            self.lr_motors_vec = args["grey_model"]["lr_motors_vec"]
            self.lr_motors_pos = args["grey_model"]["lr_motors_pos"]
        else:
            self.grey_model = False
            self.poly_expand_U = False

        self.learn_correction = args["cor_model"]["learn_correction"]
        if self.learn_correction:
            self.lr_cor = args["cor_model"]["lr_cor"]
  
        self.load_model = args["learn"]["load_model"]
        self.model_path = args["learn"]["model_path"]
        self.learning_rate = args["learn"]["learning_rate"]
        self.nb_epochs = args["learn"]["nb_epochs"]
        self.nb_batches = args["learn"]["nb_batches"]
        self.batch_size = args["learn"]["batch_size"]
        self.testing_share = args["learn"]["testing_share"]

    def save(self):
        args = {}
        args["data"] = {
            "load_data": self.load_data,
            "series": self.series
        }
        if self.model_type == "DHO" or self.model_type == "CSTR" or self.model_type == "HolohoverBlack":
            args["black_model"] = {
                "controlled_system": self.controlled_system,
                "lyapunov_correction": self.lyapunov_correction
            }
        if self.model_type == "HolohoverGrey" or self.model_type == "Signal2Thrust":
            args["grey_model"] = {
                "learn_center_of_mass": self.learn_center_of_mass,
                "learn_inertia": self.learn_inertia,
                "learn_mass":self.learn_mass,
                "learn_signal2thrust": self.learn_signal2thrust,
                "learn_motors_vec": self.learn_motors_vec,
                "learn_motors_pos": self.learn_motors_pos,
                "poly_expand_U": self.poly_expand_U
            }
        args["learn"] = {
            "load_model": self.load_model,
            "model_path": self.model_path,
            "learning_rate": self.learning_rate,
            "nb_epochs": self.nb_epochs,
            "nb_batches": self.nb_batches,
            "batch_size": self.batch_size,
            "testing_share": self.testing_share
        }
        
        # Serializing json
        json_object = json.dumps(args, indent=4)
        
        # Writing to sample.json
        with open(os.path.join(self.dir_path, "args_"+self.model_type+".json"), "w") as outfile:
            outfile.write(json_object)


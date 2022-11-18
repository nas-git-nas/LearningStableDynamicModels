import os
import json

class Args():
    def __init__(self, model_type) -> None:
        args_path = os.path.join("args", "args_"+model_type+".json")

        with open(args_path) as f:
            args = json.load(f)

        self.model_type = model_type

        self.load_data = args["data"]["load_data"]
        self.series = args["data"]["series"]

        if model_type == "DHO" or model_type == "CSTR" or model_type == "HolohoverBlack":
            self.black_model = True
            self.controlled_system = args["black_model"]["controlled_system"]
            self.lyapunov_correction = args["black_model"]["lyapunov_correction"]
        else:
            self.black_model = False

        if model_type == "HolohoverGrey":
            self.grey_model = True
            self.learn_center_of_mass = args["grey_model"]["learn_center_of_mass"]
            self.learn_inertia = args["grey_model"]["learn_inertia"]
        else:
            self.grey_model = False       
  
        self.load_model = args["learn"]["load_model"]
        self.model_path = args["learn"]["model_path"]
        self.learning_rate = args["learn"]["self.learning_rate"]
        self.nb_epochs = args["learn"]["self.nb_epochs"]
        self.nb_batches = args["learn"]["self.nb_batches"]
        self.batch_size = args["learn"]["self.batch_size"]
        self.testing_share = args["learn"]["self.testing_share"]


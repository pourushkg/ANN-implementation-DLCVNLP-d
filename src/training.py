from src.utils.common import read_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model
import argparse

def training(config_path):
    config = read_config(config_path)
    validation_data = config["params"]["validation_data"]
    (x_train,y_train),(x_valid,y_valid),(x_test,y_test) = get_data(validation_data)
    LOSS_FUNCTION = config["params"]["loss_function"]
    OPTIMIZER = config["params"]["optimizer"]
    METRICS = config["params"]["metrics"]
    NUM_CLASSES = config["params"]["num_classes"]

    model = create_model(LOSS_FUNCTION,OPTIMIZER,METRICS,NUM_CLASSES)

    EPOCHS=config["params"]["epochs"]
    VALIDATION = (x_valid,y_valid)
    history = model.fit(x_train,y_train,validation_data=VALIDATION,epochs=EPOCHS)

if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config.yaml")

    parsed_args = args.parse_args()

    training(config_path= parsed_args.config)
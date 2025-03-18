from utils import *
from data_processing import upload_dataset
from forecasting import Forecasting


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = upload_dataset("train_set")
validation_set = upload_dataset("validation_set")

# flag = [scene, traffic_light, state] (0: yes, 1: no)
# forecaster = Forecasting(concat_dim=1280, flag=[0, 0, 0], device=device).to(device)
forecaster = Forecasting(concat_dim=1216, flag=[1, 0, 0], device=device).to(device)
# forecaster = Forecasting(concat_dim=1152, flag=[1, 1, 0], device=device).to(device)
# forecaster = Forecasting(concat_dim=1152, flag=[1, 0, 1], device=device).to(device)
# forecaster = Forecasting(concat_dim=1088, flag=[1, 1, 1], device=device).to(device)

forecaster.train(train_set=train_set, validation_set=validation_set, epochs=50, name='your_model.pt')
forecaster.save(name='your_model.pt')

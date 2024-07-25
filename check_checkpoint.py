import torch
from torchsummary import summary

MODEL_LOAD_PATH = "semantickitti_coda_40epoch_25class/model_save.pt"

model = torch.load(MODEL_LOAD_PATH)

print(summary(model, (1, 100, 9), (1, 100, 3), 1))

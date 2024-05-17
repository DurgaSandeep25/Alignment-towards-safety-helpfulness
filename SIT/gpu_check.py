import torch

if torch.cuda.is_available():
    print("GPU Available : ", torch.cuda.is_available())
    print("GPU Count : ", torch.cuda.device_count())
else:
    print("GPU Not Available, CPU is being used")

import torch 

print(torch.cuda.device_count())

torch.load(f"trained_models/mml_best.pt")
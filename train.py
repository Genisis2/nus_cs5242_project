from torchvision import datasets, transforms
import torch

BATCH_SIZE = 32

dataset = datasets.ImageFolder('/content/dataset', transform=transforms.Compose([transforms.ToTensor(),  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small
from tqdm import tqdm


def train(loader: DataLoader, num_epochs: int):
    # using a tiny model because we do not want to be compute bound when
    # benchmarking the dataloading
    model = mobilenet_v3_small()
    model.to("cuda")
    prediction_head = nn.Linear(1000, 10)
    prediction_head.to("cuda")
    model.train()
    params = list(model.parameters()) + list(prediction_head.parameters())
    optim = torch.optim.Adam(params, lr=1e-3)
    for epoch in range(num_epochs):
        epoch_start = time.time()
        for batch in tqdm(loader, desc=f"Epoch {epoch}"):
            x = batch.to("cuda")
            embeddings = model(x)
            logits = prediction_head(embeddings)
            # just random labels
            y = torch.randint(0, 10, (len(x),), device="cuda")
            loss = nn.CrossEntropyLoss()(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

        epoch_end = time.time()
        print(f"Epoch {epoch} took {epoch_end - epoch_start:.3f}s")

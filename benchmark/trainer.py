import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small
from tqdm import tqdm


def train(loader: DataLoader, num_epochs: int) -> list[float]:
    # using a tiny model because we do not want to be compute bound when
    # benchmarking the dataloading
    model = mobilenet_v3_small()
    model.to("cuda")
    prediction_head = nn.Linear(1000, 10)
    prediction_head.to("cuda")
    model.train()
    params = list(model.parameters()) + list(prediction_head.parameters())
    optim = torch.optim.Adam(params, lr=1e-3)

    times = []
    for epoch in range(num_epochs):
        torch.cuda.synchronize()
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

        torch.cuda.synchronize()  # wait for all computations to finish
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        times.append(epoch_time)
        print(f"Epoch {epoch} took {epoch_time:.3f}s")
    return times

import argparse

from torch.utils.data import DataLoader

from benchmark.dataset import DummyDataset
from benchmark.trainer import train

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default="/data2/dummy_data")
parser.add_argument("--cache-limit-gib", type=int, default=0)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--pin-memory", type=bool, default=True)
parser.add_argument("--epochs", type=int, default=2)


def main(args):
    dataset = DummyDataset(args.data_dir, args.cache_limit_gib)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=args.pin_memory,
    )
    train(loader, args.epochs)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

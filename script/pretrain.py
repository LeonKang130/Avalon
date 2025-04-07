from model import Avalon
from util import save_exr_image, timeit
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import argparse

@timeit
def train(model: Avalon, dataloader: DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, num_epochs: int, num_round: int):
    for epoch in range(num_epochs):
        epoch_loss = 0
        for xs, ys in tqdm(dataloader):
            optimizer.zero_grad()
            loss = criterion(model(xs), ys)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {num_round * num_epochs + epoch} training loss: {epoch_loss / len(dataloader)}")

@timeit
def validate(model: Avalon, validation_features: torch.Tensor, validation_labels: torch.Tensor, criterion: torch.nn.Module, num_round: int):
    with torch.no_grad():
        prediction = model(validation_features)
        loss = criterion(prediction, validation_labels)
        print(f"Round {num_round} validation loss: {loss.item()}")
        prediction = prediction.exp_().sub_(1.0).reshape(32 ** 2, 32 ** 2, -1)
        save_exr_image(prediction, f"output/validation-{num_round + 1}.exr")

@timeit
def save_model(model: Avalon, num_round: int):
    torch.save(model.encoder.state_dict(), f"checkpoint/encoder-{num_round + 1}.pt")
    torch.save(model.decoder.state_dict(), f"checkpoint/decoder-{num_round + 1}.pt")

def main():
    parser = argparse.ArgumentParser(description="Encoder/Decoder pretraining script")
    parser.add_argument("--num_rounds", type=int, default=8, help="Number of training rounds")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of epochs per round")
    parser.add_argument("--save", action="store_true", help="Save the model after each round")
    args = parser.parse_args()
    pretraining_dataset = torch.load("dataset/pretrain.pt")
    pretraining_features = pretraining_dataset[..., :-3].to(torch.float32).cuda()
    pretraining_labels = pretraining_dataset[..., -3:].to(torch.float32).cuda().add_(1.0).log_()
    dataset = TensorDataset(pretraining_features, pretraining_labels)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    model = Avalon().cuda()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.L1Loss()
    validation_dataset = torch.load("dataset/validation.pt")
    validation_features = validation_dataset[..., :-3].to(torch.float32).cuda()
    validation_labels = validation_dataset[..., -3:].to(torch.float32).cuda().add_(1.0).log_()
    for i in range(args.num_rounds):
        print(f"Starting round {i + 1}...")
        train(model, dataloader, optimizer, criterion, args.num_epochs, i)
        validate(model, validation_features, validation_labels, criterion, i)
        if args.save:
            save_model(model, i)

if __name__ == "__main__":
    main()
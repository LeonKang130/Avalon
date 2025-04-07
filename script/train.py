from model import Avalon, Decoder
from util import *
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange, tqdm
import argparse

def load_model(decoder_path: str) -> Decoder:
    model = Avalon()
    model.decoder.load_state_dict(torch.load(decoder_path, weights_only=True))
    model.eval()
    return model.decoder

def main():
    parser = argparse.ArgumentParser(description="Latent code training script")
    parser.add_argument("--decoder", type=str, required=True, help="Path to the decoder checkpoint")
    parser.add_argument("--num_epochs", type=int, default=32, help="Number of training epochs")
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    decoder = load_model(args.decoder).to(device)
    dataset = torch.load("dataset/simulation.pt").to(torch.float32).cuda()
    features = dataset[..., :-3]
    labels = dataset[..., -3:].add(1.0).log()
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    latent_code = torch.randn(32, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam(decoder.parameters())
    criterion = torch.nn.L1Loss()
    """
    for _ in trange(args.num_epochs):
        for xs, ys in dataloader:
            optimizer.zero_grad()
            ws = xs.reshape(xs.shape[:-1] + (2, 3))
            prediction = decoder(latent_code, ws)
            loss = criterion(prediction, ys)
            loss.backward()
            optimizer.step()
    """
    for xs, ys in tqdm(dataloader):
        for _ in range(args.num_epochs):
            optimizer.zero_grad()
            ws = xs.reshape(xs.shape[:-1] + (2, 3))
            prediction = decoder(latent_code, ws)
            loss = criterion(prediction, ys)
            loss.backward()
            optimizer.step()
    ws = generate_visualization_ws(32, device)
    with torch.no_grad():
        prediction = decoder(latent_code, ws.reshape(-1, 2, 3))
        prediction = prediction.exp_().sub_(1.0).reshape(32 ** 2, 32 ** 2, -1)
        save_exr_image(prediction, f"output/prediction.exr")

if __name__ == "__main__":
    main()
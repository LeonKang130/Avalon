from model import Avalon
from util import save_exr_image, timeit
import torch
import argparse

@timeit
def load_model(encoder_path: str, decoder_path: str) -> Avalon:
    model = Avalon()
    model.encoder.load_state_dict(torch.load(encoder_path, weights_only=True))
    model.decoder.load_state_dict(torch.load(decoder_path, weights_only=True))
    return model

@timeit
def validate(model: Avalon, dataset: torch.Tensor, criterion: torch.nn.Module) -> None:
    with torch.no_grad():
        features = dataset[..., :-3]
        labels = dataset[..., -3:].add(1.0).log()
        prediction = model(features)
        loss = criterion(prediction, labels)
        print(f"Validation loss: {loss.item()}")
        save_exr_image(prediction.exp_().sub_(1.0).reshape(32 ** 2, 32 ** 2, -1), f"output/validation.exr")

def main():
    parser = argparse.ArgumentParser(description="Checkpoint validation script")
    parser.add_argument("--encoder", type=str, required=True, help="Path to the encoder checkpoint")
    parser.add_argument("--decoder", type=str, required=True, help="Path to the decoder checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the validation dataset")
    args = parser.parse_args()
    print(f"Loading model from {args.encoder} and {args.decoder}...")
    model = load_model(args.encoder, args.decoder).cuda()
    print("Model loaded successfully.")
    dataset = torch.load(args.dataset).to(torch.float32).cuda()
    criterion = torch.nn.L1Loss()
    validate(model, dataset, criterion)

if __name__ == "__main__":
    main()
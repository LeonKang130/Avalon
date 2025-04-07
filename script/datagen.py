from util import *
from tqdm import trange
from typing import NamedTuple
import argparse
import random

class Aggregate(NamedTuple):
    eigenvectors: torch.Tensor
    eigenvalues: torch.Tensor
    bsdf_sh_coefficients: torch.Tensor

@torch.jit.script
def extract_aggregate_feature(eigenvectors: torch.Tensor, eigenvalues: torch.Tensor, bsdf_sh_coefficients: torch.Tensor) -> torch.Tensor:
    s = eigenvectors.T @ torch.diag(eigenvalues) @ eigenvectors
    sigma = torch.sqrt(torch.diag(s))
    r = torch.hstack([s[0, 1] / (sigma[0] * sigma[1]),
                        s[0, 2] / (sigma[0] * sigma[2]),
                        s[1, 2] / (sigma[1] * sigma[2])])
    return torch.hstack((sigma, r, bsdf_sh_coefficients.flatten()))

@torch.jit.script
def evaluate_sggx(eigenvectors: torch.Tensor, eigenvalues: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    s = (1 / eigenvalues) * torch.square(ws @ eigenvectors.T)
    return 1 / torch.square(s.sum(-1))

@torch.jit.script
def evaluate_disney_lambda(roughness2: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    tan2theta = torch.square(w[..., 2])
    tan2theta = torch.where(tan2theta < 1e-6, 0, (1 - tan2theta) / tan2theta)
    return (-1 + torch.sqrt(1 + roughness2 * tan2theta)) * 0.5

@torch.jit.script
def evaluate_disney_d(roughness2, w: torch.Tensor) -> torch.Tensor:
    return roughness2 / (torch.pi * torch.square(1 + (roughness2 - 1) * w[..., 2] * w[..., 2]) + 1e-6)

@torch.jit.script
def evaluate_disney_bsdf(metallic: torch.Tensor, roughness: torch.Tensor, ior: torch.Tensor, base_color: torch.Tensor, wi_local: torch.Tensor, wo_local: torch.Tensor) -> torch.Tensor:
    roughness2 = torch.square(roughness)
    wh = wi_local + wo_local
    wh /= torch.linalg.norm(wh, dim=-1).unsqueeze(-1) + 1e-6
    fc = torch.pow(1.0 - (wi_local * wh).sum(-1), 5)
    reflectance = torch.square((ior - 1) / (ior + 1))
    f0 = metallic.unsqueeze(-1) * base_color + ((1 - metallic) * reflectance).unsqueeze(-1)
    f = f0 + fc.unsqueeze(-1) * (1 - f0)
    specular_bsdf = (evaluate_disney_d(roughness2, wh) / (1 + evaluate_disney_lambda(roughness2, wi_local) + evaluate_disney_lambda(roughness2, wo_local))).unsqueeze(-1) * f
    specular_bsdf /= 4 * torch.abs(wi_local[..., -1:] * wo_local[..., -1:]) + 1e-6
    front_facing = torch.where(torch.minimum(wi_local[..., -1:], wo_local[..., -1:]) > 0, 1, 0)
    diffuse_bsdf = ((1 - metallic).unsqueeze(-1) / torch.pi) * base_color
    return (diffuse_bsdf + specular_bsdf) * front_facing

@torch.jit.script
def sample_sggx_vndf(u: torch.Tensor, eigenvectors: torch.Tensor, eigenvalues: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    num_samples = u.shape[0] // ws.shape[0]
    s = eigenvectors.T @ torch.diag(eigenvalues) @ eigenvectors
    sqrt_det = eigenvalues.prod().sqrt_().unsqueeze(0)
    ns = sample_hemisphere_cosine(u).reshape(ws.shape[0], num_samples, 3)
    onb_ws = make_onb(ws)
    s_local = onb_ws @ s @ onb_ws.transpose(-2, -1)
    s_kk, s_jj, s_ii = s_local[..., 0, 0], s_local[..., 1, 1], s_local[..., 2, 2]
    s_kj, s_ki, s_ji = s_local[..., 0, 1], s_local[..., 0, 2], s_local[..., 1, 2]
    tmp = torch.sqrt(s_jj * s_ii - s_ji * s_ji)
    inv_sqrt_s_ii = 1.0 / torch.sqrt(s_ii).unsqueeze(-1)
    zeros = torch.zeros_like(tmp)
    mk = torch.stack((sqrt_det / tmp, zeros, zeros), dim=-1)
    mj = inv_sqrt_s_ii * torch.stack((-(s_ki * s_ji - s_kj * s_ii) / tmp, tmp, zeros), dim=-1)
    mi = inv_sqrt_s_ii * torch.stack((s_ki, s_ji, s_ii), dim=-1)
    ns = ns @ torch.stack((mk, mj, mi), dim=-2)
    ns.div_(torch.linalg.norm(ns, dim=-1).unsqueeze(-1) + 1e-6)
    return ns

def evaluate_aggregate(eigenvectors: torch.Tensor, eigenvalues: torch.Tensor, bsdf_sh_coefficients: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor, num_samples: int = 32, batch_size: int = 1024) -> torch.Tensor:
    evaluation = torch.empty(wi.shape[:-1] + (3,), dtype=torch.float32, device=wi.device)
    for i in range(0, wi.shape[0], batch_size):
        batch_slice = slice(i, min(i + batch_size, wi.shape[0]))
        batch_wi, batch_wo = wi[batch_slice], wo[batch_slice]
        onb_wo = make_onb(batch_wo)
        u = torch.rand(batch_wo.shape[0] * num_samples, 2, dtype=torch.float32, device=wi.device)
        ns = sample_sggx_vndf(u, eigenvectors, eigenvalues, batch_wo) @ onb_wo
        onb_ns = make_onb(ns)
        wi_local = torch.einsum('nmji,ni->nmj', onb_ns, batch_wi).reshape(-1, 3)
        wo_local = torch.einsum('nmji,ni->nmj', onb_ns, batch_wo).reshape(-1, 3)
        batch_sh_basis = evaluate_sh(ns.reshape(-1, 3))
        batch_bsdf_parameters = batch_sh_basis @ bsdf_sh_coefficients
        metallic = batch_bsdf_parameters[..., 0].clamp_(0, 1)
        roughness = batch_bsdf_parameters[..., 1].clamp_(0.05, 1)
        ior = batch_bsdf_parameters[..., 2].clamp_(1, 2)
        base_color = batch_bsdf_parameters[..., 3:].clamp_(0, 1)
        batch_bsdf_eval = evaluate_disney_bsdf(metallic, roughness, ior, base_color, wi_local, wo_local)
        # batch_bsdf_eval = torch.ones(batch_bsdf_parameters.shape[:-1] + (3,), dtype=torch.float32, device=wi.device)
        evaluation[batch_slice] = \
            (batch_bsdf_eval * wi_local[..., 2].clamp_min(0).unsqueeze(-1)).reshape(-1, num_samples, 3).sum(-2)
    evaluation /= num_samples
    return evaluation

@torch.jit.script
def generate_random_aggregate(u: torch.Tensor, force_specular: bool = False) -> Aggregate:
    ws, ys = evaluate_sf(4, u.device) @ sample_onb_uniform(u[:3]).squeeze(0), u[3:27].reshape(4, 6)
    ys[..., 2].add_(1.0)
    if force_specular:
        ys[..., 1].clamp_max_(0.2)
    bsdf_sh_coefficients = fit_sh_coefficients(ws, ys)
    eigenvectors = sample_onb_uniform(u[27:30]).squeeze(0)
    eigenvalues = u[30:].clamp_min_(1e-6)
    return Aggregate(eigenvectors, eigenvalues, bsdf_sh_coefficients)

@timeit
def generate_pretraining_dataset(num_aggregate: int, sphere_resolution: int, device: torch.device):
    dataset = torch.empty(num_aggregate, sphere_resolution * sphere_resolution, 39, dtype=torch.float32)
    print("Generating pretraining dataset...")
    sf_grid = evaluate_sf(sphere_resolution, device)
    for i in trange(num_aggregate):
        aggregate = generate_random_aggregate(torch.rand(33, dtype=torch.float32, device=device))
        dataset[i, ..., :30] = extract_aggregate_feature(*aggregate)
        onb = sample_onb_uniform(torch.rand(2, 3, dtype=torch.float32, device=device))
        wi = torch.tile(sf_grid @ onb[0], (sphere_resolution, 1)) # 1, 1, 1,...
        wo = (sf_grid @ onb[1]).repeat(1, sphere_resolution).reshape(-1, 3)
        dataset[i, ..., 30:33] = wi
        dataset[i, ..., 33:36] = wo
        dataset[i, ..., 36:] = evaluate_aggregate(*aggregate, wi, wo, 4096)
    torch.save(dataset.reshape(-1, 39), f"dataset/pretrain.pt")

@timeit
def generate_validation_dataset(square_resolution: int, device: torch.device):
    aggregate = generate_random_aggregate(torch.rand(33, dtype=torch.float32, device=device), True)
    ws = generate_visualization_ws(square_resolution, device)
    image = evaluate_aggregate(*aggregate, ws[..., :3], ws[..., 3:], 1024)
    save_exr_image(image.reshape(square_resolution ** 2, square_resolution ** 2, -1), "dataset/reference.exr")
    dataset = torch.empty(square_resolution ** 4, 39, dtype=torch.float32)
    dataset[..., :30] = extract_aggregate_feature(*aggregate)
    dataset[..., 30:36] = ws
    dataset[..., 36:] = image
    torch.save(dataset, "dataset/validation.pt")

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    parser = argparse.ArgumentParser(description="dataset generator")
    parser.add_argument("--validation", action="store_true", help="generate validation dataset")
    parser.add_argument("--pretraining", action="store_true", help="generate pretraining dataset")
    parser.add_argument("--num_aggregate", type=int, default=4096, help="number of aggregates")
    parser.add_argument("--sphere_resolution", type=int, default=64, help="sphere resolution")
    parser.add_argument("--seed", type=int, default=random.randint(0, 2 ** 32 - 1), help="random seed")
    parser.add_argument("--replay", action="store_true", help="replay the random seed")
    args = parser.parse_args()
    if args.replay:
        with open("seed.txt", "r") as f:
            seed = int(f.read())
        print("Replay random seed:", seed)
    else:
        seed = args.seed
        with open("seed.txt", "w") as f:
            f.write(str(seed))
        print("Using random seed:", seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0")
    if args.validation:
        generate_validation_dataset(32, device)
    if args.pretraining:
        generate_pretraining_dataset(args.num_aggregate, args.sphere_resolution, device)

if __name__ == "__main__":
    main()
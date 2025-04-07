from util import *
from typing import NamedTuple
import argparse

class TriangleList(NamedTuple):
    surface_areas: torch.Tensor
    ns: torch.Tensor
    metallic: torch.Tensor
    roughness: torch.Tensor
    ior: torch.Tensor
    base_color: torch.Tensor

@torch.jit.script
def generate_random_triangle_list(num_triangles: int, device: torch.device) -> TriangleList:
    surface_areas = torch.rand(num_triangles, dtype=torch.float32, device=device)
    ns = sample_sphere_uniform(torch.rand(num_triangles, 2, dtype=torch.float32, device=device))
    metallic = torch.rand(num_triangles, dtype=torch.float32, device=device)
    roughness = torch.rand(num_triangles, dtype=torch.float32, device=device).clamp_min_(0.1)
    ior = torch.rand(num_triangles, dtype=torch.float32, device=device) + 1.0
    base_color = torch.rand(num_triangles, 3, dtype=torch.float32, device=device)
    return TriangleList(surface_areas, ns, metallic, roughness, ior, base_color)

@torch.jit.script
def evaluate_triangle_list(triangle_list: TriangleList, wi: torch.Tensor, wo: torch.Tensor) -> torch.Tensor:
    onb_ns = make_onb(triangle_list.ns)
    wi_local = torch.einsum("mij,nj->nmi", onb_ns, wi)
    wo_local = torch.einsum("mij,nj->nmi", onb_ns, wo)
    orientation = torch.where(wo_local[..., 2:] > 0, 1, -1)
    wi_local = wi_local * orientation
    wo_local = wo_local * orientation
    denominator = wo_local[..., 2] * triangle_list.surface_areas
    metallic = triangle_list.metallic
    roughness2 = torch.square(triangle_list.roughness)
    ior = triangle_list.ior
    base_color = triangle_list.base_color
    wh = wi_local + wo_local
    wh /= torch.linalg.norm(wh, dim=-1).unsqueeze(-1) + 1e-6
    fc = torch.pow(1.0 - (wi_local * wh).sum(-1), 5)
    reflectance = torch.square((ior - 1) / (ior + 1))
    f0 = metallic.unsqueeze(-1) * base_color + ((1 - metallic) * reflectance).unsqueeze(-1)
    f = f0 + fc.unsqueeze(-1) * (1 - f0)
    d = evaluate_disney_d(roughness2, wh)
    g = 1.0 / (1.0 + evaluate_disney_lambda(roughness2, wi_local) + evaluate_disney_lambda(roughness2, wo_local))
    specular_bsdf = (d * g).unsqueeze(-1) * f / (4 * torch.abs(wi_local[..., 2:] * wo_local[..., 2:]) + 1e-6)
    diffuse_bsdf = ((1 - metallic).unsqueeze(-1) / torch.pi) * base_color
    nominator = (specular_bsdf + diffuse_bsdf) * wi_local[..., 2:].clamp_min(0) * denominator.unsqueeze(-1)
    return nominator.sum(-2) / denominator.sum(-1, keepdim=True)

def visualize_triangle_list(triangle_list: TriangleList, square_resolution: int, device: torch.device):
    ws = generate_visualization_ws(square_resolution, device)
    image = evaluate_triangle_list(triangle_list, ws[..., :3], ws[..., 3:])
    save_exr_image(image.reshape(square_resolution ** 2, square_resolution ** 2, -1), "dataset/simulation.exr")

def generate_simulation_dataset(triangle_list, sphere_resolution: int, device: torch.device):
    dataset = torch.empty(sphere_resolution * sphere_resolution, 9, dtype=torch.float32)
    print("Generating simulation dataset...")
    sf_grid = evaluate_sf(sphere_resolution, device)
    onb = sample_onb_uniform(torch.rand(2, 3, dtype=torch.float32, device=device))
    wi = torch.tile(sf_grid @ onb[0], (sphere_resolution, 1))
    wo = (sf_grid @ onb[1]).repeat(1, sphere_resolution).reshape(-1, 3)
    dataset[..., :3] = wi
    dataset[..., 3:6] = wo
    dataset[..., 6:] = evaluate_triangle_list(triangle_list, wi, wo)
    torch.save(dataset.reshape(-1, 9), "dataset/simulation.pt")

def main():
    parser = argparse.ArgumentParser(description="Triangle aggregation simulation script")
    parser.add_argument("--num_triangles", type=int, default=32, help="Number of triangles to simulate")
    parser.add_argument("--sphere_resolution", type=int, default=32, help="Resolution of the sphere")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    triangle_list = generate_random_triangle_list(args.num_triangles, device)
    visualize_triangle_list(triangle_list, 32, device)
    generate_simulation_dataset(triangle_list, args.sphere_resolution, device)

if __name__ == "__main__":
    main()
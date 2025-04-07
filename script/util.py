import torch
import OpenEXR, Imath
import time
from functools import wraps

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} Took {total_time:.4f} seconds")
        return result
    return timeit_wrapper

@torch.jit.script
def sample_sphere_uniform(u: torch.Tensor) -> torch.Tensor:
    phi = 2.0 * torch.pi * u[..., 0]
    cos_theta = 1.0 - 2.0 * u[..., 1]
    sin_theta = torch.sqrt(1.0 - torch.square(cos_theta))
    return torch.stack((sin_theta * torch.cos(phi), sin_theta * torch.sin(phi), cos_theta), dim=-1)

@torch.jit.script
def sample_hemisphere_uniform(u: torch.Tensor) -> torch.Tensor:
    phi = 2.0 * torch.pi * u[..., 0]
    sin_theta = torch.sqrt(1.0 - torch.square(u[..., 1]))
    return torch.stack((sin_theta * torch.cos(phi), sin_theta * torch.sin(phi), u[..., 1]), dim=-1)

@torch.jit.script
def sample_hemisphere_cosine(u: torch.Tensor) -> torch.Tensor:
    phi = 2.0 * torch.pi * u[..., 0]
    sin_theta = torch.sqrt(1.0 - u[..., 1])
    return torch.stack((torch.cos(phi) * sin_theta, torch.sin(phi) * sin_theta, torch.sqrt(u[..., 1])), dim=-1)

@torch.jit.script
def make_onb(ns: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(ns[..., 0])
    c1 = torch.stack([-ns[..., 1], ns[..., 0], zeros], dim=-1)
    c2 = torch.stack([zeros, -ns[..., 2], ns[..., 1]], dim=-1)
    ts = torch.where(ns[..., :1].abs() > ns[..., -1:].abs(), c1, c2)
    ts /= torch.linalg.norm(ts, dim=-1).unsqueeze_(-1) + 1e-6
    bs = torch.linalg.cross(ns, ts)
    return torch.stack((ts, bs, ns), dim=-2)

@torch.jit.script
def sample_onb_uniform(u: torch.Tensor) -> torch.Tensor:
    onb = make_onb(sample_sphere_uniform(u[..., :2]))
    t, b, n = onb[..., 0, :], onb[..., 1, :], onb[..., 2, :]
    ksi = 2.0 * torch.pi * u[..., 2:]
    cos_ksi, sin_ksi = torch.cos(ksi), torch.sin(ksi)
    return torch.stack([
        cos_ksi * t + sin_ksi * b,
        cos_ksi * b - sin_ksi * t,
        n
    ], dim=-2)

# Generate a Spherical Fibonacci lattice of n points
@torch.jit.script
def evaluate_sf(n: int, device: torch.device) -> torch.Tensor:
    inv_golden_ratio = (5 ** 0.5 - 1) / 2
    i = torch.arange(n, dtype=torch.float32, device=device)
    phi = 2.0 * torch.pi * torch.frac(inv_golden_ratio * i)
    cos_theta = 1.0 - (2.0 * i + 1.0) / n
    sin_theta = torch.sqrt(1.0 - torch.square(cos_theta))
    return torch.stack((torch.cos(phi) * sin_theta, torch.sin(phi) * sin_theta, cos_theta), dim=-1)

# Evaluate spherical harmonics of order 2 at the given points
@torch.jit.script
def evaluate_sh(ws: torch.Tensor) -> torch.Tensor:
    ks = 0.28209479177387814, 0.4886025119029199
    num_spherical_harmonic_basis = 4
    sh = torch.empty(ws.shape[:-1] + (num_spherical_harmonic_basis,), dtype=ws.dtype, device=ws.device)
    sh[..., 0] = ks[0]
    sh[..., 1] = -ks[1] * ws[..., 1]
    sh[..., 2] = ks[1] * ws[..., 2]
    sh[..., 3] = -ks[1] * ws[..., 0]
    return sh

# Fit 2 order spherical harmonics coefficients to the given points
@torch.jit.script
def fit_sh_coefficients(ws: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    return torch.linalg.lstsq(evaluate_sh(ws), ys, rcond=None).solution

@torch.jit.script
def evaluate_disney_lambda(roughness2: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    tan2theta = torch.square(w[..., 2])
    tan2theta = torch.where(tan2theta < 1e-6, 0, (1 - tan2theta) / tan2theta)
    return (-1 + torch.sqrt(1 + roughness2 * tan2theta)) * 0.5

@torch.jit.script
def evaluate_disney_d(roughness2, w: torch.Tensor) -> torch.Tensor:
    return roughness2 / (torch.pi * torch.square(1 + (roughness2 - 1) * w[..., 2] * w[..., 2]) + 1e-6)

@torch.jit.script
def evaluate_octahedron(uv: torch.Tensor) -> torch.Tensor:
    uv = 2 * uv - 1
    uvp = torch.abs(uv)
    signed_distance = 1 - uvp[..., 0] - uvp[..., 1]
    r = 1 - signed_distance.abs()
    phi = torch.where(r == 0, 1, (uvp[..., 1] - uvp[..., 0]) / r + 1) * (torch.pi / 4)
    z = (1 - r * r).copysign_(signed_distance)
    cos_phi = torch.cos(phi).copysign_(uv[..., 0])
    sin_phi = torch.sin(phi).copysign_(uv[..., 1])
    return torch.stack((r * cos_phi, r * sin_phi, z), dim=-1)

@torch.jit.script
def generate_visualization_ws(resolution: int, device: torch.device) -> torch.Tensor:
    i = torch.arange(resolution ** 2, dtype=torch.float32, device=device)
    xy = torch.stack(torch.meshgrid(i, i, indexing="xy"), dim=-1).reshape(-1, 2)
    wi = evaluate_octahedron((xy // resolution + 0.5) / resolution)
    wo = evaluate_octahedron((xy % resolution + 0.5) / resolution)
    return torch.hstack((wi, wo))

@timeit
def save_exr_image(image: torch.Tensor, filename: str):
    assert(image.ndim == 3 and image.shape[-1] in (3, 4))
    image = image.to(torch.float32).cpu().numpy()
    exr_data = [image[..., i].tobytes() for i in range(image.shape[-1])]
    header = OpenEXR.Header(*image.shape[:2])
    channels = "RGBA"[:len(exr_data)]
    header["channels"] = {c: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) for c in channels}
    exr_file = OpenEXR.OutputFile(filename, header)
    exr_file.writePixels(dict(zip(channels, exr_data)))
    exr_file.close()

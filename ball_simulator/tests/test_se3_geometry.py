import torch
from ball_world_model.geometry.se3 import (
    carrier_acceleration, carrier_state, carrier_velocity, compose, inverse, so3_exp,
)

def test_compose_inverse_identity():
    torch.manual_seed(1)
    p = torch.randn(7, 3, dtype=torch.float64)
    r = so3_exp(torch.randn(7, 3, dtype=torch.float64) * 0.3)
    pi, ri = inverse(p, r)
    p0, r0 = compose(p, r, pi, ri)
    assert torch.allclose(p0, torch.zeros_like(p0), atol=1e-10)
    assert torch.allclose(r0, torch.eye(3, dtype=r.dtype).expand_as(r0), atol=1e-10)

def test_affine_carrier_first_and_second_derivatives():
    torch.manual_seed(2)
    dtype = torch.float64
    p = torch.randn(2, 3, dtype=dtype)
    r = so3_exp(torch.randn(2, 3, dtype=dtype) * 0.2)
    v = torch.randn(2, 3, dtype=dtype)
    w = torch.randn(2, 3, dtype=dtype)
    a = torch.randn(2, 3, dtype=dtype)
    alpha = torch.randn(2, 3, dtype=dtype)
    b = torch.randn(11, 3, dtype=dtype) * 0.1
    eps = 1e-4

    def state(t):
        pt = p + t*v + 0.5*t*t*a
        rt = so3_exp(t*w + 0.5*t*t*alpha) @ r
        return carrier_state(pt, rt, b)
    
    x0, xp, xm = state(0.0), state(eps), state(-eps)
    fd1 = (xp - xm) / (2*eps)
    fd2 = (xp - 2*x0 + xm) / (eps*eps)
    analytic1 = carrier_velocity(r, v, w, b)
    analytic2 = carrier_acceleration(r, w, a, alpha, b)
    assert torch.allclose(fd1, analytic1, atol=2e-7, rtol=2e-6)
    assert torch.allclose(fd2, analytic2, atol=2e-5, rtol=2e-4)

def test_global_frame_equivariance():
    torch.manual_seed(3)
    dtype = torch.float64
    p, v, w, a, alpha = [torch.randn(4, 3, dtype=dtype) for _ in range(5)]
    r = so3_exp(torch.randn(4, 3, dtype=dtype))
    q = so3_exp(torch.randn(4, 3, dtype=dtype))
    shift = torch.randn(4, 3, dtype=dtype)
    b = torch.randn(9, 3, dtype=dtype)
    x = carrier_state(p, r, b)
    dx = carrier_velocity(r, v, w, b)
    ddx = carrier_acceleration(r, w, a, alpha, b)

    p2 = shift + torch.einsum("bij,bj->bi", q, p)
    r2 = q @ r
    rotate = lambda z: torch.einsum("bij,bkj->bki", q, z)
    x2 = carrier_state(p2, r2, b)
    dx2 = carrier_velocity(r2, torch.einsum("bij,bj->bi", q, v), torch.einsum("bij,bj->bi", q, w), b)
    ddx2 = carrier_acceleration(r2, torch.einsum("bij,bj->bi", q, w), torch.einsum("bij,bj->bi", q, a), torch.einsum("bij,bj->bi", q, alpha), b)
    
    assert torch.allclose(x2, shift[:, None] + rotate(x), atol=1e-10)
    assert torch.allclose(dx2, rotate(dx), atol=1e-10)
    assert torch.allclose(ddx2, rotate(ddx), atol=1e-10)

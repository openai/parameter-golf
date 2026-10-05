"""CPU regressions for Muon's momentum rule and input ownership."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_gpt import Muon, zeropower_via_newtonschulz5


def reference_update(gradient, steps):
    x = gradient.to(torch.bfloat16).clone()
    x = x / (x.norm() + 1e-7)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a = x @ x.T
        b = -4.7750 * a + (2.0315 * a) @ a
        x = 3.4445 * x + b @ x
    return x.T if transposed else x


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 3), (3, 2), (2, 2)])
@pytest.mark.parametrize("nesterov", [False, True])
@pytest.mark.parametrize("steps", [0, 3])
def test_muon_uses_the_selected_momentum_rule(dtype, shape, nesterov, steps):
    parameter = torch.nn.Parameter(torch.zeros(shape, dtype=dtype))
    optimizer = Muon([parameter], lr=0.125, momentum=0.75, backend_steps=steps, nesterov=nesterov)
    expected_parameter = parameter.detach().clone()
    momentum = torch.zeros_like(parameter)
    generator = torch.Generator().manual_seed(731)
    for _ in range(3):
        gradient = torch.randn(shape, generator=generator).to(dtype)
        parameter.grad = gradient.clone()
        momentum.mul_(0.75).add_(gradient)
        direction = gradient.add(momentum, alpha=0.75) if nesterov else momentum
        update = reference_update(direction, steps)
        update *= max(1, shape[0] / shape[1]) ** 0.5
        expected_parameter.add_(update.to(dtype), alpha=-0.125)
        optimizer.step()
        torch.testing.assert_close(parameter, expected_parameter, rtol=0, atol=0)
        torch.testing.assert_close(
            optimizer.state[parameter]["momentum_buffer"], momentum, rtol=0, atol=0
        )
        torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("steps", [0, 3])
def test_orthogonalization_does_not_mutate_its_input(dtype, steps):
    gradient = torch.tensor([[1.0, 2.0], [3.0, -4.0]], dtype=dtype)
    original = gradient.clone()
    output = zeropower_via_newtonschulz5(gradient, steps=steps)
    torch.testing.assert_close(gradient, original, rtol=0, atol=0)
    torch.testing.assert_close(output, reference_update(original, steps), rtol=0, atol=0)


def test_parameters_without_gradients_are_unchanged():
    parameter = torch.nn.Parameter(torch.ones((2, 2)))
    optimizer = Muon([parameter], lr=0.1, momentum=0.9, backend_steps=3, nesterov=False)
    optimizer.step()
    torch.testing.assert_close(parameter, torch.ones((2, 2)))
    assert not optimizer.state

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class PackedLinearProjection(nn.Module):
    def __init__(self, d_in: int, split_sizes: tuple[int, ...], *, bias: bool = True) -> None:
        super().__init__()
        if not split_sizes or any(size <= 0 for size in split_sizes):
            raise ValueError(f"invalid packed projection split sizes: {split_sizes}")
        self.split_sizes = split_sizes
        self.projection = CastedLinear(d_in, sum(split_sizes), bias=bias)

    def forward(self, inputs: Tensor) -> tuple[Tensor, ...]:
        packed = self.projection(inputs)
        return tuple(packed.split(self.split_sizes, dim=-1))


class BlockDiagonalLinear(nn.Module):
    def __init__(self, width: int, blocks: int) -> None:
        super().__init__()
        if width % blocks != 0:
            raise ValueError(f"block-diagonal width {width} must be divisible by blocks {blocks}")
        self.width = width
        self.blocks = blocks
        self.block_width = width // blocks
        # Keep this as a 2D parameter so the existing Muon split can treat it as a matrix.
        self.weight = nn.Parameter(torch.empty(width, self.block_width))
        self.bias = nn.Parameter(torch.empty(width))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        bound = 1.0 / self.block_width**0.5
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: Tensor) -> Tensor:
        leading = inputs.shape[:-1]
        reshaped_inputs = inputs.reshape(-1, self.blocks, self.block_width)
        reshaped_weight = self.weight.reshape(self.blocks, self.block_width, self.block_width)
        projected = torch.einsum("bgi,goi->bgo", reshaped_inputs, reshaped_weight)
        projected = projected.reshape(*leading, self.width)
        return projected + self.bias.to(dtype=projected.dtype)


def triton_runtime_available() -> bool:
    return triton is not None and tl is not None


def ensure_triton_runtime_available() -> None:
    if not triton_runtime_available():
        raise RuntimeError(
            "P20_RUNTIME_BACKEND=triton requires a CUDA environment with Triton installed"
        )


def validate_p20_runtime_backend(runtime_backend: str) -> None:
    if runtime_backend not in {"torch", "triton"}:
        raise ValueError(f"P20 runtime backend must be torch|triton, got {runtime_backend}")
    if runtime_backend == "triton":
        ensure_triton_runtime_available()


def _next_power_of_two(value: int) -> int:
    power = 1
    while power < value:
        power <<= 1
    return power


if triton_runtime_available():

    @triton.jit
    def _p20_forward_kernel(
        update_gate_ptr,
        retain_gate_ptr,
        transformed_state_ptr,
        candidate_ptr,
        output_gate_ptr,
        next_state_ptr,
        emitted_output_ptr,
        input_row_stride,
        output_row_stride,
        width,
        BLOCK_SIZE: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        block_index = tl.program_id(1)
        offsets = block_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < width
        input_row_offsets = batch_index * input_row_stride + offsets
        output_row_offsets = batch_index * output_row_stride + offsets

        update_gate = tl.load(update_gate_ptr + input_row_offsets, mask=mask, other=0.0)
        retain_gate = tl.load(retain_gate_ptr + input_row_offsets, mask=mask, other=0.0)
        transformed_state = tl.load(transformed_state_ptr + input_row_offsets, mask=mask, other=0.0)
        candidate = tl.load(candidate_ptr + input_row_offsets, mask=mask, other=0.0)
        output_gate = tl.load(output_gate_ptr + input_row_offsets, mask=mask, other=0.0)

        next_state = update_gate * transformed_state + retain_gate * candidate
        emitted_output = output_gate * next_state

        tl.store(next_state_ptr + output_row_offsets, next_state, mask=mask)
        tl.store(emitted_output_ptr + output_row_offsets, emitted_output, mask=mask)


    @triton.jit
    def _p20_backward_kernel(
        grad_next_state_ptr,
        grad_emitted_output_ptr,
        update_gate_ptr,
        retain_gate_ptr,
        transformed_state_ptr,
        candidate_ptr,
        output_gate_ptr,
        next_state_ptr,
        grad_update_gate_ptr,
        grad_retain_gate_ptr,
        grad_transformed_state_ptr,
        grad_candidate_ptr,
        grad_output_gate_ptr,
        grad_row_stride,
        input_row_stride,
        next_state_row_stride,
        output_row_stride,
        width,
        BLOCK_SIZE: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        block_index = tl.program_id(1)
        offsets = block_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < width
        grad_row_offsets = batch_index * grad_row_stride + offsets
        input_row_offsets = batch_index * input_row_stride + offsets
        next_state_row_offsets = batch_index * next_state_row_stride + offsets
        output_row_offsets = batch_index * output_row_stride + offsets

        grad_next_state = tl.load(grad_next_state_ptr + grad_row_offsets, mask=mask, other=0.0)
        grad_emitted_output = tl.load(grad_emitted_output_ptr + grad_row_offsets, mask=mask, other=0.0)
        update_gate = tl.load(update_gate_ptr + input_row_offsets, mask=mask, other=0.0)
        retain_gate = tl.load(retain_gate_ptr + input_row_offsets, mask=mask, other=0.0)
        transformed_state = tl.load(transformed_state_ptr + input_row_offsets, mask=mask, other=0.0)
        candidate = tl.load(candidate_ptr + input_row_offsets, mask=mask, other=0.0)
        output_gate = tl.load(output_gate_ptr + input_row_offsets, mask=mask, other=0.0)
        next_state = tl.load(next_state_ptr + next_state_row_offsets, mask=mask, other=0.0)

        total_grad_next_state = grad_next_state + grad_emitted_output * output_gate
        grad_update_gate = total_grad_next_state * transformed_state
        grad_retain_gate = total_grad_next_state * candidate
        grad_transformed_state = total_grad_next_state * update_gate
        grad_candidate = total_grad_next_state * retain_gate
        grad_output_gate = grad_emitted_output * next_state

        tl.store(grad_update_gate_ptr + output_row_offsets, grad_update_gate, mask=mask)
        tl.store(grad_retain_gate_ptr + output_row_offsets, grad_retain_gate, mask=mask)
        tl.store(grad_transformed_state_ptr + output_row_offsets, grad_transformed_state, mask=mask)
        tl.store(grad_candidate_ptr + output_row_offsets, grad_candidate, mask=mask)
        tl.store(grad_output_gate_ptr + output_row_offsets, grad_output_gate, mask=mask)


class _P20FusedUpdateReadout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, update_gate: Tensor, retain_gate: Tensor, transformed_state: Tensor, candidate: Tensor, output_gate: Tensor):
        if not triton_runtime_available():
            raise RuntimeError("P20 Triton fused update requires Triton to be installed")
        if update_gate.device.type != "cuda":
            raise RuntimeError("P20 Triton fused update requires CUDA tensors")
        for tensor in (retain_gate, transformed_state, candidate, output_gate):
            if tensor.device != update_gate.device or tensor.shape != update_gate.shape:
                raise RuntimeError("P20 Triton fused update requires matching CUDA tensor shapes")

        batch_size, width = update_gate.shape
        block_size = min(1024, _next_power_of_two(width))
        input_row_stride = update_gate.stride(0)
        next_state = torch.empty_like(update_gate)
        emitted_output = torch.empty_like(update_gate)
        output_row_stride = next_state.stride(0)

        grid = (batch_size, triton.cdiv(width, block_size))
        _p20_forward_kernel[grid](
            update_gate,
            retain_gate,
            transformed_state,
            candidate,
            output_gate,
            next_state,
            emitted_output,
            input_row_stride,
            output_row_stride,
            width,
            BLOCK_SIZE=block_size,
        )
        ctx.block_size = block_size
        ctx.input_row_stride = input_row_stride
        ctx.output_row_stride = output_row_stride
        ctx.width = width
        ctx.save_for_backward(update_gate, retain_gate, transformed_state, candidate, output_gate, next_state)
        return next_state, emitted_output

    @staticmethod
    def backward(ctx, grad_next_state: Tensor, grad_emitted_output: Tensor):
        update_gate, retain_gate, transformed_state, candidate, output_gate, next_state = ctx.saved_tensors
        grad_update_gate = torch.empty_like(update_gate)
        grad_retain_gate = torch.empty_like(retain_gate)
        grad_transformed_state = torch.empty_like(transformed_state)
        grad_candidate = torch.empty_like(candidate)
        grad_output_gate = torch.empty_like(output_gate)
        grad_next_state = grad_next_state.contiguous()
        grad_emitted_output = grad_emitted_output.contiguous()

        grid = (update_gate.shape[0], triton.cdiv(ctx.width, ctx.block_size))
        _p20_backward_kernel[grid](
            grad_next_state,
            grad_emitted_output,
            update_gate,
            retain_gate,
            transformed_state,
            candidate,
            output_gate,
            next_state,
            grad_update_gate,
            grad_retain_gate,
            grad_transformed_state,
            grad_candidate,
            grad_output_gate,
            grad_next_state.stride(0),
            ctx.input_row_stride,
            ctx.output_row_stride,
            grad_update_gate.stride(0),
            ctx.width,
            BLOCK_SIZE=ctx.block_size,
        )
        return grad_update_gate, grad_retain_gate, grad_transformed_state, grad_candidate, grad_output_gate


if triton_runtime_available():

    @triton.jit
    def _p20_block_diagonal_sequence_forward_kernel(
        update_gate_ptr,
        angle_cos_ptr,
        angle_sin_ptr,
        candidate_ptr,
        output_gate_ptr,
        initial_state_ptr,
        transform_weight_ptr,
        transform_bias_ptr,
        state_history_ptr,
        emitted_output_ptr,
        tensor_batch_stride,
        tensor_seq_stride,
        pair_batch_stride,
        pair_seq_stride,
        state_batch_stride,
        history_batch_stride,
        history_seq_stride,
        output_batch_stride,
        output_seq_stride,
        weight_block_stride,
        weight_row_stride,
        weight_col_stride,
        block_width,
        pair_width,
        SEQ_LEN: tl.constexpr,
        BLOCK_PAIR_WIDTH: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        block_index = tl.program_id(1)

        pair_offsets = tl.arange(0, BLOCK_PAIR_WIDTH)
        pair_mask = pair_offsets < pair_width
        even_offsets = pair_offsets * 2
        odd_offsets = even_offsets + 1
        block_base = block_index * block_width
        global_even_offsets = block_base + even_offsets
        global_odd_offsets = block_base + odd_offsets

        bias_even = tl.load(transform_bias_ptr + global_even_offsets, mask=pair_mask, other=0.0).to(tl.float32)
        bias_odd = tl.load(transform_bias_ptr + global_odd_offsets, mask=pair_mask, other=0.0).to(tl.float32)

        weight_block_ptr = transform_weight_ptr + block_index * weight_block_stride
        matrix_mask = pair_mask[:, None] & pair_mask[None, :]
        row_even = even_offsets[:, None]
        row_odd = odd_offsets[:, None]
        col_even = even_offsets[None, :]
        col_odd = odd_offsets[None, :]
        weight_even_even = tl.load(
            weight_block_ptr + row_even * weight_row_stride + col_even * weight_col_stride,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        weight_even_odd = tl.load(
            weight_block_ptr + row_even * weight_row_stride + col_odd * weight_col_stride,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        weight_odd_even = tl.load(
            weight_block_ptr + row_odd * weight_row_stride + col_even * weight_col_stride,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        weight_odd_odd = tl.load(
            weight_block_ptr + row_odd * weight_row_stride + col_odd * weight_col_stride,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)

        state_base = batch_index * state_batch_stride
        state_even = tl.load(initial_state_ptr + state_base + global_even_offsets, mask=pair_mask, other=0.0).to(
            tl.float32
        )
        state_odd = tl.load(initial_state_ptr + state_base + global_odd_offsets, mask=pair_mask, other=0.0).to(
            tl.float32
        )

        history_batch_base = batch_index * history_batch_stride
        tl.store(state_history_ptr + history_batch_base + global_even_offsets, state_even, mask=pair_mask)
        tl.store(state_history_ptr + history_batch_base + global_odd_offsets, state_odd, mask=pair_mask)

        tensor_batch_base = batch_index * tensor_batch_stride
        pair_batch_base = batch_index * pair_batch_stride
        output_batch_base = batch_index * output_batch_stride

        for position in range(SEQ_LEN):
            projected_even = (
                tl.sum(weight_even_even * state_even[None, :], axis=1)
                + tl.sum(weight_even_odd * state_odd[None, :], axis=1)
                + bias_even
            )
            projected_odd = (
                tl.sum(weight_odd_even * state_even[None, :], axis=1)
                + tl.sum(weight_odd_odd * state_odd[None, :], axis=1)
                + bias_odd
            )

            pair_step_base = pair_batch_base + position * pair_seq_stride + block_index * pair_width
            cos = tl.load(angle_cos_ptr + pair_step_base + pair_offsets, mask=pair_mask, other=0.0).to(tl.float32)
            sin = tl.load(angle_sin_ptr + pair_step_base + pair_offsets, mask=pair_mask, other=0.0).to(tl.float32)
            rotated_even = projected_even * cos - projected_odd * sin
            rotated_odd = projected_even * sin + projected_odd * cos

            tensor_step_base = tensor_batch_base + position * tensor_seq_stride + block_base
            update_even = tl.load(update_gate_ptr + tensor_step_base + even_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            update_odd = tl.load(update_gate_ptr + tensor_step_base + odd_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            retain_even = 1.0 - update_even
            retain_odd = 1.0 - update_odd
            candidate_even = tl.load(candidate_ptr + tensor_step_base + even_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            candidate_odd = tl.load(candidate_ptr + tensor_step_base + odd_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            output_even = tl.load(output_gate_ptr + tensor_step_base + even_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            output_odd = tl.load(output_gate_ptr + tensor_step_base + odd_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )

            next_even = update_even * rotated_even + retain_even * candidate_even
            next_odd = update_odd * rotated_odd + retain_odd * candidate_odd
            emitted_even = output_even * next_even
            emitted_odd = output_odd * next_odd

            output_step_base = output_batch_base + position * output_seq_stride + block_base
            tl.store(emitted_output_ptr + output_step_base + even_offsets, emitted_even, mask=pair_mask)
            tl.store(emitted_output_ptr + output_step_base + odd_offsets, emitted_odd, mask=pair_mask)

            history_step_base = history_batch_base + (position + 1) * history_seq_stride + block_base
            tl.store(state_history_ptr + history_step_base + even_offsets, next_even, mask=pair_mask)
            tl.store(state_history_ptr + history_step_base + odd_offsets, next_odd, mask=pair_mask)

            state_even = next_even
            state_odd = next_odd


    @triton.jit
    def _p20_block_diagonal_sequence_backward_kernel(
        grad_emitted_output_ptr,
        grad_final_state_ptr,
        update_gate_ptr,
        angle_cos_ptr,
        angle_sin_ptr,
        candidate_ptr,
        output_gate_ptr,
        state_history_ptr,
        transform_weight_ptr,
        transform_bias_ptr,
        grad_update_gate_ptr,
        grad_angle_cos_ptr,
        grad_angle_sin_ptr,
        grad_candidate_ptr,
        grad_output_gate_ptr,
        grad_initial_state_ptr,
        grad_transform_weight_ptr,
        grad_transform_bias_ptr,
        tensor_batch_stride,
        tensor_seq_stride,
        pair_batch_stride,
        pair_seq_stride,
        state_batch_stride,
        history_batch_stride,
        history_seq_stride,
        weight_block_stride,
        weight_row_stride,
        weight_col_stride,
        grad_weight_batch_stride,
        grad_weight_block_stride,
        grad_weight_row_stride,
        grad_weight_col_stride,
        grad_bias_batch_stride,
        block_width,
        pair_width,
        SEQ_LEN: tl.constexpr,
        BLOCK_PAIR_WIDTH: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        block_index = tl.program_id(1)

        pair_offsets = tl.arange(0, BLOCK_PAIR_WIDTH)
        pair_mask = pair_offsets < pair_width
        even_offsets = pair_offsets * 2
        odd_offsets = even_offsets + 1
        block_base = block_index * block_width
        global_even_offsets = block_base + even_offsets
        global_odd_offsets = block_base + odd_offsets

        weight_block_ptr = transform_weight_ptr + block_index * weight_block_stride
        matrix_mask = pair_mask[:, None] & pair_mask[None, :]
        row_even = even_offsets[:, None]
        row_odd = odd_offsets[:, None]
        col_even = even_offsets[None, :]
        col_odd = odd_offsets[None, :]
        weight_even_even = tl.load(
            weight_block_ptr + row_even * weight_row_stride + col_even * weight_col_stride,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        weight_even_odd = tl.load(
            weight_block_ptr + row_even * weight_row_stride + col_odd * weight_col_stride,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        weight_odd_even = tl.load(
            weight_block_ptr + row_odd * weight_row_stride + col_even * weight_col_stride,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        weight_odd_odd = tl.load(
            weight_block_ptr + row_odd * weight_row_stride + col_odd * weight_col_stride,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        bias_even = tl.load(transform_bias_ptr + global_even_offsets, mask=pair_mask, other=0.0).to(tl.float32)
        bias_odd = tl.load(transform_bias_ptr + global_odd_offsets, mask=pair_mask, other=0.0).to(tl.float32)

        grad_state_even = tl.load(
            grad_final_state_ptr + batch_index * state_batch_stride + global_even_offsets,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)
        grad_state_odd = tl.load(
            grad_final_state_ptr + batch_index * state_batch_stride + global_odd_offsets,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)

        grad_weight_even_even = tl.zeros((BLOCK_PAIR_WIDTH, BLOCK_PAIR_WIDTH), dtype=tl.float32)
        grad_weight_even_odd = tl.zeros((BLOCK_PAIR_WIDTH, BLOCK_PAIR_WIDTH), dtype=tl.float32)
        grad_weight_odd_even = tl.zeros((BLOCK_PAIR_WIDTH, BLOCK_PAIR_WIDTH), dtype=tl.float32)
        grad_weight_odd_odd = tl.zeros((BLOCK_PAIR_WIDTH, BLOCK_PAIR_WIDTH), dtype=tl.float32)
        grad_bias_even = tl.zeros((BLOCK_PAIR_WIDTH,), dtype=tl.float32)
        grad_bias_odd = tl.zeros((BLOCK_PAIR_WIDTH,), dtype=tl.float32)

        tensor_batch_base = batch_index * tensor_batch_stride
        pair_batch_base = batch_index * pair_batch_stride
        history_batch_base = batch_index * history_batch_stride

        for position in range(SEQ_LEN - 1, -1, -1):
            state_history_base = history_batch_base + position * history_seq_stride + block_base
            next_state_history_base = history_batch_base + (position + 1) * history_seq_stride + block_base
            state_even = tl.load(state_history_ptr + state_history_base + even_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            state_odd = tl.load(state_history_ptr + state_history_base + odd_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            next_even = tl.load(
                state_history_ptr + next_state_history_base + even_offsets,
                mask=pair_mask,
                other=0.0,
            ).to(tl.float32)
            next_odd = tl.load(
                state_history_ptr + next_state_history_base + odd_offsets,
                mask=pair_mask,
                other=0.0,
            ).to(tl.float32)

            projected_even = (
                tl.sum(weight_even_even * state_even[None, :], axis=1)
                + tl.sum(weight_even_odd * state_odd[None, :], axis=1)
                + bias_even
            )
            projected_odd = (
                tl.sum(weight_odd_even * state_even[None, :], axis=1)
                + tl.sum(weight_odd_odd * state_odd[None, :], axis=1)
                + bias_odd
            )

            pair_step_base = pair_batch_base + position * pair_seq_stride + block_index * pair_width
            cos = tl.load(angle_cos_ptr + pair_step_base + pair_offsets, mask=pair_mask, other=0.0).to(tl.float32)
            sin = tl.load(angle_sin_ptr + pair_step_base + pair_offsets, mask=pair_mask, other=0.0).to(tl.float32)
            rotated_even = projected_even * cos - projected_odd * sin
            rotated_odd = projected_even * sin + projected_odd * cos

            tensor_step_base = tensor_batch_base + position * tensor_seq_stride + block_base
            update_even = tl.load(update_gate_ptr + tensor_step_base + even_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            update_odd = tl.load(update_gate_ptr + tensor_step_base + odd_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            retain_even = 1.0 - update_even
            retain_odd = 1.0 - update_odd
            candidate_even = tl.load(candidate_ptr + tensor_step_base + even_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            candidate_odd = tl.load(candidate_ptr + tensor_step_base + odd_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            output_even = tl.load(output_gate_ptr + tensor_step_base + even_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            output_odd = tl.load(output_gate_ptr + tensor_step_base + odd_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )

            grad_output_base = tensor_batch_base + position * tensor_seq_stride + block_base
            grad_emitted_even = tl.load(
                grad_emitted_output_ptr + grad_output_base + even_offsets,
                mask=pair_mask,
                other=0.0,
            ).to(tl.float32)
            grad_emitted_odd = tl.load(
                grad_emitted_output_ptr + grad_output_base + odd_offsets,
                mask=pair_mask,
                other=0.0,
            ).to(tl.float32)

            total_grad_next_even = grad_state_even + grad_emitted_even * output_even
            total_grad_next_odd = grad_state_odd + grad_emitted_odd * output_odd
            grad_output_even_step = grad_emitted_even * next_even
            grad_output_odd_step = grad_emitted_odd * next_odd

            grad_update_even = total_grad_next_even * rotated_even
            grad_update_odd = total_grad_next_odd * rotated_odd
            grad_retain_even = total_grad_next_even * candidate_even
            grad_retain_odd = total_grad_next_odd * candidate_odd
            grad_candidate_even = total_grad_next_even * retain_even
            grad_candidate_odd = total_grad_next_odd * retain_odd

            grad_rotated_even = total_grad_next_even * update_even
            grad_rotated_odd = total_grad_next_odd * update_odd
            grad_projected_even = grad_rotated_even * cos + grad_rotated_odd * sin
            grad_projected_odd = -grad_rotated_even * sin + grad_rotated_odd * cos
            grad_cos = grad_rotated_even * projected_even + grad_rotated_odd * projected_odd
            grad_sin = -grad_rotated_even * projected_odd + grad_rotated_odd * projected_even

            tl.store(
                grad_update_gate_ptr + grad_output_base + even_offsets,
                grad_update_even - grad_retain_even,
                mask=pair_mask,
            )
            tl.store(
                grad_update_gate_ptr + grad_output_base + odd_offsets,
                grad_update_odd - grad_retain_odd,
                mask=pair_mask,
            )
            tl.store(grad_candidate_ptr + grad_output_base + even_offsets, grad_candidate_even, mask=pair_mask)
            tl.store(grad_candidate_ptr + grad_output_base + odd_offsets, grad_candidate_odd, mask=pair_mask)
            tl.store(grad_output_gate_ptr + grad_output_base + even_offsets, grad_output_even_step, mask=pair_mask)
            tl.store(grad_output_gate_ptr + grad_output_base + odd_offsets, grad_output_odd_step, mask=pair_mask)
            tl.store(grad_angle_cos_ptr + pair_step_base + pair_offsets, grad_cos, mask=pair_mask)
            tl.store(grad_angle_sin_ptr + pair_step_base + pair_offsets, grad_sin, mask=pair_mask)

            grad_weight_even_even += grad_projected_even[:, None] * state_even[None, :]
            grad_weight_even_odd += grad_projected_even[:, None] * state_odd[None, :]
            grad_weight_odd_even += grad_projected_odd[:, None] * state_even[None, :]
            grad_weight_odd_odd += grad_projected_odd[:, None] * state_odd[None, :]
            grad_bias_even += grad_projected_even
            grad_bias_odd += grad_projected_odd

            grad_state_even = (
                tl.sum(weight_even_even * grad_projected_even[:, None], axis=0)
                + tl.sum(weight_odd_even * grad_projected_odd[:, None], axis=0)
            )
            grad_state_odd = (
                tl.sum(weight_even_odd * grad_projected_even[:, None], axis=0)
                + tl.sum(weight_odd_odd * grad_projected_odd[:, None], axis=0)
            )

        grad_initial_base = batch_index * state_batch_stride + block_base
        tl.store(grad_initial_state_ptr + grad_initial_base + even_offsets, grad_state_even, mask=pair_mask)
        tl.store(grad_initial_state_ptr + grad_initial_base + odd_offsets, grad_state_odd, mask=pair_mask)

        grad_bias_base = batch_index * grad_bias_batch_stride + block_base
        tl.store(grad_transform_bias_ptr + grad_bias_base + even_offsets, grad_bias_even, mask=pair_mask)
        tl.store(grad_transform_bias_ptr + grad_bias_base + odd_offsets, grad_bias_odd, mask=pair_mask)

        grad_weight_block_ptr = (
            grad_transform_weight_ptr
            + batch_index * grad_weight_batch_stride
            + block_index * grad_weight_block_stride
        )
        tl.store(
            grad_weight_block_ptr + row_even * grad_weight_row_stride + col_even * grad_weight_col_stride,
            grad_weight_even_even,
            mask=matrix_mask,
        )
        tl.store(
            grad_weight_block_ptr + row_even * grad_weight_row_stride + col_odd * grad_weight_col_stride,
            grad_weight_even_odd,
            mask=matrix_mask,
        )
        tl.store(
            grad_weight_block_ptr + row_odd * grad_weight_row_stride + col_even * grad_weight_col_stride,
            grad_weight_odd_even,
            mask=matrix_mask,
        )
        tl.store(
            grad_weight_block_ptr + row_odd * grad_weight_row_stride + col_odd * grad_weight_col_stride,
            grad_weight_odd_odd,
            mask=matrix_mask,
        )


class _P20BlockDiagonalSequenceScan(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        update_gate: Tensor,
        angle_cos: Tensor,
        angle_sin: Tensor,
        candidate: Tensor,
        output_gate: Tensor,
        initial_state: Tensor,
        transform_weight: Tensor,
        transform_bias: Tensor,
    ) -> tuple[Tensor, Tensor]:
        ensure_triton_runtime_available()
        tensors = (
            update_gate,
            angle_cos,
            angle_sin,
            candidate,
            output_gate,
            initial_state,
            transform_weight,
            transform_bias,
        )
        if any(tensor.device.type != "cuda" for tensor in tensors):
            raise RuntimeError("P20 Triton sequence scan requires CUDA tensors")

        update_gate = update_gate.contiguous()
        angle_cos = angle_cos.contiguous()
        angle_sin = angle_sin.contiguous()
        candidate = candidate.contiguous()
        output_gate = output_gate.contiguous()
        initial_state = initial_state.contiguous()
        transform_weight = transform_weight.contiguous()
        transform_bias = transform_bias.contiguous()

        batch_size, seq_len, width = update_gate.shape
        if candidate.shape != update_gate.shape or output_gate.shape != update_gate.shape:
            raise RuntimeError("P20 Triton sequence scan requires matching [batch, seq, width] tensor shapes")
        if initial_state.shape != (batch_size, width):
            raise RuntimeError("P20 Triton sequence scan requires initial_state shape [batch, width]")
        if angle_cos.shape != angle_sin.shape:
            raise RuntimeError("P20 Triton sequence scan requires matching cosine and sine runtime plans")
        if angle_cos.shape[:2] != update_gate.shape[:2]:
            raise RuntimeError("P20 Triton sequence scan requires angle tensors aligned with [batch, seq]")

        blocks, block_width, block_width_cols = transform_weight.shape
        if block_width != block_width_cols:
            raise RuntimeError("P20 Triton sequence scan requires square block-diagonal transform weights")
        if blocks * block_width != width:
            raise RuntimeError("P20 Triton sequence scan requires block-diagonal weight width to match state width")
        if block_width % 2 != 0:
            raise RuntimeError("P20 Triton sequence scan requires an even per-block width for rotary pairs")
        pair_width = block_width // 2
        expected_angle_width = width // 2
        if angle_cos.shape[2] != expected_angle_width:
            raise RuntimeError("P20 Triton sequence scan requires angle width equal to state_width // 2")
        block_pair_width = _next_power_of_two(pair_width)

        emitted_outputs = torch.empty_like(update_gate)
        state_history = torch.empty(
            batch_size,
            seq_len + 1,
            width,
            device=update_gate.device,
            dtype=update_gate.dtype,
        )

        grid = (batch_size, blocks)
        _p20_block_diagonal_sequence_forward_kernel[grid](
            update_gate,
            angle_cos,
            angle_sin,
            candidate,
            output_gate,
            initial_state,
            transform_weight,
            transform_bias,
            state_history,
            emitted_outputs,
            update_gate.stride(0),
            update_gate.stride(1),
            angle_cos.stride(0),
            angle_cos.stride(1),
            initial_state.stride(0),
            state_history.stride(0),
            state_history.stride(1),
            emitted_outputs.stride(0),
            emitted_outputs.stride(1),
            transform_weight.stride(0),
            transform_weight.stride(1),
            transform_weight.stride(2),
            block_width,
            pair_width,
            SEQ_LEN=seq_len,
            BLOCK_PAIR_WIDTH=block_pair_width,
        )

        ctx.seq_len = seq_len
        ctx.block_pair_width = block_pair_width
        ctx.pair_width = pair_width
        ctx.save_for_backward(
            update_gate,
            angle_cos,
            angle_sin,
            candidate,
            output_gate,
            initial_state,
            transform_weight,
            transform_bias,
            state_history,
        )
        return emitted_outputs, state_history[:, -1, :].contiguous()

    @staticmethod
    def backward(ctx, grad_emitted_outputs: Tensor | None, grad_final_state: Tensor | None):
        (
            update_gate,
            angle_cos,
            angle_sin,
            candidate,
            output_gate,
            initial_state,
            transform_weight,
            transform_bias,
            state_history,
        ) = ctx.saved_tensors
        if grad_emitted_outputs is None:
            grad_emitted_outputs = torch.zeros_like(update_gate)
        else:
            grad_emitted_outputs = grad_emitted_outputs.contiguous()
        if grad_final_state is None:
            grad_final_state = torch.zeros_like(initial_state)
        else:
            grad_final_state = grad_final_state.contiguous()

        batch_size, _seq_len, width = update_gate.shape
        blocks, block_width, _block_width_cols = transform_weight.shape

        grad_update_gate = torch.empty_like(update_gate)
        grad_angle_cos = torch.empty_like(angle_cos)
        grad_angle_sin = torch.empty_like(angle_sin)
        grad_candidate = torch.empty_like(candidate)
        grad_output_gate = torch.empty_like(output_gate)
        grad_initial_state = torch.empty_like(initial_state)
        grad_transform_weight_contrib = torch.empty(
            batch_size,
            blocks,
            block_width,
            block_width,
            device=transform_weight.device,
            dtype=transform_weight.dtype,
        )
        grad_transform_bias_contrib = torch.empty(
            batch_size,
            width,
            device=transform_bias.device,
            dtype=transform_bias.dtype,
        )

        grid = (batch_size, blocks)
        _p20_block_diagonal_sequence_backward_kernel[grid](
            grad_emitted_outputs,
            grad_final_state,
            update_gate,
            angle_cos,
            angle_sin,
            candidate,
            output_gate,
            state_history,
            transform_weight,
            transform_bias,
            grad_update_gate,
            grad_angle_cos,
            grad_angle_sin,
            grad_candidate,
            grad_output_gate,
            grad_initial_state,
            grad_transform_weight_contrib,
            grad_transform_bias_contrib,
            update_gate.stride(0),
            update_gate.stride(1),
            angle_cos.stride(0),
            angle_cos.stride(1),
            initial_state.stride(0),
            state_history.stride(0),
            state_history.stride(1),
            transform_weight.stride(0),
            transform_weight.stride(1),
            transform_weight.stride(2),
            grad_transform_weight_contrib.stride(0),
            grad_transform_weight_contrib.stride(1),
            grad_transform_weight_contrib.stride(2),
            grad_transform_weight_contrib.stride(3),
            grad_transform_bias_contrib.stride(0),
            block_width,
            ctx.pair_width,
            SEQ_LEN=ctx.seq_len,
            BLOCK_PAIR_WIDTH=ctx.block_pair_width,
        )
        grad_transform_weight = grad_transform_weight_contrib.sum(dim=0)
        grad_transform_bias = grad_transform_bias_contrib.sum(dim=0)
        return (
            grad_update_gate,
            grad_angle_cos,
            grad_angle_sin,
            grad_candidate,
            grad_output_gate,
            grad_initial_state,
            grad_transform_weight,
            grad_transform_bias,
        )


def _rotate_state_pairs_with_trig(state: Tensor, *, cos: Tensor, sin: Tensor) -> Tensor:
    batch_size, width = state.shape
    pair_count = width // 2
    state_pairs = state.reshape(batch_size, pair_count, 2)
    first = state_pairs[..., 0]
    second = state_pairs[..., 1]
    rotated_first = first * cos - second * sin
    rotated_second = first * sin + second * cos
    return torch.stack((rotated_first, rotated_second), dim=-1).reshape(batch_size, width)


class P20RuntimeSequenceMixer(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        state_transform_blocks: int = 2,
        runtime_backend: str = "triton",
        scan_chunk_size: int = 0,
    ) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"P20 requires even d_model, got {d_model}")
        validate_p20_runtime_backend(runtime_backend)
        if scan_chunk_size < 0:
            raise ValueError(f"P20 scan_chunk_size must be non-negative, got {scan_chunk_size}")
        self.d_model = d_model
        self.runtime_backend = runtime_backend
        self.scan_chunk_size = scan_chunk_size
        self.in_projection = PackedLinearProjection(d_model, (d_model, d_model // 2, d_model, d_model))
        self.state_transform_projection = BlockDiagonalLinear(d_model, blocks=state_transform_blocks)

    def forward(self, inputs: Tensor) -> Tensor:
        update_gate_inputs, angle_inputs, candidate_inputs, output_gate_inputs = self.in_projection(inputs)
        update_gates = torch.sigmoid(update_gate_inputs)
        angle_cos = torch.cos(angle_inputs)
        angle_sin = torch.sin(angle_inputs)
        candidates = torch.tanh(candidate_inputs)
        output_gates = torch.sigmoid(output_gate_inputs)

        batch_size, seq_len, _ = inputs.shape
        state = torch.zeros(batch_size, self.d_model, device=inputs.device, dtype=inputs.dtype)
        if self.runtime_backend == "triton":
            if not inputs.is_cuda:
                raise RuntimeError("P20 Triton sequence scan requires CUDA inputs")
            transform_weight = self.state_transform_projection.weight.reshape(
                self.state_transform_projection.blocks,
                self.state_transform_projection.block_width,
                self.state_transform_projection.block_width,
            )
            scan_batch_size = batch_size
            scan_seq_len = seq_len
            if self.scan_chunk_size > 0 and seq_len > self.scan_chunk_size:
                chunk_size = self.scan_chunk_size
                if seq_len % chunk_size != 0:
                    raise RuntimeError(
                        f"P20_SCAN_CHUNK_SIZE={chunk_size} must evenly divide sequence length {seq_len}"
                    )
                chunk_count = seq_len // chunk_size

                def chunk_tensor(tensor: Tensor) -> Tensor:
                    return tensor.reshape(batch_size, chunk_count, chunk_size, *tensor.shape[2:]).reshape(
                        batch_size * chunk_count,
                        chunk_size,
                        *tensor.shape[2:],
                    )

                update_gates = chunk_tensor(update_gates)
                angle_cos = chunk_tensor(angle_cos)
                angle_sin = chunk_tensor(angle_sin)
                candidates = chunk_tensor(candidates)
                output_gates = chunk_tensor(output_gates)
                scan_batch_size = batch_size * chunk_count
                scan_seq_len = chunk_size
                state = torch.zeros(scan_batch_size, self.d_model, device=inputs.device, dtype=inputs.dtype)
            outputs, _final_state = _P20BlockDiagonalSequenceScan.apply(
                update_gates,
                angle_cos,
                angle_sin,
                candidates,
                output_gates,
                state,
                transform_weight,
                self.state_transform_projection.bias,
            )
            if self.scan_chunk_size > 0 and seq_len > self.scan_chunk_size:
                outputs = outputs.reshape(batch_size, seq_len // scan_seq_len, scan_seq_len, self.d_model).reshape(
                    batch_size,
                    seq_len,
                    self.d_model,
            )
            return outputs

        retain_gates = 1.0 - update_gates
        outputs = torch.empty(batch_size, seq_len, self.d_model, device=inputs.device, dtype=inputs.dtype)
        for position in range(seq_len):
            projected_state = self.state_transform_projection(state)
            transformed_state = _rotate_state_pairs_with_trig(
                projected_state,
                cos=angle_cos[:, position, :],
                sin=angle_sin[:, position, :],
            )
            state = update_gates[:, position, :] * transformed_state + retain_gates[:, position, :] * candidates[:, position, :]
            outputs[:, position, :] = output_gates[:, position, :] * state
        return outputs

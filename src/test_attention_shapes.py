"""Expected contracts:
- MultiHeadAttention: input x (b, T, d_model) -> output (b, T, d_out)
- In this project usage, d_out is set to d_model for residual-shape compatibility.
"""

from __future__ import annotations

import torch

from src.attention import MultiHeadAttention


def _run_device_case(device: torch.device) -> None:
    torch.manual_seed(42)

    b, T, d_model, num_heads = 2, 6, 8, 2
    x = torch.randn(b, T, d_model, device=device, dtype=torch.float32)

    mha = MultiHeadAttention(
        d_in=d_model,
        d_out=d_model,
        context_length=T,
        dropout=0.0,
        num_heads=num_heads,
    ).to(device)
    mha.eval()

    out = mha(x)
    assert out.shape == (b, T, d_model), f"Unexpected output shape: {tuple(out.shape)}"
    assert torch.is_floating_point(out), f"Output must be floating dtype, got {out.dtype}"
    # MPS may report as "mps:0" while requested device is "mps".
    assert out.device.type == device.type, (
        f"Output device type mismatch: {out.device.type} vs {device.type}"
    )
    if device.index is not None:
        assert out.device.index == device.index, (
            f"Output device index mismatch: {out.device.index} vs {device.index}"
        )

    # Causality check: changing a future token should not change earlier outputs.
    x2 = x.clone()
    x2[:, 4, :] = x2[:, 4, :] + 10.0

    out1 = mha(x)
    out2 = mha(x2)

    assert torch.allclose(
        out1[:, :4, :],
        out2[:, :4, :],
        atol=1e-5,
        rtol=1e-5,
    ), "Causal masking violated: earlier positions changed after future-token edit"

    print(f"[{device.type}] x: {tuple(x.shape)} -> out: {tuple(out.shape)}, dtype={out.dtype}")


def main() -> int:
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))

    for device in devices:
        _run_device_case(device)

    print("attention shape/causality checks: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Tests for neural network models."""

import pytest
import torch

from diffusionsde.models import ConditionalUNet, ResidualBlock, SinusoidalPositionalEncoding
from tests.conftest import assert_finite_with_shape


class TestSinusoidalPositionalEncoding:
    """Test suite for SinusoidalPositionalEncoding."""

    def test_output_shape(self):
        """Test that output shape is correct and values are finite."""
        dim = 128
        batch_size = 4

        encoder = SinusoidalPositionalEncoding(dim)
        t = torch.randint(0, 1000, (batch_size,))

        encoding = encoder(t)

        assert_finite_with_shape(encoding, (batch_size, dim))

    def test_different_timesteps_different_encodings(self):
        """Test that different timesteps produce different encodings."""
        encoder = SinusoidalPositionalEncoding(64)

        t1 = torch.tensor([0])
        t2 = torch.tensor([500])

        enc1 = encoder(t1)
        enc2 = encoder(t2)

        assert not torch.allclose(enc1, enc2)

    def test_encoding_bounded(self):
        """Test that sinusoidal encoding values are bounded."""
        encoder = SinusoidalPositionalEncoding(128)
        t = torch.arange(0, 1000)

        encoding = encoder(t)

        assert encoding.min() >= -1.0
        assert encoding.max() <= 1.0


class TestResidualBlock:
    """Test suite for ResidualBlock."""

    def test_output_shape_same_channels(self):
        """Test output shape when in_channels == out_channels."""
        block = ResidualBlock(64, 64, 128)

        x = torch.randn(4, 64, 50)
        time_emb = torch.randn(4, 128)

        output = block(x, time_emb)

        assert output.shape == (4, 64, 50)
        assert not torch.isnan(output).any()

    def test_output_shape_different_channels(self):
        """Test output shape when in_channels != out_channels."""
        block = ResidualBlock(32, 64, 128)

        x = torch.randn(4, 32, 50)
        time_emb = torch.randn(4, 128)

        output = block(x, time_emb)

        assert output.shape == (4, 64, 50)
        assert not torch.isnan(output).any()


class TestConditionalUNet:
    """Test suite for ConditionalUNet."""

    def test_physical_position_encoding_matches_trajectory_length(self):
        """Test that the U-Net carries a fixed physical-time encoding."""
        input_dim = 50
        hidden_dim = 32
        unet = ConditionalUNet(
            input_dim=input_dim, hidden_dim=hidden_dim, n_layers=2, use_position_encoding=True
        )

        encoding = unet.physical_position_encoding

        assert encoding.shape == (1, hidden_dim, input_dim)
        assert torch.allclose(
            encoding[0, 0],
            torch.linspace(-1.0, 1.0, input_dim, dtype=encoding.dtype),
        )

    def test_forward_pass(self, unet, config):
        """Test ConditionalUNet forward pass."""
        batch_size = config.batch_size
        trajectory_length = config.trajectory_length

        x_t = torch.randn(batch_size, trajectory_length)
        t = torch.randint(0, config.n_timesteps, (batch_size,))
        conditioning = torch.randn(batch_size, trajectory_length)

        output = unet(x_t, t, conditioning)

        assert_finite_with_shape(output, (batch_size, trajectory_length))

    def test_forward_pass_multi_channel_conditioning(self, config):
        """Test ConditionalUNet with two conditioning channels."""
        batch_size = config.batch_size
        trajectory_length = config.trajectory_length
        unet = ConditionalUNet(
            input_dim=trajectory_length,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            time_emb_dim=32,
            n_input_channels=3,
        )

        x_t = torch.randn(batch_size, trajectory_length)
        t = torch.randint(0, config.n_timesteps, (batch_size,))
        conditioning = torch.randn(batch_size, 2, trajectory_length)

        output = unet(x_t, t, conditioning)

        assert_finite_with_shape(output, (batch_size, trajectory_length))

    def test_gradient_flow(self, unet, config):
        """Test that gradients flow through the model."""
        x_t = torch.randn(2, config.trajectory_length, requires_grad=True)
        t = torch.randint(0, config.n_timesteps, (2,))
        conditioning = torch.randn(2, config.trajectory_length, requires_grad=True)

        output = unet(x_t, t, conditioning)
        loss = output.sum()
        loss.backward()

        assert x_t.grad is not None
        assert conditioning.grad is not None

    @pytest.mark.parametrize("input_dim", [50, 100, 250])
    def test_different_input_dims(self, input_dim):
        unet = ConditionalUNet(input_dim=input_dim, hidden_dim=32, n_layers=2)
        x_t = torch.randn(2, input_dim)
        t = torch.randint(0, 100, (2,))
        c = torch.randn(2, input_dim)
        output = unet(x_t, t, c)
        assert output.shape == (2, input_dim)

    @pytest.mark.parametrize("n_layers", [2, 3, 4])
    def test_different_n_layers(self, n_layers):
        unet = ConditionalUNet(input_dim=50, hidden_dim=32, n_layers=n_layers)
        x_t = torch.randn(2, 50)
        t = torch.randint(0, 100, (2,))
        c = torch.randn(2, 50)
        output = unet(x_t, t, c)
        assert output.shape == (2, 50)

    def test_batch_independence(self, unet, config):
        """Test that batch samples are processed independently."""
        x_t = torch.randn(4, config.trajectory_length)
        t = torch.randint(0, config.n_timesteps, (4,))
        c = torch.randn(4, config.trajectory_length)

        output_batch = unet(x_t, t, c)

        outputs_single = []
        for i in range(4):
            out = unet(x_t[i : i + 1], t[i : i + 1], c[i : i + 1])
            outputs_single.append(out)
        output_single = torch.cat(outputs_single, dim=0)

        assert torch.allclose(output_batch, output_single, atol=1e-5)

    def test_causal_unet_future_invariance(self):
        """Causal outputs must not change when future inputs change."""
        input_dim = 64
        cutoff = input_dim // 2
        unet = ConditionalUNet(input_dim=input_dim, hidden_dim=32, n_layers=2, causal=True)
        unet.eval()

        batch = 2
        x_t = torch.randn(batch, input_dim)
        t = torch.randint(0, 100, (batch,))
        c = torch.randn(batch, input_dim)

        with torch.no_grad():
            output = unet(x_t, t, c)

        x_t_future = x_t.clone()
        c_future = c.clone()
        x_t_future[:, cutoff:] += torch.randn(batch, input_dim - cutoff) * 5.0
        c_future[:, cutoff:] += torch.randn(batch, input_dim - cutoff) * 5.0

        with torch.no_grad():
            output_future = unet(x_t_future, t, c_future)

        assert torch.allclose(output[:, :cutoff], output_future[:, :cutoff], atol=1e-5)
        assert not torch.allclose(output[:, cutoff:], output_future[:, cutoff:], atol=1e-3)

    def test_non_causal_unet_uses_future(self):
        """The non-causal U-Net must change earlier outputs when future inputs change."""
        input_dim = 64
        cutoff = input_dim // 2
        unet = ConditionalUNet(input_dim=input_dim, hidden_dim=32, n_layers=2, causal=False)
        unet.eval()

        batch = 2
        x_t = torch.randn(batch, input_dim)
        t = torch.randint(0, 100, (batch,))
        c = torch.randn(batch, input_dim)

        with torch.no_grad():
            output = unet(x_t, t, c)

        x_t_future = x_t.clone()
        c_future = c.clone()
        x_t_future[:, cutoff:] += torch.randn(batch, input_dim - cutoff) * 5.0
        c_future[:, cutoff:] += torch.randn(batch, input_dim - cutoff) * 5.0

        with torch.no_grad():
            output_future = unet(x_t_future, t, c_future)

        assert not torch.allclose(output[:, :cutoff], output_future[:, :cutoff], atol=1e-5)

    def test_causal_unet_same_parameter_count(self):
        """Causality must not change the number of trainable parameters."""
        baseline = ConditionalUNet(input_dim=64, hidden_dim=32, n_layers=2)
        causal = ConditionalUNet(input_dim=64, hidden_dim=32, n_layers=2, causal=True)

        baseline_params = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
        causal_params = sum(p.numel() for p in causal.parameters() if p.requires_grad)

        assert causal_params == baseline_params

    def test_causal_unet_forward_shape(self, config):
        """Causal mode must preserve the sequence length for every U-Net level."""
        unet = ConditionalUNet(
            input_dim=config.trajectory_length,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            time_emb_dim=32,
            causal=True,
            dilation_schedule=[1, 2],
        )
        x_t = torch.randn(config.batch_size, config.trajectory_length)
        t = torch.randint(0, config.n_timesteps, (config.batch_size,))
        c = torch.randn(config.batch_size, config.trajectory_length)

        output = unet(x_t, t, c)

        assert_finite_with_shape(output, (config.batch_size, config.trajectory_length))

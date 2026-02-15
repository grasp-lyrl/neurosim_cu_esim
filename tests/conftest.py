"""Shared fixtures for neurosim_cu_esim tests."""

import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: marks tests that require a CUDA GPU")


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked ``@pytest.mark.cuda`` when no GPU is available."""
    if torch.cuda.is_available():
        return
    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)


# ---- Fixtures -----------------------------------------------------------


@pytest.fixture
def device():
    return torch.device("cuda")


@pytest.fixture
def resolution():
    """Default sensor resolution (W, H)."""
    return 640, 480


@pytest.fixture
def sim(resolution):
    """A fresh EventSimulator at the default resolution."""
    from neurosim_cu_esim import EventSimulator

    w, h = resolution
    return EventSimulator(width=w, height=h, device="cuda")


@pytest.fixture
def constant_frame(resolution, device):
    """A uniform grey frame (value=0.5)."""
    w, h = resolution
    return torch.full((h, w), 0.5, dtype=torch.float32, device=device)


@pytest.fixture
def random_frame(resolution, device):
    """A random frame in (0, 1]."""
    w, h = resolution
    return torch.rand(h, w, dtype=torch.float32, device=device).clamp(min=1e-4)

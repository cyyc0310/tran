import pytest
import torch

from transcif.data.dataset import load_bundle
from transcif.data.synthetic import make_bundle


@pytest.fixture(scope="session", autouse=True)
def single_thread():
    torch.set_num_threads(1)


@pytest.fixture
def bundle(tmp_path):
    manifest = make_bundle(tmp_path / "data", days=12)
    doc, grids = load_bundle(manifest)
    return manifest, doc, grids

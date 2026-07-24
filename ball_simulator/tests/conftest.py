import os
import tempfile
from pathlib import Path
import pytest

@pytest.fixture
def smoke_dataset():
    current_dir = os.path.dirname(__file__)
    dataset_path = os.path.abspath(
        os.path.join(current_dir, "data", "single_wall_dataset.h5")
    )
    return dataset_path

@pytest.fixture
def single_wall_dataset():
    current_dir = os.path.dirname(__file__)
    dataset_path = os.path.abspath(
        os.path.join(current_dir, "data", "single_wall_dataset.h5")
    )
    return dataset_path

@pytest.fixture
def u_box_dataset():
    current_dir = os.path.dirname(__file__)
    dataset_path = os.path.abspath(
        os.path.join(current_dir, "data", "u_box_dataset.h5")
    )
    return dataset_path

@pytest.fixture
def temp_path():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)
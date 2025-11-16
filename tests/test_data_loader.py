import pytest
import pandas as pd
from pathlib import Path
from src.data.data_loader import DataLoader


@pytest.fixture
def sample_data(tmp_path):
    """Create sample data file"""
    data = {
        'qtype': ['treatment', 'symptoms', 'prevention'],
        'Question': ['Q1', 'Q2', 'Q3'],
        'Answer': ['A1', 'A2', 'A3']
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


def test_data_loader_load(sample_data):
    """Test data loading"""
    loader = DataLoader(sample_data)
    df = loader.load_data()
    assert df is not None
    assert len(df) == 3


def test_data_loader_info(sample_data):
    """Test getting data info"""
    loader = DataLoader(sample_data)
    loader.load_data()
    info = loader.get_data_info()
    assert 'shape' in info
    assert info['shape'] == (3, 3)


def test_data_loader_query(sample_data):
    """Test SQL querying"""
    loader = DataLoader(sample_data)
    loader.load_data()
    result = loader.query_data("SELECT * FROM df WHERE qtype = 'treatment'")
    assert len(result) == 1


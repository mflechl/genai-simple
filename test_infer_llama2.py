"""
Test main
"""

from infer_llama2 import main


def test_main():
    """testing inference main function"""
    result = main(n_examples=2)
    assert result == 0

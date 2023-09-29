"""
Test main
"""

from ft_llama2 import main


def test_main():
    """testing fine-tuning main function"""
    result = main(n_examples=3)
    assert result == 0

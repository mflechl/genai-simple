"""
Test main
"""

from ft_llama2 import main


def test_main():
    """testing fine-tuning main function"""
    result = main(n_examples=10, is_test=True)
    assert result == 0

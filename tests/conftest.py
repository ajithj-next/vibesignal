"""Pytest configuration and fixtures."""

import pytest

from vibesignal.config import reset_config


@pytest.fixture(autouse=True)
def reset_config_after_test():
    """Reset global config after each test."""
    yield
    reset_config()


@pytest.fixture
def sample_notebook_data():
    """Sample notebook data for testing."""
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "source": "# First Principles Analysis\n\nThis explores the fundamentals.",
                "metadata": {},
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "source": "import numpy as np\nimport matplotlib.pyplot as plt",
                "outputs": [],
                "metadata": {},
            },
            {
                "cell_type": "code",
                "execution_count": 2,
                "source": "plt.plot([1, 2, 3], [1, 4, 9])\nplt.show()",
                "outputs": [
                    {
                        "data": {
                            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                        },
                        "metadata": {},
                        "output_type": "display_data",
                    }
                ],
                "metadata": {},
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

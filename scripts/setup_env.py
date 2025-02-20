import sys
import pkg_resources
import torch
import platform


def check_environment():
    """Verify Python version, GPU availability, and package versions."""

    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Platform: {platform.platform()}")

    required_packages = [
        'transformers',
        'datasets',
        'requests',
        'numpy',
        'llama-cpp-python',
        'langchain'
    ]

    print("\nPackage Versions:")
    for package in required_packages:
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"{package}: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"{package}: Not installed")


if __name__ == "__main__":
    check_environment()
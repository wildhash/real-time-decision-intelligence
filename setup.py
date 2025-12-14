"""Setup configuration for real-time-decision-intelligence package."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="real-time-decision-intelligence",
    version="0.1.0",
    author="Real-Time Decision Intelligence Team",
    description="A machine learning framework for probabilistic world modeling and real-time decision-making",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wildhash/real-time-decision-intelligence",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=2.0.0",
        "scipy>=1.7.0",
        "gymnasium>=0.28.0",
        "tqdm>=4.65.0",
        "pydantic>=2.0.0",
        "typing-extensions>=4.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
)

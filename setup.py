"""Packaging setup for NOESIS."""

from setuptools import setup, find_packages

setup(
    name="noesis",
    version="0.1.0",
    description="Memory-centric, self-improving inference system",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.3.0",
        "bitsandbytes>=0.43.0",
        "qdrant-client>=1.9.0",
        "firecrawl-py>=0.1.0",
        "unstructured>=0.14.0",
        "PyPDF2>=3.0.0",
        "sentence-transformers>=2.7.0",
        "psutil>=5.9.0",
        "safetensors>=0.4.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0"],
    },
    entry_points={
        "console_scripts": [
            "noesis=noesis.cli:main",
        ],
    },
)

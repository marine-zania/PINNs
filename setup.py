"""
Setup script for PINNs package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pinns",
    version="0.1.0",
    author="Zania",
    description="Physics-Informed Neural Networks for solving PDEs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/marine-zania/PINNs",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
    ],
)

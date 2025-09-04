from setuptools import setup, find_packages
from pathlib import Path

# Read the README.md file for PyPI
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="automated_neural_adapter_ana",
    version="7.6.1",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "datasets",
        "peft",
        "accelerate",
        "bitsandbytes",
    ],
    author="Rudransh Joshi",
    author_email="rudransh20septmber@gmail.com",
    description="A library for LoRA fine-tuning and model merging",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",  # <-- replace with GitHub repo or docs later
    entry_points={
        "console_scripts": [
            "ana=ana.train:run",  # or :main depending on your function name
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)

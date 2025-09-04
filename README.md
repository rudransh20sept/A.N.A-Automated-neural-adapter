
# ANA – Automated Neural Adapter

ANA is a lightweight tool that makes fine-tuning AI models with LoRA simple and beginner-friendly.
It provides a ready-to-run GUI, so you don’t need to mess with complicated scripts.


![Logo](https://i.postimg.cc/g0Gfr5B5/logo-ana-sh.png))


## Installation

Install directly from PyPI:

```bash
  pip install ana
```
    
## Features

Automated LoRA fine-tuning workflow

Simple one-command GUI launch

Runs out of the box on Windows

No complicated setup required



## Requirements

Python 3.10 or newer

Windows (Linux/Mac support coming soon)

For the best performance, install a CUDA-enabled PyTorch build that matches your GPU.
Without CUDA, training will run only on CPU, which is much slower.

## Usage/Examples
After installation, simply run:
```bash
    python -m ana
```


This will launch the GUI window where you can select your model, dataset, and training options.


## Errors & Troubleshooting

If you encounter issues installing or running, you can try the pre-built executable version here:



[Download ANA (Windows EXE)](https://drive.google.com/file/d/1jeBkmLz9x5qZMUdDthbeyDiIL4zFM4Y2/view?usp=drive_link)


## Tips

Install PyTorch with CUDA support (matching your GPU + driver) for much faster training. (Highly recommended)

If you don’t have a physical GPU, CPU training will still work, but it will be very slow.


## Authors

[Rudransh joshi](https://rudransh.kafalfpc.com/)
# DragDiffusion Reproduction Scaffold

Colab free T4 target implementation scaffold for:

`DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing`

## Colab First

This project is intended to run on Colab T4. See [COLAB.md](COLAB.md).

## Optional Local Python 3.12 Environment

Your current local `python` points to Python 3.14. If you also want a local environment, use a Python 3.12 executable for the virtual environment.

```powershell
# Example if Python 3.12 is installed at the default location.
& "$env:LocalAppData\Programs\Python\Python312\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-colab.txt
```

If Python 3.12 is not installed locally, use Colab first or install Python 3.12 separately.

## Run Gradio

```powershell
python app.py
```

In Colab:

```python
!pip install -r requirements-colab.txt
!python app.py
```

## Project Layout

```text
dragdiff_repro/
  config.py
  pipeline.py
  models/
    loader.py
    feature_hooks.py
    attention_control.py
  methods/
    lora_finetune.py
    ddim_inversion.py
    latent_optimization.py
    point_tracking.py
  ui/
    gradio_app.py
  utils/
    image.py
    logging.py
app.py
requirements-colab.txt
```

## Implementation Status

Implemented:

- Gradio UI
- image/mask/point preprocessing
- SD 1.5 loader with fp16, xformers fallback, and VAE slicing
- Generated Image mode with intermediate latent caching
- Real Image mode path with LoRA fine-tuning and DDIM inversion
- latent optimization with motion supervision
- feature-map point tracking
- reference-latent-control denoising through self-attention K/V swap processors

The first practical test should be Generated Image mode at 384 resolution with a small drag step count. Real Image mode is heavier because it adds LoRA fine-tuning and DDIM inversion.

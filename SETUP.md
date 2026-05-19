# Setup Notes

## Colab

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
pip install -r requirements-colab.txt
python app.py
```

Use the public Gradio URL printed by `app.py`.

## Windows Local

The current PATH uses Python 3.14. Create the project venv with Python 3.12 instead:

```powershell
& "$env:LocalAppData\Programs\Python\Python312\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-colab.txt
python app.py
```

If that path does not exist, install Python 3.12 first or run the project on Colab.

## First Test

Start with:

- mode: `Generated Image`
- resolution: `384`
- LoRA steps: `0`
- drag steps: `10`
- handle points: `180,220`
- target points: `250,220`

Then move to Real Image mode after the generated path runs.


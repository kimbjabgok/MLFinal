# 설정 안내

## Colab

Google Drive를 마운트합니다.

```python
from google.colab import drive
drive.mount("/content/drive")
```

필요한 패키지를 설치하고 앱을 실행합니다.

```bash
pip install -r requirements-colab.txt
python app.py
```

`app.py`가 출력하는 public Gradio URL을 열어서 사용합니다.

## Windows 로컬 실행

현재 PATH의 Python이 3.14일 수 있습니다. 프로젝트 가상환경은 Python 3.12로 만드는 것을 권장합니다.

```powershell
& "$env:LocalAppData\Programs\Python\Python312\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-colab.txt
python app.py
```

위 경로가 없다면 Python 3.12를 먼저 설치하거나 Colab에서 프로젝트를 실행하세요.

## 첫 테스트

처음에는 다음 값으로 시작하세요.

- mode: `Generated Image`
- resolution: `384`
- LoRA steps: `0`
- drag steps: `10`
- handle points: `180,220`
- target points: `250,220`

Generated Image 모드가 정상 실행되는 것을 확인한 뒤 Real Image 모드로 넘어가는 것을 권장합니다.

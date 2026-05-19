# Colab 실행 가이드

Colab 무료 T4 기준으로 실행합니다. Colab에는 CUDA용 PyTorch가 기본 설치되어 있으므로 `torch`를 강제로 재설치하지 않습니다.

## 1. 런타임 설정

Colab 메뉴에서:

```text
런타임 → 런타임 유형 변경 → T4 GPU
```

GPU 확인:

```python
import torch
print(torch.__version__)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
```

## 2. Drive 마운트

```python
from google.colab import drive
drive.mount("/content/drive")
```

## 3. 코드 업로드 방식

가장 단순한 방식:

1. 이 프로젝트 폴더를 Google Drive에 업로드
2. Colab에서 해당 폴더로 이동

```python
%cd /content/drive/MyDrive/DragDiffusion
```

GitHub에 올린 뒤 clone하는 방식도 가능:

```bash
git clone <your-repo-url>
%cd <repo-dir>
```

## 4. 패키지 설치

```bash
pip install -r requirements-colab.txt
```

설치 후 런타임 재시작이 필요하다는 메시지가 나오면 재시작 후 다시 폴더로 이동합니다.

## 5. Gradio 실행

```bash
python app.py
```

출력되는 public Gradio URL을 열어서 사용합니다.

## 6. 첫 테스트 권장값

먼저 Generated Image 모드로 테스트합니다.

| 항목 | 값 |
|---|---|
| mode | Generated Image |
| resolution | 384 |
| LoRA steps | 0 |
| Drag steps | 10 |
| prompt | a photo of a cat |
| handle points | 180,220 |
| target points | 250,220 |

Generated Image 모드가 정상 동작한 뒤 Real Image 모드로 넘어갑니다.

## 7. Real Image 모드 권장값

| 항목 | 값 |
|---|---|
| resolution | 384 |
| LoRA steps | 20 또는 50 |
| Drag steps | 30 또는 50 |
| mask | 흰색 영역이 편집 가능인 PNG |

OOM이 나면 LoRA steps, Drag steps를 낮추고, 런타임을 재시작한 뒤 다시 실행합니다.


# DragDiffusion 재현 프로젝트

Colab 무료 T4 환경을 기준으로 만든 구현 스캐폴드입니다.

대상 논문:

`DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing`

## Colab 우선 실행

이 프로젝트는 Colab T4에서 실행하는 것을 우선으로 합니다. 자세한 실행 순서는 [COLAB.md](COLAB.md)를 참고하세요.

## 선택 사항: 로컬 Python 3.12 환경

현재 로컬 `python`이 Python 3.14를 가리킬 수 있습니다. 로컬에서도 실행하려면 Python 3.12 실행 파일로 가상환경을 만드는 것을 권장합니다.

```powershell
# Python 3.12가 기본 경로에 설치되어 있는 경우의 예시입니다.
& "$env:LocalAppData\Programs\Python\Python312\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-colab.txt
```

로컬에 Python 3.12가 없다면 먼저 Colab에서 실행하거나 Python 3.12를 별도로 설치하세요.

## Gradio 실행

로컬에서는 다음 명령을 사용합니다.

```powershell
python app.py
```

Colab에서는 다음 순서로 실행합니다.

```python
!pip install -r requirements-colab.txt
!python app.py
```

## 프로젝트 구조

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

## 구현 상태

구현된 기능:

- Gradio UI
- 이미지, 마스크, 포인트 전처리
- fp16, xformers fallback, VAE slicing을 포함한 Stable Diffusion 1.5 로더
- 중간 latent를 저장하는 Generated Image 모드
- LoRA 미세조정과 DDIM inversion을 사용하는 Real Image 모드 경로
- motion supervision을 이용한 latent 최적화
- feature map 기반 point tracking
- self-attention K/V swap processor를 통한 reference-latent-control denoising

첫 테스트는 `384` 해상도에서 Generated Image 모드와 작은 drag step 값으로 시작하는 것을 권장합니다. Real Image 모드는 LoRA 미세조정과 DDIM inversion이 추가되므로 더 무겁습니다.

# DragDiffusion 재현 프로젝트

Colab 무료 T4 GPU 환경을 기준으로 DragDiffusion을 재현하는 프로젝트입니다.

참고 논문:

`DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing`

## 실행

Colab 실행 순서는 [COLAB.md](COLAB.md)를 참고하세요.

로컬에서 실행하려면 Python 3.12 가상환경을 권장합니다.

```powershell
& "$env:LocalAppData\Programs\Python\Python312\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-colab.txt
python app.py
```

## T4 권장 설정

현재 기본 설정은 Colab 무료 T4에서 돌아가는 것을 우선으로 맞춰져 있습니다.

```text
resolution: 384 x 384
dtype: float16
num_ddim_steps: 50
target_timestep_index: 35
lora_rank: 8
lora_batch_size: 2
lora_steps: 60
lora_lr: 5e-4
vae_tiling: False
cpu_offload: False
```

OOM이 발생하면 우선 `lora_batch_size`를 1로 낮추거나 해상도를 줄이세요. 고해상도 테스트에서는 `vae_tiling=True`를 사용할 수 있지만, 기본 384 해상도에서는 `vae_slicing`만 사용하는 현재 설정이 더 무난합니다.

## 반영된 최적화

- LoRA fine-tuning은 T4를 고려해 rank 8, steps 60, batch 2로 설정했습니다.
- LoRA 학습은 DDPMScheduler 기반 noise sampling을 사용합니다.
- LoRA target module에 `to_out.0`을 추가했습니다.
- DDIM inversion은 VAE latent `sample()` 대신 `mean`을 사용해 재현성을 높였습니다.
- DDIM inversion loop 안에서 prompt embedding을 반복 계산하지 않도록 캐싱했습니다.
- 공식 구현에 가까운 DDIM inverse step을 별도 함수로 분리했습니다.
- loader에서 공식 DragDiffusion에 맞춘 DDIM scheduler 설정을 사용합니다.
- prompt embedding cache를 추가해 같은 prompt의 text encoder 반복 호출을 줄였습니다.
- `vae_tiling` 옵션을 추가했으며 기본값은 `False`입니다.

## 디버그

일반 Gradio 실행에서는 reconstruction 검증이나 중간 latent 저장을 하지 않습니다. DDIM inversion 품질을 확인할 때만 별도 명령으로 실행합니다.

```bash
python scripts/check_inversion.py --image path/to/image.png --prompt "a photo of a cat"
```

결과는 기본적으로 `outputs/inversion_debug`에 저장됩니다.

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
scripts/
  check_inversion.py
app.py
requirements-colab.txt
```

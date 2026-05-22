# Colab 실행 가이드

이 프로젝트는 Colab 무료 T4 GPU를 기준으로 실행합니다. Colab에는 CUDA와 PyTorch가 기본으로 준비되어 있으므로 `torch`를 강제로 재설치하지 않는 것을 권장합니다.

## 1. 런타임 설정

Colab 메뉴에서 다음을 선택합니다.

```text
런타임 -> 런타임 유형 변경 -> T4 GPU
```

GPU 확인:

```python
import torch
print(torch.__version__)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
```

## 2. 코드 준비

Google Drive를 사용하는 경우:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/DragDiffusion
```

GitHub에서 clone하는 경우:

```bash
git clone https://github.com/kimbjabgok/MLFinal.git
cd MLFinal
```

## 3. 패키지 설치

```bash
pip install -r requirements-colab.txt
```

설치 후 런타임 재시작 안내가 나오면 재시작한 뒤 프로젝트 폴더로 다시 이동하세요.

## 4. Gradio 실행

```bash
python app.py
```

출력되는 public Gradio URL을 열어 사용합니다.

## 5. 현재 T4 권장값

기본 설정은 `dragdiff_repro/config.py`에 있습니다.

| 항목 | 값 |
|---|---|
| resolution | 384 x 384 |
| dtype | float16 |
| num_ddim_steps | 50 |
| target_timestep_index | 35 |
| LoRA rank | 8 |
| LoRA batch size | 2 |
| LoRA steps | 60 |
| LoRA lr | 5e-4 |
| guidance_scale_real | 1.0 |
| guidance_scale_generated | 7.5 |
| vae_tiling | False |
| cpu_offload | False |

## 6. 테스트 순서

먼저 Generated Image 모드로 확인하는 것을 권장합니다.

```text
mode: Generated Image
resolution: 384
drag_steps: 10 또는 30
prompt: a photo of a cat
```

Generated Image 모드가 정상 동작하면 Real Image 모드를 테스트합니다. Real Image 모드는 LoRA fine-tuning과 DDIM inversion이 추가되어 더 오래 걸립니다.

## 7. OOM 대응

T4에서 메모리 부족이 나면 다음 순서로 낮춰보세요.

```text
1. lora_batch_size: 2 -> 1
2. drag_steps 감소
3. resolution 감소
4. 고해상도 테스트에서만 vae_tiling=True 사용
```

`cpu_offload=True`는 VRAM을 줄일 수 있지만 속도가 크게 느려질 수 있습니다. 또한 latent optimization 중 GPU/CPU 이동 문제가 생길 수 있으므로 기본값은 `False`로 유지합니다.

## 8. DDIM inversion 디버그

일반 Gradio 실행에서는 reconstruction 검증을 하지 않습니다. 필요할 때만 별도 명령으로 실행합니다.

```bash
python scripts/check_inversion.py --image path/to/image.png --prompt "a photo of a cat"
```

LoRA 없이 inversion만 빠르게 확인하려면:

```bash
python scripts/check_inversion.py --image path/to/image.png --prompt "a photo of a cat" --skip-lora
```

결과는 `outputs/inversion_debug/source.png`, `outputs/inversion_debug/reconstruction.png`로 저장됩니다.

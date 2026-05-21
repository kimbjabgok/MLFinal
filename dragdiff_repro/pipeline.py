from __future__ import annotations

import torch
from PIL import Image

from dragdiff_repro.config import DragConfig, EditRequest
from dragdiff_repro.methods.ddim_inversion import (
    ddim_invert,    #실제 이미지를 diffustion 중간 timestep latent 되돌림/
    image_to_latent,    #이미지를 VAE encoder로 latent space로 변환
    latent_to_image,    #latent space를 VAE decoder로 이미지 tensor로 변환
    predict_noise,    #UNet으로 현재 timestep의 noise를 예측
)


from dragdiff_repro.methods.lora_finetune import finetune_lora  #lora 미세조정

#latent space에서 최적화하는 함수, handle points와 target points를 이용하여 latent space에서 최적화 진행
from dragdiff_repro.methods.latent_optimization import optimize_latent

from dragdiff_repro.models.attention_control import reference_latent_control #attention 정보 활용
from dragdiff_repro.models.loader import ModelBundle #Stable Diffusion 관련 모델 묶음 타입
from dragdiff_repro.utils.image import tensor_to_pil  # torch tensor 이미지를 PIL 이미지로 바꾸는 유틸 함수


 #이 함수 안에서는 gradient 계산을 하지 않음. 학습이 아니라 추론이기 때문에 가중치 바꿀 필요 없음
 #++메모리 절약과 속도 향상
@torch.no_grad() 
#프롬프트로 이미지 생성하고 생성 동충 특정 timestep의 latent를 저장 후 반환
def generate_image_and_latent(
    bundle: ModelBundle,
    prompt: str,    #텍스트 프롬프트
    config: DragConfig,
) -> tuple[Image.Image, torch.Tensor]:
    generator = torch.Generator(device=bundle.device).manual_seed(config.seed)  #config는 실행 설정임.
    
    #shape 값은 이후에 최적화 하면됨
    shape = (
        1,  #batch size
        bundle.unet.config.in_channels, #보통 4
        config.height // 8,   #latent height
        config.width // 8,  #latent width
    )


    #Stable Diffusion에서 이미지 생성 과정은 다음과 같음: 픽셀이 아닌 조작하는 압축된 이미지
    
    # 텍스트 prompt
    #     ↓
    # UNet이 latent 공간에서 noise 예측
    #     ↓
    # scheduler가 latent를 조금씩 denoise
    #     ↓
    # VAE decoder가 latent를 이미지로 변환



    #정규분포 랜덤 노이즈 latent 생성.
    #stable diffusion은 노이즈에서 시작해서 점점 선명한 이미지로 만들어 나가는 방식이기 때문에 초기 노이즈 latent가 필요함.
    latents = torch.randn(shape, generator=generator, device=bundle.device, dtype=bundle.dtype)
    latents = latents * bundle.scheduler.init_noise_sigma #scheduler가 요구하는 초기 노이즈 크기에 맞게 latent 스캐일링.
    cached = None

    #scheduler의 timesteps를 순회하면서 매 timestep마다 UNet으로 노이즈 예측하고 scheduler로 latent 업데이트.
    #즉, 노이즈 이미지를 점점 denoise하면서 실제 이미지에 가깝게 만드는 과정.
    for index, timestep in enumerate(bundle.scheduler.timesteps):
        #Unet에게 현재 latent에서 제거해야 할 noise를 예측하게함.
        noise_pred = predict_noise(
            bundle,
            latents,
            timestep,
            prompt,
            config.guidance_scale_generated,
        )

        #스케줄러가 예측된 noise를 사용해서 latent를 한 단계 denoise.
        #prev_sample은 다음 반복에서 사용할 더 깨끗해진 latent
        latents = bundle.scheduler.step(noise_pred, timestep, latents).prev_sample
        
        #현재 denoising 단계가 사용자가 설정한 target timestep인지 확인.
        if index == config.target_timestep_index:
            cached = latents.detach().clone() #detach는 gradient 그래프에서 분리.이 latent가 나중에 DragDiffusion 최적화의 시작점

    
    image_tensor = latent_to_image(bundle, latents) #최종 latent를 VAE decoder로 이미지 tensor로 변환
    image = tensor_to_pil(image_tensor) #이미지 tensor를 PIL 이미지로 변환해서 반환. PIL 이미지는 일반적으로 이미지 저장이나 표시할 때 사용.
    return image, cached if cached is not None else latents.detach().clone() #생성된 이미지와 저장된 중간 latent를 반환.


@torch.no_grad()

#timestep의 latent에서 시작해서 남은 diffustion step을 진행하여 최종 이미지를 만드는 함수
def denoise_from_timestep(
    bundle: ModelBundle,
    latents: torch.Tensor,
    reference_latents: torch.Tensor,
    prompt: str,
    start_index: int,
    guidance_scale: float,
) -> Image.Image:
    current = torch.nan_to_num(latents.detach(), nan=0.0, posinf=1.0, neginf=-1.0)  #편집된 latent를 current로 복사.
    reference_current = torch.nan_to_num(reference_latents.detach(), nan=0.0, posinf=1.0, neginf=-1.0)
    timesteps = bundle.scheduler.timesteps[start_index:] # 편집된 중간 latent에서 최종 이미지까지 denoise할 남은 단계들.

    #Unet에 attention control 적용. 원본 latent의 attention 정보 저장 후 편집 latent denoise할 때 활용.
    with reference_latent_control(bundle.unet) as controller:
        for timestep in timesteps:
            controller.mode = "write" #reference latent를 통과시키면서 key/value 정보 저장
            controller.kv_bank.clear()    #이전 단계의 attention 정보 지우기

            #노이즈 예측. 이 과정에서 컨트롤러가 attention 정보를 기록
            reference_noise = predict_noise(bundle, reference_current, timestep, prompt, guidance_scale)

            reference_current = bundle.scheduler.step(reference_noise, timestep, reference_current).prev_sample  #reference latent도 한 단계 denoise 
            reference_current = torch.nan_to_num(reference_current, nan=0.0, posinf=1.0, neginf=-1.0)  #이상값 발생 시 다시 정리

            controller.mode = "read"

            #앞에 저장한 reference attestion 정보를 편집 latent denoise에 사용.
            noise_pred = predict_noise(bundle, current, timestep, prompt, guidance_scale)   #편집 latent에서 노이즈 예측.
            current = bundle.scheduler.step(noise_pred, timestep, current).prev_sample    #편집 latent도 한 단계 denoise
            current = torch.nan_to_num(current, nan=0.0, posinf=1.0, neginf=-1.0)   #이상값 발생 시 다시 정리

    image_tensor = latent_to_image(bundle, current)
    return tensor_to_pil(image_tensor)



#전체 DragDiffusion 실행 함수.
def run_dragdiffusion(bundle: ModelBundle, request: EditRequest) -> dict:
    config = request.config #요청 안에 들어있는 설정.

    if request.mode == "generated":   #generated 모드. 나중에 처리.
        
        #generated_image는 생성된 원본 이미지, latent_Zt는 target timestep에서의 latent.
        generated_image, latent_zt = generate_image_and_latent(bundle, request.prompt, config)
        original_latent_zt = latent_zt.detach().clone() #편집 전 latent를 원본 refernece로 저장함.
        source_image = generated_image  #사용자에게 보여줄 원본 이미지를 생성 이미지로 설정.
        guidance_scale = config.guidance_scale_generated #생성 이미지 모드에서 사용할 guidance scale을 설정.
        denoise_start_index = config.target_timestep_index #생성 이미지 모드에서 사용할 guidance scale을 설정.
    
    else: #real image 모드. 여기부터 처리.
        if request.image is None:
            raise ValueError("Real image mode requires an image tensor.")
        
        #z0은 깨끗한 원본 이미지 latent이고, 입력 이미지를 VAE encoder로 latent로 변환함.
        latent_z0 = image_to_latent(bundle, request.image)

        finetune_lora(bundle, latent_z0, request.prompt, config) #입력 이미지에 맞게 LoRA를 미세조정
        
        #DDIM inversion 수행.(실제 이미지를 Stable Diffusion의 중간 생성 상태로 되돌림.)
        #입력 이미지에서 target timestep까지의 latent 역전파.
        #latent_zt는 target timestep에서의 latent, denoise_start_index는 나중에 편집된 latent를 denoise할 때 사용할 시작 timestep index.
        latent_zt, _, denoise_start_index = ddim_invert(
            bundle,
            latent_z0,
            request.prompt,
            config.target_timestep_index,
            guidance_scale=config.guidance_scale_real, #실제 이미지 모드에서 사용할 guidance scale 설정.
        )

        original_latent_zt = latent_zt.detach().clone()
        source_image = tensor_to_pil(request.image)
        guidance_scale = config.guidance_scale_real

    #DragDiffusion의 핵심 최적화를 실행. 
    #handle points와 target points를 이용하여 latent space에서 편집된 latent를 최적화.
    optimized_latent, log = optimize_latent(
        bundle=bundle,
        latent_zt=latent_zt,
        original_latent_zt=original_latent_zt,
        mask=request.mask,
        prompt=request.prompt,
        handle_points=request.handle_points,
        target_points=request.target_points,
        config=config,
    )

    #최적화된 latent를 다시 이미지로 복원.
    edited_image = denoise_from_timestep(
        bundle=bundle,
        latents=optimized_latent,
        reference_latents=original_latent_zt,
        prompt=request.prompt,
        start_index=denoise_start_index,
        guidance_scale=guidance_scale,
    )

    return {
        "source_image": source_image,
        "edited_image": edited_image,
        "tracked_points": log.point_history[-1] if log.point_history else request.handle_points,
        "logs": log.to_dict(),
    }

import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "/home/hice1/xkong43/scratch/outputs/full_ft_sd15",
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

pipe.requires_safety_checker = False

prompts = [
    "eldenring gameplay style, third person view, player character fighting a massive dragon boss, dynamic motion, sparks, fire, cinematic camera angle",
    "eldenring gameplay style, boss arena, grotesque humanoid monster, detailed armor, high contrast lighting, epic composition",
    "eldenring gameplay style, open world exploration, distant castle, broken bridges, dead trees, misty environment, wide shot",
]

for i, p in enumerate(prompts):
    image = pipe(p, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save(f"/home/hice1/xkong43/scratch/outputs/full_ft_sd15/sample_{i}.png")

print("Done")
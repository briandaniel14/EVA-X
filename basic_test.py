from pathlib import Path

from eva_x import eva_x_tiny_patch16, eva_x_small_patch16, eva_x_base_patch16

print('bomboclaat')
tiny_path = Path('checkpoints/eva_x_tiny_patch16_merged520k_mim.pt')
print('bomboclaat')

model = eva_x_tiny_patch16(pretrained=tiny_path)

print(model)
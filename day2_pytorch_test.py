import torch
import numpy as np

# 1. Numpy 배열 생성
np_array = np.array([1, 2, 3])
print(f"Numpy 배열: {np_array}")

# 2. Numpy 배열을 PyTorch 텐서로 변환
torch_tensor = torch.from_numpy(np_array)
print(f"PyTorch 텐서: {torch_tensor}")

# 3. GPU 사용 가능한지 확인 (가장 중요!)
# (Nvidia GPU가 없거나 설치가 잘못되면 False가 나옵니다)
is_cuda_available = torch.cuda.is_available()
print(f"\nGPU 사용 가능 여부: {is_cuda_available}")

if is_cuda_available:
    # 4. 텐서를 CPU에서 GPU로 '보내기'
    tensor_on_gpu = torch_tensor.to('cuda')
    print(f"GPU로 보낸 텐서: {tensor_on_gpu}")

    # 5. GPU에서 계산하기 (Numpy는 못하는 일)
    result_gpu = tensor_on_gpu + 10
    print(f"GPU 계산 결과: {result_gpu}")
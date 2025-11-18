#사진 2장일 경우

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

try:
    img1 = Image.open("images/dog1.jpeg")
    img2 = Image.open("images/dog2.png")
    tensor1 = preprocess(img1)
    tensor2 = preprocess(img2)

    input_batch = torch.stack([tensor1, tensor2])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    input_batch = input_batch.to(device)

    with torch.no_grad():
        output = model(input_batch)

    predicted_class_indices = output.argmax(dim = 1)
    print(f"1번째 사진 예측된 클래스 번호(인덱스): {predicted_class_indices[0].item()}")
    print(f"2번째 사진 예측된 클래스 번호(인덱스): {predicted_class_indices[1].item()}")

except FileNotFoundError:
    print("오류: 사진 파일들을 images 폴더에 넣어 주세요!")
except Exception as e:
    print(f"오류 발생: {e}")
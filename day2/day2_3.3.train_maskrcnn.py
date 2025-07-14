# 필요한 PyTorch 및 관련 라이브러리들을 임포트합니다.
import torch
import torchvision
# torchvision 모델 라이브러리에서 Faster R-CNN과 Mask R-CNN의 최종 예측 레이어를 가져옵니다.
# 이들을 사용해 사전 학습된 모델의 헤드(head)를 우리 데이터셋에 맞게 교체할 수 있습니다.
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
# COCO 형식의 데이터셋을 로드하기 위한 torchvision 기본 클래스
from torchvision.datasets import CocoDetection
# 데이터를 미니배치(mini-batch) 단위로 효율적으로 로드하기 위한 클래스
from torch.utils.data import DataLoader
# 이미지 전처리 및 변환(augmentation)을 위한 모듈
import torchvision.transforms as T
from PIL import Image  # 이미지 로딩 및 기본 처리를 위한 Pillow 라이브러리
import os  # 파일 및 디렉토리 경로 관련 작업을 위한 라이브러리
from tqdm import tqdm  # 학습 진행 상황을 시각적인 프로그레스 바로 보여주기 위한 라이브러리
import numpy as np  # 수치 연산, 특히 배열 처리를 위해 사용 (UserWarning 해결에 필요)

# --- 설정 (Configurations) ---
# 데이터셋의 루트 디렉토리. 이미지 폴더들이 이 안에 있어야 합니다.
DATA_ROOT = os.path.join(os.getcwd(), "data/output_data")
# COCO 형식의 주석(annotation) JSON 파일 경로
ANNOTATIONS_FILE = os.path.join(DATA_ROOT, 'coco_annotations.json')
# 총 클래스 수 = 실제 객체 클래스 수 + 1 (배경 클래스)
NUM_CLASSES = 79  # 78 (ycb_object) + 1 (background)
# 전체 데이터셋을 몇 번 반복하여 학습할지 결정하는 에포크(epoch) 수
NUM_EPOCHS = 20
# 한 번의 반복(iteration)에서 모델이 처리할 이미지의 수
BATCH_SIZE = 2
# 학습률(learning rate). 경사 하강법에서 파라미터를 업데이트하는 보폭(step size)
LEARNING_RATE = 0.005

class CocoLikeDataset(CocoDetection):
    """
    torchvision의 CocoDetection 클래스를 상속받아 우리 데이터셋에 맞게 커스터마이징하는 클래스입니다.
    __getitem__ 메서드를 오버라이드하여 모델이 요구하는 형식으로 이미지와 타겟(정답) 데이터를 반환합니다.
    """
    def __init__(self, root, annFile, transforms=None):
        # 부모 클래스(CocoDetection)의 초기화 메서드를 호출하여 기본적인 COCO 데이터 파싱을 수행합니다.
        super().__init__(root, annFile)
        # 이미지 변환(augmentation 등)을 위한 transform 함수를 저장합니다.
        self.transforms = transforms

    def __getitem__(self, idx):
        """
        데이터셋에서 특정 인덱스(idx)의 샘플(이미지와 타겟)을 가져오는 메서드입니다.
        DataLoader가 이 메서드를 호출하여 미니배치를 구성합니다.
        """
        # 부모 클래스의 __getitem__을 사용하여 원본 PIL 이미지와 어노테이션 리스트를 가져옵니다.
        img, target = super().__getitem__(idx)
        # 현재 이미지의 고유 ID를 가져옵니다.
        image_id = self.ids[idx]

        # 이미지에 해당하는 어노테이션(객체)이 존재하는 경우
        if target:
            # 바운딩 박스 정보 추출 및 변환
            # COCO의 bbox 형식은 [x, y, width, height] 입니다.
            boxes = [t['bbox'] for t in target]
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            # PyTorch 모델이 요구하는 [x1, y1, x2, y2] 형식으로 변환합니다.
            boxes[:, 2:] += boxes[:, :2]
            
            # 레이블(카테고리 ID) 정보 추출
            labels = torch.as_tensor([t['category_id'] for t in target], dtype=torch.int64)

            # 세그멘테이션 마스크 정보 추출 및 변환
            # self.coco.annToMask(t)는 RLE 형식의 세그멘테이션을 NumPy 배열 마스크로 변환합니다.
            # 이 배열들을 리스트에 담은 후, np.array()로 감싸서 하나의 다차원 NumPy 배열로 만듭니다.
            masks_list = [self.coco.annToMask(t) for t in target]
            masks = torch.as_tensor(np.array(masks_list), dtype=torch.uint8)
            
            # 모델이 요구하는 최종 타겟 딕셔너리를 구성합니다.
            processed_target = {
                "boxes": boxes,
                "labels": labels,
                "masks": masks,
                "image_id": torch.tensor([image_id]),
                "area": torch.as_tensor([t['area'] for t in target], dtype=torch.float32),
                "iscrowd": torch.as_tensor([t['iscrowd'] for t in target], dtype=torch.int64)
            }
        else: # 이미지에 해당하는 어노테이션(객체)이 없는 경우 (배경만 있는 이미지)
            w, h = img.size
            # 빈 텐서들로 타겟 딕셔너리를 구성합니다. 각 텐서의 shape에 주의해야 합니다.
            processed_target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "masks": torch.zeros((0, h, w), dtype=torch.uint8),
                "image_id": torch.tensor([image_id])
            }

        # PIL 이미지를 PyTorch 텐서로 변환하는 작업은 어노테이션 유무와 관계없이 항상 수행되어야 합니다.
        # 따라서 if-else 블록 밖으로 이동하여 `img_tensor`가 항상 정의되도록 합니다.
        img_tensor = T.ToTensor()(img)

        # 만약 추가적인 변환(augmentation)이 정의되어 있다면 여기서 적용합니다.
        if self.transforms is not None:
            # 예: img_tensor, processed_target = self.transforms(img_tensor, processed_target)
            pass

        return img_tensor, processed_target

def get_model_instance_segmentation(num_classes):
    """
    사전 학습된 Mask R-CNN 모델을 로드하고, 헤드 부분을 우리 데이터셋의 클래스 수에 맞게 교체합니다.
    """
    # ImageNet으로 사전 학습된 Mask R-CNN 모델을 로드합니다. (Transfer Learning)
    # 최신 PyTorch는 'pretrained=True' 대신 'weights' 파라미터를 사용합니다.
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights="MaskRCNN_ResNet50_FPN_Weights.DEFAULT")
    
    # 박스 예측기의 입력 피처 수를 가져옵니다.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 박스 예측기를 우리의 클래스 수에 맞는 새로운 예측기로 교체합니다.
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # 마스크 예측기의 입력 채널 수를 가져옵니다.
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256 # 마스크 예측을 위한 중간 레이어 크기
    # 마스크 예측기를 우리의 클래스 수에 맞는 새로운 예측기로 교체합니다.
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model

def collate_fn(batch):
    """
    DataLoader가 데이터셋에서 가져온 샘플 리스트를 하나의 배치로 묶어주는 함수입니다.
    객체 탐지 모델의 입력은 이미지와 타겟의 크기가 각기 다를 수 있어 기본 collate 함수를 사용할 수 없습니다.
    단순히 이미지 리스트와 타겟 리스트를 튜플로 묶어 반환합니다.
    """
    return tuple(zip(*batch))

def main():
    # 학습에 사용할 디바이스를 설정합니다. (CUDA GPU 우선)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # 커스텀 데이터셋 객체를 생성합니다.
    dataset = CocoLikeDataset(root=DATA_ROOT, annFile=ANNOTATIONS_FILE)
    
    # 데이터 로더를 생성합니다. 데이터를 배치 단위로 묶고, 매 에포크마다 섞어주며(shuffle=True),
    # 커스텀 collate_fn을 사용하여 배치를 구성합니다.
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # 모델을 생성하고 지정된 디바이스로 이동시킵니다.
    model = get_model_instance_segmentation(NUM_CLASSES)
    model.to(device)

    # 옵티마이저(optimizer)를 설정합니다.
    # `requires_grad=True`인, 즉 학습 가능한 파라미터들만 옵티마이저에 전달합니다.
    params = [p for p in model.parameters() if p.requires_grad]
    # 여기서는 SGD(Stochastic Gradient Descent) 옵티마이저를 사용합니다.
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

    print("\n--- Starting Training ---")
    # 설정된 에포크 수만큼 학습을 반복합니다.
    for epoch in range(NUM_EPOCHS):
        # 모델을 학습 모드(train mode)로 설정합니다.
        model.train()
        epoch_loss = 0
        # tqdm을 사용하여 데이터 로더를 순회하며 진행 상황을 표시합니다.
        for i, (images, targets) in enumerate(tqdm(data_loader, 
                                                   desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
            # 이미지와 타겟 데이터를 모두 지정된 디바이스로 이동시킵니다.
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 모델의 순전파(forward pass)를 수행합니다.
            # 학습 모드에서는 이미지와 타겟을 모두 입력으로 받아 손실(loss) 딕셔너리를 반환합니다.
            loss_dict = model(images, targets)
            # 반환된 모든 손실(분류 손실, 박스 회귀 손실, 마스크 손실 등)을 합산합니다.
            losses = sum(loss for loss in loss_dict.values())
            
            # 역전파(backpropagation)를 위한 준비
            optimizer.zero_grad() # 이전 반복에서 계산된 그래디언트를 초기화합니다.
            losses.backward()     # 현재 손실에 대한 그래디언트를 계산합니다.
            optimizer.step()      # 계산된 그래디언트를 사용하여 모델의 파라미터를 업데이트합니다.
            
            # 현재 배치의 손실을 에포크 전체 손실에 누적합니다.
            epoch_loss += losses.item()

        # 한 에포크의 학습이 끝나면 평균 손실을 출력합니다.
        print(f"--- Epoch {epoch+1} Summary ---")
        print(f"Average Epoch Loss: {epoch_loss / len(data_loader):.4f}\n")
        
        # 10 에포크마다 모델 가중치를 중간 저장합니다.
        if (epoch + 1) % 10 == 0:
            model_save_path = os.path.join(os.getcwd(), "data/output_data", f"maskrcnn_trained_model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Checkpoint saved to {model_save_path}")

    # 모든 학습이 완료된 후 최종 모델을 저장합니다.
    model_save_path = os.path.join(os.getcwd(), "data/output_data", "maskrcnn_trained_model_refined.pth")

    torch.save(model.state_dict(), model_save_path)
    print(f"Training finished. Model saved to {model_save_path}")

# 이 스크립트가 직접 실행될 때 main() 함수를 호출합니다.
if __name__ == '__main__':
    main()
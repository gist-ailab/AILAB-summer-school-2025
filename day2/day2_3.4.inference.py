import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import random
import os

# --- 설정 ---
MODEL_PATH = os.path.join(os.getcwd(), "data/output_data", "maskrcnn_trained_model_refined.pth")
# MODEL_PATH = os.path.join(os.getcwd(), "data/checkpoint/maskrcnn_ckpt", "maskrcnn_trained_model_refined.pth") # <-- 사전 학습된 Weight
IMAGE_PATH = os.path.join(os.getcwd(), "data/output_data", "episode_000/rgb_1_0.png") # <-- 테스트하고 싶은 이미지로 변경하세요.
OUTPUT_PATH = os.path.join(os.getcwd(), "data", "inference_result_final.png")
NUM_CLASSES = 79  # 모델 구조는 학습 때와 동일해야 함
CONFIDENCE_THRESHOLD = 0.5

YCB_OBJECT_CLASSES = sorted([
        '002_master_chef_can', '008_pudding_box', '014_lemon', '021_bleach_cleanser', '029_plate', '036_wood_block', '044_flat_screwdriver', '054_softball', '061_foam_brick', '065_c_cups', '065_i_cups', '072_b_toy_airplane', '073_c_lego_duplo',
        '003_cracker_box', '009_gelatin_box', '015_peach', '022_windex_bottle', '030_fork', '037_scissors', '048_hammer', '055_baseball', '062_dice', '065_d_cups', '065_j_cups', '072_c_toy_airplane', '073_d_lego_duplo',
        '004_sugar_box', '010_potted_meat_can', '016_pear', '024_bowl', '031_spoon', '038_padlock', '050_medium_clamp', '056_tennis_ball', '063_a_marbles', '065_e_cups', '070_a_colored_wood_blocks', '072_d_toy_airplane', '073_e_lego_duplo',
        '005_tomato_soup_can', '011_banana', '017_orange', '025_mug', '032_knife', '040_large_marker', '051_large_clamp', '057_racquetball', '063_b_marbles', '065_f_cups', '070_b_colored_wood_blocks', '072_e_toy_airplane', '073_f_lego_duplo',
        '006_mustard_bottle', '012_strawberry', '018_plum', '026_sponge', '033_spatula', '042_adjustable_wrench', '052_extra_large_clamp', '058_golf_ball', '065_a_cups', '065_g_cups', '071_nine_hole_peg_test', '073_a_lego_duplo', '073_g_lego_duplo',
        '007_tuna_fish_can', '013_apple', '019_pitcher_base', '028_skillet_lid', '035_power_drill', '043_phillips_screwdriver', '053_mini_soccer_ball', '059_chain', '065_b_cups', '065_h_cups', '072_a_toy_airplane', '073_b_lego_duplo', '077_rubiks_cube'
    ])
CLASS_NAME = ['BACKGROUND'] + YCB_OBJECT_CLASSES

def get_model_instance_segmentation(num_classes):
    """ 학습 때와 동일한 구조로 Mask R-CNN 모델을 생성합니다. """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def get_random_color():
    """ 시각화를 위해 랜덤 RGB 색상을 생성합니다. """
    return [random.randint(50, 255) for _ in range(3)] # 너무 어둡지 않은 색상

def run_inference():
    """ 개선된 시각화 로직으로 추론을 수행합니다. """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # 1. 모델 로드
    print("Loading model...")
    model = get_model_instance_segmentation(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model.to(device)

    # 2. 이미지 로드 및 전처리
    print(f"Loading image: {IMAGE_PATH}")
    img_pil = Image.open(IMAGE_PATH).convert("RGB")
    transform = T.ToTensor()
    img_tensor = transform(img_pil)
    
    # 3. 추론 실행
    print("Running inference...")
    with torch.no_grad():
        prediction = model([img_tensor.to(device)])
        
    # 4. 결과 후처리 및 시각화 (개선된 방식)
    img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    # Bbox와 텍스트를 그릴 이미지 레이어
    img_with_boxes = img_np.copy()
    # 마스크를 그릴 투명한 이미지 레이어
    mask_overlay = img_np.copy()
    
    pred_scores = prediction[0]['scores'].cpu().numpy()
    pred_boxes = prediction[0]['boxes'].cpu().numpy()
    pred_masks = prediction[0]['masks'].cpu().numpy()
    pred_labels = prediction[0]['labels'].cpu().numpy()
    
    print(f"Found {len(pred_scores)} objects. Visualizing valid results...")

    for i in range(len(pred_scores)):
        score = pred_scores[i]
        label_id = pred_labels[i]

        # 신뢰도와 레이블 ID를 함께 확인하여 Background 제외
        if score > CONFIDENCE_THRESHOLD and label_id != 0:
            color = get_random_color()
            
            # --- Bbox와 텍스트 그리기 ---
            box = pred_boxes[i]
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
            
            class_names = CLASS_NAME
            label_text = f"{class_names[label_id]}: {score:.2f}"
            cv2.putText(img_with_boxes, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # --- 마스크 그리기 ---
            mask = pred_masks[i, 0]
            binary_mask = (mask > 0.5) # Boolean mask
            # 마스크 영역에만 색상 적용
            mask_overlay[binary_mask] = color

    # Bbox와 Mask를 분리해서 그린 후 마지막에 한 번만 합성
    alpha = 0.5 # 마스크 투명도
    final_result = cv2.addWeighted(mask_overlay, alpha, img_with_boxes, 1 - alpha, 0)

    # 5. 결과 저장
    cv2.imwrite(OUTPUT_PATH, final_result)
    print(f"Inference result saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    run_inference()
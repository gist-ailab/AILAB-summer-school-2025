import argparse
import os
import glob
import random
import torch
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import clip
from torchvision import transforms as T
from torchvision.transforms import functional as F
import gymnasium as gym
from collections.abc import Sequence
import open3d as o3d
import matplotlib.pyplot as plt

# Isaac Lab 관련 라이브러리 임포트
from isaaclab.app import AppLauncher

# Argparse로 CLI 인자 파싱 및 Omniverse 앱 실행
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

# AILAB-summer-school-2025/cgnet 폴더에 접근하기 위한 시스템 파일 경로 추가
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Detection 모델 라이브러리 임포트
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision

# Contact-GraspNet 모델 라이브러리 임포트
from cgnet.utils.config import cfg_from_yaml_file
from cgnet.tools import builder
from cgnet.inference_cgnet import inference_cgnet

# 카메라 렌더링 옵션 --enable_cameras flag 를 대신하기 위함
import carb
carb_settings_iface = carb.settings.get_settings()
carb_settings_iface.set_bool("/isaaclab/cameras_enabled", True)

# 커스텀 환경 시뮬레이션 환경 config 파일 임포트
from task.lift.custom_pickplace_env_cfg_3_3 import YCBPickPlaceEnvCfg

# gymnasium 라이브러리를 활용한 시뮬레이션 환경 선언
from task.lift.config.ik_abs_env_cfg_3_3 import FrankaYCBPickPlaceEnvCfg
gym.register(
    id="Isaac-Lift-Cube-Franka-Custom-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": FrankaYCBPickPlaceEnvCfg,
    },
    disable_env_checker=True,
)


# Detection 모델 설정
DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(DIR_PATH, 'data/checkpoint/maskrcnn_ckpt/maskrcnn_trained_model_refined.pth') # <-- 사전 학습된 Weight
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

def depth2pc(depth, K, rgb=None):
    """ 뎁스 이미지를 포인트 클라우드로 변환하는 함수 """
    
    mask = np.where(depth > 0)
    x, y = mask[1], mask[0]
    
    normalized_x = (x.astype(np.float32)-K[0,2])
    normalized_y = (y.astype(np.float32)-K[1,2])
    
    world_x = normalized_x * depth[y, x] / K[0,0]
    world_y = normalized_y * depth[y, x] / K[1,1]
    world_z = depth[y, x]
    
    if rgb is not None:
        rgb = rgb[y, x]
    
    pc = np.vstack([world_x, world_y, world_z]).T
    return (pc, rgb)

def get_world_bbox(depth, K, bb): 
    """ Bounding Box의 좌표를 포인트 클라우드 기준 좌표로 변환하는 함수 """

    image_width = depth.shape[1]
    image_height = depth.shape[0]

    x_min, x_max = bb[0], bb[2]
    y_min, y_max = bb[1], bb[3]
    
    if y_min < 0:
        y_min = 0
    if y_max >= image_height:
        y_max = image_height-1
    if x_min < 0:
        x_min = 0
    if x_max >=image_width:
        x_max = image_width-1

    z_0, z_1 = depth[int(y_min), int(x_min)], depth[int(y_max), int(x_max)]
    
    def to_world(x, y, z):
        """ 뎁스 포인트를 3D 포인트로 변환하는 함수 """
        world_x = (x - K[0, 2]) * z / K[0, 0]
        world_y = (y - K[1, 2]) * z / K[1, 1]
        return world_x, world_y, z
    
    x_min_w, y_min_w, z_min_w = to_world(x_min, y_min, z_0)
    x_max_w, y_max_w, z_max_w = to_world(x_max, y_max, z_1)
    
    return x_min_w, y_min_w, x_max_w, y_max_w

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

def CLIP_transform(img_tensor, output_size, fill=0, padding_mode='constant'):
    """ 이미지의 비율을 유지하면서 리사이즈하고, 목표 크기에 맞춰 패딩 """

    # 긴 축을 기준으로 리사이즈될 크기 계산
    _, h, w = img_tensor.shape
    if h > w:
        new_h = output_size
        new_w = int(w * (new_h / h))
    else: # w >= h
        new_w = output_size
        new_h = int(h * (new_w / w))

    # 비율을 유지하며 리사이즈
    img_tensor = F.resize(img_tensor, [new_h, new_w])

    # 목표 크기에 맞게 패딩을 추가 (left, top, right, bottom)
    pad_left = (output_size - new_w) // 2
    pad_top = (output_size - new_h) // 2
    pad_right = output_size - new_w - pad_left
    pad_bottom = output_size - new_h - pad_top
    padding = (pad_left, pad_top, pad_right, pad_bottom)

    # 패딩을 적용하여 이미지를 반환
    padded_img = F.pad(img_tensor, padding, fill, padding_mode)

    # mean/std를 사용해 정규화
    normalized_img = F.normalize(padded_img.to(torch.float32), 
                                 mean=[0.48145466, 0.4578275, 0.40821073], 
                                 std=[0.26862954, 0.26130258, 0.27577711])
    
    return F.pad(img_tensor, padding, fill, padding_mode)


class GripperState:
    """ 로봇 제어를 위한 그리퍼 state 정의 """
    OPEN = 1.0
    CLOSE = -1.0

class PickAndPlaceSmState:
    """ 로봇 제어를 위한  상황 state 정의 """
    REST = 0
    PREDICT = 1
    READY = 2
    PREGRASP = 3
    GRASP = 4
    CLOSE = 5
    LIFT = 6
    MOVE_TO_BIN = 7
    LOWER = 8
    RELEASE = 9
    BACK = 10
    BACK_TO_READY = 11

class PickAndPlaceSmWaitTime:
    """ 각 pick-and-place 상황 state 별 대기 시간(초) 정의 """
    REST = 3.0
    PREDICT = 0.0
    READY = 0.5
    PREGRASP = 1.0
    GRASP = 0.5
    CLOSE = 1.0
    LIFT = 0.5
    MOVE_TO_BIN = 0.5
    LOWER = 0.5
    RELEASE = 0.5
    BACK = 0.5
    BACK_TO_READY = 0.5


class PickAndPlaceSm:
    """
    로봇이 물체를 집어 옮기는(Pick-and-Place) 작업을 상태머신(State Machine)으로 구현.
    각 단계별로 End-Effector 위치와 그리퍼 상태를 지정해줌.

    0. REST: 로봇이 초기자세 상태에 있습니다.
    1. PREDICT: 파지 예측을 수행합니다.
    2. READY: 로봇이 초기자세 상태에 위치하고, 그리퍼를 CLOSE 상태로 둡니다.
    3. PREGRASP: 타겟 물체 앞쪽의 pre-grasp 자세로 이동합니다.
    4. GRASP: 엔드이펙터를 타겟 물체에 grasp 자세로 접근합니다.
    5. CLOSE: 그리퍼를 닫아 물체를 집습니다.
    6. LIFT: 물체를 들어올립니다.
    7. MOVE_TO_BIN: 물체를 목표 xy 위치(바구니)로 이동시키고, 높이도 특정 높이까지 유지합니다.
    8. LOWER: 물체를 낮은 z 위치까지 내립니다.
    9. RELEASE: 그리퍼를 열어 물체를 놓습니다.
    10. BACK: 엔드이펙터를 바구니 위의 특정 높이로 다시 이동시킵니다.
    11. BACK_TO_READY: 엔드이펙터를 원래 초기 위치로 이동시킵니다.
    """
    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.01):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        # state machine 파라미터 값(1)
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold

        # state machine 파라미터 값(2)
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # 목표 로봇 끝단(end-effector) 자세 및 그리퍼 상태
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs, 1), 0.0, device=self.device)

        # 물체 이미지를 취득하기 위한 준비 자세
        self.ready_pose = torch.tensor([[ 3.0280e-01, -5.6916e-02,  6.2400e-01, -1.4891e-10,  1.0000e+00, 8.4725e-11, -8.7813e-10]], device=self.device)  # (x, y, z, qw, qx, qy, qz)
        self.ready_pose = self.ready_pose.repeat(num_envs, 1)

        # 물체를 상자에 두기 위해 상자 위에 위치하는 자세
        self.bin_pose = torch.tensor([[ 0.2, 0.6, 0.55, 0, 1, 0, 0]], device=self.device)   # (x, y, z, qw, qx, qy, qz)
        self.bin_pose = self.bin_pose.repeat(num_envs, 1)

        # 물체를 안정적으로 상자에 두기 위한 낮은 자세
        self.bin_lower_pose = torch.tensor([[ 0.2, 0.6, 0.35, 0, 1, 0, 0]], device=self.device)   # (x, y, z, qw, qx, qy, qz)
        self.bin_lower_pose = self.bin_lower_pose.repeat(num_envs, 1)

        # Contact-GraspNet 추론 값을 담기위한 변수 선언
        self.grasp_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.pregrasp_pose = torch.zeros((self.num_envs, 7), device=self.device)

        # Gripper가 원하는 위치에 도달하지 못하는 경우, statemachine이 멈추는 것을 방지하기 위한 변수 선언
        self.stack_ee_pose = []

    # env idx 를 통한 reset 상태 실행
    def reset_idx(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = PickAndPlaceSmState.REST
        self.sm_wait_time[env_ids] = 0.0

    ##################################### State Machine #####################################
    # 로봇의 end-effector 및 그리퍼의 목표 상태 계산
    def compute(self, ee_pose: torch.Tensor, grasp_pose: torch.Tensor, pregrasp_pose: torch.Tensor, robot_data):
        ee_pos = ee_pose[:, :3]
        ee_pos[:, 2] -= 0.5

        # 각 environment에 반복적으로 적용
        for i in range(self.num_envs):
            state = self.sm_state[i]
            # 각 상태에 따른 로직 구현
            if state == PickAndPlaceSmState.REST:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                self.des_ee_pose[i] = self.ready_pose[i]
                self.des_gripper_state[i] = GripperState.OPEN

                # 특정 시간 동안 대기
                if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.REST:
                    # 다음 state 로 전환 및 state 시간 초기화
                    self.sm_state[i] = PickAndPlaceSmState.PREDICT
                    self.sm_wait_time[i] = 0.0

            elif state == PickAndPlaceSmState.PREDICT:
                # 다음 state 로 전환 및 state 시간 초기화
                
            elif state == PickAndPlaceSmState.READY:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                
                # 목표자세 도딜시 특정 시간 동안 대기

                        # 다음 state 로 전환 및 state 시간 초기화


            elif state == PickAndPlaceSmState.PREGRASP:
                # 목표 end-effector 자세 및 그리퍼 상태 정의

                # 현재 state에서의 end-effector position을 저장
                self.stack_ee_pose.append(ee_pos[i])

                # 목표자세 도딜시 특정 시간 동안 대기
                if
                        # 다음 state 로 전환 및 state 시간 초기화
                        
                # end-effector의 위치가 일정 step 이상 바뀌지 않을때, 다음 state 로 전환 및 state 시간 초기화
                else:
                    if len(self.stack_ee_pose) > 50:
                        if torch.linalg.norm(ee_pos[i] - self.stack_ee_pose[-30]) < self.position_threshold:
                            self.sm_state[i] = PickAndPlaceSmState.CLOSE
                            self.sm_wait_time[i] = 0.0
                            self.stack_ee_pose = []

            elif state == PickAndPlaceSmState.GRASP:
                # 목표 end-effector 자세 및 그리퍼 상태 정의

                # 현재 state에서의 end-effector position을 저장
                self.stack_ee_pose.append(ee_pos[i])

                # 목표자세 도딜시 특정 시간 동안 대기
                
                        # 다음 state 로 전환 및 state 시간 초기화
                        
                # end-effector의 위치가 일정 step 이상 바뀌지 않을때, 다음 state 로 전환 및 state 시간 초기화
                else:
                    if len(self.stack_ee_pose) > 50:
                        if torch.linalg.norm(ee_pos[i] - self.stack_ee_pose[-30]) < self.position_threshold:
                            self.sm_state[i] = PickAndPlaceSmState.CLOSE
                            self.sm_wait_time[i] = 0.0
                            self.stack_ee_pose = []

            elif state == PickAndPlaceSmState.CLOSE:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                
                # 특정 시간 동안 대기
                
                    # 다음 state 로 전환 및 state 시간 초기화
                    

                    # 일정 높이로 들어 올릴 위치 설정
                    self.lift_pose = grasp_pose[i]
                    self.lift_pose[2] = self.lift_pose[2] + 0.4

            elif state == PickAndPlaceSmState.LIFT:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                
                # 목표자세 도딜시 특정 시간 동안 대기
                
                        # 다음 state 로 전환 및 state 시간 초기화
                        

            elif state == PickAndPlaceSmState.MOVE_TO_BIN:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                
                # 현재 state에서의 end-effector position을 저장
                self.stack_ee_pose.append(ee_pos[i])

                # 목표자세 도딜시 특정 시간 동안 대기
                
                        # 다음 state 로 전환 및 state 시간 초기화
                        
                # end-effector의 위치가 일정 step 이상 바뀌지 않을때, 다음 state 로 전환 및 state 시간 초기화
                else:
                    if len(self.stack_ee_pose) > 50:
                        if torch.linalg.norm(ee_pos[i] - self.stack_ee_pose[-30]) < self.position_threshold:
                            self.sm_state[i] = PickAndPlaceSmState.CLOSE
                            self.sm_wait_time[i] = 0.0
                            self.stack_ee_pose = []

            elif state == PickAndPlaceSmState.LOWER:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                
                # 목표자세 도딜시 특정 시간 동안 대기
                
                        # 다음 state 로 전환 및 state 시간 초기화
                        

            elif state == PickAndPlaceSmState.RELEASE:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                
                # 목표자세 도딜시 특정 시간 동안 대기
                
                        # 다음 state 로 전환 및 state 시간 초기화
                        

            elif state == PickAndPlaceSmState.BACK:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                
                # 목표자세 도딜시 특정 시간 동안 대기
                
                        # 다음 state 로 전환 및 state 시간 초기화
                        

            elif state == PickAndPlaceSmState.BACK_TO_READY:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                
                # 목표자세 도딜시 특정 시간 동안 대기
                
                        # 남은 물체를 잡기 위해, PREDICT state 로 전환 및 state 시간 초기화
                        
            # state machine 단위시간 경과
            self.sm_wait_time[i] += self.dt

            actions = torch.cat([self.des_ee_pose, self.des_gripper_state], dim=-1)

        return actions
    ###############################################################################################

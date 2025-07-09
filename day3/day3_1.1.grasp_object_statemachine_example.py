
import argparse
import os
import random
import torch
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from torchvision import transforms as T
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
# Contact-GraspNet 모델 라이브러리 임포트
from cgnet.utils.config import cfg_from_yaml_file
from cgnet.tools import builder
from cgnet.inference_cgnet import inference_cgnet


# 커스텀 환경 시뮬레이션 환경 config 파일 임포트
from task.lift.custom_lift_env_cfg_3_1 import LiftEnvCfg
from task.lift.config.ik_abs_env_cfg_3_1 import FrankaCubeLiftEnvCfg


""" 카메라 렌더링 옵션 --enable_cameras flag 를 대신하기 위함. """
import carb
#Q1. Isaac Lab 카메라 렌더링을 활성화하세요.
carb_settings_iface = 
carb_settings_iface.set_bool(


# 뎁스 이미지를 포인트 클라우드로 변환하는 함수
def depth2pc(depth, K, rgb=None):
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

# gymnasium 라이브러리를 활용한 시뮬레이션 환경 선언
#Q2. gymnasium에 커스텀 환경 id와 등록할 환경 config을 지정하세요.
gym.register(
    id="",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ,
    },
    disable_env_checker=True,
)

########## 실습(2)-state machine 정의 ##########
# 로봇 제어를 위한 그리퍼 state 정의
#10. statemachine 에서 사용할 그리퍼의 OPEN, CLOSE 상태를 정의하세요.
class GripperState:
    """States for the gripper."""
    

# 로봇 제어를 위한  상황 state 정의
#Q11. statemachine 에서 사용할 Pick-and-Place 상황의 상태를 정의하세요.
class PickAndPlaceSmState:
    """States for the object grasping state machine."""
    

# 각 pick-and-place 상황 state 별 대기 시간(초) 정의
class PickAndPlaceSmWaitTime:
    """Additional wait times (in s) for states for before switching."""
    REST = 1.5
    PREDICT = 0.01
    READY = 0.5
    PREGRASP = 1.0
    GRASP = 0.5
    CLOSE = 1.0
    LIFT = 0.5

# 실제 Pick-and-Place statemachine 정의
class PickAndPlaceSm:
    """
    로봇이 물체를 집어 옮기는 작업을 상태머신(State Machine)으로 구현.
    각 단계별로 End-Effector 위치와 그리퍼 상태를 지정해줌.

    0. REST: 로봇이 초기자세 상태에 있습니다.
    1. PREDICT: 파지 예측을 수행합니다.
    2. READY: 로봇이 초기자세 상태에 위치하고, 그리퍼를 CLOSE 상태로 둡니다.
    3. PREGRASP: 타겟 물체 앞쪽의 pre-grasp 자세로 이동합니다.
    4. GRASP: 엔드이펙터를 타겟 물체에 grasp 자세로 접근합니다.
    5. CLOSE: 그리퍼를 닫아 물체를 집습니다.
    6. LIFT: 물체를 들어올립니다.
    """
    # Statemachine 선언시 자동으로 실행되는 메소드(__init__)
    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.01):
        # state machine 파라미터 값(1)
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device

        # state machine 파라미터 값(2)
        #Q12. statemachine 의 단위시간(sm_dt), 상태(state) 변수를 선언하세요.
        self.sm_dt = 
        self.sm_state = 
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)
        # 현재위치와 목표위치의 위치 오차에 대한 position threshold
        self.position_threshold = position_threshold

        # 목표 로봇 끝단(end-effector) 자세 및 그리퍼 상태
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs, 1), 0.0, device=self.device)

        #Q13. 물체를 집기 위한 준비 자세를 정의하세요.(global 좌표계 기준)
        # 물체 이미지를 취득하기 위한 준비 자세
        self.ready_pose = torch.tensor([[]], device=self.device)  # (x, y, z, qw, qx, qy, qz)
        self.ready_pose = self.ready_pose.repeat(num_envs, 1)

        #Q14. Contact-GraspNet 추론 값을 담고, 로봇 입력에 적합한 사이즈로 grasp_pose 와 pregrasp_pose 변수를 선언하세요.
        # Contact-GraspNet 추론 값을 담기위한 변수 선언
        self.grasp_pose = torch.zeros(, device=self.device)
        self.pregrasp_pose = torch.zeros(, device=self.device)

    #Q15. statemachine 의 에러 처리를 위해 상태를 초기화하는 reset_idx 메소드를 정의하세요.
    # env idx 를 통한 reset 상태 실행
    def reset_idx(self, env_ids: Sequence[int] | None = None):
        """Reset the state machine for specified environment indices."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = PickAndPlaceSmState.REST
        self.sm_wait_time[env_ids] = 0.0

    # 로봇의 end-effector 및 그리퍼의 목표 상태 계산
    def compute(self, ee_pose: torch.Tensor, grasp_pose: torch.Tensor, pregrasp_pose: torch.Tensor, robot_data):
        """Compute the desired state of the robot's end-effector and the gripper."""
        # end-effector의 현재 글로벌 자세를 가져옵니다. 따라서, base z축 위치인 0.5 만큼 빼줍니다.
        ee_pos = ee_pose[:, :3]
        ee_pos[:, 2] -= 0.5

        for i in range(self.num_envs):
            state = self.sm_state[i]
            # 각 상태에 따른 로직 구현
            if state == PickAndPlaceSmState.REST:   # 0
                # 로봇을 초기 자세로 이동시키고, 그리퍼를 OPEN 상태로 둡니다.
                self.des_ee_pose[i] = self.ready_pose[i]
                self.des_gripper_state[i] = GripperState.OPEN
                # 특정 시간 동안 대기
                if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.REST:
                    # 다음 state 로 전환 및 state 시간 초기화
                    self.sm_state[i] = PickAndPlaceSmState.PREDICT
                    self.sm_wait_time[i] = 0.0
            elif state == PickAndPlaceSmState.PREDICT:   # 1
                # 특정 시간 동안 대기
                if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.PREDICT:
                    # 다음 state 로 전환 및 state 시간 초기화
                    self.sm_state[i] = PickAndPlaceSmState.READY
                    self.sm_wait_time[i] = 0.0
            elif state == PickAndPlaceSmState.READY:   # 2
                # 로봇을 초기 자세로 이동시키고, 그리퍼를 CLOSE 상태로 둡니다.
                self.des_ee_pose[i] = self.ready_pose[i]
                self.des_gripper_state[i] = GripperState.CLOSE
                # 목표 위치와 현재 위치의 거리를 계산하여, 목표 위치에 도달했는지 확인합니다.
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    # 특정 시간 동안 대기
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.READY:
                        # 다음 state 로 전환 및 state 시간 초기화
                        self.sm_state[i] = PickAndPlaceSmState.PREGRASP
                        self.sm_wait_time[i] = 0.0
            elif state == PickAndPlaceSmState.PREGRASP:   # 3
                # Contact-GraspNet 모델에서 추론한 grasp_pose로 목표 자세 지정합니다.
                self.des_ee_pose[i] = pregrasp_pose[i]
                self.des_gripper_state[i] = GripperState.OPEN
                # 목표 위치와 현재 위치의 거리를 계산하여, 목표 위치에 도달했는지 확인합니다.
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    # 특정 시간 동안 대기
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.PREGRASP:
                        # 다음 state 로 전환 및 state 시간 초기화
                        self.sm_state[i] = PickAndPlaceSmState.GRASP
                        self.sm_wait_time[i] = 0.0
            elif state == PickAndPlaceSmState.GRASP:   # 4
                #Q16. Contact-GraspNet 모델 추론값을 활용하여 GRASP 단계의 statemachine을 구현하세요.
                # Contact-GraspNet 모델에서 추론한 grasp_pose로 목표 자세 지정합니다.


                # 목표 위치와 현재 위치의 거리를 계산하여, 목표 위치에 도달했는지 확인합니다.

                    # 특정 시간 동안 대기

                        # 다음 state 로 전환 및 state 시간 초기화


            elif state == PickAndPlaceSmState.CLOSE:   # 5
                # 그리퍼를 CLOSE 상태로 둡니다.
                self.des_ee_pose[i] = ee_pose[i]
                self.des_gripper_state[i] = GripperState.CLOSE
                # 목표 위치와 현재 위치의 거리를 계산하여, 목표 위치에 도달했는지 확인합니다.
                if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.CLOSE:
                    # 다음 state 로 전환 및 state 시간 초기화
                    self.sm_state[i] = PickAndPlaceSmState.LIFT
                    self.sm_wait_time[i] = 0.0
            elif state == PickAndPlaceSmState.LIFT:   # 6
                # 로봇을 초기 자세로 이동시키고, 그리퍼를 CLOSE 상태로 둡니다.
                self.des_ee_pose[i] = self.ready_pose[i]
                self.des_gripper_state[i] = GripperState.CLOSE
                # 목표 위치와 현재 위치의 거리를 계산하여, 목표 위치에 도달했는지 확인합니다.
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    # 특정 시간 동안 대기
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.LIFT:
                        # 다음 state 로 전환 및 state 시간 초기화
                        self.sm_state[i] = PickAndPlaceSmState.REST
                        self.sm_wait_time[i] = 0.0
            
            #Q17. state machine step 마다 단위 시간씩 업데이트하세요.
            # state machine 단위시간 경과


            #Q18. state machine 의 state_wait_time 이 3초가 넘어가도 한 state가 지속되면, env reset을 실행하도록 하세요.

                # state machine 의 시간이 3초 넘어가면, env reset
                
            #Q19. state machine 의 상태에 따라 robot controller 에 입력될 end-effector 및 그리퍼 actions 값을 정의하세요.
            # robot controller 에 입력될 끝단, 그리퍼 actions 값
            actions = 
        return actions


def main():
    """메인 함수"""
    # 환경 갯수(1개로 고정)
    num_envs = 1

    # 환경 및 설정 파싱
    #Q3. 환경 id 를 압력하고 파싱된 환경 config를 만드세요.
    env_cfg: LiftEnvCfg = parse_env_cfg(
        "",
        device=args_cli.device,
        num_envs=num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    #Q4. 환경 id 와 config 를 입력하여 gymnasium 환경 인스턴스를 생성하세요.
    # 환경 생성 및 초기화
    env = gym.make("", cfg=)
    env.reset()
    print(f"Environment reset. Number of environments: {env.unwrapped.num_envs}")
    
    # 환경 관측 카메라 시점 셋팅
    env.unwrapped.sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.0, 0.0, 0.5])
    
    # 환경 연산 디바이스(gpu)
    device = env.unwrapped.scene.device
    
    # 환경에서 robot handeye camera 변수 불러오기
    robot_camera = env.unwrapped.scene.sensors['camera']
    # 카메라 인트린식(intrinsics)
    K = robot_camera.data.intrinsic_matrices.squeeze().cpu().numpy()


    # Contact-GraspNet 모델 config를 불러오기 위한 경로 설정
    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(dir_path, 'cgnet/configs/config.yaml')
    config = cfg_from_yaml_file(config_path)

    #Q5. Contact-GraspNet 모델을 빌드하기 위해, cgnet config 파일을 활용하세요.
    # Contact-GraspNet 모델 선언 및 checkpoint 입력을 통한 모델 weight 불러오기
    grasp_model = builder.model_builder(config.model)
    grasp_model_path = os.path.join(dir_path, 'data/checkpoint/contact_grasp_ckpt/ckpt-iter-60000_gc6d.pth')
    #Q6. Contact-GraspNet 모델 checkpoint 파일을 활용해서 pretrained model 을 만드세요.
    # 모델 weight 불러오기
    builder.
    grasp_model.to(device)
    grasp_model.eval()

    print("[INFO]: Setup complete...")
    

    ########## 실습(2)-state machine 정의(Q 10~19) ##########
    # 로봇 pick-and-place 제어를 위한 State machine 선언
    pick_and_place_sm = PickAndPlaceSm(
        dt=env_cfg.sim.dt * env_cfg.decimation,
        num_envs=num_envs,
        device=device,
        position_threshold=0.01
    )
    ###############################################################

    #Q20. state machine 의 초기 자세를 이용해 초기화하세요.
    # 초기 action 설정
    init_ee_pose = pick_and_place_sm.ready_pose
    init_gripper_state = torch.tensor([GripperState.CLOSE], device=device).repeat(num_envs, 1)  # 그리퍼를 CLOSE 상태로 초기화
    actions = torch.cat([init_ee_pose, init_gripper_state], dim=-1)
    # 시뮬레이션 루프
    while simulation_app.is_running():
        # 모델 추론 상태 - 학습 연산 비활성화
        with torch.inference_mode():
            dones = env.step(actions)[-2]

            # env 별 시뮬레이션 루프 실행
            for env_num in range(num_envs):
                #Q21. grasp model 추론을 하기 위해 상태 지정하기
                # 만약 상태가 PREDICT라면(즉, grasp 추론 필요시)
                if pick_and_place_sm.sm_state[env_num] == PickAndPlaceSmState.:
                    # 시각화를 위한 RGB 이미지 및 Depth 이미지 얻기
                    image_ = robot_camera.data.output["rgb"][env_num]
                    img_np = image_.squeeze().detach().cpu().numpy()

                    depth = robot_camera.data.output["distance_to_image_plane"][env_num]
                    depth_np = depth.squeeze().detach().cpu().numpy()
                    depth_vis = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_np, alpha=255.0 / np.max(depth_np)), 
                        cv2.COLORMAP_JET
                    )

                    # 이미지 시각화(디버깅용)
                    plt.figure(figsize=(10, 5))

                    plt.subplot(1, 2, 1)
                    plt.imshow(img_np)
                    plt.title('Robot View RGB Image')
                    plt.axis('off')

                    plt.subplot(1, 2, 2)
                    plt.imshow(depth_vis)
                    plt.title('Robot View Depth Image')
                    plt.axis('off')

                    if not os.path.exists('day3/3-1'):
                        os.makedirs('day3/3-1')
                    plt.savefig('day3/3-1/robot_view.png')
                    print(f"[INFO] Saving robot view images to 'day3/3-1/robot_view.png'")

                    # 취득한 Depth 이미지를 통한 Point Cloud 생성
                    if num_envs > 1:
                        pc, _ = depth2pc(depth_np, K[env_num])
                    else:
                        pc, _ = depth2pc(depth_np, K)


                    #Q22. mdp config 설정을 위한 object name을 지정하세요.
                    # SceneEntityCfg를 통해 object name을 이용하여 MDP(Markov Decision Process) 조건 설정
                    object_name = ''
                    env.unwrapped.termination_manager._term_cfgs[0].params = {'object_cfg': SceneEntityCfg(object_name)}

                    # 로봇의 End-Effector(EE) 위치 및 자세 얻기
                    robot_entity_cfg = SceneEntityCfg("robot", body_names=["panda_hand"])
                    robot_entity_cfg.resolve(env.unwrapped.scene)
                    hand_body_id = robot_entity_cfg.body_ids[0]
                    hand_pose_w = env.unwrapped.scene["robot"].data.body_state_w[:, hand_body_id, :]  # (num_envs, 13)

                    # Point Cloud가 존재하면, 이를 시각화하고 grasp 추론을 수행
                    if pc is not None:
                        pc_o3d = o3d.geometry.PointCloud()
                        pc_o3d.points = o3d.utility.Vector3dVector(pc)
                        o3d.visualization.draw_geometries([pc_o3d])
                        
                        rot_ee, trans_ee, width = inference_cgnet(pc, grasp_model, device, hand_pose_w, env)
                    
                        print(f"[INFO] Received ee coordinates from inference_cgnet")
                        print(f"[INFO] Gripper width: {width}")
                        
                        # ======================== 4) IsaacLab 형식으로 변환 ========================
                        grasp_rot = rot_ee
                        pregrasp_pos = trans_ee
                        grasp_quat = R.from_matrix(grasp_rot).as_quat()  # (x, y, z, w)
                        grasp_quat = np.array([grasp_quat[3], grasp_quat[0], grasp_quat[1], grasp_quat[2]]) # (w, x, y, z)
                        
                        # grasp_pos는 pregrasp_pos의 z축을 기축으로 0.07m 떨어진곳으로 배치
                        z_axis = grasp_rot[:, 2]  # (3,)
                        grasp_pos = pregrasp_pos + z_axis * 0.07
                        pregrasp_pose = np.concatenate([pregrasp_pos, grasp_quat])
                        grasp_pose = np.concatenate([grasp_pos, grasp_quat])                    

                        # torch tensor로 변환 후 unsqueeze 적용 (배치 차원을 맞추기 위해)
                        grasp_pose = torch.tensor(grasp_pose, device=device).unsqueeze(0)  # (1, 7)
                        pregrasp_pose = torch.tensor(pregrasp_pose, device=device).unsqueeze(0)  # (1, 7)

                        #Q23. cgnet 추론 결과를 state machine에 업데이트하세요.
                        # State machine 에 grasp 및 pregrasp 자세 업데이트
                        pick_and_place_sm.grasp_pose[env_num] = grasp_pose[0]
                        pick_and_place_sm.pregrasp_pose[env_num] = pregrasp_pose[0]
            
            # 로봇의 End-Effector 위치와 자세를 기반으로 actions 계산
            robot_data = env.unwrapped.scene["robot"].data
            # robot_root_pos_w = robot_data.root_state_w[:, :3]
            # robot_root_quat_w = robot_data.root_state_w[:, 3:7]
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]

            tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            ee_pose = torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1)

            #Q24. state machine 을 통해 action 값을 계산하세요.
            # state machine 을 통한 action 값 출력
            actions = 

            # 환경에 대한 액션을 실행 (환경에서 상태를 업데이트)
            obs, rewards, terminated, truncated, info = env.step(actions)

            # 시뮬레이션 종료 여부 체크
            dones = terminated | truncated
            if dones:
                print(terminated, truncated, dones)

    # 환경 종료
    env.close()

# 메인 함수 실행 후 시뮬레이션 종료
if __name__ == "__main__":
    main()
    simulation_app.close()
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# --- 기본 라이브러리 및 명령줄 인자 파서 임포트 ---
import argparse  # 스크립트 실행 시 커맨드 라인 인자를 파싱하기 위한 라이브러리
import glob  # 파일 경로에서 와일드카드(*)를 사용하여 여러 파일 목록을 가져오기 위한 라이브러리
import math  # 수학 연산(pi, 삼각함수 등)을 위한 라이브러리
import os  # 운영체제와 상호작용(경로 생성, 파일 시스템 접근 등)을 위한 라이브러리
import random  # 무작위 숫자 생성, 샘플링 등을 위한 라이브러리
import torch  # PyTorch 라이브러리, 텐서 연산을 위해 사용
import gc  # Garbage Collector, 명시적인 메모리 수거를 위해 사용
import sys  # 파이썬 인터프리터 제어(스크립트 종료 등)를 위한 라이브러리
import json  # 메타데이터 저장을 위한 JSON 라이브러리
from isaaclab.app import AppLauncher  # Isaac Lab 시뮬레이션 앱을 시작하고 설정하기 위한 기본 클래스

# --- 스크립트 레벨 설정 ---
# 생성할 총 에피소드(데이터셋)의 개수
NUM_EPISODES_TO_GENERATE = 1000
# YCB 객체의 3D 모델(USD 파일)이 저장된 경로 패턴
YCB_OBJECTS_USD_PATH =  os.path.join(os.getcwd(), "data/assets/ycb_usd/ycb/*/final.usd")

# --- Isaac Lab 앱 실행기 (App Launcher) 설정 ---
# 스크립트 실행 시 받을 수 있는 인자에 대한 설명 추가
parser = argparse.ArgumentParser(description="Tutorial on randomizing YCB objects in IsaacLab.")
# "--num_envs" 인자 추가 (여기서는 1로 고정되지만, 확장성을 위해 존재)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# Isaac Lab의 기본 실행 인자(예: --headless)를 파서에 추가
AppLauncher.add_app_launcher_args(parser)
# 커맨드 라인에서 입력된 인자들을 파싱
args_cli = parser.parse_args()
# 파싱된 인자를 바탕으로 AppLauncher 인스턴스 생성
app_launcher = AppLauncher(args_cli)
# AppLauncher를 통해 시뮬레이션 애플리케이션(isaac sim) 자체에 대한 참조를 얻음
simulation_app = app_launcher.app

# --- Isaac Lab 관련 모듈 임포트 ---
# 이 부분은 시뮬레이션 앱이 초기화된 후에 임포트해야 합니다.
import omni.replicator.core as rep  # 데이터 생성을 위한 Replicator 모듈
import omni.usd  # USD 스테이지와 상호작용하기 위한 모듈
from pxr import Gf, Sdf, UsdGeom  # Pixar USD 라이브러리, 저수준 USD 조작에 사용
import isaaclab.sim as sim_utils  # 시뮬레이션 에셋(에셋, 조명 등) 생성을 위한 유틸리티
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObjectCfg  # 로봇(관절), 강체 등 에셋 클래스
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # 시뮬레이션 월드를 정의하고 상호작용하기 위한 클래스
from isaaclab.sensors import CameraCfg  # 카메라 센서 및 설정을 위한 클래스
from isaaclab.sim import SimulationCfg, SimulationContext  # 시뮬레이션의 물리 설정 및 컨텍스트 관리 클래스
from isaaclab.utils import configclass  # 설정 클래스를 쉽게 만들기 위한 데코레이터
from isaaclab.utils.math import quat_from_angle_axis  # 회전각과 축으로부터 쿼터니언을 계산하는 유틸리티
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # 미리 정의된 Franka Panda 로봇 설정
from isaaclab.managers import SceneEntityCfg  # 씬 내의 특정 요소(예: 로봇 관절)를 쉽게 참조하기 위한 클래스

""" camera rendering option instead of using --enable_cameras flag """
import carb
carb_settings_iface = carb.settings.get_settings()
carb_settings_iface.set_bool("/isaaclab/cameras_enabled", True)


def get_randomized_scene_cfg() -> tuple[type[InteractiveSceneCfg], list[str]]:
    """
    매번 호출될 때마다 무작위로 구성된 새로운 씬(Scene) 설정을 생성합니다.
    이 함수는 3~5개의 YCB 객체를 무작위로 선택하고, 이들의 위치, 방향, 크기를 무작위화하여
    씬 설정 클래스에 동적으로 추가합니다.

    :return: 동적으로 생성된 씬 설정 클래스(RandomizedYCBSceneCfg)와, 씬에 추가된 객체들의 에셋 키 리스트.
    """
    # -- 변경 사항: 생성할 객체 개수를 3~5개 사이에서 무작위로 결정 --
    num_objects_to_show = random.randint(3, 5)

    # 지정된 경로 패턴에 맞는 모든 YCB 객체의 USD 파일 경로를 리스트로 가져옴
    ycb_object_usd_files = glob.glob(YCB_OBJECTS_USD_PATH)
    # 스폰할 객체 수보다 사용 가능한 객체 수가 적으면 에러 발생
    if len(ycb_object_usd_files) < num_objects_to_show:
        raise ValueError(f"Not enough unique YCB objects to spawn {num_objects_to_show}")
    
    # 전체 YCB 객체 목록에서 정해진 개수만큼 중복 없이 무작위로 샘플링
    selected_usd_paths = random.sample(ycb_object_usd_files, num_objects_to_show)
    
    # 씬에 추가될 객체들의 고유 키를 저장할 리스트
    object_asset_keys = []

    # @configclass 데코레이터를 사용하여 씬 설정을 정의. 코드 자동완성 및 타입 체킹에 유용.
    @configclass
    class RandomizedYCBSceneCfg(InteractiveSceneCfg):
        # 고정된 에셋들을 먼저 정의
        # 바닥 평면
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        # 돔 라이트 (전역 조명)
        dome_light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, 
                                                                   color=(0.75, 0.75, 0.75))
        )
        # 테이블 (강체)
        table: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.CuboidCfg(
                size=(1.2, 2.0, 0.5), # 테이블 크기
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.5, 0.5, 0.5)), # 시각적 재질
                rigid_props=sim_utils.RigidBodyPropertiesCfg(), # 강체 속성
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True), # 충돌 활성화
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0) # 물리 재질
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.4, 0.25)), # 초기 위치
        )
        # Franka Panda 로봇 (관절 에셋)
        robot: Articulation = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=FRANKA_PANDA_HIGH_PD_CFG.InitialStateCfg(
                pos=(0.0, 0.0, 0.5), # 로봇 베이스 초기 위치
                joint_pos={ # 각 관절의 초기 각도 설정
                    "panda_joint1": 0.0, "panda_joint2": -0.785, "panda_joint3": 0.0,
                    "panda_joint4": -2.356, "panda_joint5": 0.0, "panda_joint6": 1.571, 
                    "panda_joint7": 0.785
                }
            ),
        )
        # 로봇 손에 부착된 카메라
        camera: CameraCfg = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/handeye_camera", # 카메라가 생성될 USD 경로
            update_period=0.1, height=480, width=640, # 업데이트 주기 및 해상도
            data_types=["rgb", 
                        "instance_id_segmentation_fast"], # 수집할 데이터 타입 (RGB, 인스턴스 분할 마스크)
            colorize_instance_id_segmentation=True, # 분할 마스크를 시각적으로 보기 좋게 색칠할지 여부
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, 
                                             horizontal_aperture=20.955), # 카메라 물리적 특성
            offset=CameraCfg.OffsetCfg(pos=(0.1, 0.035, 0.0), 
                                       rot=(0.70710678, 0.0, 0.0, 0.70710678), 
                                       convention="ros") # 부모(panda_hand)로부터의 상대적 위치/방향
        )
    
    #### 구현 예제 ####
    # 선택된 USD 파일들을 순회하며 무작위화된 객체 설정을 동적으로 생성
    for i in range(num_objects_to_show):
        usd_path = selected_usd_paths[i]
        # 파일 경로에서 YCB 객체의 고유 ID를 추출 (예: '003_cracker_box')
        ycb_id = os.path.basename(os.path.dirname(usd_path))
        
        # 이 객체에 대한 고유한 에셋 키 생성 (예: 'ycb_003_cracker_box')
        asset_key = f"ycb_{ycb_id.replace('-', '_')}"
        object_asset_keys.append(asset_key)

        #############################################################object_pose###########

        ################## 객체의 위치 설정 (랜덤화 범위 설정) ######################

        # 객체의 초기 위치, 크기, 회전을 무작위로 결정

        # (예제 1) 테이블 위 X 좌표

        # (예제 2) 테이블 위 Y 좌표

        # (예제 3) 공중에서 떨어뜨릴 Z 좌표 (테이블 높이 0.5 + 여유)

        # (예제 4) 객체 크기

        # (예제 5) Z축 기준 회전 각도 (라디안)

        ########################################################################

        z_axis_tensor = torch.tensor([0.0, 0.0, 1.0]) # Z축 벡터
        # 회전 각도와 축으로부터 쿼터니언(사원수) 계산
        quat = quat_from_angle_axis(torch.tensor(random_z_angle), z_axis_tensor).tolist()

        # 이 객체에 대한 RigidObjectCfg 생성
        config = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{asset_key}", # 씬 내에서 이 객체의 경로
            init_state=RigidObjectCfg.InitialStateCfg(pos=(pos_x, pos_y, pos_z), 
                                                      rot=quat), # 무작위화된 초기 위치/방향
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path, # 로드할 3D 모델 파일
                scale=(random_scale, random_scale, random_scale), # 무작위화된 크기
                rigid_props=sim_utils.RigidBodyPropertiesCfg(), # 강체 속성
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True), # 충돌 활성화
                mass_props=sim_utils.MassPropertiesCfg(mass=0.2), # 질량 설정
            ),
        )
        # `setattr`을 사용하여 동적으로 생성된 설정을 RandomizedYCBSceneCfg 클래스의 속성으로 추가
        # 예를 들어, `RandomizedYCBSceneCfg.ycb_003_cracker_box = config` 와 같이 동작
        setattr(RandomizedYCBSceneCfg, asset_key, config)
        
    return RandomizedYCBSceneCfg, object_asset_keys

def main():
    """메인 함수. 데이터 생성 파이프라인 전체를 관리합니다."""
    # 시뮬레이션 설정: 시간 간격(dt) 및 사용할 장치(CPU/GPU) 정의
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    # 시뮬레이션 컨텍스트 초기화 (이 컨텍스트 내에서 물리 시뮬레이션이 실행됨)
    sim = SimulationContext(sim_cfg)
    # 시뮬레이터의 뷰포트 카메라 위치 설정 (디버깅/시각화용)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    
    # 생성된 데이터를 저장할 출력 디렉토리 생성
    output_dir = os.path.join(os.getcwd(), "output_data")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO]: Saving data to: {output_dir}")
    
    # 이 스크립트에서는 병렬 환경을 1개만 사용
    num_envs = 1

    # 설정된 횟수만큼 에피소드(데이터 생성)를 반복
    for episode in range(NUM_EPISODES_TO_GENERATE):
        print(f"\n--- Starting Episode {episode + 1}/{NUM_EPISODES_TO_GENERATE} ---")
        
        # 매 에피소드마다 새로운 무작위 씬 설정을 가져옴
        CurrentSceneCfg, object_asset_keys = get_randomized_scene_cfg()
        # 가져온 설정으로 씬 설정 객체(instance)를 생성
        scene_cfg = CurrentSceneCfg(num_envs=num_envs, env_spacing=2.5)
        # 설정 객체를 바탕으로 실제 상호작용 가능한 씬을 생성
        scene = InteractiveScene(scene_cfg)
        
        # 시뮬레이션 리셋. 이전 에피소드의 씬을 지우고 새로운 씬을 로드.
        sim.reset()

        # 씬에서 로봇 객체에 대한 참조를 가져옴
        franka_robot = scene.articulations["robot"] 
        # 로봇의 특정 관절들("panda_joint"으로 시작하는 모든 관절)을 제어하기 위한 설정
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"])
        robot_entity_cfg.resolve(scene) # 씬에서 해당 관절들의 ID를 찾음
        # 로봇이 취할 목표 관절 각도를 텐서로 정의
        joint_pos_target = torch.tensor(
            [[-0.0183, -0.1576,  0.0131, -0.8471,  0.0032,  0.6895,  0.7825]], device=sim.device)
        # 로봇의 목표 관절 각도를 설정
        franka_robot.set_joint_position_target(
            joint_pos_target, joint_ids=robot_entity_cfg.joint_ids)
        # 변경된 데이터를 시뮬레이션에 적용
        scene.write_data_to_sim()

        print("Scene created. Moving robot and letting objects settle...")
        # 객체들이 중력에 의해 떨어져 안정화될 때까지 100 스텝 동안 시뮬레이션 진행
        for _ in range(100):
            sim.step() # 물리 시뮬레이션을 한 스텝 진행
            scene.update(sim.get_physics_dt()) # 씬의 상태를 업데이트

        print("Capturing and saving data...")
        # 카메라를 업데이트하여 현재 씬의 이미지를 렌더링하고 데이터를 수집
        scene["camera"].update(sim.get_physics_dt())
        
        # 현재 에피소드의 데이터를 저장할 하위 디렉토리 생성 (예: output_data/episode_000)
        episode_output_dir = os.path.join(output_dir, f"episode_{episode:03d}")
        os.makedirs(episode_output_dir, exist_ok=True)
        
        # 메타데이터 저장을 위한 딕셔너리 초기화
        metadata = {"episode": episode}
        metadata["objects"] = []
        # 현재 씬에 있는 객체들의 최종 상태(자세)를 기록
        # 이제 object_asset_keys의 길이는 매번 3~5 사이에서 변합니다.
        for asset_key in object_asset_keys:
            # 에셋 키를 사용하여 씬에서 해당 객체의 핸들(참조)을 가져옴
            obj_handle = scene.rigid_objects[asset_key]
            # 에셋 키에서 'ycb_' 접두사를 제거하여 순수 YCB ID를 얻음
            ycb_id = asset_key.replace("ycb_", "", 1)
            
            # 객체의 현재 위치와 방향(쿼터니언) 데이터를 가져옴
            pose = obj_handle.data.root_state_w[0, :7].cpu().numpy().tolist()
            # 메타데이터에 추가할 객체 정보 구성
            obj_data = {
                "ycb_id": ycb_id,
                "prim_path": obj_handle.cfg.prim_path,
                "position": pose[:3], # 위치 [x, y, z]
                "quaternion_wxyz": pose[3:] # 방향 [w, x, y, z]
            }
            metadata["objects"].append(obj_data)
        
        # 메타데이터를 JSON 파일로 저장
        json_filename = "metadata.json"
        json_filepath = os.path.join(episode_output_dir, json_filename)
        with open(json_filepath, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved to: {json_filepath}")
        
        # Replicator를 사용하여 센서 데이터(이미지, 마스크)를 파일로 저장하기 위한 writer 설정
        rep_writer = rep.BasicWriter(
            output_dir=episode_output_dir, # 출력 디렉토리
            frame_padding=0, # 파일 이름에 프레임 번호 패딩 사용 안함
            colorize_instance_id_segmentation=scene["camera"].cfg.colorize_instance_id_segmentation,
        )
        # 씬의 모든 환경(여기서는 1개)에 대해 데이터 저장
        for env_id in range(scene.num_envs):
            # 카메라 데이터를 GPU에서 CPU로 옮기고 numpy 배열로 변환
            single_env_data = {k: v[env_id].cpu().numpy() 
                               for k, v in scene["camera"].data.output.items()}
            # 카메라 데이터에 대한 추가 정보
            single_env_info = scene["camera"].data.info[env_id]
            # Replicator가 요구하는 형식으로 데이터 재구성
            rep_output = {"annotators": {}}
            for key, data, info in zip(single_env_data.keys(), 
                                       single_env_data.values(), 
                                       single_env_info.values()):
                if info is not None: 
                    rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                else: 
                    rep_output["annotators"][key] = {"render_product": {"data": data}}
            rep_output["trigger_outputs"] = {"on_time": scene["camera"].frame[0]}
            # 최종적으로 데이터를 파일로 씀 (rgb.png, instance_id_segmentation.png, ..._mapping.json 등)
            rep_writer.write(rep_output)
            print(f"Image data for episode {episode} saved to: {episode_output_dir}")
        
        # 메모리 누수 해결을 위한 올바른 씬 정리 방식
        # 마지막 에피소드가 아니라면, 다음 에피소드를 위해 현재 씬을 완전히 정리
        if episode < NUM_EPISODES_TO_GENERATE - 1:
            print("Cleaning up scene for the next episode...")
            # 파이썬 객체에 대한 참조를 명시적으로 제거하여 가비지 컬렉터가 메모리를 회수할 수 있도록 유도
            scene = None
            scene_cfg = None
            CurrentSceneCfg = None
            # 가비지 컬렉션을 수동으로 호출
            gc.collect()

            # USD 스테이지에서 직접 프리미티브(객체)를 제거. 이것이 메모리 누수를 막는 가장 확실한 방법.
            stage = omni.usd.get_context().get_stage()
            if stage.GetPrimAtPath("/World/envs"):
                stage.RemovePrim("/World/envs") # 모든 환경(로봇, 테이블, 객체 포함) 제거
            if stage.GetPrimAtPath("/World/defaultGroundPlane"):
                stage.RemovePrim("/World/defaultGroundPlane") # 바닥 제거
            if stage.GetPrimAtPath("/World/Light"):
                stage.RemovePrim("/World/Light") # 조명 제거
            
            # 프리미티브 제거 작업이 시뮬레이션에 즉시 반영되도록 앱을 한 프레임 업데이트
            simulation_app.update()
        else:
            # 마지막 에피소드가 끝나면 씬을 그대로 두고 종료
            print("Final episode complete. Leaving the scene intact.")


        # 시뮬레이션 앱이 사용자에 의해 종료되었는지 확인
        if not simulation_app.is_running():
            break

    print("\n--- Data generation complete ---")
    # 모든 작업 완료 후 스크립트를 정상적으로 종료.
    # try...except 블록에서 이 종료 신호를 잡아 앱을 안전하게 닫음.
    sys.exit(0)

if __name__ == "__main__":
    # 스크립트 실행의 메인 진입점
    try:
        main()
    except Exception as e:
        # 그 외 다른 예외가 발생했을 때 에러 메시지 출력
        print(f"An error occurred: {e}")
    finally:
        # 예외 발생 여부와 관계없이 항상 마지막에 시뮬레이션 앱을 안전하게 종료
        simulation_app.close()

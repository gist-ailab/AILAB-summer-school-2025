"""
UR5e 로봇과 빨간 큐브를 포함한 Isaac Lab 시뮬레이션 환경

이 스크립트는 다음과 같은 기능을 제공합니다:
1. UR5e 로봇을 시뮬레이션 환경에 추가
2. 빨간 큐브를 로봇 근처에 배치
3. 로봇의 순기구학(Forward Kinematics) 동작 시연
4. 주기적인 로봇 리셋 및 동작 제어
"""

# ============================================================================
# 1. 필요한 라이브러리 임포트 및 명령행 인자 설정
# ============================================================================
import argparse
from isaaclab.app import AppLauncher

# 명령행 인자 파서 설정
parser = argparse.ArgumentParser(
    description="이 스크립트는 Isaac Lab 환경에 커스텀 로봇을 추가하는 방법을 보여줍니다."
)
parser.add_argument("--num_envs", type=int, default=1, help="생성할 환경의 개수")
# AppLauncher의 명령행 인자들을 파서에 추가
AppLauncher.add_app_launcher_args(parser)
# 인자 파싱
args_cli = parser.parse_args()

# Isaac Lab 앱 실행
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab 관련 모듈들 임포트
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.actuators import ImplicitActuatorCfg

# 유틸리티 모듈들 임포트
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import numpy as np
import torch


# ============================================================================
# 2. 로봇 및 오브젝트 설정
# ============================================================================

# UR5e 로봇 설정
# - USD 파일 경로: Isaac Lab의 로봇 에셋 디렉토리에서 UR10e 모델 사용
# - 액추에이터: 모든 조인트에 대해 actuatorcfg 설정 (제어를 위한 설정)
UR5E_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd"),
    actuators={"arm_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)

# 빨간 큐브 설정
# - 크기: 5cm x 5cm x 5cm
# - 질량: 1kg
# - 색상: 빨간색 (RGB: 1.0, 0.0, 0.0)
# - 물리적 속성: 강체(RigidBody)로 설정하여 시뮬레이터에 로드
cfg_cube = sim_utils.CuboidCfg(
    size=(0.05, 0.05, 0.05),  # 큐브 크기 (미터 단위)
    rigid_props=sim_utils.RigidBodyPropertiesCfg(),  # 강체 속성
    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),  # 질량 설정
    collision_props=sim_utils.CollisionPropertiesCfg(),  # 충돌 속성
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # 빨간색 재질
)


# ============================================================================
# 3. 시뮬레이션 씬 설정 클래스
# ============================================================================
class UR5eSceneCfg(InteractiveSceneCfg):
    """
    UR5e 로봇과 빨간 큐브를 포함한 새로운 씬을 설계하는 클래스
    
    이 클래스는 시뮬레이션 환경의 모든 요소들을 정의합니다:
    - 바닥면 (Ground Plane)
    - 조명 (Dome Light)
    - 로봇 (UR5e)
    - 큐브 (빨간색)
    """
    
    # 바닥면 설정 - 기본 지면을 생성
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0)),
        spawn=GroundPlaneCfg(),
    )

    # 조명 설정 - 돔 라이트로 전체 환경을 밝게 조명
    dome_light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # 로봇 설정 - UR5e를 각 환경에 배치
    # {ENV_REGEX_NS}는 환경 네임스페이스를 자동으로 생성하는 플레이스홀더
    UR5e = UR5E_CONFIG.replace(prim_path="{ENV_REGEX_NS}/UR5e")
    UR5e.init_state.pos = (0.0, 0.0, 0.0)


# ============================================================================
# 4. 시뮬레이션 실행 함수
# ============================================================================
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """
    시뮬레이션의 메인 루프를 실행하는 함수
    
    이 함수는 다음 작업들을 수행합니다:
    1. 로봇의 주기적 리셋 (500 스텝마다)
    2. 로봇 조인트 제어 (위치 타겟 설정)
    3. 시뮬레이션 스텝 진행
    4. 씬 상태 업데이트
    """
    
    # 시뮬레이션 시간 설정
    sim_dt = sim.get_physics_dt()  # 물리 시뮬레이션 시간 간격
    sim_time = 0.0
    count = 0  # 스텝 카운터

    # 시뮬레이션 메인 루프
    while simulation_app.is_running():
        
        # 500 스텝마다 로봇 상태 리셋
        if count % 500 == 0:
            count = 0
            print("[INFO]: 로봇 상태를 초기화합니다...")
            
            # 로봇의 루트 상태(위치, 방향, 속도) 가져오기
            root_ur5e_state = scene["UR5e"].data.default_root_state.clone()
            print("리셋 전 루트 상태: ", root_ur5e_state)
            
            # 환경 원점을 기준으로 위치 조정
            root_ur5e_state[:, :3] += scene.env_origins
            print("리셋 후 루트 상태: ", root_ur5e_state)

            # 시뮬레이션에 로봇의 위치와 방향 설정 (처음 7개 값: x,y,z + quaternion)
            scene["UR5e"].write_root_pose_to_sim(root_ur5e_state[:, :7])
            # 시뮬레이션에 로봇의 속도 설정 (나머지 6개 값: linear + angular velocity)
            scene["UR5e"].write_root_velocity_to_sim(root_ur5e_state[:, 7:])

            # 조인트 상태 초기화
            joint_pos = scene["UR5e"].data.default_joint_pos.clone()  # 조인트 위치
            joint_vel = scene["UR5e"].data.default_joint_vel.clone()  # 조인트 속도
            scene["UR5e"].write_joint_state_to_sim(joint_pos, joint_vel)
            
            # 씬 전체 리셋
            scene.reset()
            print("[INFO]: UR5e와 큐브 상태 리셋 완료!")
            
        # 로봇 동작 제어 (200 스텝 주기로 동작 변경)
        if count % 200 < 100:
            # 첫 100 스텝: 특정 조인트 각도로 이동 (라디안 단위)
            # [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
            action = torch.tensor([[0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]])
        else:
            # 다음 100 스텝: 모든 조인트를 0도로 이동
            action = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        # 로봇 조인트 제어
        # set_joint_velocity_target: 속도 제어 (사용하지 않음)
        # set_joint_position_target: 위치 제어 (사용)
        scene["UR5e"].set_joint_position_target(action)

        # 시뮬레이션 진행
        scene.write_data_to_sim()  # 씬 데이터를 시뮬레이션에 쓰기
        sim.step()  # 시뮬레이션 한 스텝 진행
        sim_time += sim_dt  # 시뮬레이션 시간 업데이트
        count += 1  # 스텝 카운터 증가
        scene.update(sim_dt)  # 씬 상태 업데이트


# ============================================================================
# 5. 메인 함수
# ============================================================================
def main():
    """
    프로그램의 메인 함수
    
    이 함수는 다음 작업들을 수행합니다:
    1. 시뮬레이션 컨텍스트 생성
    2. 카메라 뷰 설정
    3. 씬 생성 및 초기화
    4. 시뮬레이션 실행
    """
    
    # 시뮬레이션 설정 및 컨텍스트 생성
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # 카메라 뷰 설정
    # eye: 카메라 위치 (x, y, z)
    # target: 카메라가 바라보는 지점 (x, y, z)
    sim.set_camera_view(
        eye=(3.5, 0.0, 3.2),      # 카메라 위치: 로봇을 위에서 바라보는 위치
        target=(0.0, 0.0, 0.5)   # 카메라 타겟: 로봇 중앙 부근
    )
    
    # 씬 생성 및 설정
    # num_envs: 환경 개수, env_spacing: 환경 간 간격 (미터)
    scene_cfg = UR5eSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # 큐브를 올바른 위치에 배치
    cfg_cube.func("/World/RedCube", cfg_cube, translation=(0.1, 1.0, 0.05))
    
    # 시뮬레이션 초기화
    sim.reset()
    print("[INFO]: UR5e와 큐브 시뮬레이션이 시작되었습니다!")
    
    # 시뮬레이션 실행
    run_simulator(sim, scene)


# ============================================================================
# 6. 프로그램 실행
# ============================================================================
if __name__ == "__main__":
    main()
    simulation_app.close()  # 시뮬레이션 종료 후 앱 정리
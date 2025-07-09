import glob
import random
import os
from dataclasses import MISSING

# Isaac Lab 관련 라이브러리 임포트
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.spawners import materials

# mdp 관련 함수와 config
from . import mdp_3_3 as mdp

@configclass
class CustomUsdFileCfg(UsdFileCfg):
    """커스텀 USD 파일 config - 물리 소재 경로를 지정하기 위함(기존의 UsdFileCfg 에는 물리 소재가 없음)"""

    # Prim에 적용할 물리 소재 경로 (상대 경로 가능)
    physics_material_path: str = "material"

    # 물리 소재를 명시적으로 지정. None이면 적용 안 함.
    physics_material: materials.PhysicsMaterialCfg | None = None


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """로봇과 물체가 포함된 기본 Scene 구성 Config"""

    # 로봇, end-effector 센서, 카메라는 Agent 환경 config에서 채워짐 (MISSING)
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    camera: CameraCfg = MISSING

    # 테이블 오브젝트 (기본 환경 오브젝트)
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
                size=(1.6, 2.0, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5), metallic=0.2, roughness=0.5),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.5, restitution=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, 0.4, 0.25)),
    )

    # 바닥 Plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0)),
        spawn=GroundPlaneCfg(),
    )

    # 조명
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # 바구니(Bin) 오브젝트
    bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/bin",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.6, 0.555), rot=[0.7071, 0.7071, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path='data/assets/basket/basket.usd',
            scale=(0.8, 0.25, 0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.7, 0.5), metallic=0.2, roughness=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
        ),
    )
    
    #################################### Set Random YCB Obects in Scene ####################################
    def __post_init__(self):
        """환경 생성 후 자동으로 실행되는 추가 세팅 코드"""

        # YCB object들 전체 읽어오기
        ycb_obj_usd_paths = glob.glob('data/assets/ycb_usd/ycb/*/final.usd')

        # YCB object 중 3가지 물체 random하게 설정
        selected_ycb_obj_usd_paths = 

        # YCB object 놓을 위치 지정(카메라 view에 맞게)

        # 각 물체 로드
        for i in range(len(selected_ycb_obj_usd_paths)):

            # 지정된 위치에서 일정 거리 내에 random하게 위치 재설정 (0.05 이내)
            
            # YCB object 경로를 절대 경로로 설정
            ycb_obj_usd_path = os.path.join(os.getcwd(), selected_ycb_obj_usd_paths[i])

            # 각 객체 이름 설정 및 material 경로 지정
            attr_name = f"object_{i}"
            physical_material_path = f"{{ENV_REGEX_NS}}/{attr_name}/physical_material"

            # 실제로 사용할 object config 세팅 (physics_material에서 friction을 설정해줘야 물체를 잘 잡을 수 있음)
            obj_cfg = 
            
            # config에 객체 속성 추가
            setattr(self, attr_name, obj_cfg)

    ############################################################################################################



""" MDP 세팅 (명령어, 액션, 관측, 보상, 이벤트 등) """

@configclass
class CommandsCfg:
    """MDP에 사용되는 명령어(command) 정의"""


@configclass
class ActionsCfg:
    """MDP에 사용되는 액션(action) 정의"""
    # Agent 환경에서 채워지는 값
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """MDP에 사용되는 관측(observation) 정의"""


@configclass
class EventCfg:
    """이벤트(event) 처리에 대한 설정"""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    """보상(reward) 항목 설정 - 강화학습 개발시 필요"""


@configclass
class TerminationsCfg:
    """에피소드 종료 조건(termination) 설정"""
    # 모든 물체가 원하는 지점에 들어왔을때, 에피소드 종료
    object_reach_goal = DoneTerm(func=mdp.object_pickplace_goal)


@configclass
class CurriculumCfg:
    """커리큘럼(curriculum) 보상 가중치 변경 등"""




""" 최종 환경 config """
@configclass
class YCBPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """환경 전체 설정"""

    # Scene 구성
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # MDP 세팅
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    # commands: CommandsCfg = CommandsCfg()  # 필요시 사용
    # curriculum: CurriculumCfg = CurriculumCfg()  # 필요시 사용

    def __post_init__(self):
        """환경 생성 후 추가 세팅"""
        self.decimation = 2
        self.episode_length_s = 5.0
        # 시뮬레이션 기본 설정
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation
        # PhysX 물리엔진 세부 튜닝
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

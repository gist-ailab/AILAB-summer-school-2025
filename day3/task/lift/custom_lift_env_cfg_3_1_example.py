from dataclasses import MISSING

# Isaac Lab 관련 라이브러리 임포트
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
from . import mdp_3_1

import os

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
    camera: CameraCfg = MISSING

    # grasp 대상 오브젝트
    object_0: RigidObjectCfg | DeformableObjectCfg = MISSING


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
    
    # 5. 오브젝트 설정 (큐브)
    object_0 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0, 0.6], rot=[1, 0, 0, 0]),
        spawn=CustomUsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            physics_material_path=f"{{ENV_REGEX_NS}}/object_0/physical_material",
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=10.0, dynamic_friction=10.0, restitution=0.0),
        ),
    )

        # setattr(self, 'object_0', self.scene.object)
    
""" MDP 세팅 (명령어, 액션, 관측, 보상, 이벤트 등) """

@configclass
class CommandsCfg:
    """MDP에 사용되는 명령어(command) 정의"""
    # object_pose = mdp_3_1.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name=MISSING,      # agent 환경에서 지정됨
    #     resampling_time_range=(5.0, 5.0),
    #     debug_vis=False,
    #     ranges=mdp_3_1.UniformPoseCommandCfg.Ranges(
    #         pos_x=(0.5, 0.5), pos_y=(-0.0, 0.0), pos_z=(0.525, 0.53), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
    #     ),
    # )


@configclass
class ActionsCfg:
    """MDP에 사용되는 액션(action) 정의"""
    # Agent 환경에서 채워지는 값
    arm_action: mdp_3_1.JointPositionActionCfg | mdp_3_1.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp_3_1.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """MDP에 사용되는 관측(observation) 정의"""
    # @configclass
    # class PolicyCfg(ObsGroup):
    #     """policy 네트워크 입력에 사용되는 관측 항목 그룹"""
    #     joint_pos = ObsTerm(func=mdp_3_1.joint_pos_rel)
    #     joint_vel = ObsTerm(func=mdp_3_1.joint_vel_rel)
    #     object_position = ObsTerm(func=mdp_3_1.object_position_in_robot_root_frame)
    #     # target_object_position = ObsTerm(func=mdp_3_1.generated_commands, params={"command_name": "object_pose"})
    #     actions = ObsTerm(func=mdp_3_1.last_action)

    #     def __post_init__(self):
    #         self.enable_corruption = True
    #         self.concatenate_terms = True

    # # 관측 그룹 등록
    # policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """이벤트(event) 처리에 대한 설정"""
    #Q25. event 함수 적용, random pose range 적용
    reset_all = EventTerm(func=, mode="reset")
    reset_object_position = EventTerm(
        func=mdp_3_1.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": ,
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object_0", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """보상(reward) 항목 설정 - 강화학습 개발시 필요"""

    # reaching_object = RewTerm(func=mdp_3_1.object_ee_distance, params={"std": 0.1}, weight=1.0)

    # lifting_object = RewTerm(func=mdp_3_1.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0)

    # object_goal_tracking = RewTerm(
    #     func=mdp_3_1.object_goal_distance,
    #     params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
    #     weight=16.0,
    # )

    # object_goal_tracking_fine_grained = RewTerm(
    #     func=mdp_3_1.object_goal_distance,
    #     params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
    #     weight=5.0,
    # )

    # # action penalty
    # action_rate = RewTerm(func=mdp_3_1.action_rate_l2, weight=-1e-4)

    # joint_vel = RewTerm(
    #     func=mdp_3_1.joint_vel_l2,
    #     weight=-1e-4,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )


@configclass
class TerminationsCfg:
    """에피소드 종료 조건(termination) 설정"""
    # time_out = DoneTerm(func=mdp_3_1.time_out, time_out=False)
    # object_dropping = DoneTerm(func=mdp_3_1.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object_0")})
    #Q25. termination 조건을 계산하는 함수 적용
    object_reach_goal = DoneTerm(func=)


@configclass
class CurriculumCfg:
    """커리큘럼(curriculum) 보상 가중치 변경 등"""
    # action_rate = CurrTerm(
    #     func=mdp_3_1.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    # )
    # joint_vel = CurrTerm(
    #     func=mdp_3_1.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    # )




""" 최종 환경 config """
@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """리프팅 환경 전체 설정"""

    # Scene 구성
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5)
    # MDP 세팅
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
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


"""
Isaac Lab 환경에서 리프팅 태스크를 위한 기본 환경 설정

1. 리프팅 태스크의 기본 씬 구조 정의
2. MDP(Markov Decision Process) 구성 요소 설정
3. 보상 함수, 종료 조건, 관측 설정
4. 강화학습을 위한 환경 구성
"""

# ============================================================================
# 1. 필요한 라이브러리 임포트
# ============================================================================
from dataclasses import MISSING

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

from . import mdp

# ============================================================================
# 2. 씬 정의 (Scene Definition)
# ============================================================================

@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """
    로봇과 오브젝트가 포함된 리프팅 씬의 설정
    
    구체적인 씬은 자식 클래스에서 정의.
    자식 클래스에서는 타겟 오브젝트, 로봇, 엔드 이펙터 프레임을 설정.
    """

    # 로봇: 자식 클래스에서 구체적으로 설정됨
    robot: ArticulationCfg = MISSING
    # 엔드 이펙터 센서: 자식 클래스에서 구체적으로 설정됨
    ee_frame: FrameTransformerCfg = MISSING
    # 타겟 오브젝트: 자식 클래스에서 구체적으로 설정됨
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # ============================================================================
    # 3. 커스텀 테이블 설정
    # ============================================================================
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",  # 테이블의 USD 경로
        spawn=sim_utils.CuboidCfg(
                size=(2.0, 1.5, 0.5),  # 테이블 크기 (가로, 세로, 높이)
                # 시각적 재질 설정
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.5, 0.5, 0.5),  # 회색
                    metallic=0.2,  # 금속성
                    roughness=0.5  # 거칠기
                ),
                # 물리적 재질 설정
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.8,   # 정지 마찰 계수
                    dynamic_friction=0.5,  # 동적 마찰 계수
                    restitution=0.1        # 탄성 계수
                ),
                # 충돌 속성 설정
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            ),
        # 초기 상태 설정
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0, 0.25)),  # 테이블 위치
    )

    ############# Plane 생성 #############
    #### GroundPlaneCfg 사용 ####
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0)),
        spawn=GroundPlaneCfg(),
    )

    ############# Light 생성 #############
    #### sim_utils.DomeLightCfg 사용 ####
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    
# ============================================================================
# 6. MDP 설정 (Markov Decision Process)
# ============================================================================

@configclass
class CommandsCfg:
    """MDP의 명령(Command) 설정"""

    # 오브젝트 목표 자세 생성 명령
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",  # 명령을 받을 로봇
        body_name=MISSING,   # 자식 클래스에서 설정됨 (엔드 이펙터 body)
        resampling_time_range=(5.0, 5.0),  # 명령 재샘플링 시간 범위
        debug_vis=True,  # 디버그 시각화 활성화
        # 목표 위치 범위 설정
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6),     # x 위치 범위 (로봇 앞쪽)
            pos_y=(-0.25, 0.25),  # y 위치 범위 (좌우)
            pos_z=(0.3, 0.7),     # z 위치 범위 (높이 - 리프팅 목표!)
            roll=(0.0, 0.0),      # 롤 회전 (고정)
            pitch=(0.0, 0.0),     # 피치 회전 (고정)
            yaw=(0.0, 0.0)        # 요 회전 (고정)
        ),
    )


@configclass
class ActionsCfg:
    """MDP의 액션(Action) 설정"""

    # 팔 액션: 자식 클래스에서 구체적으로 설정됨
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    # 그리퍼 액션: 자식 클래스에서 구체적으로 설정됨
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """MDP의 관측(Observation) 설정"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy를 위한 설정"""

        # 조인트 위치 (상대적)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # 조인트 속도 (상대적)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # 로봇 루트 프레임에서의 오브젝트 위치
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # 생성된 명령 (목표 오브젝트 위치)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        # 이전 액션
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            """초기화 후 실행되는 메서드"""
            self.enable_corruption = True    # 관측 노이즈 활성화
            self.concatenate_terms = True    # 관측 항목들을 연결

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """이벤트 설정"""

    # 전체 씬을 기본 상태로 리셋
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # 오브젝트 위치만 랜덤하게 리셋
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # 오브젝트 초기 위치 범위
            "pose_range": {
                "x": (-0.1, 0.1),    # x 위치 범위
                "y": (-0.25, 0.25),  # y 위치 범위
                "z": (0.0, 0.0)      # z 위치 범위 (테이블 위)
            },
            "velocity_range": {},  # 속도 범위 (비어있음)
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),  # 오브젝트 설정
        },
    )


@configclass
class RewardsCfg:
    """MDP의 보상(Reward) 설정, 현재는 사용하지 않음"""


@configclass
class TerminationsCfg: 
    """MDP의 종료 조건 설정"""
    
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 오브젝트 낙하
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,  # 최소 높이 이하
        params={
            "minimum_height": 0.45,         # 최소 높이 (45cm) = 책상으로 설정한 큐브보다 낮은 높이
            "asset_cfg": SceneEntityCfg("object")  # 오브젝트 설정
        }
    )


@configclass
class CurriculumCfg:
    """커리큘럼 학습 설정, 현재는 사용하지 않음"""
    
    
    # 액션 변화 페널티 가중치 조정
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,  # 보상 가중치 수정
        params={
            "term_name": "action_rate",  # 수정할 보상 항목
            "weight": -1e-1,             # 새로운 가중치
            "num_steps": 10000           # 적용할 스텝 수
        }
    )

    # 조인트 속도 페널티 가중치 조정
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,  # 보상 가중치 수정
        params={
            "term_name": "joint_vel",    # 수정할 보상 항목
            "weight": -1e-1,             # 새로운 가중치
            "num_steps": 10000           # 적용할 스텝 수
        }
    )


# ============================================================================
# 7. 환경 설정 (Environment Configuration)
# ============================================================================

@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """리프팅 환경의 설정"""

    # 씬 설정
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # 기본 설정
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP 설정
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # 커리큘럼 설정 (현재 비활성화)
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """초기화 후 실행되는 메서드"""
        # ============================================================================
        # 8. 일반 설정
        # ============================================================================
        self.decimation = 2              # 액션 반복 횟수
        self.episode_length_s = 5.0      # 에피소드 길이 (5초)
        
        # ============================================================================
        # 9. 시뮬레이션 설정
        # ============================================================================
        self.sim.dt = 0.01  # 시뮬레이션 시간 스텝 (100Hz)
        self.sim.render_interval = self.decimation  # 렌더링 간격

        # ============================================================================
        # 10. PhysX 물리 엔진 설정
        # ============================================================================
        # 튀어오름 임계 속도 설정
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        
        # GPU 메모리 설정
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4  # 4MB
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024  # 16KB
        
        # 마찰 상관 거리 설정
        self.sim.physx.friction_correlation_distance = 0.00625

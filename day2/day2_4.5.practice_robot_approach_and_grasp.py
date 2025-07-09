# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Isaac Lab 환경에서 Franka 로봇을 사용한 Object Approach 예제

1. Isaac Lab 시뮬레이션 환경 초기화
2. Franka 로봇과 큐브가 포함된 리프팅 환경 생성
3. State Machine을 통한 로봇의 Object Approach 제어
"""

# ============================================================================
# 1. 필요한 라이브러리 임포트 및 명령행 인자 설정
# ============================================================================
import argparse
from isaaclab.app import AppLauncher

# 명령행 인자 파서 설정 - 사용자가 다양한 옵션을 설정할 수 있도록 함
parser = argparse.ArgumentParser(description="Pick and lift state machine for lift environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
# AppLauncher의 명령행 인자들을 파서에 추가
AppLauncher.add_app_launcher_args(parser)
# 인자 파싱
args_cli = parser.parse_args()

# Isaac Lab 앱 초기화 - 시뮬레이션 환경을 시작
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

# ============================================================================
# 2. 시뮬레이션 관련 라이브러리 임포트
# ============================================================================
import gymnasium as gym
import torch
from collections.abc import Sequence

from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData

import isaaclab_tasks  # noqa: F401
# 커스텀 환경 설정 파일들 임포트
from task.lift.custom_lift_env_cfg import LiftEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# 커스텀 환경 등록 - Gymnasium 환경 레지스트리에 추가
from task.lift.config.ik_abs_env_cfg import FrankaCubeLiftEnvCfg

# Gymnasium 환경 레지스트리에 커스텀 환경 등록
# 이렇게 하면 gym.make()로 환경을 생성할 수 있음
gym.register(
    id="Isaac-Lift-Cube-Franka-Custom-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": FrankaCubeLiftEnvCfg,
    },
    disable_env_checker=True,
)

# ============================================================================
# 3. State Machine 상수 정의
# ============================================================================

class GripperState:
    """그리퍼의 상태를 정의하는 클래스"""
    OPEN = 1.0    # 그리퍼 열림 상태
    CLOSE = -1.0  # 그리퍼 닫힘 상태


class PickSmState:
    """Pick and Lift State Machine의 상태들을 정의하는 클래스"""
    REST = 0                    # 대기 상태
    APPROACH_ABOVE_OBJECT = 1   # 오브젝트 위로 접근
    APPROACH_OBJECT = 2         # 오브젝트로 접근
    GRASP_OBJECT = 3            # 오브젝트 잡기
    LIFT_OBJECT = 4             # 오브젝트 들어올리기


class PickSmWaitTime:
    """각 상태에서 전환하기 전 대기 시간을 정의하는 클래스 (초 단위)"""
    REST = 0.2                  # 대기 상태에서 0.2초 대기
    APPROACH_ABOVE_OBJECT = 0.5 # 오브젝트 위 접근에서 0.5초 대기
    APPROACH_OBJECT = 0.6       # 오브젝트 접근에서 0.6초 대기
    GRASP_OBJECT = 0.3          # 오브젝트 잡기에서 0.3초 대기
    LIFT_OBJECT = 1.0           # 오브젝트 들어올리기에서 1.0초 대기

# ============================================================================
# 4. State Machine 구현 클래스
# ============================================================================

class PickAndLiftSm:
    """
    로봇의 태스크 공간에서 오브젝트를 잡고 들어올리는 간단한 State Machine
    
    1. REST: 로봇이 대기 상태
    2. APPROACH_ABOVE_OBJECT: 로봇이 오브젝트 위로 이동
    3. APPROACH_OBJECT: 로봇이 오브젝트로 이동
    4. GRASP_OBJECT: 로봇이 오브젝트를 잡음
    5. LIFT_OBJECT: 로봇이 오브젝트를 원하는 자세로 들어올림
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.01):
        """상태 머신을 초기화합니다.

        Args:
            dt: 환경의 시간 스텝
            num_envs: 시뮬레이션할 환경의 개수
            device: 상태 머신을 실행할 디바이스
            position_threshold: 상태 전환을 위한 위치 오차 임계값
        """
        self.dt = float(dt)  # 시간 스텝
        self.num_envs = num_envs  # 환경 개수
        self.device = device  # 디바이스
        self.position_threshold = position_threshold  # 위치 임계값

        # 상태 머신 변수들 초기화
        # 각 환경의 현재 상태를 REST로 초기화
        self.sm_state = torch.full((self.num_envs,), PickSmState.REST, dtype=torch.int32, device=self.device)
        # 각 환경의 대기 시간을 0으로 초기화
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # 원하는 상태들 초기화
        # 엔드 이펙터의 원하는 자세 (위치 3개 + 쿼터니언 4개 = 7개)
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        # 그리퍼의 원하는 상태 (열림 상태로 초기화)
        self.des_gripper_state = torch.full((self.num_envs, 1), GripperState.OPEN, device=self.device)

        # 오브젝트 위로 접근하기 위한 오프셋 설정
        # [x, y, z, w, x, y, z] 형태: 위치 3개 + 쿼터니언 4개 (w,x,y,z)
        self.offset = torch.tensor([0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0], device=self.device)
        # 쿼터니언 부분만 추출하여 모든 환경에 적용
        self.offset_quat = self.offset[3:].repeat(self.num_envs, 1)
        # 위치 부분만 추출하여 모든 환경에 적용
        self.offset_pos = self.offset[:3].repeat(self.num_envs, 1)

    def reset_idx(self, env_ids: Sequence[int] | None = None):
        """지정된 환경 인덱스에 대해 상태 머신을 리셋합니다."""
        if env_ids is None:
            env_ids = slice(None)  # 모든 환경을 리셋
        self.sm_state[env_ids] = PickSmState.REST  # 상태를 REST로 리셋
        self.sm_wait_time[env_ids] = 0.0  # 대기 시간을 0으로 리셋

    def _quat_mul(self, q1, q2):
        """쿼터니언 곱셈을 수행합니다."""
        # 쿼터니언을 각 성분으로 분해
        w1, x1, y1, z1 = q1.unbind(dim=-1)
        w2, x2, y2, z2 = q2.unbind(dim=-1)
        # 쿼터니언 곱셈 공식 적용
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([w, x, y, z], dim=-1)

    def _quat_apply(self, q, v):
        """쿼터니언 q로 벡터 v를 회전시킵니다."""
        # 쿼터니언 계산
        q_conj = q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=self.device)
        # 벡터를 쿼터니언 형태로 변환 (w=0)
        v_quat = torch.cat([torch.zeros(v.shape[0], 1, device=self.device), v], dim=-1)
        # 쿼터니언 회전 적용: q * v_quat * q_conj
        return self._quat_mul(self._quat_mul(q, v_quat), q_conj)[:, 1:]

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pos: torch.Tensor, robot_data: torch.Tensor) -> torch.Tensor:
        """
        State Machine을 기반으로 원하는 엔드 이펙터 자세와 그리퍼 상태를 계산합니다.
        
        Args:
            ee_pose: 현재 엔드 이펙터 자세 (위치 + 쿼터니언)
            object_pose: 현재 오브젝트 자세 (위치 + 쿼터니언)
            des_object_pos: 원하는 오브젝트 위치
            robot_data: 로봇 데이터
            
        Returns:
            액션 텐서 (위치 + 쿼터니언 + 그리퍼)
        """
        # Isaac Lab은 쿼터니언을 (w, x, y, z) 형태로 사용합니다.
        ee_pos = ee_pose[:, :3]      # 엔드 이펙터 위치
        object_pos = object_pose[:, :3]  # 오브젝트 위치
        object_quat = object_pose[:, 3:]  # 오브젝트 쿼터니언
        
        # 위치로부터 원하는 오브젝트 자세 생성
        des_object_quat = object_quat.clone()  # 같은 방향 유지
        des_object_pose = torch.cat([des_object_pos, des_object_quat], dim=-1)# 물체를 원하는 자세로 들어올릴 때 사용

        # 각 환경에 대해 반복 (현재는 하나만 있지만)
        for i in range(self.num_envs):
            state = self.sm_state[i]
            # --- State Machine 로직 ---
            if state == PickSmState.REST:
                # 대기 상태: 현재 자세 유지, 그리퍼 열기
                self.des_ee_pose[i] = ee_pose[i]
                self.des_gripper_state[i] = GripperState.OPEN
                # 대기 시간이 충분하면 APPROACH_ABOVE_OBJECT 상태로 전환 및 대기 시간 초기화
                if self.sm_wait_time[i] >= PickSmWaitTime.REST:
                    self.sm_state[i] = PickSmState.APPROACH_ABOVE_OBJECT
                    self.sm_wait_time[i] = 0.0

            elif state == PickSmState.APPROACH_ABOVE_OBJECT:
                # 오브젝트 위로 접근: 오브젝트 위치 + 오프셋, ee_pose의 방향은 object_quat으로 유지
                self.des_ee_pose[i, :3] = object_pos[i] + self.offset_pos[i]
                self.des_ee_pose[i, 3:] = object_quat[i]  # 오브젝트 방향 유지

                # 그리퍼 상태 설정 = 열기
                self.des_gripper_state[i] = GripperState.OPEN
                # 위치 오차가 임계값보다 작고, 만약 작다면 대기 시간이 충분하면 APPROACH_OBJECT 상태로 전환하고 대기 시간 초기화
                # ee_pos와 self.des_ee_pose의 차이를 self.position_threshold와 비교
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickSmWaitTime.APPROACH_ABOVE_OBJECT:
                        self.sm_state[i] = PickSmState.APPROACH_OBJECT
                        self.sm_wait_time[i] = 0.0

            
            elif state == PickSmState.APPROACH_OBJECT:
                # 오브젝트 x,y,z 위치로 이동
                self.des_ee_pose[i, :3] = object_pos[i]
                # 방향은 유지
                self.des_ee_pose[i, 3:] = object_quat[i]
                # 그리퍼 상태 설정 = 열기
                self.des_gripper_state[i] = GripperState.OPEN
                # 위치 오차가 임계값보다 작고, 만약 작다면 대기 시간이 충분하면 GRASP_OBJECT 상태로 전환하고 대기 시간 초기화
                # ee_pos와 self.des_ee_pose의 차이를 self.position_threshold와 비교
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickSmWaitTime.APPROACH_OBJECT:
                        self.sm_state[i] = PickSmState.GRASP_OBJECT
                        self.sm_wait_time[i] = 0.0

            elif state == PickSmState.GRASP_OBJECT:
                # 현재 자세 유지
                self.des_ee_pose[i] = self.des_ee_pose[i]
                # 그리퍼 상태 설정 = 닫기
                self.des_gripper_state[i] = GripperState.CLOSE
                # 대기 시간이 충분하면 LIFT_OBJECT 상태로 전환 및 대기 시간 초기화
                if self.sm_wait_time[i] >= PickSmWaitTime.GRASP_OBJECT:
                    self.sm_state[i] = PickSmState.LIFT_OBJECT
                    self.sm_wait_time[i] = 0.0

            # 대기 시간 증가
            self.sm_wait_time[i] += self.dt

            # 월드 좌표계에서 로봇 베이스 좌표계로 변환
            if state != PickSmState.LIFT_OBJECT:
                # 로봇 루트의 월드 좌표계 위치와 쿼터니언
                robot_root_pos_w = robot_data.root_state_w[:, :3]
                robot_root_quat_w = robot_data.root_state_w[:, 3:7]
                
                # 위치와 방향 추출
                action_pos_w = self.des_ee_pose[:, :3]  # 월드 좌표계 위치
                action_quat_w = self.des_ee_pose[:, 3:7]  # 월드 좌표계 쿼터니언 (w,x,y,z)
                
                # 위치를 로봇 베이스 좌표계로 변환
                from isaaclab.utils.math import subtract_frame_transforms
                action_pos_b, action_quat_b = subtract_frame_transforms(
                    robot_root_pos_w, robot_root_quat_w, action_pos_w, action_quat_w
                )
                
                # 로봇 베이스 좌표계에서 액션 재구성
                actions = torch.cat([action_pos_b, action_quat_b, self.des_gripper_state], dim=-1)
            
            else:
                # LIFT_OBJECT 상태에서는 월드 좌표계 사용
                actions = torch.cat([self.des_ee_pose, self.des_gripper_state], dim=-1)

        return actions


# ============================================================================
# 5. 메인 함수
# ============================================================================

def main():
    """메인 함수 - 시뮬레이션의 진입점"""
    # 환경 개수를 1로 고정
    num_envs = 1

    # 환경 설정 파싱
    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Cube-Franka-Custom-v0",
        device=args_cli.device,
        num_envs=num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    
    # 환경 생성
    env = gym.make("Isaac-Lift-Cube-Franka-Custom-v0", cfg=env_cfg)
    
    # 시작 시 환경 리셋
    env.reset()
    print(f"Environment reset. Number of environments: {env.unwrapped.num_envs}")
    
    # 액션 버퍼 생성 (IK 절대 타겟: 위치 + 쿼터니언 + 그리퍼)
    # 형태는 (num_envs, 8)
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0  # 초기 유효한 쿼터니언 (w=1)

    # 원하는 방향 설정 (x축 방향)
    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 1] = 1.0

    # 상태 머신 생성
    pick_sm = PickAndLiftSm(
        dt=env_cfg.sim.dt * env_cfg.decimation,  # 환경 시간 스텝
        num_envs=env.unwrapped.num_envs,
        device=env.unwrapped.device,
        position_threshold=0.01  # 위치 임계값
    )

    # ============================================================================
    # 6. 시뮬레이션 메인 루프
    # ============================================================================
    while simulation_app.is_running():
        # 모든 연산을 추론 모드에서 실행 (메모리 효율성)
        with torch.inference_mode():
            # 환경 스텝 실행
            dones = env.step(actions)[-2]

            # -- 현재 관측값 가져오기 --
            # 엔드 이펙터 자세 (위치, 쿼터니언 wxyz)
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            # 일관성을 위해 로봇 베이스 좌표계 사용
            tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()

            # 오브젝트 자세 (위치, 쿼터니언 wxyz)
            object_data: RigidObjectData = env.unwrapped.scene["object"].data
            object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins

            # 명령 관리자로부터 원하는 오브젝트 위치 가져오기
            desired_position = env.unwrapped.command_manager.get_command("object_pose")[..., :3]
            robot_data = env.unwrapped.scene["robot"].data

            # -- 상태 머신을 진행하여 액션 계산 --
            actions = pick_sm.compute( 
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),  # 엔드 이펙터 자세
                torch.cat([object_position, desired_orientation], dim=-1),     # 오브젝트 자세
                torch.cat([desired_position], dim=-1),                         # 원하는 오브젝트 위치
                robot_data                                                      # 로봇 데이터
                )

            # -- 환경 스텝 진행 --
            obs, rewards, terminated, truncated, info = env.step(actions)

            # 종료 조건 확인
            dones = terminated | truncated

            # -- 리셋 시 상태 머신 리셋 --
            if dones.any():
                # 단일 환경의 경우 dones.nonzero()는 tensor([0])가 됨
                pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))
                print("Environment reset. Resetting state machine.")

    # ============================================================================
    # 7. 정리 작업
    # ============================================================================
    # 환경 종료
    env.close()


# ============================================================================
# 8. 프로그램 실행
# ============================================================================
if __name__ == "__main__":
    # 메인 함수 실행
    main()
    # 시뮬레이션 앱 종료
    simulation_app.close()
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Isaac Lab 환경에서 Franka 로봇을 사용한 큐브 리프팅 태스크 실행 스크립트

이 스크립트는 다음과 같은 기능을 제공합니다:
1. Isaac Lab 시뮬레이션 환경 초기화
2. Franka 로봇과 큐브가 포함된 리프팅 환경 생성
3. 로봇의 역기구학(IK) 기반 동작 제어
4. 시뮬레이션 루프 실행 및 환경 관리
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
# AppLauncher의 명령행 인자들을 파서에 추가 (headless, device, enable_cameras 등)
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Isaac Lab 앱 초기화 - 시뮬레이션 환경을 시작
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

# ============================================================================
# 2. 시뮬레이션 관련 라이브러리 임포트
# ============================================================================
"""Rest everything else."""
import gymnasium as gym
import torch
from collections.abc import Sequence

from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData

import isaaclab_tasks  # noqa: F401
from task.lift.custom_lift_env_cfg_2_4_2 import LiftEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from task.lift.config.joint_pos_env_cfg_2_4_2 import FrankaCubeLiftEnvCfg 

# ============================================================================
# 3. 커스텀 환경 등록
# ============================================================================
# Gymnasium 환경 레지스트리에 커스텀 환경을 등록
# 이렇게 등록하면 gym.make()로 환경을 생성할 수 있음
gym.register(
    id="Isaac-Lift-Cube-Franka-Custom-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": FrankaCubeLiftEnvCfg,
    },
    disable_env_checker=True,
)


def main():
    """
    메인 함수 - 시뮬레이션의 전체 파이프라인 관리
    
    이 함수는 다음 단계로 구성됩니다:
    1. 환경 설정 파싱
    2. 환경 생성 및 초기화
    3. 액션 버퍼 생성
    4. 시뮬레이션 루프 실행
    """
    # ============================================================================
    # 4. 환경 설정 및 초기화
    # ============================================================================
    # 하드코딩된 환경 개수 (학습 시에는 더 많은 환경을 사용할 수 있음)
    num_envs = 1

    # parse configuration
    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Cube-Franka-Custom-v0",
        device=args_cli.device,
        num_envs=num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    
    # 환경 생성 - 실제 시뮬레이션 환경을 생성
    env = gym.make("Isaac-Lift-Cube-Franka-Custom-v0", cfg=env_cfg)
    
    # 환경 리셋 - 초기 상태로 환경을 설정
    env.reset()
    
    # 환경 정보 출력
    print(f"Environment reset. Number of environments: {env.unwrapped.num_envs}")
    
    # ============================================================================
    # 5. 액션 버퍼 생성 및 초기화
    # ============================================================================
    # IK 절대 위치 타겟을 위한 액션 버퍼 생성 (팔 joint 7개+ 그리퍼)
    # 형태: (num_envs, 8) - 7개 joint+ 1개 그리퍼
    if env.action_space.shape is not None:
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    else:
        # action_space.shape가 None인 경우 기본값 사용
        actions = torch.zeros((num_envs, 8), device=args_cli.device)
    
    # 초기 joint 각도 설정
    actions[:, 0] = 1.57  # 0번 joint
    actions[:, 1] = -1.57  # 1번 joint
    
    # # 그리퍼 액션 설정 (7번 인덱스)
    # actions[:, 7] = 1.0  # 그리퍼 열기 (1.0) 또는 닫기 (0.0)

    # ============================================================================
    # 6. 시뮬레이션 메인 루프
    # ============================================================================
    print("Starting simulation loop...")
    
    # 시뮬레이션 앱이 실행 중인 동안 반복
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # 환경에 액션을 적용하고 결과 받기
            # env.step()은 (observation, reward, terminated, truncated, info)를 반환
            step_result = env.step(actions)
            dones = step_result[-2]  # done 상태들 (terminated + truncated) -> 현재 tutorial에서는 사용하지 않음
            
            # 모든 환경이 종료되었는지 확인
            if dones.any():
                print("Some environments finished. Resetting...")
                env.reset()

    # ============================================================================
    # 7. 정리 작업
    # ============================================================================
    # 환경 종료
    env.close()


if __name__ == "__main__":
    # 메인 함수 실행
    main()
    # 시뮬레이션 앱 종료
    simulation_app.close()
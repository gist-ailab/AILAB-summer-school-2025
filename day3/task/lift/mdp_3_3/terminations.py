# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_pickplace_goal(
    env: ManagerBasedRLEnv,
    threshold: float = 0.25,
    threshold_z: float = 0.05,
    object_0_cfg: SceneEntityCfg = SceneEntityCfg("object_0"),
    object_1_cfg: SceneEntityCfg = SceneEntityCfg("object_1"),
    object_2_cfg: SceneEntityCfg = SceneEntityCfg("object_2"),
    bin_cfg: SceneEntityCfg = SceneEntityCfg("bin"),
) -> torch.Tensor:
    """객체를 들어서 특정 위치에 놓는(Pick and Place) 작업의 종료 조건을 정의

    이 함수는 세 개의 객체(object_0, object_1, object_2)가 모두
    목표 지점(bin)의 특정 반경 내에 위치했는지 확인하여 작업 완료 여부를 판단
    
    Args:
        env: 현재 시뮬레이션 환경 객체입니다.
        threshold: 목표 위치(bin)와 객체 간의 수평(x, y) 거리 임계값입니다. 기본값은 0.25입니다.
        threshold_z: 목표 위치(bin)와 객체 간의 수직(z) 거리 임계값입니다. 기본값은 0.07입니다.
        object_0_cfg: 첫 번째 객체의 설정 정보입니다. 기본값은 "object_0"이라는 이름의 SceneEntityCfg입니다.
        object_1_cfg: 두 번째 객체의 설정 정보입니다. 기본값은 "object_1"이라는 이름의 SceneEntityCfg입니다.
        object_2_cfg: 세 번째 객체의 설정 정보입니다. 기본값은 "object_2"라는 이름의 SceneEntityCfg입니다.
        bin_cfg: 객체를 놓을 목표 지점(bin)의 설정 정보입니다. 기본값은 "bin"이라는 이름의 SceneEntityCfg입니다.
    
    Returns:
        각 환경(environment)별로 작업 완료 여부를 나타내는 boolean 값의 Tensor를 반환합니다.
        (True: 작업 완료, False: 작업 미완료)
    """
    # 사용될 객체 추출
    object_0: RigidObject = env.scene[object_0_cfg.name]
    object_1: RigidObject = env.scene[object_1_cfg.name]
    object_2: RigidObject = env.scene[object_2_cfg.name]
    bin: RigidObject = env.scene[bin_cfg.name]
    
    # 각 환경의 종료 여부를 저장할 리스트를 초기화
    dones = []
    # 여러 환경이 동시에 실행될 경우를 대비하여 각 환경에 대해 반복
    for i in range(len(bin.data.root_pos_w)):
        # 목표 지점(bin)과 각 객체 사이의 수평(x, y축) 거리를 계산
        distance_0 = torch.norm(bin.data.root_pos_w[:, :2] - object_0.data.root_pos_w[:, :2], dim=1)
        distance_1 = torch.norm(bin.data.root_pos_w[:, :2] - object_1.data.root_pos_w[:, :2], dim=1)
        distance_2 = torch.norm(bin.data.root_pos_w[:, :2] - object_2.data.root_pos_w[:, :2], dim=1)
        distances = [distance_0, distance_1, distance_2]

        # 목표 지점(bin)과 각 객체 사이의 수직(z축) 높이 차이를 계산
        distance_0_z = torch.abs(bin.data.root_pos_w[:, 2] - object_0.data.root_pos_w[:, 2])
        distance_1_z = torch.abs(bin.data.root_pos_w[:, 2] - object_1.data.root_pos_w[:, 2])
        distance_2_z = torch.abs(bin.data.root_pos_w[:, 2] - object_2.data.root_pos_w[:, 2])
        distances_z = [distance_0_z, distance_1_z, distance_2_z]

        # 현재 환경의 종료 여부를 False로 초기화
        done = False
        # 모든 객체가 수평 거리(threshold) 및 수직 거리(threshold_z) 임계값 이내에 있는지 확인
        # 모든 조건을 만족하면 현재 환경의 작업을 완료 처리합니다.
        if all(d < threshold for d in distances) and all(dz < threshold_z for dz in distances_z):
            done = True 
        # 현재 환경의 종료 상태를 리스트에 추가합니다.
        dones.append(done)

    # 종료 상태 리스트를 PyTorch 텐서로 변환하여 반환합니다.
    return torch.tensor(dones, device=bin.data.root_pos_w.device)

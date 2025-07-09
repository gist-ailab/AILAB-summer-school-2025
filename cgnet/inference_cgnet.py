import torch
import cgnet.utils.utils as utils
import numpy as np
from cgnet.utils.collision_detector import ModelFreeCollisionDetector
import open3d as o3d
from scipy.spatial.transform import Rotation as R

def inference_cgnet(pc, model, device, hand_pose_w, env):
    pc_torch = torch.from_numpy(pc).float().to(device)
    pred = model(pc_torch.unsqueeze(0))
    pred_grasps = pred['pred_grasps'].detach().cpu() #(B, N, 4, 4)
    pred_scores = pred['pred_scores'] # (B, N)
    pred_points = pred['pred_points'].detach().cpu() # (B, N, 3)
    pred_graps_width_bin = pred['pred_width'] # (B, N, 1)

    pred_rot = pred_grasps[:, :, :3, :3] # (B, N, 3, 3)
    pred_trans = pred_grasps[:, :, :3, 3] # (B, N, 3)
    
    sorted_pred_score, sorted_idx = torch.topk(pred_scores.squeeze(), k=2048, largest=True)
    sorted_idx = sorted_idx.detach().cpu()
    sorted_pred_rot = pred_rot[:, sorted_idx, :, :]
    sorted_pred_trans = pred_trans[:, sorted_idx, :]
    sorted_pred_width_bin = pred_graps_width_bin[:, sorted_idx, :]
    
    sorted_pred_rot = sorted_pred_rot.detach().cpu().numpy()[0] #(2048, 3, 3)
    sorted_pred_trans = sorted_pred_trans.detach().cpu().numpy()[0] #(2048, 3)
    sorted_pred_score = sorted_pred_score.detach().cpu().numpy()
    sorted_pred_width_bin = sorted_pred_width_bin.detach().cpu().numpy()[0, :, 0]
    
    
    #* visualize non-filtered grasps, this check model inference is correctely working
    # fix for numpy.float issue in graspnetAPI
    np.float = float
    np.float_ = np.float64

    # from graspnetAPI.graspnetAPI import GraspNet, GraspNetEval, GraspGroup, Grasp
    from graspnetAPI.graspnetAPI import GraspGroup
    #from graspnetAPI import GraspGroup
    
    g_array = []
    for i in range(len(sorted_idx)):
        score = sorted_pred_score[i]
        width = sorted_pred_width_bin[i]
        rot = sorted_pred_rot[i] # (3, 3)
        trans = sorted_pred_trans[i].reshape(-1) # (3,)
        
        trans = trans.reshape(1, 3) - (rot @ np.array([[0.00,0,0]]).reshape(3, 1)).reshape(1,3)
        trans = trans.reshape(-1)
        
        rot = rot.reshape(-1)
        g_array.append([score, width, 0.02, 0.02, *rot, *trans, -1])
    
    g_array = np.array(g_array)
    gg = GraspGroup(g_array)
    
    # check collisoin
    mfcdetector = ModelFreeCollisionDetector(pc, voxel_size=0.005)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=0.01)
    gg = gg[~collision_mask]
    gg = gg.nms()
    gg = gg.sort_by_score()
    if gg.__len__() > 10:
        gg = gg[:10]
    gg_vis = gg[:1]
    gg_vis_trans = gg[:1]
    gg_vis_trans.translations += (gg_vis_trans.rotation_matrices[0] @ np.array([[0.1, 0, 0]]).reshape(3, 1)).reshape(1,3)
    grippers = gg_vis.to_open3d_geometry_list()
    grippers_trans = gg_vis_trans.to_open3d_geometry_list()
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(pc)
    
    rot_top1 = gg.rotation_matrices[0]           # (3, 3)
    trans_top1 = gg.translations[0]              # (3,)
    width_top1 = gg.widths[0]                    # scalar

    # Pregrasp offset 적용
    tcp_offset = np.array([0.05, 0, 0])          # 5cm
    trans_top1_tcp = trans_top1 - rot_top1 @ tcp_offset

    # ======================== 1) Origin & Base Pose ========================
    origin = env.unwrapped.scene.env_origins.cpu().numpy()

    rs = env.unwrapped.scene["robot"].data.root_state_w[0]
    base_pos_w = rs[:3].cpu().numpy() - origin
    base_quat_w = rs[3:7].cpu().numpy()  # (x, y, z, w)

    base_rot = R.from_quat(base_quat_w)
    T_w_base = np.eye(4)
    T_w_base[:3, :3] = base_rot.as_matrix()
    T_w_base[:3, 3] = base_pos_w

    # ======================== 2) Handeye Camera Pose ========================
    # offset_pos = np.array([0.1, 0.035, 0.0])
    offset_pos = np.array([0.1, 0.035, 0.0])
    offset_quat_wxyz = np.array([0.70710678, 0.0, 0.0, 0.70710678])  # (w, x, y, z)
    offset_rot = R.from_quat([
        offset_quat_wxyz[1], offset_quat_wxyz[2], offset_quat_wxyz[3], offset_quat_wxyz[0]
    ])  # (x, y, z, w)

    # offset_rot = R.from_quat(offset_quat_wxyz)
    T_hand_cam = np.eye(4)
    T_hand_cam[:3, :3] = offset_rot.as_matrix()
    T_hand_cam[:3, 3] = offset_pos

    hand_pos_w = hand_pose_w[0, 0:3].cpu().numpy()
    hand_quat_w = hand_pose_w[0, 3:7].cpu().numpy()  # (x, y, z, w)
    
    # rotate hand z축으로 180도
    z_rotation = R.from_euler('z', 180, degrees=True)
    hand_rot = R.from_quat(hand_quat_w) * z_rotation
    T_w_hand = np.eye(4)
    T_w_hand[:3, :3] = hand_rot.as_matrix()
    T_w_hand[:3, 3] = hand_pos_w

    # World to EE frame 시각화
    ee_frame_sensor = env.unwrapped.scene["ee_frame"]
    ee_pos_w = ee_frame_sensor.data.target_pos_w[0, :].cpu().numpy()
    ee_quat_w = ee_frame_sensor.data.target_quat_w[0, :].cpu().numpy()  # (x, y, z, w) or (w, x, y, z) 확인 필요

    # 쿼터니언 → 회전행렬 변환 (scipy는 (x, y, z, w) 순서)
    ee_rot_w = R.from_quat(ee_quat_w)
    ee_rot_w = ee_rot_w * R.from_euler('z', 180, degrees=True)  # z축 180도 회전 추가
    ee_rot_w_mat = ee_rot_w.as_matrix()

    T_w_ee_isaac = np.eye(4)
    T_w_ee_isaac[:3, :3] = ee_rot_w_mat
    T_w_ee_isaac[:3, 3] = ee_pos_w

    # 카메라 pose 계산
    T_w_cam = T_w_hand @ T_hand_cam

    # ======================== 3) PointCloud & Grippers → world ========================
    pc_world = (T_w_cam[:3, :3] @ pc.T).T + T_w_cam[:3, 3]
    pc_o3d.points = o3d.utility.Vector3dVector(pc_world)

    # Grippers → world
    for g in grippers:
        g.transform(T_w_cam)
    for g in grippers_trans:
        g.transform(T_w_cam)

    # ======================== 4) Grasp Pose Axes ========================
    # GraspNetAPI 결과를 World 좌표계로 변환
    rot_world = T_w_cam[:3, :3] @ rot_top1
    trans_world = T_w_cam[:3, :3] @ trans_top1_tcp + T_w_cam[:3, 3]
    # trans_world_pre = T_w_cam[:3, :3] @ trans_top1_pre_tcp + T_w_cam[:3, 3]

    # Grasp pose (world)
    T_w_grasp = np.eye(4)
    T_w_grasp[:3, :3] = rot_world
    T_w_grasp[:3, 3] = trans_world

    # Pregrasp pose (world)
    T_w_pregrasp = np.eye(4)
    T_w_pregrasp[:3, :3] = rot_world

    # 시각화용 좌표축 생성
    grasp_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    grasp_axis.transform(T_w_grasp)

    # ======================== 4-1) EE 좌표축으로 변환 ========================
    reorder = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ])
    
    rot_ee = rot_world @ reorder.T
    r = R.from_matrix(rot_ee)
    euler = r.as_euler('xyz', degrees=True)
    euler[0] = -euler[0]
    euler[1] = 180+euler[1]
    rot_ee = R.from_euler('xyz', euler, degrees=True).as_matrix()
    print("EE frame Euler angles (deg):", euler)
    z_180_rot = R.from_euler('z', 180, degrees=True).as_matrix()
    rot_ee = rot_ee @ z_180_rot  # 또는 rot_ee = y_180_rot @ rot_ee (순서에 따라 다름)
    
    trans_ee = trans_world  # 위치는 그대로 (필요시 변환)

    # EE 프레임 기준 grasp pose
    T_w_grasp_ee = np.eye(4)
    T_w_grasp_ee[:3, :3] = rot_ee
    T_w_grasp_ee[:3, 3] = trans_ee

    # EE 프레임 기준 grasp pose 시각화
    grasp_ee_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
    grasp_ee_axis.transform(T_w_grasp_ee)

    # ======================== 5) Coordinate Frames ========================
    # World frame
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    
    # Camera frame
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    camera_frame.transform(T_w_cam)
    
    # Robot base frame
    robot_base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    robot_base_frame.transform(T_w_base)
    
    # Robot hand frame (hand_pose_w 기준)
    robot_hand_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12)
    robot_hand_frame.transform(T_w_hand)

    # IsaacSim EE frame (ee_frame_sensor 기준)
    ee_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    ee_frame.transform(T_w_ee_isaac)

    # # ======================== 6) 최종 시각화 ========================
    o3d.visualization.draw_geometries([
        pc_o3d,
        # grasp_axis,  # GraspNetAPI 축
        # grasp_ee_axis,  # EE 프레임 기준 grasp pose
        *grippers,
        
    #     # 좌표 프레임들
    #     # world_frame,
    #     camera_frame,          # 빨간색: 카메라 frame
    #     robot_base_frame,      # 초록색: 로봇 base frame
    #     robot_hand_frame,      # 파란색: 로봇 hand frame (hand_pose_w)
    #     ee_frame,        # 주황색: IsaacSim EE frame (ee_frame_sensor)
    ])
    
    # 최종적으로 world-to-base로 변환
    trans_ee_base = trans_ee - np.array([0, 0, 0.5])
    
    return rot_ee, trans_ee_base, width_top1


# AILAB-summer-school-2025
제12회 인공지능 및 로보틱스 여름학교 (AI &amp; Robotics Summer School 2025)
<br/>
<br/>

*참고용 repo 입니다.(이번 여름학교에서는 사전에 배포드린 docker 기반 환경 구성 메뉴얼을 따라주세요.) 
<br/>
<br/>

# Download Data
아래의 링크에서 ```AILAB-summer-school-2025/data``` 에 아래의 데이터 트리 구조를 참고해서 데이터를 다운받아주세요.<br/><br/>
**Google Drive Link**: https://drive.google.com/drive/folders/1UxQlTQ8C91Eq6qDCELKuXrTaPHWjxTCZ?usp=drive_link
```
./data
├── assets
│     └── basket
│       └── basket.usd
│     └── ycb_usd
│        └── ycb
│            └── 006_mustard_bottle
│                ├── config.yaml
│                ├── final.usd
│                └── textures
│                    └── texture_map.png
│            └── ...
└── checkpoint
    ├── contact_grasp_ckpt
    │   └── ckpt-iter-60000_gc6d.pth
    └── maskrcnn_ckpt
       └── maskrcnn_trained_model_refined.pth

```
# Local Installation
로컬 컴퓨터에 설치 필요시 참고
1. Isaac Lab 설치(Installation documentation) - https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
2. 여름학교 강의 내 추가 라이브러리 설치
```
# torch version(Isaac Lab 설치시 설치됨.)
torch                     2.5.1+cu124                    pypi_0    pypi
torchvision               0.20.1                   pypi_0    pypi

# graspnetAPI
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install .

# Pointnet_Pointnet2_pytorch
git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git

# CLIP
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# Contact-GraspNet
git clone https://github.com/NVlabs/contact_graspnet.git

pip install pycocotools open3d==0.19.0 opencv-python==4.11 scipy 
```
---
### 참고 repo

https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git <br/>
https://github.com/graspnet/graspnetAPI.git <br/>
https://github.com/NVlabs/contact_graspnet.git <br/>

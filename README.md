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
---
### 참고 repo

https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git <br/>
https://github.com/graspnet/graspnetAPI.git <br/>
https://github.com/NVlabs/contact_graspnet.git <br/>

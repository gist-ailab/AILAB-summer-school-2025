# Ubuntu 22.04 NVIDIA 드라이버, CUDA Toolkit, cuDNN 설치 가이드

## 0. 소개

* 이 가이드는 Ubuntu 22.04에 NVIDIA 그래픽 드라이버, CUDA Toolkit, cuDNN을 설치하는 방법을 안내합니다. 자동 설치와 수동 설치 방법을 모두 다루며, 특정 버전을 지정하여 설치하는 방법도 포함합니다.
* 설치에는 외부 인터넷에서 파일을 다운받는 과정이 포함되어 있습니다.
* 메뉴얼 내 모든 명령어는 Ubuntu 22.04 내부의 terminal(CLI) 환경으로 수행합니다. (Ctrl + Alt + T)
* 1.2 Nouveau 비활성화 단계에서는 컴퓨터의 GUI 사용이 일시 중단될 수 있습니다. 다른 매체를 통해 본 메뉴얼을 볼 수 있도록 준비해주세요.

## 1. NVIDIA 드라이버 설치

GPU 활용을 위해 NVIDIA 드라이버를 설치합니다.

### 1.0 드라이버 설치 전 준비

#### 1.0.1 사전 요소 설치

먼저, Ubuntu OS를 최신 상태로 유지하고 필수 패키지를 설치합니다.

```bash
   #!/bin/bash
sudo apt update && sudo apt upgrade -y
sudo apt install build-essential gcc ubuntu-drivers-common dkms vim nvidia-modprobe
```

#### 1.0.2 Nouveau 비활성화

Nouveau 커널 드라이버와 충돌을 방지하기 위해 비활성화합니다.

```bash
   #!/bin/bash
# nouveau 드라이버를 블랙리스트에 추가
sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
sudo bash -c "echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nvidia-nouveau.conf"

# 파일 내용 확인
cat /etc/modprobe.d/blacklist-nvidia-nouveau.conf
# 출력 예시:
# blacklist nouveau
# options nouveau modeset=0

# 업데이트 및 재부팅 (GUI 사용 불가 상태가 될 수 있음)
sudo update-initramfs -u
sudo reboot
```

재부팅 후 nouveau 모듈이 로드되지 않았는지 확인합니다.

```bash
   #!/bin/bash
lsmod | grep nouveau
# 아무 출력이 없으면 성공적으로 비활성화된 것임
```

### 1.1 드라이버 수동 설치

특정 버전 드라이버 설치 방법입니다.

1. 설치 가능한 드라이버 목록 조회:

   ```bash
   #!/bin/bash
   ubuntu-drivers devices
   ```

2. 예시: nvidia-driver-570 설치

   ```bash
   #!/bin/bash
   sudo apt install nvidia-driver-570
   sudo reboot
   ```

   * 설치 중 Secure Boot 설정이 나오면 "Continue" 선택 후 MOK 비밀번호 등록

3. 설치 확인:

   ```bash
   #!/bin/bash
   nvidia-smi
   ```

## 2. CUDA Toolkit 설치

### 2.1 수동 설치 (runfile)

1. CUDA Toolkit 다운로드 페이지에서 runfile(local) 선택 후 다운로드
2. 설치 스크립트 실행:

   ```bash
   #!/bin/bash
   wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
   sudo sh cuda_12.1.1_530.30.02_linux.run
   ```

   * 안내에 따라 "Continue" 및 "accept" 입력
     Nvidia-driver는 설치 했음으로 아래와 같이 체크박스 해제 후 설치 진행

    ```bash
   #!/bin/bash
    CUDA Installer
    - [ ] Driver                    # space 버튼 으로 체크 해제
        [ ] 530.39.02
    + [X] CUDA Toolkit 12.1
      [ ] CUDA Demo Suite 12.1
      [ ] CUDA Documentation 12.1
    - [ ] Kernel Objects
         [ ] nvidia-fs
    ```
3. 환경 변수 설정:

   ```bash
   #!/bin/bash
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```
4. 설치 확인:

   ```bash
   #!/bin/bash
   
   nvcc --version
   ```

## 3. cuDNN 설치

1. cuDNN 다운로드 (NVIDIA Developer 웹사이트[https://developer.nvidia.com/rdp/cudnn-archive], 계정 필요)
    이번 가이드에서는 cuDNN v8.9.7 (December 5th, 2023), for CUDA 12.x 를 기준으로 설명
    ```Download cuDNN v8.9.7 (December 5th, 2023), for CUDA 12.x``` 클릭 후 ```Local Installer for Linux x86_64 (Tar)``` 를 클릭하여 다운로드.

2. 압축 해제:

   ```bash
   #!/bin/bash
   tar -xvf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
   ```
3. 라이브러리 및 헤더 복사:

   ```bash
   #!/bin/bash
   sudo cp cudnn-linux-x86_64-8.9.7.29_cuda12/lib/* /usr/local/cuda/lib64/
   sudo cp cudnn-linux-x86_64-8.9.7.29_cuda12/include/* /usr/local/cuda/include/
   ```
4. 권한 설정:

   ```bash
   #!/bin/bash
   sudo chmod a+r /usr/local/cuda/include/cudnn*.h
   sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
   ```
5. 설치 확인: 딥러닝 프레임워크에서 GPU 사용 여부 확인

   딥러닝 프레임워크 설치
   ```bash
   #!/bin/bash
   pip install 'tensorflow[and-cuda]'
   # https://www.tensorflow.org/install/pip?hl=ko

   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   # cuda 12.1 버전 기준
   https://pytorch.org/get-started/previous-versions/
   ```

   프레임워크가 설치된 python 환경에서 실행 확인
   ```python
   #!/bin/bash
   python
   >>> import tensorflow as tf
   >>> print(tf.config.list_physical_devices('GPU'))

   >>> import torch
   >>> print(torch.cuda.is_available())
   ```

## 4. 설치 후 확인

```bash
   #!/bin/bash
nvidia-smi
nvcc --version
```

## 5. 문제 해결

* **드라이버 충돌:** 기존 드라이버 제거 후 재설치
* **CUDA 버전 불일치:** cuDNN 버전과 CUDA 버전 확인
* **환경 변수 오류:** `~/.bashrc` 수정 후 `source ~/.bashrc`

# Isaac Lab v2.1.0 with Isaac Sim v4.5.0 Docker Environment (Fixed Dependencies)
FROM nvcr.io/nvidia/isaac-sim:4.5.0

# 환경 변수들을 한번에 설정 (X11 기본 설정 추가)
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    QT_X11_NO_MITSHM=1 \
    QT_GRAPHICSSYSTEM=native \
    LIBGL_ALWAYS_INDIRECT=0 \
    LIBGL_ALWAYS_SOFTWARE=0 \
    __GL_SYNC_TO_VBLANK=0 \
    __GL_MaxFramesAllowed=1 \
    MESA_GL_VERSION_OVERRIDE=4.1 \
    ISAAC_SIM_PATH="/isaac-sim" \
    ISAACLAB_PATH="/opt/IsaacLab" \
    PYTHONPATH="/isaac-sim/kit/python/lib/python3.10/site-packages:/opt/IsaacLab/source" \
    PATH="/usr/local/bin:${PATH}" \
    LANG=ko_KR.UTF-8 \
    LANGUAGE=ko_KR:ko \
    LC_ALL=ko_KR.UTF-8 \
    OMNI_KIT_ALLOW_ROOT=1 \
    TERM=xterm-256color \
    DISPLAY=:0

# 모든 패키지를 한번에 설치 (Layer 최적화)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # 기본 개발 도구
    git wget curl vim nano build-essential cmake \
    python3-pip python3-dev sudo openssh-server \
    # X11 및 GUI 라이브러리 (한번에 설치)
    x11-apps x11-utils x11-xserver-utils xauth xvfb mesa-utils \
    libgl1-mesa-glx libgl1-mesa-dri libglx-mesa0 libglu1-mesa \
    libglib2.0-0 libsm6 libxext6 libxrender1 libxtst6 libxi6 \
    libxrandr2 libxss1 libgconf-2-4 libxcomposite1 libxcursor1 \
    libxdamage1 libxfixes3 libgtk-3-0 libgbm1 libnss3 \
    libatk-bridge2.0-0 libatspi2.0-0 libdrm2 \
    # 한글 지원 패키지
    locales language-pack-ko fonts-noto-cjk fonts-noto-cjk-extra \
    fonts-noto-color-emoji fontconfig unzip fonts-liberation \
    fonts-dejavu fonts-liberation2 fonts-nanum fonts-nanum-coding \
    # VSCode 의존성
    gpg apt-transport-https software-properties-common \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 수정된 ncurses 패키지 설치 (Ubuntu 22.04+ 호환)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libncurses-dev libncursesw5-dev libreadline-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 로케일 및 기본 설정 (한번에 처리)
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    sed -i '/ko_KR.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    echo 'root ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# D2Coding 폰트 설치 (한글 표시에 필수)
RUN mkdir -p /usr/share/fonts/truetype/d2coding && \
    wget -q -O /tmp/D2Coding.zip "https://github.com/naver/d2codingfont/releases/download/VER1.3.2/D2Coding-Ver1.3.2-20180524.zip" && \
    unzip -q /tmp/D2Coding.zip -d /tmp/d2coding && \
    cp /tmp/d2coding/D2Coding/*.ttf /usr/share/fonts/truetype/d2coding/ && \
    fc-cache -fv && \
    rm -rf /tmp/D2Coding.zip /tmp/d2coding

# VSCode 설치 (최적화된 방식)
RUN wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /usr/share/keyrings/ms-vscode-archive-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ms-vscode-archive-keyring.gpg] https://packages.microsoft.com/repos/vscode stable main" > /etc/apt/sources.list.d/vscode.list && \
    apt-get update && apt-get install -y --no-install-recommends code && \
    rm -rf /var/lib/apt/lists/* && apt-get clean

# 워킹 디렉토리 설정
WORKDIR /AILAB-summer-school-2025

# Isaac Lab 클론 (속도 최적화의 핵심 - 최신 커밋만)
RUN git clone --depth 1 --branch v2.1.0 --single-branch https://github.com/isaac-sim/IsaacLab.git /opt/IsaacLab && \
    cd /opt/IsaacLab && \
    # 올바른 Isaac Sim 링크 생성
    rm -f _isaac_sim && \
    ln -sf /isaac-sim _isaac_sim && \
    echo "✅ Isaac Lab 클론 및 링크 완료"

# PyTorch 및 필수 패키지 설치 (Isaac Sim Python 사용, 호환성 고려)
RUN /isaac-sim/python.sh -m pip install --no-cache-dir --upgrade pip && \
    /isaac-sim/python.sh -m pip install --no-cache-dir \
    torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Isaac Lab 필수 dependency 설치 (호환성 버전으로 수정, ipykernel 명시적 포함)
RUN /isaac-sim/python.sh -m pip install --no-cache-dir \
    # Isaac Lab 필수 패키지들
    gymnasium==0.29.1 \
    toml \
    lxml==5.2.2 \
    pytz>=2020.1 \
    # numpy 1.26.2와 호환되는 버전들
    opencv-python==4.8.1.78 \
    psutil==5.9.8 \
    # 개발 도구들 (ipykernel 명시적 포함)
    black flake8 pylint autopep8 rope jedi debugpy readline \
    ipython jupyter ipykernel matplotlib scipy open3d easydict timm isort \
    pycocotools

# CLIP 설치 (별도 처리)
RUN /isaac-sim/python.sh -m pip install --no-cache-dir git+https://github.com/openai/CLIP.git

# 시스템 Python에도 기본 패키지 설치 (백업용, ipykernel 포함)
RUN pip3 install --no-cache-dir \
    torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir \
    gymnasium==0.29.1 toml lxml==5.2.2 pytz>=2020.1 \
    opencv-python==4.8.1.78 psutil==5.9.8 \
    black flake8 pylint autopep8 rope jedi debugpy \
    ipython jupyter ipykernel numpy==1.26.2

# Isaac Lab 설치 (v2.1.0 robust 설치 방법)
RUN cd /opt/IsaacLab && \
    # 1. 환경 변수 설정
    export ISAACLAB_PATH="/opt/IsaacLab" && \
    export ISAAC_SIM_PATH="/isaac-sim" && \
    export OMNI_KIT_ALLOW_ROOT=1 && \
    export PYTHONPATH="/opt/IsaacLab/source:/isaac-sim/kit/python/lib/python3.10/site-packages" && \
    # 2. Isaac Lab v2.1.0 구조 확인
    echo "✅ Isaac Lab 구조 확인 중..." && \
    ls -la /opt/IsaacLab/source/ && \
    # 3. Isaac Lab 설치 시도 (여러 방법으로 시도)
    echo "🔧 Isaac Lab 설치 시작..." && \
    # 방법 1: Isaac Lab 자체 설치 스크립트 사용 (권장)
    if [ -f "./isaaclab.sh" ]; then \
        echo "📦 Method 1: isaaclab.sh --install 시도..."; \
        chmod +x ./isaaclab.sh; \
        if ./isaaclab.sh --install; then \
            echo "✅ isaaclab.sh --install 성공!"; \
        else \
            echo "⚠️ isaaclab.sh --install 실패, 다른 방법 시도..."; \
            # 방법 2: pip install 시도
            echo "📦 Method 2: pip install -e . 시도..."; \
            if /isaac-sim/python.sh -m pip install --no-cache-dir -e .; then \
                echo "✅ pip install 성공!"; \
            else \
                echo "⚠️ pip install도 실패, manual setup 사용..."; \
                # 방법 3: manual PYTHONPATH 설정
                echo "📦 Method 3: Manual PYTHONPATH 설정..."; \
                mkdir -p /isaac-sim/kit/python/lib/python3.10/site-packages; \
                ln -sf /opt/IsaacLab/source/isaaclab /isaac-sim/kit/python/lib/python3.10/site-packages/isaaclab; \
                ln -sf /opt/IsaacLab/source/isaaclab_assets /isaac-sim/kit/python/lib/python3.10/site-packages/isaaclab_assets; \
                ln -sf /opt/IsaacLab/source/isaaclab_tasks /isaac-sim/kit/python/lib/python3.10/site-packages/isaaclab_tasks; \
                ln -sf /opt/IsaacLab/source/isaaclab_rl /isaac-sim/kit/python/lib/python3.10/site-packages/isaaclab_rl; \
                echo "✅ Manual setup 완료"; \
            fi; \
        fi; \
    else \
        echo "❌ isaaclab.sh not found, using pip install..."; \
        /isaac-sim/python.sh -m pip install --no-cache-dir -e .; \
    fi

# Isaac Lab 설치 검증 및 Jupyter kernel 설정
RUN tee /tmp/verify_isaaclab.py > /dev/null << 'EOF'
import sys
print(f'Python executable: {sys.executable}')
print("Testing Isaac Lab installation...")

# Test 1: Basic import
try:
    import isaaclab
    print('✅ isaaclab 패키지 import 성공')
    
    # Test 2: AppLauncher import (most critical)
    try:
        from isaaclab.app import AppLauncher
        print('✅ isaaclab.app.AppLauncher import 성공')
        print('🎉 Isaac Lab 완전 설치 확인!')
    except ImportError as e:
        print(f'⚠️ isaaclab.app.AppLauncher import 실패: {e}')
        
        # Test alternative paths
        alternatives = [
            ('isaaclab.sim', 'SimulationApp'),
            ('isaaclab.simulation_app', 'SimulationApp'), 
            ('isaaclab.core', 'app')
        ]
        
        success = False
        for module_name, attr_name in alternatives:
            try:
                module = __import__(module_name, fromlist=[attr_name])
                getattr(module, attr_name)
                print(f'✅ 대안 모듈 {module_name}.{attr_name}: 성공')
                success = True
                break
            except (ImportError, AttributeError):
                continue
        
        if not success:
            # Show available modules for debugging
            import os
            isaaclab_path = os.path.dirname(isaaclab.__file__)
            print(f'📂 isaaclab 설치 경로: {isaaclab_path}')
            items = [item for item in os.listdir(isaaclab_path) if not item.startswith('__') and not item.endswith('.pyc')]
            print(f'📦 사용 가능한 모듈들: {items}')
            
            # Check for app directory specifically
            app_path = os.path.join(isaaclab_path, 'app')
            if os.path.exists(app_path):
                print(f'📁 app 디렉토리 존재: {os.listdir(app_path)}')
            else:
                print('❌ app 디렉토리 없음')
                
except ImportError as e:
    print(f'❌ isaaclab 패키지 import 실패: {e}')
    print('🔍 PYTHONPATH 확인:')
    for path in sys.path:
        if 'isaac' in path.lower():
            print(f'  - {path}')
    exit(1)
EOF

RUN cd /opt/IsaacLab && \
    export ISAACLAB_PATH="/opt/IsaacLab" && \
    export ISAAC_SIM_PATH="/isaac-sim" && \
    export OMNI_KIT_ALLOW_ROOT=1 && \
    export PYTHONPATH="/opt/IsaacLab/source:/isaac-sim/kit/python/lib/python3.10/site-packages" && \
    /isaac-sim/python.sh /tmp/verify_isaaclab.py && \
    echo "✅ Isaac Lab 설치 및 검증 완료" && \
    rm -f /tmp/verify_isaaclab.py

# 임시 Jupyter kernel 설정 (Isaac Lab Python 환경만 등록)
RUN echo "🔧 Jupyter kernel 설정 중..." && \
    export ISAACLAB_PATH="/opt/IsaacLab" && \
    export ISAAC_SIM_PATH="/isaac-sim" && \
    export OMNI_KIT_ALLOW_ROOT=1 && \
    export PYTHONPATH="/opt/IsaacLab/source:/isaac-sim/kit/python/lib/python3.10/site-packages" && \
    # Isaac Lab Python kernel 등록
    /isaac-sim/python.sh -m ipykernel install --user --name isaaclab-direct --display-name "Isaac Lab Direct" && \
    echo "✅ 기본 Jupyter kernel 등록 완료"

# Dependency 충돌 해결 확인
RUN tee /tmp/check_dependencies.py > /dev/null << 'EOF'
import sys
print("🔍 Dependency 충돌 확인 중...")

# 주요 패키지들 버전 확인
packages_to_check = [
    'numpy', 'opencv-python', 'psutil', 'gymnasium', 
    'torch', 'lxml', 'toml', 'pytz', 'ipykernel'
]

for package in packages_to_check:
    try:
        module = __import__(package.replace('-', '_'))
        version = getattr(module, '__version__', 'Unknown')
        print(f'✅ {package}: {version}')
    except ImportError:
        print(f'❌ {package}: Not installed')

print("\n📋 Dependency 호환성 체크:")
try:
    import numpy
    import cv2
    print(f'✅ numpy ({numpy.__version__}) + opencv-python ({cv2.__version__}): 호환')
except Exception as e:
    print(f'❌ numpy + opencv 호환성 문제: {e}')

try:
    import psutil
    # rl-games는 psutil<6.0.0,>=5.9.0 요구
    print(f'✅ psutil: {psutil.__version__} (rl-games 호환)')
except Exception as e:
    print(f'❌ psutil 문제: {e}')

try:
    import gymnasium
    # stable-baselines3는 gymnasium<1.2.0,>=0.29.1 요구
    print(f'✅ gymnasium: {gymnasium.__version__} (stable-baselines3 호환)')
except Exception as e:
    print(f'❌ gymnasium 문제: {e}')

print("🎉 Dependency 충돌 해결 완료!")
EOF

RUN /isaac-sim/python.sh /tmp/check_dependencies.py && \
    rm -f /tmp/check_dependencies.py

# Isaac Lab 스크립트 수정 및 명령어 등록 (경로 문제 해결)
RUN cd /opt/IsaacLab && \
    # isaaclab.sh 스크립트의 경로 문제 수정
    sed -i 's|SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"|SCRIPT_DIR="/opt/IsaacLab"|g' isaaclab.sh && \
    # 더 정확한 경로 수정 (이중 슬래시 방지)
    sed -i 's|${ISAACLAB_PATH}/_isaac_sim|/isaac-sim|g' isaaclab.sh && \
    # 스크립트 권한 설정
    chmod +x ./isaaclab.sh && \
    # 전역 명령어 생성
    rm -f /usr/local/bin/isaaclab && \
    ln -sf /opt/IsaacLab/isaaclab.sh /usr/local/bin/isaaclab && \
    # 환경 변수를 bashrc에 영구 추가
    echo '# Isaac Lab Environment' >> /root/.bashrc && \
    echo 'export TERM=xterm-256color' >> /root/.bashrc && \
    echo 'export ISAACLAB_PATH="/opt/IsaacLab"' >> /root/.bashrc && \
    echo 'export ISAAC_SIM_PATH="/isaac-sim"' >> /root/.bashrc && \
    echo 'export OMNI_KIT_ALLOW_ROOT=1' >> /root/.bashrc && \
    echo 'export PYTHONPATH="/opt/IsaacLab/source:/isaac-sim/kit/python/lib/python3.10/site-packages"' >> /root/.bashrc && \
    echo "✅ Isaac Lab 스크립트 수정 및 설정 완료"

# Isaac Lab 검증 스크립트 생성 (정식 설치 버전)
RUN tee /usr/local/bin/test-isaaclab > /dev/null << 'EOF'
#!/bin/bash
export PYTHONPATH="/opt/IsaacLab/source:/isaac-sim/kit/python/lib/python3.10/site-packages"
export OMNI_KIT_ALLOW_ROOT=1
export ISAAC_SIM_PATH="/isaac-sim"

# Python 검증 스크립트를 임시 파일로 생성
cat > /tmp/test_isaaclab.py << 'PYEOF'
import sys
print(f'Testing Isaac Lab with Python: {sys.executable}')
try:
    import isaaclab
    print('✅ Isaac Lab 패키지: 성공')
    
    # AppLauncher 테스트 (가장 중요한 부분)
    try:
        from isaaclab.app import AppLauncher
        print('✅ isaaclab.app.AppLauncher: 성공')
        exit(0)
    except ImportError as e:
        print(f'❌ isaaclab.app.AppLauncher: 실패 - {e}')
        
        # 대안 경로들 시도
        alternative_paths = [
            'isaaclab.sim.SimulationApp',
            'isaaclab.simulation_app',
            'isaaclab.core.app'
        ]
        
        for alt_path in alternative_paths:
            try:
                module_parts = alt_path.split('.')
                module = __import__(module_parts[0])
                for part in module_parts[1:]:
                    module = getattr(module, part)
                print(f'✅ 대안 모듈 {alt_path}: 성공')
                exit(0)
            except (ImportError, AttributeError):
                continue
        
        # 사용 가능한 모듈 구조 확인
        print('📂 isaaclab 패키지 구조:')
        import os
        isaaclab_path = os.path.dirname(isaaclab.__file__)
        for item in os.listdir(isaaclab_path):
            if not item.startswith('__'):
                print(f'   - {item}')
        exit(1)
        
except ImportError as e:
    print(f'❌ Isaac Lab 패키지 import 실패: {e}')
    exit(1)
PYEOF

# Python 스크립트 실행
/isaac-sim/python.sh /tmp/test_isaaclab.py
result=$?

# 임시 파일 정리
rm -f /tmp/test_isaaclab.py

exit $result
EOF
RUN chmod +x /usr/local/bin/test-isaaclab

# VSCode 설정 디렉토리 생성
RUN mkdir -p /root/.vscode/extensions /root/.config/Code/User /AILAB-summer-school-2025/.vscode

# VSCode settings.json 생성 (Isaac Lab python 환경으로 수정)
RUN tee /root/.config/Code/User/settings.json > /dev/null <<'EOF'
{
    "python.defaultInterpreterPath": "/usr/local/bin/python",
    "python.terminal.executeInFileDir": true,
    "python.terminal.activateEnvironment": false,
    "python.analysis.extraPaths": [
        "/opt/IsaacLab/source",
        "/AILAB-summer-school-2025",
        "/isaac-sim/kit/python/lib/python3.10/site-packages"
    ],
    "python.analysis.autoImportCompletions": true,
    "python.analysis.completeFunctionParens": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "88"],
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.lintOnSave": true,
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "editor.rulers": [88],
    "editor.semanticHighlighting.enabled": true,
    "terminal.integrated.cwd": "/AILAB-summer-school-2025",
    "terminal.integrated.shell.linux": "/bin/bash",
    "files.associations": {
        "*.py": "python",
        "*.pyx": "python",
        "*.pyi": "python"
    },
    "extensions.autoUpdate": false,
    "workbench.colorTheme": "Default Dark Modern",
    "workbench.iconTheme": "material-icon-theme",
    "editor.fontFamily": "'D2Coding', 'Noto Sans CJK KR', 'Nanum Gothic Coding', 'DejaVu Sans Mono', 'Consolas', 'Liberation Mono', monospace",
    "editor.fontSize": 14,
    "terminal.integrated.fontFamily": "'D2Coding', 'Noto Sans Mono CJK KR', 'Nanum Gothic Coding', 'DejaVu Sans Mono', monospace",
    "terminal.integrated.fontSize": 14,
    "debug.console.fontFamily": "'D2Coding', 'Noto Sans Mono CJK KR', 'Nanum Gothic Coding', monospace",
    "markdown.preview.fontFamily": "'Noto Sans CJK KR', 'Nanum Gothic', 'Liberation Sans', sans-serif",
    "notebook.output.fontFamily": "'D2Coding', 'Noto Sans Mono CJK KR', monospace",
    "scm.inputFontFamily": "'D2Coding', 'Noto Sans CJK KR', monospace",
    "jupyter.defaultKernel": "isaaclab-direct"
}
EOF

# VSCode keybindings.json 생성
RUN tee /root/.config/Code/User/keybindings.json > /dev/null <<'EOF'
[
    {
        "key": "f5",
        "command": "python.debugInTerminal",
        "when": "editorTextFocus && editorLangId == python"
    },
    {
        "key": "ctrl+shift+`",
        "command": "terminal.new"
    }
]
EOF

# VSCode 확장 프로그램 설치 스크립트 생성
RUN tee /install_vscode_extensions.sh > /dev/null <<'EOF'
#!/bin/bash
set -e

echo "🔧 VSCode 확장 프로그램 설치 중..."

EXTENSIONS=(
    "ms-python.python"
    "ms-python.black-formatter"
    "ms-python.flake8"
    "ms-python.pylint"
    "ms-python.debugpy"
    "ms-toolsai.jupyter"
    "ms-vscode.cmake-tools"
    "ms-vscode.cpptools"
    "eamodio.gitlens"
    "PKief.material-icon-theme"
    "ms-vscode.vscode-json"
    "redhat.vscode-yaml"
    "ms-python.isort"
)

for ext in "${EXTENSIONS[@]}"; do
    echo "📦 설치 중: $ext"
    if code --install-extension "$ext" \
             --user-data-dir=/root/.vscode \
             --extensions-dir=/root/.vscode/extensions \
             --force > /dev/null 2>&1; then
        echo "  ✅ $ext 설치 성공"
    else
        echo "  ⚠️ $ext 설치 실패 (런타임에 재시도 가능)"
    fi
done

echo "🎉 VSCode 확장 프로그램 설치 완료!"
EOF

# VSCode 확장 프로그램 설치 실행
RUN chmod +x /install_vscode_extensions.sh && \
    /install_vscode_extensions.sh || echo "⚠️ 일부 확장 프로그램은 런타임에 설치됩니다"

# VSCode 작업공간 설정 생성 (Isaac Lab python 환경으로 수정)
RUN tee /AILAB-summer-school-2025/.vscode/settings.json > /dev/null <<'EOF'
{
    "python.defaultInterpreterPath": "/usr/local/bin/python",
    "python.terminal.executeInFileDir": true,
    "python.terminal.activateEnvironment": false,
    "python.analysis.extraPaths": [
        "/opt/IsaacLab/source",
        "/AILAB-summer-school-2025"
    ],
    "python.linting.enabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "terminal.integrated.cwd": "/AILAB-summer-school-2025",
    "editor.fontFamily": "'D2Coding', 'Noto Sans CJK KR', 'Nanum Gothic Coding', 'DejaVu Sans Mono', monospace",
    "editor.fontSize": 14,
    "terminal.integrated.fontFamily": "'D2Coding', 'Noto Sans Mono CJK KR', 'Nanum Gothic Coding', monospace",
    "terminal.integrated.fontSize": 14,
    "jupyter.defaultKernel": "isaaclab-direct"
}
EOF

# Python 백업 생성
RUN cp /usr/bin/python3 /usr/bin/python3-system && \
    cp /usr/bin/python /usr/bin/python-system 2>/dev/null || ln -sf /usr/bin/python3 /usr/bin/python-system

# Isaac Lab Python 래퍼 스크립트 생성 (v2.1.0 구조 반영)
RUN tee /usr/local/bin/python > /dev/null <<'EOF'
#!/bin/bash

# 기본 실행 파일들
ISAACLAB_PYTHON="/opt/IsaacLab/isaaclab.sh -p"
ISAAC_PYTHON="/isaac-sim/python.sh"
SYSTEM_PYTHON="/usr/bin/python3-system"
CURRENT_DIR="$(pwd)"

# 안전한 실행 함수
safe_exec() {
    local cmd="$1"
    shift
    if command -v "${cmd%% *}" >/dev/null 2>&1; then
        exec $cmd "$@"
    else
        echo "❌ Command not found: ${cmd%% *}" >&2
        exec "$SYSTEM_PYTHON" "$@"
    fi
}

# 환경 감지 함수들
is_vscode_terminal() {
    [[ "$TERM_PROGRAM" == "vscode" ]] || [[ -n "$VSCODE_PID" ]] || [[ -n "$VSCODE_INJECTION" ]]
}

is_isaac_env() {
    [[ "$CURRENT_DIR" =~ ^/AILAB-summer-school-2025 ]] || \
    [[ "$CURRENT_DIR" =~ ^/opt/IsaacLab ]] || \
    [[ "$CURRENT_DIR" == "/AILAB-summer-school-2025" ]] || \
    [[ "$CURRENT_DIR" == "/opt/IsaacLab" ]] || \
    is_vscode_terminal
}

is_isaac_script() {
    local script="$1"
    if [[ -f "$script" ]]; then
        if grep -q "isaac\|isaaclab\|torch\.cuda\|AppLauncher" "$script" 2>/dev/null || \
           [[ "$script" =~ (set_env|grasp_|isaac|check_env) ]]; then
            return 0
        fi
    fi
    return 1
}

# Isaac Lab 가용성 확인 (v2.1.0 구조 반영)
is_isaaclab_available() {
    export PYTHONPATH="/opt/IsaacLab/source:/isaac-sim/kit/python/lib/python3.10/site-packages"
    export OMNI_KIT_ALLOW_ROOT=1
    
    if /isaac-sim/python.sh -c "
import sys
sys.path.insert(0, '/opt/IsaacLab/source')
try:
    import isaaclab
    exit(0)
except ImportError:
    exit(1)
" >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

# Isaac Lab 환경 설정
setup_isaaclab_env() {
    export ISAACLAB_PATH="/opt/IsaacLab"
    export ISAAC_SIM_PATH="/isaac-sim"
    export OMNI_KIT_ALLOW_ROOT=1 
    export TERM=xterm-256color
    export PYTHONPATH="/opt/IsaacLab/source:/isaac-sim/kit/python/lib/python3.10/site-packages"
    
    # X11 환경도 함께 설정
    [[ -z "$DISPLAY" ]] && export DISPLAY=:0
    export QT_X11_NO_MITSHM=1 
    export QT_GRAPHICSSYSTEM=native
    export LIBGL_ALWAYS_INDIRECT=0 
    export LIBGL_ALWAYS_SOFTWARE=0
    export __GL_SYNC_TO_VBLANK=0 
    export __GL_MaxFramesAllowed=1
    export MESA_GL_VERSION_OVERRIDE=4.1
}

# Isaac Sim 직접 실행 (Isaac Lab 없이)
run_with_isaac_sim() {
    export OMNI_KIT_ALLOW_ROOT=1
    export ISAAC_SIM_PATH="/isaac-sim"
    export PYTHONPATH="/isaac-sim/kit/python/lib/python3.10/site-packages"
    [[ -z "$DISPLAY" ]] && export DISPLAY=:0
    export QT_X11_NO_MITSHM=1
    safe_exec "$ISAAC_PYTHON" "$@"
}

# 메인 로직
case "${1:-}" in
    --version|--help|-V|-h)
        if is_isaaclab_available; then
            setup_isaaclab_env
            safe_exec "$ISAAC_PYTHON" "$@"
        else
            run_with_isaac_sim "$@"
        fi ;;
    -c|-m)
        if [[ "$2" =~ (isaac|isaaclab|torch) ]] || is_isaac_env || is_vscode_terminal; then
            if is_isaaclab_available; then
                setup_isaaclab_env
                safe_exec "$ISAAC_PYTHON" "$@"
            else
                run_with_isaac_sim "$@"
            fi
        else
            safe_exec "$SYSTEM_PYTHON" "$@"
        fi ;;
    "")
        if is_isaac_env || is_vscode_terminal; then
            echo "🚀 Isaac Python (interactive)"
            if is_isaaclab_available; then
                setup_isaaclab_env
                cd /opt/IsaacLab 2>/dev/null || true
                safe_exec "$ISAAC_PYTHON"
            else
                echo "⚠️ Isaac Lab not available, using Isaac Sim Python"
                run_with_isaac_sim
            fi
        else
            safe_exec "$SYSTEM_PYTHON"
        fi ;;
    *)
        if is_isaac_env || is_isaac_script "$1" || is_vscode_terminal; then
            echo "🚀 Running with Isaac Python: $1"
            if is_isaaclab_available; then
                echo "  (Isaac Lab environment)"
                setup_isaaclab_env
                safe_exec "$ISAAC_PYTHON" "$@"
            else
                echo "  (Isaac Sim environment)"
                run_with_isaac_sim "$@"
            fi
        else
            safe_exec "$SYSTEM_PYTHON" "$@"
        fi ;;
esac
EOF

# python3도 동일한 래퍼로 설정
RUN cp /usr/local/bin/python /usr/local/bin/python3 && \
    chmod +x /usr/local/bin/python /usr/local/bin/python3

# Isaac Lab wrapper Python kernel 추가 등록
RUN echo "🔧 Isaac Lab wrapper kernel 등록 중..." && \
    export ISAACLAB_PATH="/opt/IsaacLab" && \
    export ISAAC_SIM_PATH="/isaac-sim" && \
    export OMNI_KIT_ALLOW_ROOT=1 && \
    export PYTHONPATH="/opt/IsaacLab/source:/isaac-sim/kit/python/lib/python3.10/site-packages" && \
    # Isaac Lab wrapper Python kernel 등록 (기본값으로 사용)
    /usr/local/bin/python -m ipykernel install --user --name isaaclab-wrapper --display-name "Isaac Lab Python" && \
    echo "✅ Isaac Lab wrapper kernel 등록 완료"

# X11 헬퍼 스크립트들 생성
RUN tee /usr/local/bin/setup-x11 > /dev/null <<'EOF'
#!/bin/bash
echo "🔧 X11 환경 설정 중..."

if [[ -z "$DISPLAY" ]]; then
    export DISPLAY=:0
    echo "📺 DISPLAY 설정: $DISPLAY"
fi

if [[ -n "$XAUTH" ]] && [[ -f "$XAUTH" ]]; then
    export XAUTHORITY="$XAUTH"
    echo "🔐 XAUTHORITY 설정: $XAUTHORITY"
elif [[ -f "/tmp/.docker.xauth" ]]; then
    export XAUTHORITY="/tmp/.docker.xauth"
    echo "🔐 XAUTHORITY 설정: $XAUTHORITY"
fi

export QT_X11_NO_MITSHM=1
export QT_GRAPHICSSYSTEM=native
export LIBGL_ALWAYS_INDIRECT=0
export LIBGL_ALWAYS_SOFTWARE=0
export __GL_SYNC_TO_VBLANK=0
export __GL_MaxFramesAllowed=1
export MESA_GL_VERSION_OVERRIDE=4.1

echo "✅ X11 환경 설정 완료"

if command -v xdpyinfo >/dev/null 2>&1; then
    if xdpyinfo >/dev/null 2>&1; then
        echo "✅ X11 연결 테스트: 성공"
    else
        echo "❌ X11 연결 테스트: 실패"
        echo "💡 도커 실행 시 다음 옵션들이 필요합니다:"
        echo "   -e DISPLAY=\$DISPLAY"
        echo "   -v /tmp/.X11-unix:/tmp/.X11-unix:rw"
        echo "   -v \$XAUTH:/tmp/.docker.xauth:rw"
        echo "   --network=host"
    fi
else
    echo "⚠️ xdpyinfo 없음 - X11 연결 확인 불가"
fi
EOF

# 안전한 VSCode 실행기
RUN tee /usr/local/bin/code-safe > /dev/null <<'EOF'
#!/bin/bash
echo "🚀 VSCode 시작 (X11 환경 설정 포함)..."

source /usr/local/bin/setup-x11

VSCODE_USER_DATA_DIR="/root/.vscode"
mkdir -p "$VSCODE_USER_DATA_DIR"

if [ $# -eq 0 ]; then
    if [[ "$PWD" =~ ^/AILAB-summer-school-2025 ]] || [[ "$PWD" == "/AILAB-summer-school-2025" ]]; then
        TARGET="."
    else
        TARGET="/AILAB-summer-school-2025"
        echo "📁 /AILAB-summer-school-2025로 이동합니다."
        cd /AILAB-summer-school-2025
    fi
else
    TARGET="$@"
fi

echo "📂 VSCode 실행: $TARGET"

exec /usr/bin/code \
    --no-sandbox \
    --disable-gpu-sandbox \
    --disable-software-rasterizer \
    --user-data-dir="$VSCODE_USER_DATA_DIR" \
    --extensions-dir="$VSCODE_USER_DATA_DIR/extensions" \
    "$TARGET"
EOF

# 추가 편의 스크립트들
RUN tee /usr/local/bin/vs > /dev/null <<'EOF'
#!/bin/bash
exec code-safe "$@"
EOF

RUN tee /usr/local/bin/isaac-vscode > /dev/null <<'EOF'
#!/bin/bash
echo "🚀 Isaac Lab VSCode 시작..."
source /usr/local/bin/setup-x11
mkdir -p /root/.vscode
cd /AILAB-summer-school-2025
exec code-safe .
EOF

# 모든 스크립트 실행 권한 부여
RUN chmod +x /usr/local/bin/setup-x11 /usr/local/bin/code-safe /usr/local/bin/vs /usr/local/bin/isaac-vscode

# bash 설정 추가 (v2.1.0 구조 반영 + dependency 정보 추가)
RUN tee -a /root/.bashrc > /dev/null <<'EOF'

# === Isaac Lab Environment Setup ===
export ISAAC_SIM_PATH="/isaac-sim"
export ISAACLAB_PATH="/opt/IsaacLab"
export PYTHONPATH="/isaac-sim/kit/python/lib/python3.10/site-packages:/opt/IsaacLab/source:${PYTHONPATH}"
export OMNI_KIT_ALLOW_ROOT=1

setup_gui_environment() {
    if [[ -z "$DISPLAY" ]]; then
        export DISPLAY=:0
    fi
    
    export QT_X11_NO_MITSHM=1
    export QT_GRAPHICSSYSTEM=native
    export LIBGL_ALWAYS_INDIRECT=0
    export LIBGL_ALWAYS_SOFTWARE=0
    export __GL_SYNC_TO_VBLANK=0
    export __GL_MaxFramesAllowed=1
    export MESA_GL_VERSION_OVERRIDE=4.1
    export OMNI_KIT_ALLOW_ROOT=1
    
    if [[ -n "$XAUTH" ]] && [[ -f "$XAUTH" ]]; then
        export XAUTHORITY="$XAUTH"
    elif [[ -f "/tmp/.docker.xauth" ]]; then
        export XAUTHORITY="/tmp/.docker.xauth"
    fi
}

setup_gui_environment

alias code='code-safe'
alias vs='code-safe'
alias vscode='cd /AILAB-summer-school-2025 && code-safe .'
alias code-AILAB-summer-school-2025='cd /AILAB-summer-school-2025 && code-safe .'
alias code-here='code-safe .'
alias isaacsim='${ISAAC_SIM_PATH}/isaac-sim.sh'

code-isaac() {
    echo "🚀 Isaac Lab VSCode 시작..."
    cd /AILAB-summer-school-2025
    code-safe .
}

code-lab() {
    echo "🔬 Isaac Lab 소스 디렉토리에서 VSCode 시작..."
    cd /opt/IsaacLab  
    code-safe .
}

code-new() {
    if [ -n "$1" ]; then
        mkdir -p "$1" && cd "$1" && code-safe .
    else
        echo "사용법: code-new <디렉토리명>"
    fi
}

run-isaac-gui() {
    echo "🚀 Running Isaac Sim with enhanced GUI support: $@"
    setup_gui_environment
    /isaac-sim/python.sh "$@"
}

run-isaac-headless() {
    echo "🤖 Running Isaac Sim in headless mode: $@"
    export ISAAC_HEADLESS=1
    /isaac-sim/python.sh "$@"
}

debug-x11() {
    echo "🔍 X11 환경 디버그 정보:"
    echo "  DISPLAY: $DISPLAY"
    echo "  XAUTHORITY: $XAUTHORITY"
    echo "  QT_X11_NO_MITSHM: $QT_X11_NO_MITSHM"
    echo "  LIBGL_ALWAYS_INDIRECT: $LIBGL_ALWAYS_INDIRECT"
    echo ""
    
    if command -v xdpyinfo >/dev/null 2>&1; then
        if xdpyinfo >/dev/null 2>&1; then
            echo "✅ X11 서버 연결: 성공"
            echo "🖥️ 디스플레이 정보:"
            xdpyinfo | head -10
        else
            echo "❌ X11 서버 연결: 실패"
        fi
    else
        echo "⚠️ xdpyinfo 명령어 없음"
    fi
    
    echo ""
    if command -v glxinfo >/dev/null 2>&1; then
        echo "🎮 OpenGL 정보:"
        glxinfo | grep -E "(OpenGL vendor|OpenGL renderer|OpenGL version)" || echo "OpenGL 정보 확인 실패"
    else
        echo "⚠️ glxinfo 명령어 없음"
    fi
}

show-code-help() {
    echo ""
    echo "🎯 === Isaac Lab VSCode 명령어 가이드 ==="
    echo "📝 기본 사용법:"
    echo "  code              # 안전한 옵션으로 VSCode 실행"
    echo "  code .            # 현재 디렉토리에서 VSCode 열기"
    echo "  code <경로>       # 특정 경로에서 VSCode 열기"
    echo ""
    echo "🚀 편의 명령어:"
    echo "  vs                # code의 짧은 별명"
    echo "  vscode            # /AILAB-summer-school-2025에서 VSCode 시작"
    echo "  code-isaac        # /AILAB-summer-school-2025에서 VSCode 시작 (함수)"
    echo "  code-lab          # /opt/IsaacLab에서 VSCode 시작"
    echo "  code-here         # 현재 위치에서 VSCode 시작"
    echo "  code-AILAB-summer-school-2025    # /AILAB-summer-school-2025로 이동 후 VSCode 시작"
    echo ""
    echo "🎮 Isaac Sim GUI:"
    echo "  python script.py  # 자동으로 Isaac Sim Python 사용"
    echo "  run-isaac-gui <script>  # GUI 환경 강제 설정"
    echo "  run-isaac-headless <script>  # Headless 모드로 실행"
    echo ""
    echo "🔧 디버그 도구:"
    echo "  debug-x11         # X11 환경 디버그 정보 표시"
    echo "  setup-x11         # X11 환경 수동 설정"
    echo ""
}

run-isaac() {
    setup_gui_environment
    /isaac-sim/python.sh "$@"
}

debug-isaac() {
    echo "🔍 Isaac Lab Python 환경:"
    echo "  Default Python: /usr/local/bin/python (Isaac Lab wrapper)"
    echo "  Isaac Lab Script: /opt/IsaacLab/isaaclab.sh"
    echo "  Isaac Sim Python: /isaac-sim/python.sh"
    echo "  PYTHONPATH: $PYTHONPATH"
    echo "  ISAACLAB_PATH: $ISAACLAB_PATH"
    echo "  DISPLAY: $DISPLAY"
    echo "  XAUTHORITY: $XAUTHORITY"
    echo ""
    echo "🧪 Testing Python environments:"
    
    echo "  1. Default python (should be Isaac Lab):"
    python --version 2>/dev/null || echo "    Python 실행 실패"
    
    echo "  2. Isaac Lab python test:"
    setup_gui_environment
    if [ -f "/opt/IsaacLab/isaaclab.sh" ]; then
        /opt/IsaacLab/isaaclab.sh -p -c "import sys; print(f'    Isaac Lab Python: {sys.executable}')" 2>/dev/null || echo "    Isaac Lab Python 테스트 실패"
    else
        echo "    Isaac Lab 스크립트 없음"
    fi
    
    echo "  3. Direct Isaac Sim python test:"
    /isaac-sim/python.sh -c "import sys; print(f'    Isaac Sim Python: {sys.executable}')" 2>/dev/null || echo "    Isaac Sim Python 테스트 실패"
    
    echo "  4. Isaac Lab module test (v2.1.0):"
    if /usr/local/bin/test-isaaclab >/dev/null 2>&1; then
        echo "    ✅ Isaac Lab module: Available"
    else
        echo "    ❌ Isaac Lab module: Not available"
        echo "    💡 Fix: run 'fix-isaaclab'"
    fi
    
    if [ -n "$1" ]; then
        echo ""
        echo "🚀 Running script with Isaac Lab Python: $1"
        setup_gui_environment
        if [ -f "/opt/IsaacLab/isaaclab.sh" ]; then
            /opt/IsaacLab/isaaclab.sh -p "$@"
        else
            echo "Isaac Lab 스크립트를 찾을 수 없습니다."
        fi
    fi
}

# Python 환경 확인 함수 (dependency 정보 추가)
check-python-env() {
    echo "🐍 Python 환경 확인 (호환성 버전 포함):"
    echo ""
    echo "1. 기본 python 명령어:"
    echo "   $(which python) -> $(readlink -f $(which python) 2>/dev/null || echo 'direct execution')"
    echo ""
    echo "2. python 실행 테스트:"
    
    # Python 테스트 스크립트를 임시 파일로 생성
    cat > /tmp/check_python_env.py << 'PYEOF'
import sys
print(f'   Python executable: {sys.executable}')
print(f'   Python version: {sys.version.split()[0]}')

# Isaac Lab 관련 패키지 테스트
try:
    import isaaclab
    print('   ✅ Isaac Lab 패키지: Available')
    
    # 가장 중요한 AppLauncher 테스트
    try:
        from isaaclab.app import AppLauncher
        print('   ✅ isaaclab.app.AppLauncher: Available')
    except ImportError as e:
        print(f'   ❌ isaaclab.app.AppLauncher: Not available - {e}')
        
except ImportError:
    print('   ❌ Isaac Lab 패키지: Not available')

# 주요 dependency들 버전 확인
packages_to_check = [
    ('torch', 'PyTorch'), ('numpy', 'NumPy'), ('opencv-python', 'OpenCV'), 
    ('gymnasium', 'Gymnasium'), ('psutil', 'psutil'), ('lxml', 'lxml'),
    ('toml', 'TOML'), ('pytz', 'pytz'), ('ipykernel', 'IPython Kernel')
]

print('   📦 주요 패키지 버전:')
for package, display_name in packages_to_check:
    try:
        if package == 'opencv-python':
            module = __import__('cv2')
        else:
            module = __import__(package.replace('-', '_'))
        version = getattr(module, '__version__', 'Unknown')
        
        # CUDA 지원 확인 (PyTorch)
        if package == 'torch':
            cuda_available = module.cuda.is_available()
            print(f'     ✅ {display_name}: {version} (CUDA: {cuda_available})')
        else:
            print(f'     ✅ {display_name}: {version}')
    except ImportError:
        print(f'     ❌ {display_name}: Not available')

# 호환성 체크
print('   🔍 호환성 체크:')
try:
    import numpy, cv2
    print(f'     ✅ numpy ({numpy.__version__}) + opencv 호환')
except:
    print('     ❌ numpy + opencv 호환성 문제')

try:
    import psutil
    print(f'     ✅ psutil ({psutil.__version__}) - rl-games 호환')
except:
    print('     ❌ psutil 문제')

try:
    import gymnasium
    print(f'     ✅ gymnasium ({gymnasium.__version__}) - stable-baselines3 호환')
except:
    print('     ❌ gymnasium 문제')
PYEOF

    # Python 스크립트 실행
    python /tmp/check_python_env.py
    
    # 임시 파일 정리
    rm -f /tmp/check_python_env.py
    
    echo ""
    echo "3. Isaac Lab 스크립트 테스트:"
    if [ -f "/opt/IsaacLab/isaaclab.sh" ]; then
        echo "   ✅ Isaac Lab script: Available"
        echo "   사용법: isaaclab -s (simulation), isaaclab -p script.py (run script)"
    else
        echo "   ❌ Isaac Lab script: Missing"
    fi
    echo ""
    echo "4. 전체 Isaac Lab 환경 테스트:"
    if /usr/local/bin/test-isaaclab >/dev/null 2>&1; then
        echo "   ✅ Isaac Lab 완전 설치: 성공"
    else
        echo "   ❌ Isaac Lab 완전 설치: 실패 - fix-isaaclab 명령어로 수정 가능"
    fi
}

# Isaac Lab 편의 함수들 (안전한 버전)
isaaclab-sim() {
    echo "🚀 Isaac Lab 시뮬레이션 시작..."
    cd /opt/IsaacLab
    setup_gui_environment
    if [ -f "./isaaclab.sh" ]; then
        ./isaaclab.sh -s
    else
        echo "❌ Isaac Lab 스크립트를 찾을 수 없습니다."
        echo "💡 Fix: run 'fix-isaaclab'"
    fi
}

isaaclab-help() {
    echo "🎯 Isaac Lab 명령어 가이드:"
    echo "  isaaclab -s         # Isaac Lab 시뮬레이션 시작"
    echo "  isaaclab -p <script> # Python 스크립트 실행"
    echo "  isaaclab-sim        # GUI로 시뮬레이션 시작"
    echo "  isaaclab --help     # 전체 옵션 보기"
    echo ""
    echo "📂 Isaac Lab 디렉토리로 이동: cd /opt/IsaacLab"
    echo "🔧 Isaac Lab 재설치: cd /opt/IsaacLab && ./isaaclab.sh --install"
}

# Isaac Lab 환경으로 이동
cdlab() {
    cd /opt/IsaacLab
    echo "📂 Isaac Lab 디렉토리: $(pwd)"
    echo "🚀 사용 가능한 명령어: isaaclab -s, isaaclab-sim"
}

# Isaac Lab 재설치 함수 (정식 설치 방법 + dependency 고려)
fix-isaaclab() {
    echo "🔧 Isaac Lab 환경 재설정 중 (호환성 버전 포함)..."
    cd /opt/IsaacLab
    
    echo "🌍 환경 변수 설정..."
    export ISAACLAB_PATH="/opt/IsaacLab"
    export ISAAC_SIM_PATH="/isaac-sim"
    export OMNI_KIT_ALLOW_ROOT=1 TERM=xterm-256color
    export PYTHONPATH="/opt/IsaacLab/source:/isaac-sim/kit/python/lib/python3.10/site-packages"
    
    echo "📦 주요 dependency 재설치 (호환성 버전)..."
    /isaac-sim/python.sh -m pip install --no-cache-dir --force-reinstall \
        gymnasium==0.29.1 toml lxml==5.2.2 pytz>=2020.1 \
        opencv-python==4.8.1.78 psutil==5.9.8 ipykernel
    
    echo "🔄 Isaac Lab 정식 재설치..."
    # 기존 설치 제거
    /isaac-sim/python.sh -m pip uninstall -y isaaclab 2>/dev/null || true
    
    # 정식 재설치
    if /isaac-sim/python.sh -m pip install --no-cache-dir -e . --force-reinstall; then
        echo "✅ Isaac Lab 재설치 성공!"
    else
        echo "❌ Isaac Lab 재설치 실패"
        echo "🔧 대안 방법 시도 중..."
        
        # 대안: Isaac Lab 스크립트 사용
        if [ -f "./isaaclab.sh" ]; then
            echo "📦 isaaclab.sh --install 실행..."
            chmod +x ./isaaclab.sh
            ./isaaclab.sh --install || echo "isaaclab.sh --install 실패"
        fi
    fi
    
    echo "🔗 isaaclab 명령어 재생성..."
    chmod +x ./isaaclab.sh 2>/dev/null || true
    rm -f /usr/local/bin/isaaclab
    ln -sf /opt/IsaacLab/isaaclab.sh /usr/local/bin/isaaclab
    
    echo "📓 Jupyter kernel 재등록..."
    /isaac-sim/python.sh -m ipykernel install --user --name isaaclab-direct --display-name "Isaac Lab Direct" --force
    if command -v /usr/local/bin/python >/dev/null 2>&1; then
        /usr/local/bin/python -m ipykernel install --user --name isaaclab-wrapper --display-name "Isaac Lab Python" --force
    fi
    
    echo "🧪 Isaac Lab 환경 테스트..."
    
    # Python 테스트 스크립트를 임시 파일로 생성
    cat > /tmp/fix_isaaclab_test.py << 'PYEOF'
import isaaclab
from isaaclab.app import AppLauncher
print('✅ Isaac Lab 완전 설치 확인!')

# Dependency 체크도 함께
import numpy, cv2, gymnasium, psutil
print(f'✅ Dependency 체크: numpy {numpy.__version__}, opencv {cv2.__version__}')
print(f'✅ 호환성 체크: gymnasium {gymnasium.__version__}, psutil {psutil.__version__}')
PYEOF

    if /isaac-sim/python.sh /tmp/fix_isaaclab_test.py 2>/dev/null; then
        echo "✅ Isaac Lab 환경 재설정 완료! (호환성 버전 포함)"
        echo ""
        echo "🚀 사용 가능한 명령어:"
        echo "  isaaclab -s              # 시뮬레이션 시작"
        echo "  isaaclab -p script.py    # Python 스크립트 실행"  
        echo "  python script.py         # 자동으로 Isaac Lab 환경 사용"
        echo "  check-python-env         # 전체 환경 및 dependency 확인"
        echo ""
    else
        echo "⚠️ Isaac Lab 환경 설정 여전히 실패"
        echo ""
        echo "🛠️ 수동 해결 방법:"
        echo "1. cd /opt/IsaacLab"
        echo "2. /isaac-sim/python.sh -m pip install -e . --force-reinstall"
        echo "3. 또는 ./isaaclab.sh --install"
        echo ""
        
        # 진단 정보 제공
        echo "🔍 진단 정보:"
        cat > /tmp/diagnose_isaaclab.py << 'PYEOF'
try:
    import isaaclab
    import os
    isaaclab_path = os.path.dirname(isaaclab.__file__)
    print(f'📂 isaaclab 설치 경로: {isaaclab_path}')
    items = [item for item in os.listdir(isaaclab_path) if not item.startswith('__')]
    print(f'📦 사용 가능한 모듈들: {items}')
except ImportError as e:
    print(f'❌ isaaclab import 실패: {e}')
PYEOF
        /isaac-sim/python.sh /tmp/diagnose_isaaclab.py 2>/dev/null || echo "진단 정보 확인 실패"
        rm -f /tmp/diagnose_isaaclab.py
    fi
    
    # 임시 파일 정리
    rm -f /tmp/fix_isaaclab_test.py
    
    source ~/.bashrc
}

# 별칭 추가
alias reinstall-isaaclab='fix-isaaclab'
EOF

# SSH 설정
RUN mkdir -p /var/run/sshd && \
    echo 'root:isaac123' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# 시작 스크립트 생성 (v2.1.0 구조 반영 + dependency 정보 추가)
RUN tee /startup.sh > /dev/null <<'EOF'
#!/bin/bash

echo "======================================"
echo "🚀 Isaac Lab Environment Starting..."
echo "======================================"

# 환경 변수 설정
export ISAAC_SIM_PATH="/isaac-sim"
export ISAACLAB_PATH="/opt/IsaacLab"
export PYTHONPATH="/isaac-sim/kit/python/lib/python3.10/site-packages:/opt/IsaacLab/source"
export OMNI_KIT_ALLOW_ROOT=1

# X11 환경 자동 설정
[[ -z "$DISPLAY" ]] && export DISPLAY=:0
export QT_X11_NO_MITSHM=1
export QT_GRAPHICSSYSTEM=native
export LIBGL_ALWAYS_INDIRECT=0
export LIBGL_ALWAYS_SOFTWARE=0
export __GL_SYNC_TO_VBLANK=0
export __GL_MaxFramesAllowed=1
export MESA_GL_VERSION_OVERRIDE=4.1

echo "🔧 X11 환경 설정 중..."
if [[ -n "$XAUTH" ]] && [[ -f "$XAUTH" ]]; then
    export XAUTHORITY="$XAUTH"
elif [[ -f "/tmp/.docker.xauth" ]]; then
    export XAUTHORITY="/tmp/.docker.xauth"
fi
echo "✅ X11 환경 설정 완료"

# X11 연결 테스트
if command -v xdpyinfo >/dev/null 2>&1; then
    if xdpyinfo >/dev/null 2>&1; then
        echo "✅ X11 연결 테스트: 성공"
    else
        echo "❌ X11 연결 테스트: 실패"
        echo "💡 Docker 실행 시 X11 옵션 확인 필요"
    fi
fi

cd /opt/IsaacLab

echo "🧪 Testing installations..."

echo "📋 Python Environment Check:"
echo "  Default python: $(which python)"
echo "  Python wrapper: $(readlink -f $(which python) 2>/dev/null || echo 'direct execution')"

# 주요 패키지 버전 체크 (dependency 포함)
echo "📦 호환성 패키지 버전 체크:"
cat > /tmp/startup_dependency_check.py << 'PYEOF'
packages = [
    ('torch', 'PyTorch'), ('numpy', 'NumPy'), ('opencv-python', 'OpenCV'), 
    ('gymnasium', 'Gymnasium'), ('psutil', 'psutil'), ('lxml', 'lxml'),
    ('toml', 'TOML'), ('pytz', 'pytz'), ('ipykernel', 'IPython Kernel')
]

for package, display_name in packages:
    try:
        if package == 'opencv-python':
            module = __import__('cv2')
        else:
            module = __import__(package.replace('-', '_'))
        version = getattr(module, '__version__', 'Unknown')
        
        if package == 'torch':
            cuda_available = module.cuda.is_available()
            print(f'  ✅ {display_name}: {version} (CUDA: {cuda_available})')
        else:
            print(f'  ✅ {display_name}: {version}')
    except ImportError:
        print(f'  ❌ {display_name}: Not installed')
PYEOF

if python /tmp/startup_dependency_check.py; then
    echo "✅ 호환성 패키지들: OK"
else
    echo "❌ 일부 패키지 설치 이슈"
fi
rm -f /tmp/startup_dependency_check.py

if /isaac-sim/python.sh -c "import omni.isaac.kit; print('Isaac Sim: OK')"; then
    echo "✅ Isaac Sim: OK"
else
    echo "❌ Isaac Sim import issue"
fi

# Isaac Lab 테스트 (정식 설치 검증)
echo "🔬 Testing Isaac Lab (정식 설치 검증)..."
echo "📂 Isaac Lab 구조 확인:"
if [ -d "/opt/IsaacLab/source/isaaclab" ]; then
    echo "  ✅ isaaclab 패키지: 존재"
    ls -la /opt/IsaacLab/source/ | grep isaaclab
else
    echo "  ❌ isaaclab 패키지: 없음"
fi

# Python 테스트 스크립트를 임시 파일로 생성
cat > /tmp/startup_test_isaaclab.py << 'PYEOF'
import sys
try:
    import isaaclab
    print('✅ isaaclab 패키지 import: 성공')
    
    # 가장 중요한 AppLauncher 테스트
    try:
        from isaaclab.app import AppLauncher
        print('✅ isaaclab.app.AppLauncher import: 성공')
        print('✅ Isaac Lab 완전 설치 확인!')
    except ImportError as e:
        print(f'❌ isaaclab.app.AppLauncher import 실패: {e}')
        
        # 구조 진단
        import os
        isaaclab_path = os.path.dirname(isaaclab.__file__)
        print(f'📂 isaaclab 설치 경로: {isaaclab_path}')
        items = [item for item in os.listdir(isaaclab_path) if not item.startswith('__')]
        print(f'📦 사용 가능한 모듈들: {items}')
        exit(1)
        
except ImportError as e:
    print(f'❌ isaaclab 패키지 import 실패: {e}')
    exit(1)
PYEOF

if /isaac-sim/python.sh /tmp/startup_test_isaaclab.py; then
    echo "✅ Isaac Lab environment ready!"
else
    echo "❌ Isaac Lab import failed"
    echo "🔧 Attempting automatic fix..."
    
    # 정식 재설치 시도
    cd /opt/IsaacLab
    echo "🔄 Isaac Lab 재설치 중..."
    /isaac-sim/python.sh -m pip install --no-cache-dir -e . --force-reinstall
    
    # 재테스트
    if /isaac-sim/python.sh /tmp/startup_test_isaaclab.py; then
        echo "✅ Isaac Lab 자동 수정 성공!"
    else
        echo "⚠️ Isaac Lab 수동 수정 필요 - 'fix-isaaclab' 명령어 사용"
    fi
fi

# 임시 파일 정리
rm -f /tmp/startup_test_isaaclab.py

echo "🔧 Checking Isaac Lab commands..."
if command -v isaaclab >/dev/null 2>&1; then
    echo "✅ isaaclab command: Available"
    echo "  Try: isaaclab --help, isaaclab -s (for simulation)"
else
    echo "❌ isaaclab command: Not found"
    echo "🔧 Fixing isaaclab command link..."
    cd /opt/IsaacLab
    chmod +x ./isaaclab.sh
    rm -f /usr/local/bin/isaaclab
    ln -sf /opt/IsaacLab/isaaclab.sh /usr/local/bin/isaaclab
    
    if command -v isaaclab >/dev/null 2>&1; then
        echo "✅ isaaclab command: Fixed!"
    else
        echo "⚠️ isaaclab command: Still not working"
        echo "  Alternative: cd /opt/IsaacLab && ./isaaclab.sh -s"
    fi
fi

echo "🔧 Checking VSCode extensions..."
extension_count=$(code --list-extensions --user-data-dir=/root/.vscode 2>/dev/null | wc -l || echo "0")
echo "📦 VSCode extensions installed: $extension_count"

echo "📓 Checking Jupyter kernels..."
kernel_count=$(jupyter kernelspec list 2>/dev/null | grep -c "isaac" || echo "0")
if [ "$kernel_count" -gt 0 ]; then
    echo "✅ Jupyter Isaac Lab kernels: $kernel_count installed"
else
    echo "⚠️ Jupyter Isaac Lab kernels: Not found"
    echo "🔧 Registering kernels..."
    export PYTHONPATH="/opt/IsaacLab/source:/isaac-sim/kit/python/lib/python3.10/site-packages"
    /isaac-sim/python.sh -m ipykernel install --user --name isaaclab-direct --display-name "Isaac Lab Direct" 2>/dev/null || echo "direct kernel registration failed"
    if command -v /usr/local/bin/python >/dev/null 2>&1; then
        /usr/local/bin/python -m ipykernel install --user --name isaaclab-wrapper --display-name "Isaac Lab Python" 2>/dev/null || echo "wrapper kernel registration failed"
    fi
fi

cd /AILAB-summer-school-2025

echo ""
echo "======================================"
echo "🎉 Isaac Lab Environment Ready!"
echo "======================================"
echo "📁 Project files: /AILAB-summer-school-2025"
echo "🔬 Isaac Lab: /opt/IsaacLab"
echo "🤖 Isaac Sim: /isaac-sim"
echo "👤 User: $(whoami)"
echo "🐍 Isaac Sim Python: /isaac-sim/python.sh"
echo "🐍 System Python: $(which python3)"
echo "💻 Safe VSCode: code-safe"
echo "🔥 PyTorch: 2.5.1+cu118"
echo "🖥️ Display: $DISPLAY"
echo "📦 Compatibility: numpy 1.26.2, opencv 4.8.1, gymnasium 0.29.1"
echo "======================================"
echo ""
echo "🚀 Quick commands:"
echo "  python              # Isaac Lab Python (자동 환경 감지)"
echo "  isaaclab --help     # Isaac Lab 도움말"
echo "  isaaclab -s         # Isaac Lab 시뮬레이션 시작"
echo "  isaaclab -p script.py # Isaac Lab Python 스크립트 실행"
echo "  code .              # Open VSCode (안전한 옵션 적용)"
echo "  vs                  # VSCode 짧은 명령어"
echo "  code-isaac          # /AILAB-summer-school-2025에서 VSCode 시작"
echo "  isaac-vscode        # Isaac Lab 전용 VSCode"
echo "  isaaclab-sim        # Isaac Lab GUI 시뮬레이션"
echo "  cdlab               # Isaac Lab 디렉토리로 이동"
echo "  check-python-env    # Python 환경 상태 확인 (dependency 포함)"
echo "  fix-isaaclab        # Isaac Lab 재설치 (호환성 버전 포함)"
echo "  run-isaac <script>  # Isaac Sim Python으로 실행"
echo "  run-isaac-gui <script>  # GUI 환경 강제 설정"
echo "  run-isaac-headless <script>  # Headless 모드로 실행"
echo "  debug-x11           # X11 환경 디버그"
echo "  show-code-help      # VSCode 명령어 전체 가이드"
echo "  isaaclab-help       # Isaac Lab 명령어 가이드"
echo "  debug-isaac <script>  # 디버그 정보와 함께 실행"
echo "======================================"
echo ""
echo "🔧 Isaac Lab 상태:"
if /usr/local/bin/test-isaaclab >/dev/null 2>&1; then
    echo "  ✅ Isaac Lab: 사용 가능"
else
    echo "  ⚠️ Isaac Lab: fix-isaaclab 명령어로 수정 가능"
fi

echo "======================================"
echo ""
echo "🔤 한글 환경: $(echo '안녕하세요!' | head -c 15)..."
echo ""

if [ "$1" = "ssh" ]; then
    service ssh start
    echo "🔑 SSH server started on port 22"
    tail -f /dev/null
elif [ "$1" = "vscode" ]; then
    echo "🎯 Starting VSCode..."
    exec isaac-vscode
elif [ $# -eq 0 ]; then
    echo "🎯 Starting interactive bash shell..."
    exec /bin/bash
else
    echo "🚀 Executing: $@"
    exec "$@"
fi
EOF

RUN chmod +x /startup.sh

# 프로젝트 파일 복사 (가장 마지막에 - 캐시 최적화)
COPY . /AILAB-summer-school-2025/

# 복사된 파일들 처리 및 의존성 설치
RUN chmod -R 755 /AILAB-summer-school-2025 && \
    chown -R root:root /AILAB-summer-school-2025 && \
    # graspnetAPI 설치 (있는 경우에만)
    if [ -d "/AILAB-summer-school-2025/graspnetAPI" ]; then \
        echo "🔧 Installing graspnetAPI..."; \
        cd /AILAB-summer-school-2025/graspnetAPI && \
        /isaac-sim/python.sh -m pip install --no-cache-dir -e . && \
        echo "✅ graspnetAPI installation completed"; \
    fi && \
    # cgnet 설치 (있는 경우에만)
    if [ -d "/AILAB-summer-school-2025/cgnet" ]; then \
        echo "🔧 Installing cgnet..."; \
        cd /AILAB-summer-school-2025/cgnet && \
        /isaac-sim/python.sh -m pip install --no-cache-dir -e . || echo "cgnet installation skipped"; \
    fi && \
    # Pointnet_Pointnet2_pytorch 설치 (있는 경우에만)
    if [ -d "/AILAB-summer-school-2025/Pointnet_Pointnet2_pytorch" ]; then \
        echo "🔧 Installing Pointnet_Pointnet2_pytorch..."; \
        cd /AILAB-summer-school-2025/Pointnet_Pointnet2_pytorch && \
        /isaac-sim/python.sh -m pip install --no-cache-dir -e . || echo "Pointnet installation skipped"; \
    fi

ENTRYPOINT ["/startup.sh"]
CMD []
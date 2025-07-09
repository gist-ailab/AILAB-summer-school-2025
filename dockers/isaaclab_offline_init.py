import argparse

# Isaac Lab 관련 라이브러리 임포트
from isaaclab.app import AppLauncher


# Argparse로 CLI 인자 파싱 및 Omniverse 앱 실행
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


simulation_app.close()

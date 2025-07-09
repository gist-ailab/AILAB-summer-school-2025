# -*- coding: utf-8 -*-
# 필요한 라이브러리들을 임포트합니다.
import os  # 파일 및 디렉토리 경로 관련 작업을 위한 라이브러리
import json  # JSON 파일을 읽고 쓰기 위한 라이브러리
import numpy as np  # 수치 계산, 특히 배열(행렬) 연산을 위한 라이브러리
import cv2  # OpenCV, 이미지 처리(읽기, 쓰기, 변환 등)를 위한 라이브러리
import glob  # 특정 패턴에 맞는 파일 경로들을 찾기 위한 라이브러리
from pycocotools import mask as mask_util  # COCO 데이터셋 유틸리티, 특히 마스크(segmentation) 인코딩/디코딩을 위해 사용
from tqdm import tqdm  # 반복문(loop) 진행 상황을 시각적으로 보여주는 라이브-러리
import re  # 정규 표현식을 사용하여 문자열 패턴을 검색하고 조작하기 위한 라이브러리


class CocoConverter:
    """
    NVIDIA Isaac Lab에서 생성된 시뮬레이션 데이터셋을 MS COCO 형식으로 변환하는 클래스입니다.
    이 클래스의 주요 특징은 미리 정의된 전체 YCB 객체 목록을 기준으로, 카테고리 ID를 부여하는 것입니다.
    이를 통해 여러 데이터셋을 일관성 있게 통합하고 학습시킬 수 있습니다.

    - 입력: Isaac Lab 데이터 형식 (episode_* 폴더 구조)
    - 출력: COCO 주석 형식을 따르는 단일 JSON 파일
    """
    # -- 변경: 보내주신 YCB 객체 목록 전체를 클래스 변수로 정의 --
    # 모든 데이터셋 변환에 걸쳐 일관된 카테고리 ID를 부여하기 위한 마스터 목록입니다.
    # sorted()를 사용하여 항상 동일한 순서를 보장합니다.
    YCB_OBJECT_CLASSES = sorted([
        '002_master_chef_can', '008_pudding_box', '014_lemon', '021_bleach_cleanser', '029_plate', '036_wood_block', '044_flat_screwdriver', '054_softball', '061_foam_brick', '065_c_cups', '065_i_cups', '072_b_toy_airplane', '073_c_lego_duplo',
        '003_cracker_box', '009_gelatin_box', '015_peach', '022_windex_bottle', '030_fork', '037_scissors', '048_hammer', '055_baseball', '062_dice', '065_d_cups', '065_j_cups', '072_c_toy_airplane', '073_d_lego_duplo',
        '004_sugar_box', '010_potted_meat_can', '016_pear', '024_bowl', '031_spoon', '038_padlock', '050_medium_clamp', '056_tennis_ball', '063_a_marbles', '065_e_cups', '070_a_colored_wood_blocks', '072_d_toy_airplane', '073_e_lego_duplo',
        '005_tomato_soup_can', '011_banana', '017_orange', '025_mug', '032_knife', '040_large_marker', '051_large_clamp', '057_racquetball', '063_b_marbles', '065_f_cups', '070_b_colored_wood_blocks', '072_e_toy_airplane', '073_f_lego_duplo',
        '006_mustard_bottle', '012_strawberry', '018_plum', '026_sponge', '033_spatula', '042_adjustable_wrench', '052_extra_large_clamp', '058_golf_ball', '065_a_cups', '065_g_cups', '071_nine_hole_peg_test', '073_a_lego_duplo', '073_g_lego_duplo',
        '077_rubiks_cube', '007_tuna_fish_can', '013_apple', '019_pitcher_base', '028_skillet_lid', '035_power_drill', '043_phillips_screwdriver', '053_mini_soccer_ball', '059_chain', '065_b_cups', '065_h_cups', '072_a_toy_airplane', '073_b_lego_duplo'
    ])

    def __init__(self, data_root: str, output_path: str):
        """
        CocoConverter 클래스의 생성자(initializer)입니다.
        :param data_root: 변환할 원본 데이터가 있는 루트 디렉토리 경로.
        :param output_path: 변환된 COCO JSON 파일을 저장할 경로.
        """
        self.data_root = data_root  # 원본 데이터 루트 경로 저장
        self.output_path = output_path  # 최종 결과물 파일 경로 저장
        self._init_coco_structure()  # COCO JSON의 기본 구조를 초기화하는 메서드 호출

    def _init_coco_structure(self):
        """
        COCO JSON 파일의 기본 구조를 생성하고, YCB_OBJECT_CLASSES를 기반으로 카테고리 정보를 미리 채웁니다.
        이 메서드는 변환 작업 시작 전에 호출되어야 합니다.
        """
        # 최종적으로 JSON으로 변환될 파이썬 딕셔너리
        self.coco_output = {
            "info": {"description": "Isaac Lab YCB Object Dataset"},
            "licenses": [],
        "images": [],
            "annotations": [],
            "categories": []  # 카테고리 정보가 이 리스트에 저장됩니다.
        }
        # YCB 객체 클래스 이름(str)을 COCO 카테고리 ID(int)에 매핑하는 딕셔너리
        self.categories_map = {}
        # 각 이미지와 주석(annotation)에 고유한 ID를 부여하기 위한 카운터
        self.image_id_counter = 1
        self.annotation_id_counter = 1

        # 미리 정의된 목록으로 카테고리 정보와 맵을 생성 --
        print("Pre-populating COCO categories based on master list...")
        # 클래스 변수 YCB_OBJECT_CLASSES를 순회하며 각 객체에 대한 카테고리 정보를 생성
        for i, class_name in enumerate(self.YCB_OBJECT_CLASSES):
            # COCO 표준에 따라 카테고리 ID는 0이 아닌 1부터 시작합니다.
            category_id = i + 1
            # 클래스 이름을 카테고리 ID에 매핑
            self.categories_map[class_name] = category_id
            # COCO 형식에 맞는 카테고리 딕셔너리를 생성하여 리스트에 추가
            self.coco_output["categories"].append({
                "id": category_id,
                "name": class_name,
                "supercategory": "ycb_object"  # 모든 객체의 상위 카테고리 지정
            })
        print(f"Initialized {len(self.categories_map)} categories.")

    def convert(self):
        """
        데이터셋 변환을 시작하는 메인 메서드입니다.
        data_root에서 'episode_*' 패턴의 모든 디렉토리를 찾아 순차적으로 처리합니다.
        """
        # glob을 사용하여 data_root 내의 모든 에피소드 디렉토리 경로를 리스트로 가져옴
        episode_dirs = sorted(glob.glob(os.path.join(self.data_root, 'episode_*')))
        # 만약 에피소드 디렉토리가 하나도 없다면 오류 메시지를 출력하고 종료
        if not episode_dirs:
            print(f"Error: No 'episode_*' directories found in '{self.data_root}'. Please check the path.")
            return

        print(f"Found {len(episode_dirs)} episode directories. Starting conversion...")

        # tqdm을 사용하여 각 에피소드 처리 진행 상황을 프로그레스 바로 표시
        for episode_dir in tqdm(episode_dirs, desc="Processing episodes"):
            self._process_episode(episode_dir)

        # 모든 에피소드 처리가 끝나면 최종 결과를 JSON 파일로 저장
        self._save_to_file()

    #### 구현 예제 함수 ####
    def _process_episode(self, episode_dir: str):
        """
        단일 에피소드 디렉토리를 처리하여 이미지와 주석(annotation) 정보를 추출합니다.
        :param episode_dir: 처리할 단일 에피소드 디렉토리의 경로.
        """
        
        # (예제 1) 에피소드 내의 RGB 이미지 파일을 찾습니다. (보통 하나만 존재)
        rgb_files = glob.glob(os.path.join(episode_dir, 'rgb_*.png'))
        if not rgb_files:
            return  # RGB 파일이 없으면 이 에피소드는 건너뜁니다.
        rgb_path = rgb_files[0]

        # (예제 2) 파일 이름에서 숫자 부분을 추출하여 다른 파일(마스크, JSON)의 이름을 구성합니다.
        # 예: 'rgb_0000.png' -> '_0000'
        base_name = os.path.basename(rgb_path)
        suffix = os.path.splitext(base_name[len("rgb"):] )[0]

        # (예제 3) 해당 RGB 이미지에 대응하는 마스크 이미지와 매핑 JSON 파일의 경로를 구성
        mask_path = os.path.join(episode_dir, f"instance_id_segmentation{suffix}.png")
        json_path = os.path.join(episode_dir, f"instance_id_segmentation_mapping{suffix}.json")

        # (예제 4) 마스크 파일이나 JSON 파일 둘 중 하나라도 없으면 처리를 중단하고 다음으로 넘어감
        if not (os.path.exists(mask_path) and os.path.exists(json_path)):
            return

        # (예제 5) OpenCV를 사용하여 RGB 이미지 로드
        image = cv2.imread(rgb_path)
        height, width, _ = image.shape  # 이미지의 높이, 너비 추출

        # (예제 6) COCO JSON에 저장될 이미지의 상대 경로 생성 (예: 'episode_0/rgb_0000.png')
        relative_rgb_path = os.path.join(os.path.basename(episode_dir), base_name)

        # 현재 이미지에 대한 ID 할당 및 COCO 'images' 섹션에 정보 추가
        current_image_id = self.image_id_counter
        self.coco_output["images"].append({
            "id": current_image_id,
            "file_name": relative_rgb_path,
            "width": width,
            "height": height
        })
        self.image_id_counter += 1  # 다음 이미지를 위해 ID 카운터 증가

        # 인스턴스 마스크 이미지를 로드합니다. `cv2.IMREAD_UNCHANGED`는 RGBA의 4개 채널(알파 채널 포함)을 모두 읽기 위함입니다.
        instance_mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        # 마스크 이미지가 유효하지 않은 경우 (로드 실패, 3차원 아님, 4채널 아님) 건너뜁니다.
        if instance_mask_img is None or len(instance_mask_img.shape) < 3 or instance_mask_img.shape[2] != 4:
            return
        # OpenCV는 기본적으로 BGR(A) 순서로 이미지를 읽으므로, 표준인 RGB(A) 순서로 변경합니다.
        instance_mask_img = cv2.cvtColor(instance_mask_img, cv2.COLOR_BGRA2RGBA)

        # 매핑 정보가 담긴 JSON 파일을 열고 데이터를 로드
        with open(json_path, 'r') as f:
            mapping_data = json.load(f)

        # 마스크의 RGBA 색상 값을 객체 정보(카테고리 ID)에 매핑하는 딕셔너리를 생성
        color_to_info = {}
        for color_str, prim_path in mapping_data.items():
            # prim_path(객체 경로)에 'ycb_' 문자열이 포함된 경우에만 처리 (YCB 객체 필터링)
            if "ycb_" in prim_path:
                # 정규 표현식을 사용하여 'ycb_...' 형태의 객체 이름을 추출합니다.
                match = re.search(r'(ycb_([a-zA-Z0-9_]+))', prim_path)
                if match:
                    # e.g., "ycb_019_pitcher_base"
                    ycb_id_with_prefix = match.group(1)
                    # 접두사 "ycb_"를 제거하여 클래스 이름만 남깁니다. e.g., "019_pitcher_base"
                    ycb_id = ycb_id_with_prefix.replace("ycb_", "", 1)

                    # -- 변경: 미리 생성된 카테고리 맵에서 ID 조회 --
                    # YCB ID를 사용하여 미리 만들어 둔 `categories_map`에서 카테고리 ID를 조회
                    category_id = self.categories_map.get(ycb_id)
                    if category_id:  # 마스터 목록에 있는 유효한 객체인 경우
                        # JSON의 색상 문자열 `'(R,G,B,A)'`를 파싱하여 정수 튜플로 변환
                        color_tuple = tuple(map(int, color_str.strip('()').split(',')))
                        # 색상 튜플을 key로, 카테고리 ID를 value로 하는 딕셔너리에 저장
                        color_to_info[color_tuple] = {"category_id": category_id}

        # 마스크 이미지에 존재하는 모든 고유한 RGBA 색상 값을 찾습니다.
        unique_colors = np.unique(instance_mask_img.reshape(-1, 4), axis=0)

        # 찾은 고유 색상들을 하나씩 순회하며 각 객체 인스턴스에 대한 주석(annotation)을 생성합니다.
        for color_rgba in unique_colors:
            color_tuple = tuple(color_rgba)
            # 이 색상이 우리가 처리하기로 한 YCB 객체에 해당하지 않으면 건너뜁니다. (예: 배경색)
            if color_tuple not in color_to_info:
                continue

            # 현재 색상(color_rgba)과 일치하는 픽셀만 True(1)로, 나머지는 False(0)로 하는 이진 마스크를 생성합니다.
            binary_mask = np.all(instance_mask_img == color_rgba, axis=-1).astype(np.uint8)
            # 만약 마스크 영역의 픽셀 수가 0이면 (아주 드문 경우) 건너뜁니다.
            if np.sum(binary_mask) == 0:
                continue

            # pycocotools를 사용하여 이진 마스크를 RLE(Run-Length Encoding) 형식으로 인코딩합니다.
            # `np.asfortranarray`는 메모리 레이아웃을 Fortran 형식으로 변경하여 RLE 인코딩 효율을 높입니다.
            rle = mask_util.encode(np.asfortranarray(binary_mask))
            # RLE의 'counts'는 기본적으로 바이트 문자열이므로, JSON 저장을 위해 'utf-8' 문자열로 디코딩합니다.
            rle['counts'] = rle['counts'].decode('utf-8')
            # RLE로부터 마스크의 면적(area)과 바운딩 박스(bbox)를 계산합니다.
            area = float(mask_util.area(rle))
            bbox = list(mask_util.toBbox(rle))  # [x, y, width, height]

            # 최종 COCO 주석(annotation) 딕셔너리를 생성하여 리스트에 추가합니다.
            self.coco_output["annotations"].append({
                "id": self.annotation_id_counter,
                "image_id": current_image_id,  # 이 주석이 속한 이미지의 ID
                "category_id": color_to_info[color_tuple]["category_id"],  # 이 객체의 카테고리 ID
                "segmentation": rle,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0  # 단일 객체이므로 iscrowd는 0
            })
            self.annotation_id_counter += 1  # 다음 주석을 위해 ID 카운터 증가

    def _save_to_file(self):
        """
        메모리에 생성된 COCO 데이터를 최종 JSON 파일로 저장합니다.
        """
        print(f"\nConversion complete! Saving COCO annotations to: {self.output_path}")
        # 최종 JSON 파일의 가독성과 일관성을 위해 카테고리 목록을 ID 순서로 정렬합니다.
        self.coco_output['categories'] = sorted(self.coco_output['categories'], key=lambda x: x['id'])
        # JSON 파일을 쓰기 모드('w')로 열고, 생성된 딕셔너리를 저장합니다.
        # `indent=4`는 사람이 읽기 쉽도록 4칸 들여쓰기로 포맷팅(pretty-print)하라는 의미입니다.
        with open(self.output_path, 'w') as f:
            json.dump(self.coco_output, f, indent=4)
        # 변환 작업 요약 정보를 출력합니다.
        print(f"Total Images: {len(self.coco_output['images'])}")
        print(f"Total Annotations: {len(self.coco_output['annotations'])}")
        print(f"Total Categories: {len(self.coco_output['categories'])}")


# 이 스크립트가 직접 실행될 때만 아래 코드가 동작합니다. (모듈로 임포트될 때는 실행되지 않음)
if __name__ == '__main__':
    # 원본 데이터가 있는 루트 디렉토리 설정
    DATA_ROOT_DIR = os.path.join(os.getcwd(), "data/output_data")
    # 최종 결과물인 COCO 주석 JSON 파일이 저장될 경로와 이름 설정
    OUTPUT_JSON_FILE = os.path.join(DATA_ROOT_DIR, 'coco_annotations.json')

    # CocoConverter 클래스의 인스턴스를 생성
    converter = CocoConverter(data_root=DATA_ROOT_DIR, output_path=OUTPUT_JSON_FILE)
    # 변환 작업을 시작
    converter.convert()
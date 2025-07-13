import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide") # 넓은 레이아웃 사용

st.title("외계행성 중력렌즈 시뮬레이터")

# 세션 상태 초기화: 사용자가 지나온 점의 궤적을 기록하기 위해 사용
if 'angle_history' not in st.session_state:
    st.session_state.angle_history = []
if 'magnification_history' not in st.session_state:
    st.session_state.magnification_history = []

# --- 사이드바: 외계 행성 위치 조절 및 시뮬레이션 파라미터 ---
st.sidebar.header("외계 행성 위치 조절")

# 행성의 렌즈별로부터의 상대적 각도 (0 ~ 360도)
planet_angle_deg = st.sidebar.slider(
    "행성 각도 (도)", 0, 360, 270, help="렌즈별 주위 행성의 상대적 각도를 조절합니다."
)
planet_angle_rad = np.radians(planet_angle_deg)

# 행성의 렌즈별로부터의 거리 (렌즈별의 아인슈타인 반지름 단위)
planet_distance = st.sidebar.slider(
    "행성 거리 (아인슈타인 반지름)", 0.1, 2.0, 1.0, step=0.05, help="렌즈별로부터 행성까지의 거리를 조절합니다."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**시뮬레이션 파라미터**")
lens_mass_ratio = st.sidebar.slider("행성-렌즈별 질량비 (q)", 0.0001, 0.1, 0.01, format="%.4f", help="행성의 질량이 렌즈별 질량의 몇 배인지 나타냅니다.")
source_offset_x = st.sidebar.slider("배경별 렌즈 중심 오프셋 X", -0.5, 0.5, 0.0, step=0.01, help="배경별이 렌즈 중심에서 벗어난 정도")
source_offset_y = st.sidebar.slider("배경별 렌즈 중심 오프셋 Y", -0.5, 0.5, 0.0, step=0.01, help="배경별이 렌즈 중심에서 벗어난 정도")

# --- 메인 영역: 시뮬레이션 시각화 ---
col1, col2 = st.columns(2) # 화면을 두 부분으로 나눔

# --- 마이크로렌즈링 광도 계산 함수 (수정된 부분) ---
# 이 함수는 인자로 받은 행성 위치(planet_pos_x, planet_pos_y)에 대해 밝기를 계산합니다.
def calculate_magnification(q_ratio, planet_pos_x, planet_pos_y, source_x, source_y):
    # 배경별과 렌즈 중심 간의 거리 (u_source)
    dist_source_lens_center = np.sqrt(source_x**2 + source_y**2)
    einstein_radius = 1.0

    # 1. 단일 점-렌즈에 의한 확대 계수 (기본 밝기)
    # 배경별이 렌즈 중심에서 멀어질수록 밝기는 1에 가까워집니다.
    if dist_source_lens_center == 0:
        base_mag = 2.5 # 중심에 있을 때 (무한대 방지)
    else:
        u = dist_source_lens_center
        base_mag = (u**2 + 2) / (u * np.sqrt(u**2 + 4))

    # 2. 행성에 의한 교란 효과 (가우시안 근사 모델)
    # 이 부분은 행성 위치(p_x, p_y)에 따라 밝기 변화가 발생하도록 유도합니다.
    
    p_x = planet_pos_x
    p_y = planet_pos_y

    # 현재 행성 위치의 각도 (0~360도)
    delta_angle = np.degrees(np.arctan2(p_y, p_x)) % 360 
    target_angle_deg = 270 # 배경별이 Y축 음수 방향에 있을 때 (시선 방향)
    angle_spread_deg = 30 # 효과가 나타나는 각도 범위
    
    # 목표 각도(270도)와 현재 행성 각도의 차이
    angle_dist = np.abs(delta_angle - target_angle_deg)
    angle_dist = np.min([angle_dist, 360 - angle_dist]) # 최소 각도 차이

    # 행성-렌즈 중심 간 거리
    current_planet_distance = np.sqrt(p_x**2 + p_y**2)
    
    # 행성 영향의 크기: 질량비(q_ratio)에 비례하고, 
    # 각도 차이가 작을수록, 그리고 아인슈타인 반지름에 가까울수록 커집니다.
    
    # 렌즈 효과가 아인슈타인 반지름 근처에서 가장 강함을 반영
    distance_influence = 1.0 / (0.1 + np.abs(current_planet_distance - einstein_radius))
    
    # 가우시안 형태의 각도 영향
    angle_influence = np.exp(-((angle_dist / angle_spread_deg)**2))

    # 최종 행성 효과
    planet_influence = q_ratio * angle_influence * distance_influence
    
    # 최종 밝기 (렌즈별 기본 밝기 + 행성 효과)
    final_magnification = base_mag + planet_influence * 5 # 임의의 스케일링으로 효과 강조
    
    return max(1.0, final_magnification) # 기본적으로 1배 이상

# -------------------------------------------------------------

with col1:
    st.subheader("중력렌즈 시스템 시각화")

    fig_system, ax_system = plt.subplots(figsize=(6, 6))
    ax_system.set_aspect('equal', adjustable='box')
    ax_system.set_xlim(-2.5, 2.5) # 아인슈타인 반지름 범위 고려
    ax_system.set_ylim(-2.5, 2.5)
    ax_system.set_title("렌즈별-행성-배경별 시스템")
    ax_system.set_xlabel("X 위치 (아인슈타인 반지름 단위)")
    ax_system.set_ylabel("Y 위치 (아인슈타인 반지름 단위)")
    ax_system.grid(True, linestyle='--', alpha=0.7)

    # 렌즈별 (중심 항성) - 원점 고정
    ax_system.plot(0, 0, 'o', color='red', markersize=15, label="렌즈별 (중심 항성)")

    # 외계 행성 - 슬라이더에 따라 위치 변화
    planet_x = planet_distance * np.cos(planet_angle_rad)
    planet_y = planet_distance * np.sin(planet_angle_rad)
    ax_system.plot(planet_x, planet_y, 'o', color='blue', markersize=8, label="외계 행성")

    # 배경별 (광원) - 렌즈별 뒤에 고정되어 있고, 오프셋 조절
    background_star_x = source_offset_x
    background_star_y = source_offset_y
    ax_system.plot(background_star_x, background_star_y, '*', color='yellow', markersize=10, label="배경별 (광원)")

    # 아인슈타인 반지름 원 (단위: 1)
    einstein_radius = 1.0
    circle = plt.Circle((0, 0), einstein_radius, color='gray', linestyle='--', fill=False, alpha=0.5, label="아인슈타인 반지름")
    ax_system.add_patch(circle)

    # 지구 (관측자) 위치 - 270도 방향으로 고정
    observer_direction_x = 0
    observer_direction_y = -2.3 
    ax_system.arrow(0, 0, observer_direction_x, observer_direction_y, head_width=0.15, head_length=0.2, fc='green', ec='green', label="지구 방향")
    ax_system.text(observer_direction_x * 1.1, observer_direction_y * 1.1, "지구 (관측자)", color='green', ha='center', va='top')

    ax_system.legend(loc='upper right')
    st.pyplot(fig_system)
    plt.close(fig_system) # 메모리 누수 방지

with col2:
    st.subheader("배경별 광도 변화 (밝기 곡선)")

    # --- 광도 곡선 전체 계산 ---
    # 0도부터 360도까지의 모든 각도에 대한 광도 계산
    angles = np.linspace(0, 360, 360) 
    magnifications = []

    for ang_deg in angles:
        ang_rad = np.radians(ang_deg)
        
        # 각도에 따른 행성 위치 계산 (광도 곡선 경로)
        px = planet_distance * np.cos(ang_rad)
        py = planet_distance * np.sin(ang_rad)

        mag = calculate_magnification(
            q_ratio=lens_mass_ratio,
            planet_pos_x=px, # 루프 변수 px, py를 전달
            planet_pos_y=py,
            source_x=source_offset_x,
            source_y=source_offset_y
        )
        magnifications.append(mag)
    
    # --- 현재 슬라이더 위치의 광도 값 계산 ---
    current_magnification = calculate_magnification(
        q_ratio=lens_mass_ratio,
        planet_pos_x=planet_x, # 슬라이더로 조절된 현재 행성 위치
        planet_pos_y=planet_y,
        source_x=source_offset_x,
        source_y=source_offset_y
    )

    # --- 현재 위치를 세션 상태에 기록 (자취를 남기기 위함) ---
    st.session_state.angle_history.append(planet_angle_deg)
    st.session_state.magnification_history.append(current_magnification)
    
    # --- 광도 곡선 그래프 그리기 ---
    fig_lightcurve, ax_lightcurve = plt.subplots(figsize=(6, 4))
    
    # 1. 전체 광도 곡선 경로 (파란색 선)
    ax_lightcurve.plot(angles, magnifications, 'b-', alpha=0.7, label="전체 광도 곡선 경로")

    # 2. 사용자가 지나온 점의 자취 (빨간색 점선)
    # 슬라이더 이동에 따라 기록된 이력을 사용하여 점들을 연결
    ax_lightcurve.plot(st.session_state.angle_history, st.session_state.magnification_history, 'r--', alpha=0.5, label="이동 궤적 (자취)")

    # 3. 현재 위치 (빨간색 점)
    ax_lightcurve.plot([planet_angle_deg], [current_magnification], 'ro', markersize=10, label=f"현재 행성 위치 ({planet_angle_deg}°)")

    ax_lightcurve.set_xlim(0, 360)
    # 동적으로 Y축 범위 조절: 최소 밝기 0.8 또는 1.0, 최대 밝기의 1.2배
    ax_lightcurve.set_ylim(min(1.0, min(magnifications)) * 0.9, max(magnifications) * 1.2 if magnifications else 2.0)
    ax_lightcurve.set_xlabel("행성 각도 (도)")
    ax_lightcurve.set_ylabel("상대 밝기 (배율)")
    ax_lightcurve.set_title("배경별 겉보기 밝기 (광도 곡선)")
    ax_lightcurve.grid(True)
    ax_lightcurve.legend()
    st.pyplot(fig_lightcurve)
    plt.close(fig_lightcurve)

st.markdown("""
---
**설명:**
외계 행성에 의한 중력렌즈 효과 시뮬레이터입니다. 행성의 위치와 질량비를 조절하여 배경별의 밝기 변화를 확인합니다.

* **왼쪽 그래프:** 렌즈별(빨간색), 외계 행성(파란색), 배경별(노란색)의 상대적 위치를 보여줍니다.
* **오른쪽 그래프:** 외계 행성의 각도에 따른 배경별의 겉보기 밝기(배율) 변화를 보여줍니다. 
    * **파란색 선:** 행성이 궤도를 한 바퀴 돌았을 때 예상되는 전체 밝기 곡선입니다.
    * **빨간색 점선:** 사용자가 슬라이더를 조작하며 지나온 행성의 각도와 밝기 값의 궤적(자취)입니다.
    * **빨간색 점:** 현재 행성의 각도에서의 밝기 값입니다.
    
    행성이 **렌즈별-배경별-지구의 시선에 정렬될 때 (270도 근처)** 밝기가 급격히 증가하는 피크가 나타납니다.
""")

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide") # 넓은 레이아웃 사용

st.title("외계행성 중력렌즈 시뮬레이터")

# --- 사이드바: 사용자 입력 컨트롤 ---
st.sidebar.header("외계 행성 위치 조절")

# 행성의 렌즈별로부터의 상대적 각도 (0 ~ 360도)
planet_angle_deg = st.sidebar.slider(
    "행성 각도 (도)", 0, 360, 0, help="렌즈별 주위 행성의 상대적 각도를 조절합니다."
)
planet_angle_rad = np.radians(planet_angle_deg)

# 행성의 렌즈별로부터의 거리 (예시: 렌즈별의 아인슈타인 반지름 단위)
# 실제 시뮬레이션에서는 물리적 스케일 고려 필요
planet_distance = st.sidebar.slider(
    "행성 거리 (단위)", 0.1, 2.0, 0.5, help="렌즈별로부터 행성까지의 거리를 조절합니다."
)

# --- 메인 영역: 시뮬레이션 시각화 ---
col1, col2 = st.columns(2) # 화면을 두 부분으로 나눔

with col1:
    st.subheader("중력렌즈 시스템 시각화")

    fig_system, ax_system = plt.subplots(figsize=(6, 6))
    ax_system.set_aspect('equal', adjustable='box')
    ax_system.set_xlim(-3, 3)
    ax_system.set_ylim(-3, 3)
    ax_system.set_title("렌즈별-행성-배경별 시스템")
    ax_system.set_xlabel("X 위치")
    ax_system.set_ylabel("Y 위치")
    ax_system.grid(True, linestyle='--', alpha=0.7)

    # 렌즈별 (중심 항성)
    ax_system.plot(0, 0, 'o', color='red', markersize=15, label="렌즈별 (중심 항성)")

    # 외계 행성
    planet_x = planet_distance * np.cos(planet_angle_rad)
    planet_y = planet_distance * np.sin(planet_angle_rad)
    ax_system.plot(planet_x, planet_y, 'o', color='blue', markersize=8, label="외계 행성")

    # 배경별 (광원) - 예시 위치
    background_star_x = 0.5
    background_star_y = 2.0
    ax_system.plot(background_star_x, background_star_y, '*', color='yellow', markersize=10, label="배경별 (광원)")

    # (선택 사항) 아인슈타인 반지름 원 표현
    einstein_radius = 1.0 # 예시 값
    circle = plt.Circle((0, 0), einstein_radius, color='gray', linestyle='--', fill=False, alpha=0.5, label="아인슈타인 반지름")
    ax_system.add_patch(circle)

    ax_system.legend()
    st.pyplot(fig_system)
    plt.close(fig_system) # 메모리 누수 방지

with col2:
    st.subheader("배경별 광도 변화 (밝기 곡선)")

    # --- 실제 중력렌즈 광도 계산 로직이 여기에 들어갑니다 ---
    # 이 부분은 매우 단순화된 예시이며, 실제로는 마이크로렌즈링 공식을 사용해야 합니다.
    # U: 렌즈별-배경별 간의 상대적 거리 (아인슈타인 반지름 단위)
    # q: 행성-렌즈별 질량비
    # alpha: 행성과 배경별의 각도
    # 등 복잡한 변수들을 고려하여 A (광원 확대 계수)를 계산합니다.

    # 임의의 광도 계산 (매우 단순화된 예시)
    # 행성과 배경별 간의 거리, 행성의 위치에 따른 임의의 밝기 변화
    # 이 부분에 실제 물리 모델을 구현해야 합니다.
    distance_to_background = np.sqrt((planet_x - background_star_x)**2 + (planet_y - background_star_y)**2)
    # 단순화된 광도 모델: 행성이 배경별에 가까워질수록 밝기가 변한다고 가정
    # 실제 중력렌즈 효과는 렌즈별에 의한 큰 피크와 행성에 의한 작은 피크가 나타납니다.
    base_magnification = 1.5 # 렌즈별에 의한 기본 증폭
    planet_perturbation = 0.0 # 행성에 의한 추가적인 변화
    if distance_to_background < 0.8: # 행성이 배경별에 가까워질 때
        planet_perturbation = 0.5 * (0.8 - distance_to_background) # 가까울수록 더 큰 변화
    elif distance_to_background < 1.5:
        planet_perturbation = -0.3 * (distance_to_background - 0.8) # 특정 거리에서 감소
    
    current_magnification = base_magnification + planet_perturbation
    # --- 광도 계산 끝 ---

    # 광도 변화 그래프 그리기
    fig_lightcurve, ax_lightcurve = plt.subplots(figsize=(6, 4))
    # 여기서는 현재 한 시점의 광도만 보여주지만,
    # 실제로는 행성이 움직이는 궤적에 따른 광도 변화를 누적해서 그릴 수 있습니다.
    # 예시로 현재 행성 각도에 따른 광도 값만 표시
    ax_lightcurve.plot([planet_angle_deg], [current_magnification], 'ro', markersize=8)
    # x축을 시간으로 하고, 광도를 시간에 따라 누적하여 그리는 것이 더 일반적입니다.
    # 여기서는 슬라이더의 값을 x축으로 사용하여 해당 위치에서의 광도를 표시합니다.
    
    ax_lightcurve.set_xlim(0, 360) # 슬라이더 범위와 동일하게 설정
    ax_lightcurve.set_ylim(0.8, 2.5) # 예상되는 광도 변화 범위
    ax_lightcurve.set_xlabel("행성 각도 (도)")
    ax_lightcurve.set_ylabel("상대 밝기 (배율)")
    ax_lightcurve.set_title("배경별 겉보기 밝기")
    ax_lightcurve.grid(True)
    st.pyplot(fig_lightcurve)
    plt.close(fig_lightcurve)

st.markdown("""
---
**설명:**
이 시뮬레이션은 사용자가 슬라이더를 이용해 외계 행성의 위치를 조절하면서, 
그에 따른 배경별의 중력렌즈 효과와 겉보기 밝기 변화를 시각적으로 보여줍니다.

* **왼쪽 그래프:** 렌즈별(중심 항성)과 외계 행성, 그리고 배경별의 상대적 위치를 보여줍니다.
* **오른쪽 그래프:** 외계 행성의 위치 변화에 따라 배경별의 겉보기 밝기가 어떻게 증폭되는지를 나타냅니다.
    실제 중력렌즈 현상에서는 행성이 특정 위치를 지날 때 독특한 광도 변화 패턴이 나타납니다.

**참고:** 광도 계산 부분은 이해를 돕기 위한 매우 단순화된 모델이며, 실제 마이크로렌즈링 계산은 더 복잡한 물리 공식을 필요로 합니다.
""")

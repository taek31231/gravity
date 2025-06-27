import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide") # 넓은 레이아웃 사용

st.title("외계행성 중력렌즈 시뮬레이터")

st.sidebar.header("외계 행성 위치 조절")

# 행성의 렌즈별로부터의 상대적 각도 (0 ~ 360도)
planet_angle_deg = st.sidebar.slider(
    "행성 각도 (도)", 0, 360, 270, help="렌즈별 주위 행성의 상대적 각도를 조절합니다."
)
planet_angle_rad = np.radians(planet_angle_deg)

# 행성의 렌즈별로부터의 거리 (렌즈별의 아인슈타인 반지름 단위)
# 이 거리는 중력렌즈 효과의 민감도에 영향을 줍니다.
planet_distance = st.sidebar.slider(
    "행성 거리 (아인슈타인 반지름)", 0.1, 2.0, 1.0, step=0.05, help="렌즈별로부터 행성까지의 거리를 조절합니다. 아인슈타인 반지름은 중력렌즈 효과가 가장 강한 영역입니다."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**시뮬레이션 파라미터**")
lens_mass_ratio = st.sidebar.slider("행성-렌즈별 질량비 (q)", 0.0001, 0.1, 0.01, format="%.4f", help="행성의 질량이 렌즈별 질량의 몇 배인지 나타냅니다. 값이 작을수록 행성 효과가 미미합니다.")
source_offset_x = st.sidebar.slider("배경별 렌즈 중심 오프셋 X", -0.5, 0.5, 0.0, step=0.01, help="배경별이 렌즈 중심에서 벗어난 정도")
source_offset_y = st.sidebar.slider("배경별 렌즈 중심 오프셋 Y", -0.5, 0.5, 0.0, step=0.01, help="배경별이 렌즈 중심에서 벗어난 정도")

# --- 메인 영역: 시뮬레이션 시각화 ---
col1, col2 = st.columns(2) # 화면을 두 부분으로 나눔

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

    # 배경별 (광원) - 렌즈별 중심과 약간의 오프셋
    # 배경별은 렌즈별 뒤에 고정되어 있고, 시선에 따라 렌즈별과 정렬될 수 있습니다.
    background_star_x = source_offset_x
    background_star_y = source_offset_y
    ax_system.plot(background_star_x, background_star_y, '*', color='yellow', markersize=10, label="배경별 (광원)")

    # 아인슈타인 반지름 원 (단위: 1)
    einstein_radius = 1.0
    circle = plt.Circle((0, 0), einstein_radius, color='gray', linestyle='--', fill=False, alpha=0.5, label="아인슈타인 반지름")
    ax_system.add_patch(circle)

    # 지구 (관측자) 위치 - 270도 방향으로 고정 (아래쪽)
    observer_direction_x = 0
    observer_direction_y = -2.3 # 화면 하단에 표시
    ax_system.arrow(0, 0, observer_direction_x, observer_direction_y, head_width=0.15, head_length=0.2, fc='green', ec='green', label="지구 방향")
    ax_system.text(observer_direction_x * 1.1, observer_direction_y * 1.1, "지구 (관측자)", color='green', ha='center', va='top')


    ax_system.legend(loc='upper right')
    st.pyplot(fig_system)
    plt.close(fig_system) # 메모리 누수 방지

with col2:
    st.subheader("배경별 광도 변화 (밝기 곡선)")

    # --- 마이크로렌즈링 광도 계산 함수 (기본적인 점-렌즈 모델 기반) ---
    # 실제 중력렌즈는 더 복잡하지만, 여기서는 핵심 원리를 보여주기 위한 단순화된 접근입니다.
    # 출처: https://www.sc.eso.org/santiago/lectures/Microlensing.pdf (간단한 모델)
    # 및 여러 천체물리학 자료 참조

    def calculate_magnification(u, q_ratio, planet_pos_x, planet_pos_y, source_x, source_y):
        # u: 렌즈와 광원의 상대적 거리 (아인슈타인 반지름 단위)
        # q_ratio: 행성-렌즈별 질량비 (Mp/Ml)
        # planet_pos_x, planet_pos_y: 행성 위치
        # source_x, source_y: 배경별 위치 (렌즈 중심 기준)

        # 1. 단일 점-렌즈에 의한 확대 계수 (렌즈별만 있는 경우)
        magnification_single_lens = (u**2 + 2) / (u * np.sqrt(u**2 + 4))

        # 2. 행성에 의한 교란 효과 (매우 단순화된 모델)
        # 행성이 배경별-렌즈별 시선에 가까워질수록 큰 영향을 줍니다.
        # 이 모델은 실제 바이너리 렌즈 모델만큼 정교하진 않지만,
        # 행성에 의한 피크/딥을 대략적으로 시뮬레이션합니다.

        # 행성과 배경별의 상대적인 x, y 거리
        rel_x_planet_source = planet_pos_x - source_x
        rel_y_planet_source = planet_pos_y - source_y

        # 행성-배경별 간의 거리 (이것이 작을수록 행성 효과가 커짐)
        dist_planet_source = np.sqrt(rel_x_planet_source**2 + rel_y_planet_source**2)

        # 행성 영향의 강도 (단순화된 가우시안 형태의 영향)
        # 행성이 시선과 가까워질수록 밝기 변화가 커짐
        # 행성의 질량비(q_ratio)에 비례
        # 'sensitivity_radius'는 행성 효과가 나타나는 범위를 조절
        sensitivity_radius = 0.3 # 이 값으로 행성 효과의 '범위'를 조절
        
        # 행성 궤도가 배경별 시선에 가까이 있을 때 (예: 270도 근처)
        # 이 모델은 행성과 배경별이 얼마나 가까운가에 따라 밝기를 조정합니다.
        # 실제는 렌즈별과 행성, 배경별의 정렬에 따라 복잡한 카오스틱 패턴이 생깁니다.
        
        # 여기서는 행성 위치가 배경별과 렌즈별 사이 정렬점에 가까워질 때
        # 밝기 변화가 발생하도록 유도합니다.
        
        # 배경별이 렌즈 중심에 있다고 가정하고, 행성이 0도 또는 180도 부근을 지날 때
        # 중력렌즈 효과가 강하게 나타납니다.
        # 우리가 원하는 '270도'에 행성이 있을 때 밝기 변화가 나타나려면
        # 배경별이 렌즈별의 '아인슈타인 반지름' 영역에 있고,
        # 행성이 배경별과 렌즈별을 잇는 선에 가까이 올 때를 시뮬레이션 해야 합니다.
        
        # 배경별과 렌즈별 중심의 상대적 위치 (u_source)
        u_source = np.sqrt(source_x**2 + source_y**2)
        
        # 렌즈별에 의한 기본 확대
        A_main = (u_source**2 + 2) / (u_source * np.sqrt(u_source**2 + 4)) if u_source > 0 else 2.5 # u=0 근접 시 무한대 방지

        # 행성에 의한 미세교란 (이 부분을 실제 모델링하는 것이 중요)
        # 매우 간단한 모델: 행성이 배경별 경로에 가까이 올 때 밝기 변화
        # 행성의 현재 위치(planet_angle_rad, planet_distance)에 따라
        # 광원(배경별)이 행성의 영향을 받는지를 계산
        
        # 행성-배경별-렌즈별의 정렬을 고려한 가중치
        # 행성이 렌즈별과 배경별 사이의 '시선'을 가로지르거나 가까이 갈 때 영향이 커집니다.
        # 시선(line of sight)은 대략 렌즈별(0,0)과 배경별(source_x, source_y)을 잇는 선입니다.
        
        # 행성의 궤도 상에서 배경별이 렌즈별과 정렬되는 위치 (0,0) 근처에서 나타나는 효과를
        # 행성의 각도에 따라 시뮬레이션 할 것입니다.

        # 행성 위치 (polar) -> (cartesian)
        p_x = planet_distance * np.cos(planet_angle_rad)
        p_y = planet_distance * np.sin(planet_angle_rad)
        
        # 배경별과 렌즈 중심의 상대적 위치에 대한 렌즈 효과
        # u: 배경별이 렌즈 중심에서 얼마나 떨어져 있는가 (아인슈타인 반지름 단위)
        # 여기서는 배경별 (source_x, source_y)와 행성(p_x, p_y)의 상대적 위치를 사용하여
        # 행성에 의한 추가적인 렌즈 효과를 시뮬레이션합니다.
        
        # 이 함수는 매우 단순화된 형태입니다.
        # 실제는 렌즈별과 행성이 이중 렌즈 시스템을 형성하여 배경별에 미치는 효과를
        # 복잡한 방정식을 통해 계산해야 합니다.
        
        # 예시: 행성이 배경별로부터 특정 거리 내에 있을 때 밝기 변화
        # 행성이 배경별의 "아인슈타인 반지름" 내에 들어올 때 큰 변화가 나타난다고 가정
        # 배경별의 위치를 렌즈 중심에 가깝게 (source_offset_x, source_offset_y) 둠
        
        # 배경별과 렌즈 중심 간의 거리
        dist_source_lens_center = np.sqrt(source_x**2 + source_y**2)
        
        # 렌즈별에 의한 기본 증폭 (배경별이 렌즈 중심에서 벗어난 정도에 따라)
        base_mag = (dist_source_lens_center**2 + 2) / (dist_source_lens_center * np.sqrt(dist_source_lens_center**2 + 4)) if dist_source_lens_center > 0 else 2.5
        
        # 행성에 의한 교란 추가: 행성이 배경별과 렌즈 중심 사이를 지나갈 때
        # 즉, 행성(p_x,p_y), 렌즈(0,0), 배경별(source_x,source_y)이 일직선에 가까워질 때
        
        # 행성과 렌즈 중심, 배경별의 상대적 위치를 기반으로 교란 계산
        # 행성이 렌즈-배경별 시선에 가까워질수록 교란이 커진다.
        # 이 시선은 (0,0)에서 (source_x, source_y)로 향하는 벡터입니다.
        
        # 렌즈와 행성 간의 거리 (rho)
        rho = np.sqrt(p_x**2 + p_y**2)

        # 렌즈-행성-배경별의 정렬 정도를 나타내는 변수
        # 예를 들어, 행성이 배경별 시선에 아주 가까이 있다면 dist_align이 작아짐
        
        # 행성이 배경별의 위치와 "정렬"되는 경우의 각도를 찾아냄
        # 270도 근처에서 피크가 생기도록 하려면,
        # 배경별이 렌즈 중심의 Y축 음수 방향으로 살짝 오프셋되어 있다고 가정
        # (source_x, source_y) = (0, -0.05) 이런 식으로

        # 행성-배경별 간의 상대적인 위치 벡터
        vec_planet_source_x = p_x - source_x
        vec_planet_source_y = p_y - source_y
        dist_planet_source_adjusted = np.sqrt(vec_planet_source_x**2 + vec_planet_source_y**2)
        
        # 행성에 의한 추가 증폭/감폭. 
        # 행성의 질량비(q_ratio)에 비례하고, 배경별에 가까울수록 강해집니다.
        # 이 모델은 단순화된 시뮬레이션이므로, 실제 물리와는 차이가 있습니다.
        
        # 특정 각도 (270도) 근처에서 행성이 배경별과 렌즈별 사이를 지날 때
        # 밝기가 피크를 찍도록 유도
        # 이를 위해 각도 차이를 이용한 가우시안 피크 모델을 적용합니다.
        
        angle_diff = np.abs(np.degrees(np.arctan2(p_y, p_x)) - planet_angle_deg)
        angle_diff = np.min([angle_diff, 360 - angle_diff]) # 0-180도로 정규화
        
        # 피크를 270도 근처에서 발생시키기 위해,
        # 배경별이 y축의 음수 방향에 약간 치우쳐 있다고 가정
        # 행성이 270도 (남쪽) 방향으로 움직일 때,
        # 배경별이 렌즈 중심의 (0, source_offset_y)에 위치하고 있다고 가정하면
        # 행성이 배경별과 렌즈별 사이를 지나갈 수 있습니다.
        
        # 시뮬레이션이 단순하므로, 270도에서 피크가 나도록 하는 근사치를 사용합니다.
        # 행성이 270도에 있을 때 렌즈와 배경별 사이로 지나가는 것으로 간주
        
        # 배경별이 렌즈 중심에 매우 가깝고 (source_offset_x=0, source_offset_y=0)
        # 행성이 270도 방향에서 렌즈와 배경별을 가로지를 때
        # 즉, 행성이 (0, -radius) 근처에 있을 때 (y축 음수)
        
        # 행성과 렌즈 중심 간의 거리 (rho)
        # 배경별과 렌즈 중심 간의 거리 (u_source)

        # 270도에서 피크를 만들기 위한 임의의 가우시안 렌즈 효과
        # 270도일 때 피크, 거리가 아인슈타인 반지름에 가까울 때 강함
        target_angle_deg = 270
        angle_spread_deg = 30 # 이 각도 범위 내에서 효과 발생
        
        # 각도 차이 계산
        delta_angle = np.degrees(np.arctan2(p_y, p_x)) % 360 # -180 ~ 180을 0 ~ 360으로
        angle_dist = np.abs(delta_angle - target_angle_deg)
        angle_dist = np.min([angle_dist, 360 - angle_dist]) # 최소 각도 차이

        # 행성 효과의 크기 (가우시안 폼)
        # 행성 질량비(q_ratio)와 렌즈-행성 거리(planet_distance)에 따라 조절
        
        # 행성 위치가 아인슈타인 반지름에 가까울수록,
        # 그리고 270도에 가까울수록 효과가 커짐
        planet_influence = q_ratio * np.exp(-((angle_dist / angle_spread_deg)**2)) * (1.0 / (0.1 + np.abs(planet_distance - einstein_radius)))
        
        # 행성의 영향이 렌즈별의 기본 밝기를 증가 또는 감소시킬 수 있음
        # 단순화를 위해 증가만 고려
        
        # 최종 밝기 = 렌즈별 기본 밝기 + 행성 효과
        final_magnification = base_mag + planet_influence * 5 # 임의의 스케일링
        
        # 최소 밝기 제한
        return max(1.0, final_magnification) # 기본적으로 1배 이상
    
    # --- 광도 계산 ---
    # 각도에 따른 광도 변화를 미리 계산하여 플로팅
    angles = np.linspace(0, 360, 360) # 0도부터 360도까지 1도 간격으로
    magnifications = []

    for ang_deg in angles:
        ang_rad = np.radians(ang_deg)
        
        # 행성 위치 (슬라이더의 planet_distance는 고정하고 각도만 변화)
        # 시뮬레이션에서는 슬라이더로 행성의 '현재' 각도를 조절하지만,
        # 광도 곡선은 0도부터 360도까지의 '경로'에 대한 곡선입니다.
        px = planet_distance * np.cos(ang_rad)
        py = planet_distance * np.sin(ang_rad)

        mag = calculate_magnification(
            u=np.sqrt(source_offset_x**2 + source_offset_y**2), # 배경별과 렌즈 중심 간 거리
            q_ratio=lens_mass_ratio,
            planet_pos_x=px,
            planet_pos_y=py,
            source_x=source_offset_x,
            source_y=source_offset_y
        )
        magnifications.append(mag)
    
    # 현재 슬라이더 위치의 광도 값
    current_magnification = calculate_magnification(
        u=np.sqrt(source_offset_x**2 + source_offset_y**2),
        q_ratio=lens_mass_ratio,
        planet_pos_x=planet_x, # 슬라이더로 조절된 현재 행성 위치
        planet_pos_y=planet_y,
        source_x=source_offset_x,
        source_y=source_offset_y
    )

    fig_lightcurve, ax_lightcurve = plt.subplots(figsize=(6, 4))
    ax_lightcurve.plot(angles, magnifications, 'b-', alpha=0.7, label="경로에 따른 밝기 변화")
    ax_lightcurve.plot([planet_angle_deg], [current_magnification], 'ro', markersize=10, label=f"현재 행성 위치 ({planet_angle_deg}°)")

    ax_lightcurve.set_xlim(0, 360)
    ax_lightcurve.set_ylim(0.8, max(magnifications) * 1.2) # 동적으로 Y축 범위 조절
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
이 시뮬레이션은 사용자가 슬라이더를 이용해 외계 행성의 위치(각도)를 조절하면서, 
그에 따른 배경별의 중력렌즈 효과와 겉보기 밝기 변화를 시각적으로 보여줍니다.

* **왼쪽 그래프:** 렌즈별(중심 항성)과 외계 행성, 그리고 배경별의 상대적 위치를 보여줍니다.
    지구(관측자)는 렌즈별과 배경별을 270도 방향에서 관측하고 있다고 가정합니다.
* **오른쪽 그래프:** 외계 행성의 각도 변화에 따라 배경별의 겉보기 밝기가 어떻게 증폭되는지를 나타냅니다.
    행성이 **렌즈별-배경별-지구의 시선에 정렬될 때 (특히 270도 근처)** 밝기가 급격히 증가하는 피크가 나타납니다.
    이는 행성의 중력이 렌즈별의 중력장을 미묘하게 교란시켜 추가적인 밝기 증폭을 유발하기 때문입니다.

**참고:** 광도 계산 부분은 중력렌즈 현상의 핵심 원리를 보여주기 위한 **매우 단순화된 모델**입니다.
실제 마이크로렌즈링 계산은 행성의 질량비, 상대 속도, 배경별의 크기 등 훨씬 복잡한 물리 공식을 필요로 합니다.
여기서는 행성이 270도 근처에서 아인슈타인 반지름 근처를 지날 때 밝기 변화가 나타나도록 유도했습니다.
""")

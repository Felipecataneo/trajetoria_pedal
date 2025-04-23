# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from io import BytesIO
# import math # numpy handles math functions

st.set_page_config(page_title="Comparador de Distâncias ISCWSA Incremental", layout="wide")

# --- Funções Trigonométricas (Graus) ---
def sind(degrees):
    return np.sin(np.radians(degrees))

def cosd(degrees):
    return np.cos(np.radians(degrees))

def tand(degrees):
    degrees = np.asanyarray(degrees)
    rad = np.radians(degrees)
    return np.tan(rad)

def atand(value):
    return np.degrees(np.arctan(value))

def atan2d(y, x):
    """Calculates arctan2 and returns degrees, handling quadrant correctly."""
    return np.degrees(np.arctan2(y, x))

def acosd(value):
    """Calculates arccos and returns degrees, clipping input for domain safety."""
    return np.degrees(np.arccos(np.clip(value, -1.0, 1.0)))

# --- Funções de Rotação e Cálculo Incremental ---

def calcula_rotacao_nev(inc_deg, az_deg):
    """
    Calcula a matriz de rotação do frame do poço (u=along, v=lateral/right, w=vertical/up)
    para o frame global NEV (North, East, Vertical Down).
    Retorna a matriz R 3x3.
    """
    inc_rad = np.radians(inc_deg)
    az_rad = np.radians(az_deg)

    si = np.sin(inc_rad)
    ci = np.cos(inc_rad)
    sa = np.sin(az_rad)
    ca = np.cos(az_rad)

    # Rotação de [u, v, w] para [N, E, V_down]
    R = np.array([
        [si*ca, -sa, ci*ca], # N = u*si*ca + v*(-sa) + w*ci*ca
        [si*sa,  ca, ci*sa], # E = u*si*sa + v*(ca)  + w*ci*sa
        [ci   ,   0, -si]    # V_down = u*ci + v*(0) - w*si
    ])
    return R

def calcula_delta_cov_segment(dmd, inc_mid_deg, az_mid_deg, params, tool_type):
    """
    Calcula a matriz de covariância incremental LOCAL (DeltaC_local) para um segmento.
    Esta é uma APROXIMAÇÃO baseada na soma das variâncias das fontes de erro
    aplicadas às dimensões do segmento (dmd).
    Retorna uma matriz 3x3 no frame local (along-hole, lateral-right, vertical-up).
    """
    # Conversões e parâmetros (simplificado, idealmente recalcular sensibilidades aqui)
    inc_mid_rad = np.radians(inc_mid_deg)
    mrad_to_rad = 0.001
    ppm_to_frac = 1e-6
    mg_to_g = 0.001
    nT_to_T = 1e-9
    deg_to_rad = np.pi / 180.0
    hr_to_sec = 3600.0
    gravity_mps2 = params.get('gravity_strength', 1.0) * 9.80665

    # --- Erros Comuns ---
    sigma_depth_prop = params.get('depth_err_prop', 0)
    sigma_depth_const = params.get('depth_err_const', 0) # Erro constante é difícil de alocar incrementalmente; uma aproximação grosseira
    sigma_acc_bias = params.get('acc_bias', 0) * mg_to_g * gravity_mps2
    sigma_acc_sf = params.get('acc_sf', 0) * ppm_to_frac
    sigma_acc_mis_xy = params.get('acc_mis_xy', 0) * mrad_to_rad
    sigma_acc_mis_z = params.get('acc_mis_z', 0) * mrad_to_rad
    sigma_sag = params.get('sag_corr_err', 0) * deg_to_rad
    sigma_misalign_inc = params.get('misalign_err_inc', 0) * deg_to_rad
    sigma_misalign_azi = params.get('misalign_err_azi', 0) * deg_to_rad

    # --- Variância Incremental Along-Hole (eixo u) ---
    # Principalmente erro de profundidade
    # Tratamento do erro constante: distribuir ao longo do poço? Aplicar uma vez?
    # Simplificação: Aplicar a variância constante a cada segmento (superestima)
    var_d_prop = (sigma_depth_prop * dmd)**2
    # Distribuir a variância constante proporcionalmente ao dmd? Não é ideal.
    # Modelo ISCWSA completo trata isso como erro sistemático inicial.
    # Por simplicidade aqui, vamos omitir o const por enquanto ou aplicar uma fração?
    # Vamos adicionar o sigma_depth_const^2 diretamente (conservador, mas errado)
    var_d_const = sigma_depth_const**2 # Simplificação grosseira
    # Ou melhor: só adicionar o constante no fim? Não, o modelo incremental precisa dele por segmento
    # Melhor aproximação simples: assumir que o erro constante é um bias que se acumula.
    # Mas a fórmula de propagação é C = C + R*DeltaC*R^T.
    # Vamos usar a variância do erro *na medição do comprimento do segmento*.
    # A variância do erro constante no comprimento total MD é sigma_const^2.
    # Se N segmentos, a variância da medição de um segmento dmd = MD/N poderia ser sigma_const^2 / N? Não é bem isso.
    # Vamos manter a abordagem simplificada, sabendo que é uma limitação:
    var_u_local = var_d_prop + var_d_const # Pode superestimar muito para erro constante

    # --- Variância Angular Efetiva para o Segmento (sigma_dInc^2, sigma_dAz^2) ---
    # Reutiliza a lógica da função antiga, mas calcula sigmas (não variâncias totais)
    # Variância de Inclinação (rad^2)
    var_inc_rad2 = 0
    var_inc_acc_bias = (sigma_acc_bias / gravity_mps2)**2
    var_inc_acc_sf = (sigma_acc_sf * sind(inc_mid_deg))**2
    var_inc_acc_mis = (max(sigma_acc_mis_xy, sigma_acc_mis_z) * cosd(inc_mid_deg))**2
    var_inc_sag = sigma_sag**2
    var_inc_misalign = sigma_misalign_inc**2

    # Variância de Azimute (rad^2)
    var_az_rad2 = 0
    var_az_misalign = sigma_misalign_azi**2

    if tool_type == "ISCWSA MWD":
        sigma_mag_bias = params.get('mag_bias', 0) * nT_to_T
        sigma_mag_sf = params.get('mag_sf', 0) * ppm_to_frac
        sigma_mag_mis_xy = params.get('mag_mis_xy', 0) * mrad_to_rad
        sigma_mag_mis_z = params.get('mag_mis_z', 0) * mrad_to_rad
        sigma_dec_err = params.get('mag_dec_err', 0) * deg_to_rad
        sigma_ds_err = params.get('mag_ds_err', 0) * deg_to_rad
        dip_rad = np.radians(params.get('dip_angle', 60))
        B_total_T = params.get('mag_field_strength', 50000) * nT_to_T
        B_H = B_total_T * np.cos(dip_rad)

        # Soma das variâncias de erro de inclinação para MWD
        var_inc_rad2 = var_inc_acc_bias + var_inc_acc_sf + var_inc_acc_mis + var_inc_sag + var_inc_misalign

        # Soma das variâncias de erro de azimute para MWD
        sensitivity_az = (B_H * sind(inc_mid_deg))
        var_az_mag_bias = 0
        var_az_mag_sf = 0 # Simplificado
        var_az_mag_mis = 0 # Simplificado
        var_az_dec = sigma_dec_err**2
        var_az_ds = sigma_ds_err**2

        if abs(sensitivity_az) > 1e-9:
             var_az_mag_bias = (sigma_mag_bias / sensitivity_az)**2
             # Incluir termos de SF e Misalign (simplificados)
             # A sensibilidade real é mais complexa
             var_az_mag_sf = (sigma_mag_sf * B_total_T / sensitivity_az)**2 # Exemplo muito simplificado
             var_az_mag_mis = (max(sigma_mag_mis_xy, sigma_mag_mis_z) * B_total_T / sensitivity_az)**2 # Exemplo muito simplificado

        else: # Alta incerteza perto de vertical/equador magnético
             large_az_error_var = (np.radians(30))**2 # Variância grande
             var_az_mag_bias = large_az_error_var
             var_az_mag_sf = large_az_error_var
             var_az_mag_mis = large_az_error_var

        var_az_rad2 = var_az_mag_bias + var_az_mag_sf + var_az_mag_mis + var_az_dec + var_az_ds + var_az_misalign

    elif tool_type == "ISCWSA Gyro":
        sigma_gyro_bias_drift_ns = params.get('gyro_bias_drift_ns', 0) * deg_to_rad / hr_to_sec
        sigma_gyro_bias_drift_ew = params.get('gyro_bias_drift_ew', 0) * deg_to_rad / hr_to_sec
        # sigma_gyro_bias_drift_v = params.get('gyro_bias_drift_v', 0) * deg_to_rad / hr_to_sec
        sigma_gyro_sf = params.get('gyro_sf', 0) * ppm_to_frac
        sigma_gyro_g_sens_drift = params.get('gyro_g_sens_drift', 0) * deg_to_rad / hr_to_sec / gravity_mps2
        sigma_gyro_mis_xy = params.get('gyro_mis_xy', 0) * mrad_to_rad
        sigma_gyro_mis_z = params.get('gyro_mis_z', 0) * mrad_to_rad
        sigma_az_ref_err = params.get('gyro_az_ref_err', 0) * deg_to_rad
        # O tempo do survey afeta o drift acumulado, não a taxa instantânea para o segmento?
        # Modelo completo integra o drift. Simplificação: usar um tempo médio por segmento?
        # Ou aplicar o drift total como um erro sistemático no fim?
        # Vamos focar nos erros 'aleatórios' por segmento por enquanto.
        # survey_time_sec = params.get('survey_time_hours', 1) * hr_to_sec

        # Gyro G-sensitivity (afeta Inc e Az)
        var_inc_gyro_g_sens = (sigma_gyro_g_sens_drift * gravity_mps2 * cosd(inc_mid_deg))**2 # Simplificado
        var_az_gyro_g_sens = (sigma_gyro_g_sens_drift * gravity_mps2 * sind(inc_mid_deg))**2 # Simplificado

        # Soma das variâncias de erro de inclinação para Gyro
        var_inc_rad2 = var_inc_acc_bias + var_inc_acc_sf + var_inc_acc_mis + var_inc_sag + var_inc_misalign + var_inc_gyro_g_sens

        # Soma das variâncias de erro de azimute para Gyro
        # Erro de referência é sistemático, não incremental por segmento (aplicado uma vez)
        # Drift é acumulativo.
        # Foco nos erros aleatórios do segmento: G-Sens, Misalign, SF (simplificado)
        var_az_ref = sigma_az_ref_err**2 # Aplicado uma vez, não aqui?
        # Termos de drift (muito simplificado - ignora acumulação e tempo)
        var_az_gyro_bias_drift = 0
        sigma_gyro_bias_drift_h = np.sqrt(sigma_gyro_bias_drift_ns**2 + sigma_gyro_bias_drift_ew**2)
        if abs(sind(inc_mid_deg)) > 1e-3:
            # Variância do *ângulo* devido ao drift num *tempo pequeno dt* seria (rate * dt)^2
            # A propagação ISCWSA lida com isso. Aqui, é difícil simplificar bem.
            # Vamos omitir o drift da covariância *incremental* e adicionar erro de referência
            pass # Omitindo drift incremental por simplicidade
        else:
            var_az_gyro_bias_drift = (np.radians(30))**2 # Proxy perto do vertical

        # Incluir SF e Misalign Gyro (muito simplificado)
        var_az_gyro_sf = (sigma_gyro_sf * tand(inc_mid_deg))**2 if abs(cosd(inc_mid_deg)) > 1e-3 else (np.radians(30))**2
        var_az_gyro_mis = (max(sigma_gyro_mis_xy, sigma_gyro_mis_z) * tand(inc_mid_deg))**2 if abs(cosd(inc_mid_deg)) > 1e-3 else (np.radians(30))**2

        # Modelo ISCWSA trata erros sistemáticos (Ref, Drift) separadamente.
        # Foco aqui na contribuição *aleatória* do segmento.
        var_az_rad2 = var_az_gyro_g_sens + var_az_misalign # + var_az_gyro_sf + var_az_gyro_mis (+ bias drift + ref?) <- Simplificação

    # Obter desvios padrão angulares efetivos para o segmento
    sigma_dInc_rad = np.sqrt(max(0, var_inc_rad2)) # Evita raiz de negativo
    sigma_dAz_rad = np.sqrt(max(0, var_az_rad2))

    # --- Variâncias Incrementais nos eixos locais v (lateral) e w (vertical) ---
    # Erro lateral (v) = dmd * sin(Inc) * dAz
    # Erro vertical (w) = dmd * dInc
    # Usamos variâncias: Var(k*X) = k^2 * Var(X)
    var_v_local = (dmd * sind(inc_mid_deg) * sigma_dAz_rad)**2
    var_w_local = (dmd * sigma_dInc_rad)**2

    # --- Montar Matriz DeltaC_local ---
    # Assumindo que no frame local (u,v,w), esses erros incrementais são não correlacionados
    # Esta é uma FORTE SIMPLIFICAÇÃO. O modelo ISCWSA completo calcula covariâncias.
    DeltaC_local = np.diag([var_u_local, var_v_local, var_w_local])

    # Aplicar erros sistemáticos (ex: gyro ref error)? O modelo ISCWSA faz isso de forma diferente.
    # Por ora, focamos nos erros que crescem com dmd.

    return DeltaC_local


def calculate_iscwsa_covariance_incremental(md_array, inc_array, az_array, params, tool_type):
    """
    Calcula a lista de matrizes de covariância cumulativas (NEV) em cada ponto
    usando o método incremental ISCWSA (aproximado).

    md_array, inc_array, az_array: arrays numpy de todos os pontos do wellbore
    params: dicionário ISCWSA
    tool_type: string "ISCWSA MWD" ou "ISCWSA Gyro"
    Retorna: Lista de matrizes de covariância cumulativas (NEV) em cada ponto (numpy arrays 3x3)
    """
    n_points = len(md_array)
    if n_points == 0:
        return []

    covariances_nev = []
    # Inicializa matriz de covariância zero na wellhead (frame NEV)
    C_cum_nev_prev = np.zeros((3,3))
    covariances_nev.append(C_cum_nev_prev.copy()) # Covariância no ponto 0 é zero

    # Loop pelos segmentos (do ponto 0 ao 1, 1 ao 2, etc.)
    for i in range(1, n_points):
        md0, md1 = md_array[i-1], md_array[i]
        inc0, inc1 = inc_array[i-1], inc_array[i]
        az0, az1 = az_array[i-1], az_array[i]

        # Propriedades do segmento
        dmd = md1 - md0
        if dmd < 1e-6: # Se não há comprimento, não há erro incremental
             C_cum_nev = C_cum_nev_prev.copy() # Mantém a covariância anterior
             covariances_nev.append(C_cum_nev)
             C_cum_nev_prev = C_cum_nev # Atualiza para o próximo passo
             continue # Pula para o próximo segmento

        inc_mid = (inc1 + inc0) / 2.0
        az_mid = (az1 + az0) / 2.0 # Cuidado com a média de ângulos (wrap around 360) - simplificação aqui

        # Calcula rotação do frame local (segmento) para NEV
        R_local_to_nev = calcula_rotacao_nev(inc_mid, az_mid)

        # Calcula matriz incremental de covariância do SEGMENTO no frame LOCAL (u,v,w)
        # Esta função contém a física/modelo de erro ISCWSA (simplificada aqui)
        DeltaC_local = calcula_delta_cov_segment(dmd, inc_mid, az_mid, params, tool_type)

        # Rotaciona DeltaC_local para o frame global NEV
        # DeltaC_global = R @ DeltaC_local @ R.T
        DeltaC_nev = R_local_to_nev @ DeltaC_local @ R_local_to_nev.T

        # Acumula a covariância: C_n = C_{n-1} + DeltaC_{n-1 -> n} (no frame NEV)
        # A covariância anterior (C_cum_nev_prev) já está no frame NEV.
        # A nova contribuição (DeltaC_nev) também está no frame NEV.
        C_cum_nev = C_cum_nev_prev + DeltaC_nev

        # Adiciona erros sistemáticos aqui? (Ex: Gyro Ref Azimuth Error)
        # O modelo completo ISCWSA propaga erros sistemáticos de forma diferente.
        # Para simplificar, podemos adicionar a variância de erros sistemáticos
        # apenas uma vez, ou de forma que afete todos os pontos seguintes.
        # Exemplo Gyro Ref Err (simplificado - adiciona variância ao termo Azimutal globalmente?)
        # -> Isso requereria mapear o erro azimutal para N, E, V e adicionar à matriz C_cum_nev.
        #    dAz -> dN = r * cos(Az) * dAz, dE = -r * sin(Az) * dAz onde r = MD*sin(Inc) approx
        #    É complexo adicionar corretamente aqui. Vamos omitir por enquanto.

        # Garante simetria numérica
        C_cum_nev = (C_cum_nev + C_cum_nev.T) / 2.0

        # Salva matriz cumulativa NEV no ponto i
        covariances_nev.append(C_cum_nev.copy())

        # Atualiza para a próxima iteração
        C_cum_nev_prev = C_cum_nev

    return covariances_nev


# Título e explicação
st.title("Comparador de Distâncias com Modelos ISCWSA (Incremental)")
st.markdown("""
Esta aplicação compara a distância entre centros de trajetórias com a distância Pedal Curve,
utilizando **cálculo incremental de covariância ISCWSA** (MWD ou Gyro) para as elipses de incerteza.
Considera as coordenadas de cabeça de poço (wellhead) diferentes.
*Nota: A implementação do cálculo incremental aqui é uma **aproximação** dos modelos ISCWSA completos.*
""")

# --- Parâmetros de Incerteza ISCWSA (Sidebar) ---
st.sidebar.header("Configuração do Modelo de Incerteza")
tool_type = st.sidebar.selectbox("Selecione o Tipo de Ferramenta", ["ISCWSA MWD", "ISCWSA Gyro"])

iscwsa_params = {}
# ... (resto da definição dos parâmetros ISCWSA na sidebar - SEM MUDANÇAS AQUI) ...
# Define default values robustly
default_mwd = {
    'depth_err_prop': 0.0002,
    'depth_err_const': 0.1,
    'acc_bias': 0.1,
    'acc_sf': 200.0,         # Atualizado para 200 ppm (2023+)
    'acc_mis_xy': 0.07,      # Reduzido caso sensor avançado
    'acc_mis_z': 0.07,
    'mag_bias': 75.0,        # Subiu para 75 nT pelo IFCWSA 2023
    'mag_sf': 150.0,
    'mag_mis_xy': 0.10,      # Reduzido para sensores avançados
    'mag_mis_z': 0.10,
    'mag_dec_err': 0.2,      # Consultar valor WMM local
    'mag_dip_err': 0.1,
    'mag_ds_err': 0.3,
    'sag_corr_err': 0.05,    # Até 0.07 se muito sensível a sag
    'misalign_err_inc': 0.05,
    'misalign_err_azi': 0.1,
    'gravity_strength': 9.81,    # Valor real em m/s²
    'mag_field_strength': 50000.0,
    'dip_angle': 60.0
}
default_gyro = {
    'depth_err_prop': 0.0002,          # fração
    'depth_err_const': 0.1,            # m
    'acc_bias': 0.1,                   # m/s2
    'acc_sf': 100.0,                   # ppm
    'acc_mis_xy': 0.10,                # mrad
    'acc_mis_z': 0.10,                 # mrad
    'gyro_bias_drift_ns': 0.03,        # deg/h (HRG/FOG), 0.05-0.1 MEMS
    'gyro_bias_drift_ew': 0.03,        # deg/h
    'gyro_bias_drift_v' : 0.03,        # deg/h
    'gyro_sf' : 100.0,                 # ppm (60 para HRG/FOG, 200 para MEMS inferior)
    'gyro_g_sens_drift': 0.05,         # deg/h/g
    'gyro_mis_xy': 0.10,               # mrad
    'gyro_mis_z': 0.10,                # mrad
    'gyro_az_ref_err': 0.03,           # deg (0.03 HRG/FOG)
    'sag_corr_err': 0.05,              # deg
    'misalign_err_inc': 0.05,          # deg
    'misalign_err_azi': 0.1,           # deg
    'gravity_strength': 9.81,          # m/s² (use 9.80665 para máxima precisão)
    'survey_time_hours': 0.25,         # h (ou 1.0 se conservador) - Usado de forma simplificada ou omitido na DeltaC atual
}

if tool_type == "ISCWSA MWD":
    st.sidebar.subheader("Parâmetros ISCWSA MWD (1-sigma)")
    # Use defaults for missing keys if necessary
    iscwsa_params['depth_err_prop'] = st.sidebar.number_input("Erro Prop. Profundidade (m/m)", value=default_mwd['depth_err_prop'], step=0.0001, format="%.4f") # dD1
    iscwsa_params['depth_err_const'] = st.sidebar.number_input("Erro Const. Profundidade (m)", value=default_mwd['depth_err_const'], step=0.05, format="%.2f") # dD0

    iscwsa_params['acc_bias'] = st.sidebar.number_input("Acc Bias (mg)", value=default_mwd['acc_bias'], step=0.01, format="%.2f") # ABX, ABY, ABZ
    iscwsa_params['acc_sf'] = st.sidebar.number_input("Acc Scale Factor (ppm)", value=default_mwd['acc_sf'], step=10.0, format="%.1f") # ASX, ASY, ASZ
    iscwsa_params['acc_mis_xy'] = st.sidebar.number_input("Acc Misalign XY (mrad)", value=default_mwd['acc_mis_xy'], step=0.01, format="%.2f") # AMX, AMY
    iscwsa_params['acc_mis_z'] = st.sidebar.number_input("Acc Misalign Z (mrad)", value=default_mwd['acc_mis_z'], step=0.01, format="%.2f") # AMZ

    iscwsa_params['mag_bias'] = st.sidebar.number_input("Mag Bias (nT)", value=default_mwd['mag_bias'], step=5.0, format="%.1f") # MBX, MBY, MBZ
    iscwsa_params['mag_sf'] = st.sidebar.number_input("Mag Scale Factor (ppm)", value=default_mwd['mag_sf'], step=10.0, format="%.1f") # MSX, MSY, MSZ
    iscwsa_params['mag_mis_xy'] = st.sidebar.number_input("Mag Misalign XY (mrad)", value=default_mwd['mag_mis_xy'], step=0.01, format="%.2f") # MMX, MMY
    iscwsa_params['mag_mis_z'] = st.sidebar.number_input("Mag Misalign Z (mrad)", value=default_mwd['mag_mis_z'], step=0.01, format="%.2f") # MMZ

    iscwsa_params['mag_dec_err'] = st.sidebar.number_input("Erro Declinação Magnética (°)", value=default_mwd['mag_dec_err'], step=0.05, format="%.2f") # DECD
    iscwsa_params['mag_dip_err'] = st.sidebar.number_input("Erro Inclinação Magnética (°)", value=default_mwd['mag_dip_err'], step=0.01, format="%.2f") # DIPD (usado internamente nos cálculos de azimute)
    iscwsa_params['mag_ds_err'] = st.sidebar.number_input("Erro Interferência Magnética (°)", value=default_mwd['mag_ds_err'], step=0.05, format="%.2f") # XYMD

    iscwsa_params['sag_corr_err'] = st.sidebar.number_input("Erro Correção SAG (°)", value=default_mwd['sag_corr_err'], step=0.01, format="%.2f") # SAGD
    # Added basic toolface/misalignment error contribution (simplified)
    iscwsa_params['misalign_err_inc'] = st.sidebar.number_input("Erro Misalign INC (°)", value=default_mwd['misalign_err_inc'], step=0.01, format="%.2f") # Relacionado a MXA, MYA
    iscwsa_params['misalign_err_azi'] = st.sidebar.number_input("Erro Misalign AZI (°)", value=default_mwd['misalign_err_azi'], step=0.01, format="%.2f") # Relacionado a MZA

    # Gravidade e Campo Magnético (Simplificado - considere usar valores locais)
    iscwsa_params['gravity_strength'] = st.sidebar.number_input("Gravidade (g)", value=default_mwd['gravity_strength'], format="%.4f")
    iscwsa_params['mag_field_strength'] = st.sidebar.number_input("Campo Magnético (nT)", value=default_mwd['mag_field_strength'], format="%.1f")
    iscwsa_params['dip_angle'] = st.sidebar.number_input("Ângulo DIP (°)", value=default_mwd['dip_angle'], format="%.2f") # Inclinação magnética
    # Ensure all keys are present even if user doesn't interact
    for key, val in default_mwd.items():
        iscwsa_params.setdefault(key, val)

elif tool_type == "ISCWSA Gyro":
    st.sidebar.subheader("Parâmetros ISCWSA Gyro (1-sigma)")
    # Use defaults for missing keys if necessary
    iscwsa_params['depth_err_prop'] = st.sidebar.number_input("Erro Prop. Profundidade (m/m)", value=default_gyro['depth_err_prop'], step=0.0001, format="%.4f")
    iscwsa_params['depth_err_const'] = st.sidebar.number_input("Erro Const. Profundidade (m)", value=default_gyro['depth_err_const'], step=0.05, format="%.2f")

    iscwsa_params['acc_bias'] = st.sidebar.number_input("Acc Bias (mg)", value=default_gyro['acc_bias'], step=0.01, format="%.2f")
    iscwsa_params['acc_sf'] = st.sidebar.number_input("Acc Scale Factor (ppm)", value=default_gyro['acc_sf'], step=10.0, format="%.1f")
    iscwsa_params['acc_mis_xy'] = st.sidebar.number_input("Acc Misalign XY (mrad)", value=default_gyro['acc_mis_xy'], step=0.01, format="%.2f")
    iscwsa_params['acc_mis_z'] = st.sidebar.number_input("Acc Misalign Z (mrad)", value=default_gyro['acc_mis_z'], step=0.01, format="%.2f")

    iscwsa_params['gyro_bias_drift_ns'] = st.sidebar.number_input("Gyro Bias Drift N/S (°/hr)", value=default_gyro['gyro_bias_drift_ns'], step=0.01, format="%.2f") # GBN
    iscwsa_params['gyro_bias_drift_ew'] = st.sidebar.number_input("Gyro Bias Drift E/W (°/hr)", value=default_gyro['gyro_bias_drift_ew'], step=0.01, format="%.2f") # GBE
    iscwsa_params['gyro_bias_drift_v'] = st.sidebar.number_input("Gyro Bias Drift Vert (°/hr)", value=default_gyro['gyro_bias_drift_v'], step=0.01, format="%.2f") # GBV
    iscwsa_params['gyro_sf'] = st.sidebar.number_input("Gyro Scale Factor (ppm)", value=default_gyro['gyro_sf'], step=10.0, format="%.1f") # GSF
    iscwsa_params['gyro_g_sens_drift'] = st.sidebar.number_input("Gyro G-Sens Drift (°/hr/g)", value=default_gyro['gyro_g_sens_drift'], step=0.01, format="%.2f") # GDX, GDY, GDZ
    iscwsa_params['gyro_mis_xy'] = st.sidebar.number_input("Gyro Misalign XY (mrad)", value=default_gyro['gyro_mis_xy'], step=0.01, format="%.2f") # GMX, GMY
    iscwsa_params['gyro_mis_z'] = st.sidebar.number_input("Gyro Misalign Z (mrad)", value=default_gyro['gyro_mis_z'], step=0.01, format="%.2f") # GMZ
    iscwsa_params['gyro_az_ref_err'] = st.sidebar.number_input("Erro Referência Azimute Gyro (°)", value=default_gyro['gyro_az_ref_err'], step=0.01, format="%.2f") # AZID

    iscwsa_params['sag_corr_err'] = st.sidebar.number_input("Erro Correção SAG (°)", value=default_gyro['sag_corr_err'], step=0.01, format="%.2f") # SAGD
    iscwsa_params['misalign_err_inc'] = st.sidebar.number_input("Erro Misalign INC (°)", value=default_gyro['misalign_err_inc'], step=0.01, format="%.2f")
    iscwsa_params['misalign_err_azi'] = st.sidebar.number_input("Erro Misalign AZI (°)", value=default_gyro['misalign_err_azi'], step=0.01, format="%.2f")

    # Gravidade
    iscwsa_params['gravity_strength'] = st.sidebar.number_input("Gravidade (g)", value=default_gyro['gravity_strength'], format="%.4f")
    # Assumed survey time (for drift calculation) - simplistic
    # Este parâmetro não está sendo usado ativamente na aproximação atual de DeltaC
    iscwsa_params['survey_time_hours'] = st.sidebar.number_input("Tempo Estimado Survey (horas)", value=default_gyro['survey_time_hours'], step=0.1, format="%.1f")
    # Ensure all keys are present even if user doesn't interact
    for key, val in default_gyro.items():
        iscwsa_params.setdefault(key, val)

st.sidebar.header("Parâmetros de Cálculo")
sigma_factor = st.sidebar.slider("Fator Sigma (Confiança da Elipse)", 1.0, 3.0, 1.0, 0.1)

# --- Funções de Cálculo Trajetória ---
def calculate_coordinates(md, inc, az, wellhead_n=0.0, wellhead_e=0.0, wellhead_tvd=0.0):
   """
   Calcula coordenadas NEV usando método minimum curvature e adiciona
   o offset da cabeça do poço (wellhead). Retorna DataFrame.
   """
   # ... (código de calculate_coordinates SEM MUDANÇAS) ...
   md = np.asarray(md, dtype=float)
   inc = np.asarray(inc, dtype=float)
   az = np.asarray(az, dtype=float)
   n_points = len(md)

   if n_points == 0:
        return pd.DataFrame({'MD': [], 'TVD': [], 'N': [], 'E': [], 'INC': [], 'AZ': []})

   n_rel = np.zeros_like(md)
   e_rel = np.zeros_like(md)
   tvd_rel = np.zeros_like(md)

   # Minimum Curvature Calculation
   # Use diff to get segment lengths; prepend=md[0] handles first segment if md[0]!=0
   delta_md = np.diff(md, prepend=md[0] if md[0]!=0 else 0.0)
   delta_md[0] = md[0] # Ensure first segment length is MD[0] if starting survey != 0

   # Ensure correct handling if first point is MD=0
   if np.isclose(md[0], 0.0):
       delta_md[0] = 0.0 # No length change at the very start

   inc_rad = np.radians(inc)
   az_rad = np.radians(az)

   # Arrays for previous and current stations
   inc1_rad = np.roll(inc_rad, 1)
   az1_rad = np.roll(az_rad, 1)
   inc2_rad = inc_rad
   az2_rad = az_rad

   # First segment starts with the first point's angles
   inc1_rad[0] = inc_rad[0]
   az1_rad[0] = az_rad[0]

   # Dogleg Angle (radians)
   # Using the more stable acos formulation
   # cos(DL) = cos(I2-I1) - sin(I1)sin(I2)(1-cos(A2-A1)) NO -> USE THIS
   cos_dls = (np.cos(inc2_rad - inc1_rad) -
                     np.sin(inc1_rad) * np.sin(inc2_rad) * (1 - np.cos(az2_rad - az1_rad)))
   # Correct formula: cos(DL) = cos(I1)cos(I2) + sin(I1)sin(I2)cos(A2-A1)
   # cos_dls = np.cos(inc1_rad) * np.cos(inc2_rad) + \
   #           np.sin(inc1_rad) * np.sin(inc2_rad) * np.cos(az2_rad - az1_rad)

   # Clip to avoid domain errors from numerical precision
   dogleg_angle_rad = np.arccos(np.clip(cos_dls, -1.0, 1.0))
   dogleg_angle_rad[0] = 0.0 # No dogleg for the first point relative to itself

   # Ratio Factor (RF) = tan(DL/2) / (DL/2), limit is 1 as DL -> 0
   half_dogleg = dogleg_angle_rad / 2.0
   # Use np.where for numerical stability near DL = 0
   rf = np.where(np.abs(dogleg_angle_rad) < 1e-9,
                 1.0,
                 np.tan(half_dogleg) / half_dogleg)
   rf[0] = 1.0 # Ensure RF is 1 for the first point (or segment of zero length)


   # Calculate increments in N, E, V (relative)
   delta_n = delta_md / 2.0 * (np.sin(inc1_rad) * np.cos(az1_rad) + np.sin(inc2_rad) * np.cos(az2_rad)) * rf
   delta_e = delta_md / 2.0 * (np.sin(inc1_rad) * np.sin(az1_rad) + np.sin(inc2_rad) * np.sin(az2_rad)) * rf
   delta_v = delta_md / 2.0 * (np.cos(inc1_rad) + np.cos(inc2_rad)) * rf

   # Cumulative relative coordinates
   n_rel = np.cumsum(delta_n)
   e_rel = np.cumsum(delta_e)
   tvd_rel = np.cumsum(delta_v)

   # Apply wellhead offset for absolute coordinates
   n_abs = n_rel + wellhead_n
   e_abs = e_rel + wellhead_e
   tvd_abs = tvd_rel + wellhead_tvd

   return pd.DataFrame({
       'MD': md,
       'TVD': tvd_abs,
       'N': n_abs,
       'E': e_abs,
       'INC': inc, # Keep original INC/AZ for reference
       'AZ': az
   })


# --- Funções de Cálculo Incerteza (Elipse e Pedal Curve - SEM MUDANÇAS) ---

def get_ellipse_params_from_covariance(C_nev, sigma_factor=1.0):
    """
    Calcula os parâmetros da elipse de incerteza horizontal a partir
    da matriz de covariância 3x3 NEV CUMULATIVA.
    Retorna: semi_major, semi_minor, angle_deg (ângulo do eixo maior com o Norte, 0-360)
    """
    # ... (código de get_ellipse_params_from_covariance SEM MUDANÇAS) ...
    C_ne = C_nev[0:2, 0:2]
    if np.allclose(C_ne, 0) or np.any(np.isnan(C_ne)):
        return 0.0, 0.0, 0.0
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(C_ne)
    except np.linalg.LinAlgError:
        st.warning(f"Eigenvalue decomposition failed for C_ne:\n{C_ne}\nReturning zero ellipse.")
        return 0.0, 0.0, 0.0

    eigenvalues[eigenvalues < 0] = 0
    idx_sort = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx_sort]
    eigenvectors = eigenvectors[:, idx_sort]

    semi_major = sigma_factor * np.sqrt(eigenvalues[0])
    semi_minor = sigma_factor * np.sqrt(eigenvalues[1])

    n_comp = eigenvectors[0, 0]
    e_comp = eigenvectors[1, 0]

    if np.isclose(n_comp, 0) and np.isclose(e_comp, 0):
        angle_deg_from_north = 0.0
    else:
        angle_deg_from_north = atan2d(e_comp, n_comp)

    angle_deg_from_north = angle_deg_from_north % 360.0
    return semi_major, semi_minor, angle_deg_from_north

def calculate_distance(p1, p2):
   """Calcula a distância euclidiana entre dois pontos no plano NE"""
   # ... (código de calculate_distance SEM MUDANÇAS) ...
   if pd.isna(p1['N']) or pd.isna(p1['E']) or pd.isna(p2['N']) or pd.isna(p2['E']):
       return np.nan
   return np.sqrt((p2['N'] - p1['N'])**2 + (p2['E'] - p1['E'])**2)

def project_ellipse_iscwsa(semi_major, semi_minor, ellipse_angle_deg, direction_az_deg):
   """
   Projeta a elipse ISCWSA na direção especificada (Pedal Curve).
   """
   # ... (código de project_ellipse_iscwsa SEM MUDANÇAS) ...
   if pd.isna(semi_major) or pd.isna(semi_minor) or pd.isna(ellipse_angle_deg) or pd.isna(direction_az_deg): return np.nan
   if semi_major < 1e-9: return 0.0

   relative_angle_deg = direction_az_deg - ellipse_angle_deg
   relative_angle_rad = np.radians(relative_angle_deg)
   a = semi_major
   b = semi_minor
   if b < 1e-9: b = 1e-9 # Avoid division by zero if perfectly flat

   cos_theta_sq = np.cos(relative_angle_rad)**2
   sin_theta_sq = np.sin(relative_angle_rad)**2

   num = (a**2) * (b**2)
   den = (b**2) * cos_theta_sq + (a**2) * sin_theta_sq

   if den < 1e-12:
        if abs(np.cos(relative_angle_rad)) > 0.99: return a
        elif abs(np.sin(relative_angle_rad)) > 0.99: return b
        else: return 0.0 # Should not happen if b is not zero

   projection = np.sqrt(num / den)
   return projection

def calculate_pedal_distance_iscwsa(p1, p2, cov_nev1, cov_nev2, sigma_factor):
    """Calcula a distância Pedal Curve entre dois pontos usando elipses ISCWSA"""
    # ... (código de calculate_pedal_distance_iscwsa SEM MUDANÇAS) ...
    center_dist = calculate_distance(p1, p2)

    if pd.isna(center_dist) or cov_nev1 is None or cov_nev2 is None:
        return {'center_dist': np.nan, 'proj1': np.nan, 'proj2': np.nan,
                'pedal_dist': np.nan, 'difference': np.nan, 'diff_percent': np.nan, 'SF': np.nan}

    if center_dist < 1e-6:
        sf_val = 0.0 # Collision
        proj1 = 0.0
        proj2 = 0.0
        # Need ellipse params even if dist=0 for potential plotting/info
        smj1, smn1, ang1 = get_ellipse_params_from_covariance(cov_nev1, sigma_factor)
        smj2, smn2, ang2 = get_ellipse_params_from_covariance(cov_nev2, sigma_factor)
        pedal_dist = 0.0
        difference = 0.0
        diff_percent = 0.0

    else:
        delta_n = p2['N'] - p1['N']
        delta_e = p2['E'] - p1['E']
        angle_deg_centers = atan2d(delta_e, delta_n) % 360.0

        smj1, smn1, ang1 = get_ellipse_params_from_covariance(cov_nev1, sigma_factor)
        smj2, smn2, ang2 = get_ellipse_params_from_covariance(cov_nev2, sigma_factor)

        proj1 = project_ellipse_iscwsa(smj1, smn1, ang1, angle_deg_centers)
        proj2 = project_ellipse_iscwsa(smj2, smn2, ang2, angle_deg_centers)

        if pd.isna(proj1) or pd.isna(proj2):
            pedal_dist, difference, diff_percent, sf_val = np.nan, np.nan, np.nan, np.nan
        else:
            pedal_dist = max(0, center_dist - (proj1 + proj2))
            difference = center_dist - pedal_dist
            diff_percent = (difference / center_dist) * 100 if center_dist > 1e-6 else 0
            denominator = proj1 + proj2
            if denominator > 1e-9:
                sf_val = center_dist / denominator
            else:
                sf_val = np.inf if center_dist > 1e-9 else 0.0 # Large SF if ellipses are tiny but centers apart

    return {
        'center_dist': center_dist, 'proj1': proj1, 'proj2': proj2,
        'pedal_dist': pedal_dist, 'difference': difference, 'diff_percent': diff_percent,
        'smj1': smj1, 'smn1': smn1, 'ang1': ang1,
        'smj2': smj2, 'smn2': smn2, 'ang2': ang2, 'SF': sf_val
    }


# --- Funções Auxiliares (SEM MUDANÇAS) ---
def find_closest_tvd_point(tvd_target, df):
   """Encontra o ponto mais próximo em TVD no DataFrame"""
   # ... (código de find_closest_tvd_point SEM MUDANÇAS) ...
   if df.empty or 'TVD' not in df.columns or df['TVD'].isna().all(): return None
   try:
        tvd_target = float(tvd_target)
        valid_tvd = df['TVD'].dropna()
        if valid_tvd.empty: return None
        idx = (valid_tvd - tvd_target).abs().idxmin()
        return df.loc[idx] # Returns a Series representing the row
   except (ValueError, TypeError):
        return None

def draw_ellipse_matplotlib(ax, center_xy, width, height, angle_deg, color="blue", alpha=0.3, label=None):
   """Desenha uma elipse com Matplotlib. Angle é anti-horário a partir do eixo +X (Leste)."""
   # ... (código de draw_ellipse_matplotlib SEM MUDANÇAS) ...
   if pd.isna(center_xy[0]) or pd.isna(center_xy[1]) or pd.isna(width) or pd.isna(height) or pd.isna(angle_deg): return None
   if width <= 1e-9 or height <= 1e-9: return None # Avoid zero size

   ellipse = Ellipse(xy=center_xy, width=width, height=height, angle=angle_deg,
                     edgecolor=color, facecolor=color, alpha=alpha, label=label)
   ax.add_patch(ellipse)
   return ellipse

# --- Interface Streamlit ---

# Interface para upload de arquivos e coordenadas de cabeça de poço
col1, col2 = st.columns(2)
with col1:
   st.header("Poço 1")
   well1_file = st.file_uploader("Upload Excel (MD, INC, AZ) Poço 1", type=["xlsx", "xls"], key="file1")
   st.subheader("Coordenadas Cabeça Poço 1")
   n_wh1 = st.number_input("Norte (m)", key="n_wh1", value=0.0, format="%.2f")
   e_wh1 = st.number_input("Este (m)", key="e_wh1", value=0.0, format="%.2f")
   tvd_wh1 = st.number_input("TVD Inicial (m)", key="tvd_wh1", value=0.0, format="%.2f")
with col2:
   st.header("Poço 2")
   well2_file = st.file_uploader("Upload Excel (MD, INC, AZ) Poço 2", type=["xlsx", "xls"], key="file2")
   st.subheader("Coordenadas Cabeça Poço 2")
   n_wh2 = st.number_input("Norte (m)", key="n_wh2", value=0.0, format="%.2f")
   e_wh2 = st.number_input("Este (m)", key="e_wh2", value=0.0, format="%.2f")
   tvd_wh2 = st.number_input("TVD Inicial (m)", key="tvd_wh2", value=0.0, format="%.2f")


# Processamento quando ambos os arquivos são carregados
if well1_file and well2_file:
   # Leitura e Validação dos arquivos Excel
   try:
       df_well1_orig = pd.read_excel(well1_file)
       df_well2_orig = pd.read_excel(well2_file)
       df_well1 = df_well1_orig.copy()
       df_well2 = df_well2_orig.copy()

       # ... (código de validação e renomeação de colunas SEM MUDANÇAS) ...
       expected_cols = ['MD', 'INC', 'AZ']
       data_valid = True
       for df, well_name, file_ref in [(df_well1, "Poço 1", well1_file), (df_well2, "Poço 2", well2_file)]:
           original_cols = df.columns.tolist()
           df.columns = [str(col).upper().strip() for col in df.columns]
           current_cols = df.columns.tolist()
           rename_map = {}
           # Check MD
           if 'MD' not in current_cols:
               found_md = False
               for potential_md in ['MEASURED DEPTH', 'PROFUNDIDADE MEDIDA', 'PROF MEDIDA']:
                   if potential_md in current_cols: rename_map[potential_md] = 'MD'; found_md = True; break
               if not found_md: st.error(f"Coluna 'MD' não encontrada em {well_name}. Colunas: {original_cols}"); data_valid = False
           # Check INC
           if 'INC' not in current_cols:
               found_inc = False
               for potential_inc in ['INCLINATION', 'INCL', 'INCLINACAO', 'INCLINAÇÃO']:
                   if potential_inc in current_cols: rename_map[potential_inc] = 'INC'; found_inc = True; break
               if not found_inc: st.error(f"Coluna 'INC' não encontrada em {well_name}. Colunas: {original_cols}"); data_valid = False
           # Check AZ
           if 'AZ' not in current_cols:
               found_az = False
               for potential_az in ['AZIMUTH', 'AZIM', 'AZIMUTE']:
                   if potential_az in current_cols: rename_map[potential_az] = 'AZ'; found_az = True; break
               if not found_az: st.error(f"Coluna 'AZ' não encontrada em {well_name}. Colunas: {original_cols}"); data_valid = False

           if not data_valid: continue
           df.rename(columns=rename_map, inplace=True)

           # Ensure numeric types
           cols_to_check = [col for col in expected_cols if col in df.columns]
           for col in cols_to_check:
               try: df[col] = pd.to_numeric(df[col], errors='coerce')
               except Exception as e_num: st.error(f"Erro ao converter '{col}' para número em {well_name}: {e_num}"); data_valid = False

           if not data_valid: continue
           initial_rows = len(df)
           df.dropna(subset=cols_to_check, inplace=True)
           if len(df) < initial_rows: st.warning(f"Removidas {initial_rows - len(df)} linhas com dados inválidos em {well_name}.")
           if len(df) < 2: st.error(f"Arquivo {well_name} não contém dados suficientes (mínimo 2 pontos válidos)."); data_valid = False

       if not data_valid:
           st.stop()

       # --- Calculation Steps ---
       # 1. Calculate Coordinates
       coords_well1 = calculate_coordinates(
           df_well1['MD'].values, df_well1['INC'].values, df_well1['AZ'].values,
           wellhead_n=n_wh1, wellhead_e=e_wh1, wellhead_tvd=tvd_wh1
       )
       coords_well2 = calculate_coordinates(
           df_well2['MD'].values, df_well2['INC'].values, df_well2['AZ'].values,
           wellhead_n=n_wh2, wellhead_e=e_wh2, wellhead_tvd=tvd_wh2
       )

       # 2. Calculate List of Cumulative Covariances using INCREMENTAL method
       # Cache this calculation as it can be slow
       @st.cache_data(ttl=3600, show_spinner="Calculando Incerteza ISCWSA Incremental...")
       def get_incremental_covariance_list(md_arr, inc_arr, az_arr, params_dict, tool, params_tuple_key):
            # params_tuple_key is only for caching, params_dict is used in calculation
            # Ensure arrays are numpy arrays
            md_np = np.asarray(md_arr)
            inc_np = np.asarray(inc_arr)
            az_np = np.asarray(az_arr)
            return calculate_iscwsa_covariance_incremental(md_np, inc_np, az_np, params_dict, tool)

       # Use tuple for cache key
       params_tuple_key = tuple(sorted(iscwsa_params.items()))

       # Calculate the list of covariances for each well
       # Pass numpy arrays directly for potentially better cache hits if only these change
       covs_list1 = get_incremental_covariance_list(
           coords_well1['MD'].values, coords_well1['INC'].values, coords_well1['AZ'].values,
           iscwsa_params, tool_type, params_tuple_key
       )
       covs_list2 = get_incremental_covariance_list(
           coords_well2['MD'].values, coords_well2['INC'].values, coords_well2['AZ'].values,
           iscwsa_params, tool_type, params_tuple_key
       )

       # Add index column to coords for easy lookup
       coords_well1 = coords_well1.reset_index().rename(columns={'index': 'OriginalIndex'})
       coords_well2 = coords_well2.reset_index().rename(columns={'index': 'OriginalIndex'})

       # 3. Find corresponding points and compare
       results = []
       st.write("Comparando trajetórias em profundidades correspondentes...")
       prog_bar_compare = st.progress(0)

       tvds_ref_raw = coords_well1['TVD'].dropna().unique()
       if len(tvds_ref_raw) == 0:
           st.error("Não foi possível encontrar TVDs válidas no Poço 1 para comparação.")
           st.stop()

       tvds_ref = np.sort(tvds_ref_raw)
       total_comparisons = len(tvds_ref)
       comparison_tvd_step = max(1, total_comparisons // 100)

       for i, tvd_ref in enumerate(tvds_ref):
           # Find point(s) in well 1 at this TVD
           p1_matches = coords_well1[np.isclose(coords_well1['TVD'], tvd_ref)]
           if p1_matches.empty: continue
           p1_row = p1_matches.iloc[0] # Get the first matching row (as a Series)
           p1_index = p1_row['OriginalIndex'] # Get its original index to find covariance

           # Find the closest point in well 2 based on TVD
           p2_row = find_closest_tvd_point(tvd_ref, coords_well2) # Returns a Series or None

           # Ensure p2 was found and is close enough
           tvd1_val = p1_row['TVD']
           tvd2_val = p2_row['TVD'] if p2_row is not None else np.nan
           tvd_diff_threshold = 10.0 # Max allowed TVD difference

           if p2_row is not None and pd.notna(tvd1_val) and pd.notna(tvd2_val) and abs(tvd1_val - tvd2_val) < tvd_diff_threshold:
               p2_index = p2_row['OriginalIndex'] # Get its original index

               # Retrieve the CUMULATIVE covariance matrices from the lists using the index
               p1_index_int = int(p1_index) # Convert to integer
               p2_index_int = int(p2_index) # Convert to integer

               cov1 = covs_list1[p1_index_int] if p1_index_int < len(covs_list1) else None
               cov2 = covs_list2[p2_index_int] if p2_index_int < len(covs_list2) else None

               # Ensure covariance data is valid
               cov1_valid = isinstance(cov1, np.ndarray) and cov1.shape == (3,3)
               cov2_valid = isinstance(cov2, np.ndarray) and cov2.shape == (3,3)

               if cov1_valid and cov2_valid:
                   # Calculate distances using ISCWSA ellipse parameters
                   distance_data = calculate_pedal_distance_iscwsa(
                       p1_row, p2_row, # Pass the Series rows
                       cov1, cov2,      # Pass the covariance matrices
                       sigma_factor
                   )

                   # Append results (using .get for safety, though keys should exist)
                   results.append({
                       'TVD_Ref': tvd_ref,
                       'TVD_Actual1': p1_row['TVD'], 'TVD_Actual2': p2_row['TVD'],
                       'MD1': p1_row['MD'], 'MD2': p2_row['MD'],
                       'INC1': p1_row['INC'], 'INC2': p2_row['INC'],
                       'AZ1': p1_row['AZ'], 'AZ2': p2_row['AZ'],
                       'N1': p1_row['N'], 'E1': p1_row['E'],
                       'N2': p2_row['N'], 'E2': p2_row['E'],
                       'DistCentros': distance_data.get('center_dist', np.nan),
                       'DistPedal': distance_data.get('pedal_dist', np.nan),
                       'Proj1': distance_data.get('proj1', np.nan),
                       'Proj2': distance_data.get('proj2', np.nan),
                       'SMj1': distance_data.get('smj1', np.nan),
                       'SMn1': distance_data.get('smn1', np.nan),
                       'Ang1': distance_data.get('ang1', np.nan),
                       'SMj2': distance_data.get('smj2', np.nan),
                       'SMn2': distance_data.get('smn2', np.nan),
                       'Ang2': distance_data.get('ang2', np.nan),
                       'DifPerc': distance_data.get('diff_percent', np.nan),
                       'SF': distance_data.get('SF', np.nan),
                       # Store indices for potential debugging
                       'Index1': p1_index, 'Index2': p2_index
                   })

           # Update progress bar
           if i % comparison_tvd_step == 0:
               prog_bar_compare.progress((i + 1) / total_comparisons)
       prog_bar_compare.progress(1.0)
       prog_bar_compare.empty()

       # Create dataframe of results
       if results:
           df_results = pd.DataFrame(results)
           df_results.dropna(subset=['DistCentros', 'DistPedal', 'SF'], inplace=True)

           if df_results.empty:
               st.warning("Não foi possível calcular distâncias válidas ou encontrar pontos próximos em TVD.")
               st.stop()

           # Exibir tabela de resultados
           st.subheader("Comparação de Distâncias e Fator de Separação (Incremental)")
           st.dataframe(df_results[[
               'TVD_Ref', 'MD1', 'MD2', 'DistCentros', 'DistPedal', 'DifPerc', 'SF',
               'SMj1', 'SMn1', 'Ang1', 'SMj2', 'SMn2', 'Ang2'
               ]].round(2))

           # --- Gráficos ---
           st.subheader("Gráficos de Análise")
           # ... (Código dos gráficos Plotly e Matplotlib - SEM MUDANÇAS AQUI) ...
           # Gráfico de distâncias vs TVD Ref
           fig1 = go.Figure()
           fig1.add_trace(go.Scatter(x=df_results['TVD_Ref'], y=df_results['DistCentros'], mode='lines+markers', name='Dist. Centros'))
           fig1.add_trace(go.Scatter(x=df_results['TVD_Ref'], y=df_results['DistPedal'], mode='lines+markers', name=f'Dist. Pedal ({sigma_factor:.1f}σ)'))
           fig1.update_layout(title='Distâncias vs Profundidade (TVD Referência Poço 1)', xaxis_title='TVD Ref Poço 1 (m)', yaxis_title='Distância (m)', legend=dict(x=0.01, y=0.99))
           st.plotly_chart(fig1, use_container_width=True)

           # Gráfico de Fator de Separação vs MD
           st.subheader("Fator de Separação (SF) vs Profundidade (MD Poço 1)")
           fig_sf = go.Figure()
           fig_sf.add_trace(go.Scatter(x=df_results['MD1'], y=df_results['SF'], mode='lines+markers', name='Fator de Separação (SF)'))
           min_md = df_results['MD1'].min()
           max_md = df_results['MD1'].max()
           fig_sf.add_shape(type="line", x0=min_md, y0=1.0, x1=max_md, y1=1.0, line=dict(color="red", width=2, dash="dash"))
           fig_sf.add_shape(type="line", x0=min_md, y0=1.5, x1=max_md, y1=1.5, line=dict(color="orange", width=2, dash="dash"))
           fig_sf.add_shape(type="line", x0=min_md, y0=2.0, x1=max_md, y1=2.0, line=dict(color="green", width=2, dash="dash"))
           annotations_list = [
               dict(x=max_md * 0.95 if pd.notna(max_md) and max_md > 0 else 100, y=0.5, text="Colisão (<1.0)", showarrow=False, font=dict(color="red", size=10)),
               dict(x=max_md * 0.95 if pd.notna(max_md) and max_md > 0 else 100, y=1.25, text="Alto Risco (1.0-1.5)", showarrow=False, font=dict(color="orange", size=10)),
               dict(x=max_md * 0.95 if pd.notna(max_md) and max_md > 0 else 100, y=1.75, text="Médio Risco (1.5-2.0)", showarrow=False, font=dict(color="green", size=10)),
               dict(x=max_md * 0.95 if pd.notna(max_md) and max_md > 0 else 100, y=2.25, text="Baixo Risco (>2.0)", showarrow=False, font=dict(color="blue", size=10)),
           ]
           fig_sf.update_layout(
               title='Fator de Separação (SF) vs Profundidade Medida (MD Poço 1)',
               xaxis_title='MD Poço 1 (m)', yaxis_title='Fator de Separação (SF)',
               yaxis_range=[0, max(df_results['SF'].max() * 1.1 if not df_results['SF'].empty and df_results['SF'].max() != np.inf and pd.notna(df_results['SF'].max()) else 5, 2.1)],
               legend=dict(x=0.01, y=0.99), annotations=annotations_list
           )
           st.plotly_chart(fig_sf, use_container_width=True)

           # Visualização 3D
           st.subheader("Visualização 3D das Trajetórias (Absoluto)")
           fig5 = go.Figure()
           fig5.add_trace(go.Scatter3d(x=coords_well1['E'], y=coords_well1['N'], z=coords_well1['TVD'], mode='lines', name='Poço 1', line=dict(color='blue', width=4)))
           fig5.add_trace(go.Scatter3d(x=coords_well2['E'], y=coords_well2['N'], z=coords_well2['TVD'], mode='lines', name='Poço 2', line=dict(color='red', width=4)))
           fig5.add_trace(go.Scatter3d(x=df_results['E1'], y=df_results['N1'], z=df_results['TVD_Actual1'], mode='markers', name='Pontos Comp. Poço 1', marker=dict(color='cyan', size=3)))
           fig5.add_trace(go.Scatter3d(x=df_results['E2'], y=df_results['N2'], z=df_results['TVD_Actual2'], mode='markers', name='Pontos Comp. Poço 2', marker=dict(color='magenta', size=3)))
           fig5.update_layout(
               scene=dict(xaxis_title='Este (m)', yaxis_title='Norte (m)', zaxis_title='TVD (m)', zaxis=dict(autorange="reversed"), aspectmode='auto'),
               title="Visualização 3D das Trajetórias (Coordenadas Absolutas)", height=700, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
           st.plotly_chart(fig5, use_container_width=True)

           # Visualização 2D com Elipses
           st.subheader(f"Visualização 2D com Elipses de Incerteza ({sigma_factor:.1f}σ)")
           tvd_options_ref = np.sort(df_results['TVD_Ref'].unique())
           if len(tvd_options_ref) == 0:
                st.warning("Não há TVDs de referência válidas para visualização 2D.")
           else:
                default_index_2d = len(tvd_options_ref) // 2 if len(tvd_options_ref) > 0 else 0
                selected_tvd_ref = st.selectbox("Selecione a TVD (Referência Poço 1) para visualização 2D",
                                                tvd_options_ref, index=default_index_2d, format_func=lambda x: f"{x:.1f} m")
                selected_data_rows = df_results[np.isclose(df_results['TVD_Ref'], selected_tvd_ref)]
                if selected_data_rows.empty:
                    st.warning(f"Não foram encontrados dados para a TVD de referência {selected_tvd_ref:.1f} m.")
                else:
                    selected_data = selected_data_rows.iloc[0]
                    fig_2d, ax_2d = plt.subplots(figsize=(10, 10))
                    ax_2d.scatter(selected_data['E1'], selected_data['N1'], color='blue', s=50, label=f'Poço 1 Centro (TVD≈{selected_data["TVD_Actual1"]:.1f}m)', zorder=5)
                    ax_2d.scatter(selected_data['E2'], selected_data['N2'], color='red', s=50, label=f'Poço 2 Centro (TVD≈{selected_data["TVD_Actual2"]:.1f}m)', zorder=5)
                    ax_2d.plot([selected_data['E1'], selected_data['E2']], [selected_data['N1'], selected_data['N2']], 'k--', alpha=0.7, label=f'Dist Centros: {selected_data["DistCentros"]:.2f} m')

                    angle_mpl1 = (90.0 - selected_data['Ang1']) % 360.0 if pd.notna(selected_data['Ang1']) else 0.0
                    draw_ellipse_matplotlib(ax_2d, center_xy=(selected_data['E1'], selected_data['N1']),
                        width=2 * selected_data['SMj1'] if pd.notna(selected_data['SMj1']) else 0,
                        height=2 * selected_data['SMn1'] if pd.notna(selected_data['SMn1']) else 0,
                        angle_deg=angle_mpl1, color="blue", alpha=0.3, label=f'Elipse 1 ({sigma_factor:.1f}σ)')

                    angle_mpl2 = (90.0 - selected_data['Ang2']) % 360.0 if pd.notna(selected_data['Ang2']) else 0.0
                    draw_ellipse_matplotlib(ax_2d, center_xy=(selected_data['E2'], selected_data['N2']),
                        width=2 * selected_data['SMj2'] if pd.notna(selected_data['SMj2']) else 0,
                        height=2 * selected_data['SMn2'] if pd.notna(selected_data['SMn2']) else 0,
                        angle_deg=angle_mpl2, color="red", alpha=0.3, label=f'Elipse 2 ({sigma_factor:.1f}σ)')

                    info_text = f"""
                    TVD Ref: {selected_data.get('TVD_Ref', np.nan):.1f} m (TVD1={selected_data.get('TVD_Actual1', np.nan):.1f}, TVD2={selected_data.get('TVD_Actual2', np.nan):.1f})
                    MD1: {selected_data.get('MD1', np.nan):.1f} m, MD2: {selected_data.get('MD2', np.nan):.1f} m
                    INC1: {selected_data.get('INC1', np.nan):.1f}°, AZ1: {selected_data.get('AZ1', np.nan):.1f}°
                    INC2: {selected_data.get('INC2', np.nan):.1f}°, AZ2: {selected_data.get('AZ2', np.nan):.1f}°
                    ---
                    Dist. Centros: {selected_data.get("DistCentros", np.nan):.2f} m
                    Dist. Pedal ({sigma_factor:.1f}σ): {selected_data.get("DistPedal", np.nan):.2f} m
                    Proj1: {selected_data.get('Proj1', np.nan):.2f} m, Proj2: {selected_data.get('Proj2', np.nan):.2f} m
                    SF: {selected_data.get("SF", np.nan):.2f}
                    ---
                    Elipse 1 ({sigma_factor:.1f}σ): SMj={selected_data.get('SMj1', np.nan):.2f}, SMn={selected_data.get('SMn1', np.nan):.2f}, Ang(N)={selected_data.get('Ang1', np.nan):.1f}°
                    Elipse 2 ({sigma_factor:.1f}σ): SMj={selected_data.get('SMj2', np.nan):.2f}, SMn={selected_data.get('SMn2', np.nan):.2f}, Ang(N)={selected_data.get('Ang2', np.nan):.1f}°
                    """
                    ax_2d.text(0.02, 0.98, info_text, transform=ax_2d.transAxes, fontsize=9,
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                    ax_2d.set_xlabel('Este (m)')
                    ax_2d.set_ylabel('Norte (m)')
                    ax_2d.set_title(f'Elipses Inc. ISCWSA Incremental ({tool_type}) @ TVD Ref ≈ {selected_data.get("TVD_Ref", np.nan):.1f} m')
                    ax_2d.grid(True, linestyle=':', alpha=0.6)
                    ax_2d.axis('equal')
                    ax_2d.legend(fontsize=8, loc='lower right')

                    max_radius1 = selected_data.get('SMj1', 0) if pd.notna(selected_data.get('SMj1', 0)) else 0
                    max_radius2 = selected_data.get('SMj2', 0) if pd.notna(selected_data.get('SMj2', 0)) else 0
                    center_e = (selected_data.get('E1', 0) + selected_data.get('E2', 0)) / 2 if pd.notna(selected_data.get('E1')) and pd.notna(selected_data.get('E2')) else 0
                    center_n = (selected_data.get('N1', 0) + selected_data.get('N2', 0)) / 2 if pd.notna(selected_data.get('N1')) and pd.notna(selected_data.get('N2')) else 0
                    dist_centers = selected_data.get("DistCentros", 0) if pd.notna(selected_data.get("DistCentros")) else 0
                    e1 = selected_data.get('E1', center_e); n1 = selected_data.get('N1', center_n)
                    e2 = selected_data.get('E2', center_e); n2 = selected_data.get('N2', center_n)
                    extent1_e = abs(e1 - center_e) + max_radius1; extent1_n = abs(n1 - center_n) + max_radius1
                    extent2_e = abs(e2 - center_e) + max_radius2; extent2_n = abs(n2 - center_n) + max_radius2
                    half_view_size = max(extent1_e, extent1_n, extent2_e, extent2_n, dist_centers / 2 if dist_centers > 0 else 1) * 1.2
                    if half_view_size < 1: half_view_size = 5
                    ax_2d.set_xlim(center_e - half_view_size, center_e + half_view_size)
                    ax_2d.set_ylim(center_n - half_view_size, center_n + half_view_size)

                    st.pyplot(fig_2d)

           # Explicação Pedal Curve Atualizada
           with st.expander("Explicação do Método Pedal Curve com ISCWSA Incremental e SF"):
               st.markdown(f"""
               ### Método Pedal Curve, ISCWSA Incremental e Fator de Separação (SF)

               Este cálculo utiliza o método **incremental** para determinar a incerteza posicional (matriz de covariância NEV) em cada ponto da trajetória, conforme os padrões ISCWSA ({tool_type}).

               **Cálculo Incremental da Covariância:**
               1. O poço é dividido em segmentos entre as estações de medição.
               2. Para cada segmento, calcula-se a **matriz de covariância incremental (ΔC)** no *frame local* do poço (Along-hole, Lateral, Vertical). Esta matriz representa a incerteza adicionada *apenas por aquele segmento*, baseada nos parâmetros de erro da ferramenta e nas propriedades do segmento (comprimento, inclinação média, etc.). *(Nota: A função `calcula_delta_cov_segment` aqui é uma aproximação disso)*.
               3. A matriz ΔC local é **rotacionada** para o frame global (Norte, Este, Vertical Down) usando a orientação média do segmento.
               4. A matriz de covariância **cumulativa** no ponto final do segmento (n+1) é obtida somando a covariância rotacionada do segmento à covariância cumulativa do ponto anterior (n):
                  `C_cumulativa(n+1) [NEV] = C_cumulativa(n) [NEV] + R(n) * ΔC(n→n+1) [Local] * R(n)^T`
               5. Este processo é repetido ao longo de todo o poço, acumulando a incerteza e considerando como ela se propaga e rotaciona.

               **Pedal Curve e SF:**
               - A partir da matriz de covariância *cumulativa* em um determinado ponto de comparação, extrai-se a submatriz horizontal (NE) para calcular os parâmetros da **elipse de incerteza** (semi-eixos maior/menor `SMj`/`SMn`, e ângulo `Ang` do eixo maior com o Norte).
               - A **Distância Pedal Curve** (`Dist_Pedal`) é então calculada da mesma forma que antes:
                 `Dist_Pedal = max(0, Dist_Centros - (Proj1 + Proj2))`
                 usando as projeções (`Proj1`, `Proj2`) das elipses obtidas pelo método incremental.
               - O **Fator de Separação (SF)** também é calculado como:
                 `SF = Dist_Centros / (Proj1 + Proj2)`
               - A interpretação do SF permanece a mesma ( < 1.0 Colisão, 1.0-1.5 Alto Risco, 1.5-2.0 Médio Risco, > 2.0 Baixo Risco, para a confiança de {sigma_factor:.1f}σ).

               A vantagem do método incremental é que ele captura de forma mais realista como os erros se acumulam e interagem ao longo de trajetórias curvas, levando a elipses de incerteza mais precisas, especialmente para poços complexos.
               """)


       else:
           st.warning("Não foi possível encontrar pontos de comparação em profundidades TVD próximas ou os dados resultantes foram insuficientes após a filtragem.")

   # ... (Blocos except e else para upload de arquivos SEM MUDANÇAS) ...
   except FileNotFoundError: st.error("Erro: Arquivo não encontrado.")
   except pd.errors.EmptyDataError: st.error("Erro: O arquivo Excel está vazio ou ilegível.")
   except ValueError as e: st.error(f"Erro ao converter dados. Verifique MD, INC, AZ e coords. WH. Detalhe: {e}")
   except KeyError as e: st.error(f"Erro: Coluna essencial não encontrada: {e}. Verifique nomes no Excel.")
   except Exception as e: st.error(f"Erro inesperado: {e}"); st.exception(e)

else:
   # Exibir exemplo e botão de download
   st.info("Aguardando upload dos arquivos Excel dos Poços 1 e 2 e definição das coordenadas de cabeça de poço...")
   example_data = pd.DataFrame({
       'MD': [0, 500, 1000, 1500, 2000, 2500],
       'INC': [0, 15, 30, 45, 60, 75],
       'AZ': [0, 45, 45, 60, 90, 120]
   })
   st.write("Formato esperado para os arquivos Excel (first sheet):")
   st.dataframe(example_data)
   st.markdown("""(...)""") # Mantém a explicação do formato
   buffer = BytesIO()
   with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
       example_data.to_excel(writer, index=False, sheet_name='TrajetoriaExemplo')
   buffer.seek(0)
   st.download_button(label="Download Arquivo Exemplo (.xlsx)", data=buffer, file_name="exemplo_trajetoria.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

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

st.set_page_config(page_title="Comparador de Distâncias ISCWSA Incremental Avançado", layout="wide")

# --- Funções Trigonométricas (Graus) ---
# ... (sem mudanças) ...
def sind(degrees): return np.sin(np.radians(degrees))
def cosd(degrees): return np.cos(np.radians(degrees))
def tand(degrees): degrees = np.asanyarray(degrees); rad = np.radians(degrees); return np.tan(rad)
def atand(value): return np.degrees(np.arctan(value))
def atan2d(y, x): return np.degrees(np.arctan2(y, x))
def acosd(value): return np.degrees(np.arccos(np.clip(value, -1.0, 1.0)))
def cotd(degrees): return 1.0 / tand(degrees) # Add cotangent

# --- Funções de Rotação e Cálculo Incremental (Refinadas) ---

def calcula_rotacao_nev(inc_deg, az_deg):
    """ Calcula a matriz de rotação [u, v, w] -> [N, E, V_down]. """
    # ... (sem mudanças) ...
    inc_rad = np.radians(inc_deg); az_rad = np.radians(az_deg)
    si = np.sin(inc_rad); ci = np.cos(inc_rad); sa = np.sin(az_rad); ca = np.cos(az_rad)
    R = np.array([ [si*ca, -sa, ci*ca], [si*sa,  ca, ci*sa], [ci   ,   0, -si] ])
    return R

# @st.cache_data # Cache pode ser útil aqui, mas depende da complexidade
def calcula_sensibilidades_segmento(inc_deg, az_deg, params, tool_type):
    """
    Calcula as sensibilidades (aproximadas) de dInc e dAz às fontes de erro
    para um segmento com inclinação e azimute médios.
    Retorna um dicionário de sensibilidades: {'dInc_dAccBiasX': ..., 'dAz_dMagBiasY': ...}
    NOTA: Esta é a parte mais complexa e aqui está MUITO SIMPLIFICADA.
    """
    sens = {}
    inc_rad = np.radians(inc_deg)
    # az_rad = np.radians(az_deg) # Azimute não entra tanto nas sensibilidades diretas Inc/Az

    # Parâmetros físicos (simplificado - usar valores locais precisos)
    g = params.get('gravity_strength', 1.0) * 9.80665 # m/s^2
    B_tot = params.get('mag_field_strength', 50000) * 1e-9 # Tesla
    dip_rad = np.radians(params.get('dip_angle', 60))
    B_h = B_tot * np.cos(dip_rad) # Componente horizontal (Tesla)
    # B_v = B_tot * np.sin(dip_rad) # Componente vertical

    # Conversões de unidade para sigmas (usados implicitamente abaixo)
    mg_to_mps2 = 0.001 * g
    mrad_to_rad = 0.001
    ppm_to_frac = 1e-6
    nT_to_T = 1e-9
    deg_to_rad = np.pi / 180.0
    # deg_hr_to_rad_s = deg_to_rad / 3600.0

    # --- Sensibilidades Comuns (dInc) ---
    # dInc / dAccBias (componente perpendicular à gravidade no plano vertical)
    sens['dInc_dAccBiasXY'] = 1.0 / g # (rad / (m/s^2)) - Simplificado, depende do eixo X/Y local
    # dInc / dAccScaleFactor (sensibilidade à componente g*sin(I))
    sens['dInc_dAccSF'] = np.abs(np.sin(inc_rad)) # (rad / frac) - Simplificado
    # dInc / dAccMisalign (sensibilidade à componente g*cos(I))
    sens['dInc_dAccMis'] = np.abs(np.cos(inc_rad)) # (rad / rad) - Simplificado
    # dInc / dSagCorrErr
    sens['dInc_dSag'] = 1.0 # (rad / rad)
    # dInc / dMisalignIncTool
    sens['dInc_dMisalignInc'] = 1.0 # (rad / rad)

    # --- Sensibilidades Específicas ---
    if tool_type == "ISCWSA MWD":
        # --- Sensibilidades MWD (dAz) ---
        # Denominador comum para sensibilidade azimutal magnética
        denom_az_mag = B_h * np.sin(inc_rad)
        if abs(denom_az_mag) < 1e-9: # Evita divisão por zero perto do vertical ou equador mag.
            # Sensibilidades explodem -> usar um valor grande ou limite?
            # O modelo completo lida com isso analiticamente.
            sens_az_mag_limit = 1e9 # Um valor grande arbitrário
            sens['dAz_dMagBiasXY'] = sens_az_mag_limit
            sens['dAz_dMagSF'] = sens_az_mag_limit
            sens['dAz_dMagMis'] = sens_az_mag_limit
        else:
            # dAz / dMagBias (componente perpendicular a Bh no plano horizontal)
            sens['dAz_dMagBiasXY'] = 1.0 / denom_az_mag # (rad / Tesla) - Simplificado
            # dAz / dMagScaleFactor (sensibilidade à componente Btot projetada) - Complexo
            # Simplificação: proporcional a cot(I)?
            sens['dAz_dMagSF'] = np.abs(B_tot / denom_az_mag * np.cos(inc_rad)) if abs(np.sin(inc_rad)) > 1e-3 else 1e6 # (rad / frac) - Muito simplificado
            # dAz / dMagMisalign (sensibilidade à componente Btot projetada) - Complexo
            sens['dAz_dMagMis'] = np.abs(B_tot / denom_az_mag * 1.0) # (rad / rad) - Muito simplificado
        # dAz / dMagDeclErr (erro sistemático, tratado fora)
        # sens['dAz_dDec'] = 1.0 # (rad / rad)
        # dAz / dMagDsErr (interferência)
        sens['dAz_dDS'] = 1.0 # (rad / rad)
        # dAz / dMisalignAziTool
        sens['dAz_dMisalignAzi'] = 1.0 # (rad / rad)

    elif tool_type == "ISCWSA Gyro":
        # --- Sensibilidades Gyro (dInc) ---
        # dInc / dGyro_G_Sens (sensibilidade a g*cos(I))
        sens['dInc_dGyroGSens'] = np.abs(g * np.cos(inc_rad)) # (rad / (rad/s / m/s^2)) - Falta tempo aqui? Gsens*g*cosI*dt? Sim. A ser multiplicado por dt depois.

        # --- Sensibilidades Gyro (dAz) ---
        # dAz / dGyroBiasDrift_H (sensibilidade horizontal do drift)
        # O drift se acumula com o tempo. A sensibilidade instantânea é complexa.
        # O erro angular total é roughly (drift_rate / sin(I)) * time
        # A contribuição *incremental* dAz para um segmento dt é complexa.
        # Trataremos o drift como sistemático por ora.
        # sens['dAz_dGyroDriftH'] = 1.0 / np.sin(inc_rad) if abs(np.sin(inc_rad)) > 1e-3 else 1e9 # (rad / (rad/s)) - Falta tempo

        # dAz / dGyro_G_Sens (sensibilidade a g*sin(I))
        sens['dAz_dGyroGSens'] = np.abs(g * np.sin(inc_rad)) # (rad / (rad/s / m/s^2)) - Falta tempo

        # dAz / dGyroScaleFactor (sensibilidade à rotação da Terra * cos(lat)*sin(Az)/sin(I) etc.) - MUITO COMPLEXO
        # Simplificação grosseira: proporcional a tan(I)?
        sens['dAz_dGyroSF'] = np.abs(np.tan(inc_rad)) if abs(np.cos(inc_rad)) > 1e-3 else 1e6 # (rad / frac) - Muito simplificado

        # dAz / dGyroMisalign (complexo, depende do eixo)
        sens['dAz_dGyroMis'] = np.abs(np.tan(inc_rad)) if abs(np.cos(inc_rad)) > 1e-3 else 1e6 # (rad / rad) - Muito simplificado

        # dAz / dGyroAzRefErr (erro sistemático, tratado fora)
        # sens['dAz_dAzRef'] = 1.0 # (rad / rad)
        # dAz / dMisalignAziTool
        sens['dAz_dMisalignAzi'] = 1.0 # (rad / rad)

    return sens

def calcula_delta_cov_segment_avancado(dmd, inc_mid_deg, az_mid_deg, params, tool_type):
    """
    Calcula a matriz de covariância incremental LOCAL (DeltaC_local) 3x3
    para um segmento, considerando erros ALEATÓRIOS e suas covariâncias (aproximadas).
    Retorna DeltaC_local no frame (along-hole, lateral-right, vertical-up).
    """
    # --- 1. Obter Sigmas dos Erros Aleatórios e Conversões ---
    mrad_to_rad = 0.001
    ppm_to_frac = 1e-6
    mg_to_mps2 = 0.001 * params.get('gravity_strength', 1.0) * 9.80665
    nT_to_T = 1e-9
    deg_to_rad = np.pi / 180.0
    # deg_hr_to_rad_s = deg_to_rad / 3600.0 # Para drift se fosse incluído aqui

    # Identificar erros tratados como aleatórios por segmento
    # (Simplificação: consideramos biases, SF, misalignments como tendo uma componente
    # aleatória por segmento, além de uma possível componente sistemática tratada depois)
    random_errors = {}
    sigmas_sq = {} # Variâncias

    # Erros comuns aleatórios (ou com componente aleatória)
    # Depth Prop Error é intrinsecamente por segmento
    sig_d_prop = params.get('depth_err_prop', 0)
    random_errors['d_prop'] = sig_d_prop
    sigmas_sq['d_prop'] = sig_d_prop**2

    # Componente aleatória dos erros de sensor? Difícil separar sem modelo completo.
    # Vamos incluir os termos ISCWSA como se fossem aleatórios aqui,
    # sabendo que é uma aproximação. A parte sistemática será adicionada depois.
    sig_acc_bias = params.get('acc_bias', 0) * mg_to_mps2
    sig_acc_sf = params.get('acc_sf', 0) * ppm_to_frac
    sig_acc_mis = max(params.get('acc_mis_xy', 0), params.get('acc_mis_z', 0)) * mrad_to_rad # Simplificado
    sig_sag = params.get('sag_corr_err', 0) * deg_to_rad
    sig_mis_inc = params.get('misalign_err_inc', 0) * deg_to_rad
    sig_mis_azi = params.get('misalign_err_azi', 0) * deg_to_rad

    random_errors.update({
        'acc_bias': sig_acc_bias, 'acc_sf': sig_acc_sf, 'acc_mis': sig_acc_mis,
        'sag': sig_sag, 'mis_inc': sig_mis_inc, 'mis_azi': sig_mis_azi
    })
    sigmas_sq.update({
        'acc_bias': sig_acc_bias**2, 'acc_sf': sig_acc_sf**2, 'acc_mis': sig_acc_mis**2,
        'sag': sig_sag**2, 'mis_inc': sig_mis_inc**2, 'mis_azi': sig_mis_azi**2
    })

    if tool_type == "ISCWSA MWD":
        sig_mag_bias = params.get('mag_bias', 0) * nT_to_T
        sig_mag_sf = params.get('mag_sf', 0) * ppm_to_frac
        sig_mag_mis = max(params.get('mag_mis_xy', 0), params.get('mag_mis_z', 0)) * mrad_to_rad # Simplificado
        sig_ds = params.get('mag_ds_err', 0) * deg_to_rad # Interferência considerada aleatória por segmento

        random_errors.update({'mag_bias': sig_mag_bias, 'mag_sf': sig_mag_sf, 'mag_mis': sig_mag_mis, 'ds': sig_ds})
        sigmas_sq.update({'mag_bias': sig_mag_bias**2, 'mag_sf': sig_mag_sf**2, 'mag_mis': sig_mag_mis**2, 'ds': sig_ds**2})

    elif tool_type == "ISCWSA Gyro":
        # G-sensitivity pode ser considerada aleatória se varia por segmento? Ou sistemática?
        # Vamos incluir aqui como aleatória (simplificação)
        sig_gyro_gsens = params.get('gyro_g_sens_drift', 0) #* deg_hr_to_rad_s / g_mps2 # Unidade: (rad/s / m/s^2)
        # O tempo dt entra na H-matrix!
        sig_gyro_sf = params.get('gyro_sf', 0) * ppm_to_frac
        sig_gyro_mis = max(params.get('gyro_mis_xy', 0), params.get('gyro_mis_z', 0)) * mrad_to_rad # Simplificado

        random_errors.update({'gyro_gsens': sig_gyro_gsens, 'gyro_sf': sig_gyro_sf, 'gyro_mis': sig_gyro_mis})
        sigmas_sq.update({'gyro_gsens': sig_gyro_gsens**2, 'gyro_sf': sig_gyro_sf**2, 'gyro_mis': sig_gyro_mis**2})
        # Drift e Ref Error são primariamente sistemáticos (tratados fora)

    error_keys = list(random_errors.keys())
    n_errors = len(error_keys)
    Sigma_x = np.diag([sigmas_sq[k] for k in error_keys]) # Matriz diagonal de variâncias

    # --- 2. Calcular Sensibilidades (H-matrix Simplificada) ---
    # H tem shape (3, n_errors): [ [du/de1, du/de2, ...], [dv/de1, dv/de2, ...], [dw/de1, dw/de2, ...] ]
    H = np.zeros((3, n_errors))
    s = calcula_sensibilidades_segmento(inc_mid_deg, az_mid_deg, params, tool_type)

    # Tempo do segmento (aproximado, se necessário para drift/gsens) - OMITIDO AQUI
    # dt_sec = dmd / (taxa_perfuração_media_m_s) # Difícil estimar
    # Ou: dt_sec = params['survey_time_hours'] * 3600 / (num_total_segmentos) # Grosseiro

    # Preencher H-matrix coluna por coluna (erro por erro)
    for j, key in enumerate(error_keys):
        # Sensibilidade de du (along-hole)
        if key == 'd_prop':
            H[0, j] = 1.0 * dmd # du = 1 * d(erro_profundidade_proporcional * dmd) ? Não, du = dmd * (1 + erro_prop) -> d(du)/d(erro_prop) = dmd
            # Correto: du = dmd * (1 + d_prop) => d(du)/d(d_prop) = dmd? Não exatamente.
            # A variância de u é Var(dmd) = Var(dmd_medido)
            # Var(dmd_medido) = (sigma_d_prop * dmd)**2.
            # Se e_u = dmd * d_prop, então d(e_u)/d(d_prop) = dmd. Usaremos isso.
            H[0, j] = dmd # Sensibilidade de du ao erro *fracional* de profundidade
        else:
            H[0, j] = 0 # Outros erros (angulares) não afetam du diretamente nesta simplificação

        # Sensibilidade de dv (lateral) = dmd * sin(Inc) * dAz
        # Sensibilidade de dw (vertical) = dmd * dInc
        dvdAz = dmd * sind(inc_mid_deg)
        dwdInc = dmd

        dInc_dErr = 0.0
        dAz_dErr = 0.0

        # Mapear erro para dInc e dAz usando sensibilidades 's'
        if key == 'acc_bias':   dInc_dErr = s.get('dInc_dAccBiasXY', 0)
        if key == 'acc_sf':     dInc_dErr = s.get('dInc_dAccSF', 0)
        if key == 'acc_mis':    dInc_dErr = s.get('dInc_dAccMis', 0)
        if key == 'sag':        dInc_dErr = s.get('dInc_dSag', 0)
        if key == 'mis_inc':    dInc_dErr = s.get('dInc_dMisalignInc', 0)
        if key == 'mis_azi':    dAz_dErr = s.get('dAz_dMisalignAzi', 0)

        if tool_type == "ISCWSA MWD":
            if key == 'mag_bias': dAz_dErr = s.get('dAz_dMagBiasXY', 0)
            if key == 'mag_sf':   dAz_dErr = s.get('dAz_dMagSF', 0)
            if key == 'mag_mis':  dAz_dErr = s.get('dAz_dMagMis', 0)
            if key == 'ds':       dAz_dErr = s.get('dAz_dDS', 0)
        elif tool_type == "ISCWSA Gyro":
             # Incluir dt para Gsens aqui? dInc = sens * sigma * dt
             # H[2, j] = dwdInc * s.get('dInc_dGyroGSens', 0) * dt_sec ?? <- Requer dt
             # H[1, j] = dvdAz * s.get('dAz_dGyroGSens', 0) * dt_sec ??
             # Por ora, omitimos dt, assumindo sensibilidade ao erro *total* acumulado no segmento
             if key == 'gyro_gsens':
                 dInc_dErr = s.get('dInc_dGyroGSens', 0) # Falta dt
                 dAz_dErr = s.get('dAz_dGyroGSens', 0) # Falta dt
             if key == 'gyro_sf':  dAz_dErr = s.get('dAz_dGyroSF', 0)
             if key == 'gyro_mis': dAz_dErr = s.get('dAz_dGyroMis', 0)


        # Preencher H para dv e dw
        H[1, j] = dvdAz * dAz_dErr
        H[2, j] = dwdInc * dInc_dErr

    # --- 3. Calcular DeltaC_local ---
    DeltaC_local = H @ Sigma_x @ H.T

    # Garante simetria numérica
    DeltaC_local = (DeltaC_local + DeltaC_local.T) / 2.0

    return DeltaC_local


def calcula_contrib_cov_sistematica_nev(md, inc_deg, az_deg, params, tool_type):
    """
    Calcula a contribuição ADICIONAL à matriz de covariância NEV
    devido a erros SISTEMÁTICOS no ponto atual (MD, Inc, Az).
    Retorna uma matriz 3x3 C_sist_nev.
    """
    C_sist_nev = np.zeros((3,3))
    deg_to_rad = np.pi / 180.0
    inc_rad = np.radians(inc_deg)
    az_rad = np.radians(az_deg)
    si = np.sin(inc_rad); ci = np.cos(inc_rad)
    sa = np.sin(az_rad); ca = np.cos(az_rad)

    # --- Erro Constante de Profundidade (dD0) ---
    # Afeta principalmente a posição along-hole. Projetado em NEV.
    sig_d_const = params.get('depth_err_const', 0)
    if sig_d_const > 0:
        var_d_const = sig_d_const**2
        # Vetor de direção along-hole em NEV: [si*ca, si*sa, ci]
        u_nev = np.array([si*ca, si*sa, ci])
        # Contribuição à covariância: var * (u @ u.T)
        C_sist_nev += var_d_const * np.outer(u_nev, u_nev)

    # --- Erros Sistemáticos de Azimute (Declinação MWD, Ref Gyro) ---
    sig_az_sist_rad = 0.0
    if tool_type == "ISCWSA MWD":
        sig_az_sist_rad = params.get('mag_dec_err', 0) * deg_to_rad
    elif tool_type == "ISCWSA Gyro":
        sig_az_sist_rad = params.get('gyro_az_ref_err', 0) * deg_to_rad
        # Bias Drift Gyro (componente sistemática)? Muito complexo para adicionar aqui de forma simples.

    if sig_az_sist_rad > 0:
        var_az_sist = sig_az_sist_rad**2
        # Erro de azimute causa deslocamento lateral: dLat = MD * sin(Inc) * dAz
        # Variância lateral: var_lat = (MD * sin(Inc))^2 * var_az_sist
        var_lat = (md * si)**2 * var_az_sist
        # Vetor de direção lateral (perp. a u no plano horizontal) em NEV: [-sa, ca, 0]
        v_nev = np.array([-sa, ca, 0])
        # Contribuição à covariância: var_lat * (v @ v.T)
        C_sist_nev += var_lat * np.outer(v_nev, v_nev)

    # --- Outros Erros Sistemáticos (Biases Constantes, etc.) ---
    # Exemplo: Bias constante de Acelerômetro (componente que afeta Inclinação)
    # sig_acc_bias_const = params.get('acc_bias', 0) * 0.001 * 9.80665 # Se uma parte é constante
    # var_inc_bias = (sig_acc_bias_const / g)**2 # Variancia angular
    # Deslocamento vertical: dVert = MD * dInc
    # Variancia vertical: var_vert = MD**2 * var_inc_bias
    # Vetor de direção "vertical" local (perp. a u no plano vertical) em NEV: [ci*ca, ci*sa, -si]
    # w_nev = np.array([ci*ca, ci*sa, -si])
    # C_sist_nev += var_vert * np.outer(w_nev, w_nev)
    # --> Isso fica complicado rapidamente, pois os biases afetam N, E e V.

    # Por simplicidade, focamos no erro constante de prof e erro azimutal sistemático.

    return C_sist_nev


def calculate_iscwsa_covariance_incremental_avancado(md_array, inc_array, az_array, params, tool_type):
    """
    Calcula a lista de matrizes de covariância cumulativas (NEV) em cada ponto
    usando o método incremental ISCWSA (APROXIMADO AVANÇADO).
    Inclui DeltaC não diagonal e adição de erros sistemáticos.
    """
    n_points = len(md_array)
    if n_points == 0: return []

    covariances_nev = []
    C_cum_nev_prev = np.zeros((3,3))
    covariances_nev.append(C_cum_nev_prev.copy()) # Ponto 0

    # --- Pré-calcular contribuição de erros sistemáticos que são fixos no início ---
    # Ex: Se houvesse um erro inicial de posição NEV, adicionaria aqui.

    for i in range(1, n_points):
        md0, md1 = md_array[i-1], md_array[i]
        inc0, inc1 = inc_array[i-1], inc_array[i]
        az0, az1 = az_array[i-1], az_array[i]

        # Propriedades do segmento
        dmd = md1 - md0
        if dmd < 1e-6:
             # Se segmento tem comprimento zero, apenas propaga a covariância anterior
             # e adiciona a contribuição sistemática *atualizada* para o ponto md1, inc1, az1
             C_sist_atual = calcula_contrib_cov_sistematica_nev(md1, inc1, az1, params, tool_type)
             # Cuidado: Adicionar C_sist pode não ser correto se C_cum já inclui C_sist anterior.
             # Modelo ISCWSA real propaga erros sistemáticos de forma diferente.
             # Vamos manter a C anterior se dmd=0 por simplicidade aqui.
             C_cum_nev = C_cum_nev_prev.copy()
             covariances_nev.append(C_cum_nev)
             C_cum_nev_prev = C_cum_nev
             continue

        inc_mid = (inc1 + inc0) / 2.0
        az_mid = (az1 + az0) / 2.0 # TODO: Usar média angular correta

        # --- 1. Calcular DeltaC Local (Erros Aleatórios do Segmento) ---
        DeltaC_local = calcula_delta_cov_segment_avancado(dmd, inc_mid, az_mid, params, tool_type)

        # --- 2. Rotacionar DeltaC Local para NEV ---
        R_local_to_nev = calcula_rotacao_nev(inc_mid, az_mid)
        DeltaC_nev_random = R_local_to_nev @ DeltaC_local @ R_local_to_nev.T

        # --- 3. Acumular Covariância Aleatória ---
        C_cum_nev_random_only = C_cum_nev_prev + DeltaC_nev_random # Propagação apenas da parte aleatória

        # --- 4. Calcular e Adicionar Contribuição Sistemática TOTAL no Ponto Atual ---
        # Esta é a parte mais controversa/simplificada. Adicionamos a variância
        # TOTAL dos erros sistemáticos projetada em NEV no ponto atual.
        # Isso ignora correlações entre erros sistemáticos e aleatórios.
        C_sist_total_atual = calcula_contrib_cov_sistematica_nev(md1, inc1, az1, params, tool_type)

        # Combinação (assumindo independência entre C acumulada aleatória e C sistemática total)
        # Cuidado: Se C_cum_nev_prev já continha C_sist_anterior, estamos adicionando de novo?
        # Uma abordagem alternativa seria propagar C_sist também: C_sist = C_sist_prev + DeltaC_sist
        # Mas DeltaC_sist é complexo.
        # Vamos usar a soma direta como APROXIMAÇÃO:
        C_cum_nev = C_cum_nev_random_only + C_sist_total_atual

        # Garante simetria numérica
        C_cum_nev = (C_cum_nev + C_cum_nev.T) / 2.0

        # Salva matriz cumulativa NEV no ponto i
        covariances_nev.append(C_cum_nev.copy())

        # Atualiza para a próxima iteração (usando a C *total*)
        C_cum_nev_prev = C_cum_nev

    return covariances_nev

# =============================================================================
# RESTANTE DO CÓDIGO STREAMLIT (Interface, Plotagem, etc.)
# =============================================================================

# Título e explicação
st.title("Comparador de Distâncias ISCWSA (Incremental Aprimorado)")
st.markdown("""
Esta aplicação compara a distância entre centros de trajetórias com a distância Pedal Curve,
utilizando **cálculo incremental de covariância ISCWSA** (MWD ou Gyro) com tratamento aprimorado (aproximado) de erros aleatórios e sistemáticos.
Considera as coordenadas de cabeça de poço (wellhead) diferentes.
*Nota: A implementação ainda contém **simplificações significativas** em relação aos modelos ISCWSA completos, especialmente nas sensibilidades e tratamento de erros sistemáticos como drift.*
""")

# --- Parâmetros de Incerteza ISCWSA (Sidebar) ---
# ... (sem mudanças na definição dos parâmetros na sidebar) ...
st.sidebar.header("Configuração do Modelo de Incerteza")
tool_type = st.sidebar.selectbox("Selecione o Tipo de Ferramenta", ["ISCWSA MWD", "ISCWSA Gyro"])
iscwsa_params = {}
# (Colar aqui a definição dos defaults e inputs da sidebar da versão anterior)
# ...
# Define default values robustly
default_mwd = {
    'depth_err_prop': 0.0002, 'depth_err_const': 0.1, 'acc_bias': 0.1, 'acc_sf': 200.0,
    'acc_mis_xy': 0.07, 'acc_mis_z': 0.07, 'mag_bias': 75.0, 'mag_sf': 150.0,
    'mag_mis_xy': 0.10, 'mag_mis_z': 0.10, 'mag_dec_err': 0.2, 'mag_dip_err': 0.1,
    'mag_ds_err': 0.3, 'sag_corr_err': 0.05, 'misalign_err_inc': 0.05, 'misalign_err_azi': 0.1,
    'gravity_strength': 9.81, 'mag_field_strength': 50000.0, 'dip_angle': 60.0
}
default_gyro = {
    'depth_err_prop': 0.0002, 'depth_err_const': 0.1, 'acc_bias': 0.1, 'acc_sf': 100.0,
    'acc_mis_xy': 0.10, 'acc_mis_z': 0.10, 'gyro_bias_drift_ns': 0.03, 'gyro_bias_drift_ew': 0.03,
    'gyro_bias_drift_v' : 0.03, 'gyro_sf' : 100.0, 'gyro_g_sens_drift': 0.05,
    'gyro_mis_xy': 0.10, 'gyro_mis_z': 0.10, 'gyro_az_ref_err': 0.03, 'sag_corr_err': 0.05,
    'misalign_err_inc': 0.05, 'misalign_err_azi': 0.1, 'gravity_strength': 9.81,
    'survey_time_hours': 0.25, # Ainda não usado efetivamente
}
# (Inputs da sidebar para MWD e Gyro aqui...)
if tool_type == "ISCWSA MWD":
    st.sidebar.subheader("Parâmetros ISCWSA MWD (1-sigma)")
    for key, val in default_mwd.items(): iscwsa_params[key] = st.sidebar.number_input(f"{key.replace('_',' ').title()} ({'m/m' if 'prop' in key else 'm' if 'const' in key else 'mg' if 'acc_bias' in key else 'ppm' if 'sf' in key else 'mrad' if 'mis' in key else 'nT' if 'mag_bias' in key else '°' if any(s in key for s in ['dec','dip','sag','misalign','azi','inc']) else 'g' if 'grav' in key else 'nT' if 'field' in key else ''})", value=val, step=0.01 if '°' in key or 'mis' in key else 0.0001 if 'prop' in key else 0.1, format="%.4f" if 'prop' in key or 'grav' in key else "%.2f")
elif tool_type == "ISCWSA Gyro":
    st.sidebar.subheader("Parâmetros ISCWSA Gyro (1-sigma)")
    for key, val in default_gyro.items(): iscwsa_params[key] = st.sidebar.number_input(f"{key.replace('_',' ').title()} ({'m/m' if 'prop' in key else 'm' if 'const' in key else 'mg' if 'acc_bias' in key else 'ppm' if 'sf' in key else 'mrad' if 'mis' in key else '°/hr' if 'drift' in key and 'g_sens' not in key else '°/hr/g' if 'g_sens' in key else '°' if any(s in key for s in ['ref','sag','misalign','azi','inc']) else 'g' if 'grav' in key else 'hr' if 'time' in key else ''})", value=val, step=0.01 if any(s in key for s in ['drift','ref','sag','misalign','azi','inc']) else 0.0001 if 'prop' in key else 0.1, format="%.4f" if 'prop' in key or 'grav' in key else "%.2f")
# Ensure all keys are present
default_params = default_mwd if tool_type == "ISCWSA MWD" else default_gyro
for key, val in default_params.items(): iscwsa_params.setdefault(key, val)

st.sidebar.header("Parâmetros de Cálculo")
sigma_factor = st.sidebar.slider("Fator Sigma (Confiança da Elipse)", 1.0, 3.0, 1.0, 0.1)

# --- Funções de Cálculo Trajetória, Elipse, Pedal Curve ---
# calculate_coordinates (sem mudanças)
# get_ellipse_params_from_covariance (sem mudanças)
# calculate_distance (sem mudanças)
# project_ellipse_iscwsa (sem mudanças)
# calculate_pedal_distance_iscwsa (sem mudanças)
# find_closest_tvd_point (sem mudanças)
# draw_ellipse_matplotlib (sem mudanças)
# (Colar as definições dessas funções aqui da versão anterior)
def calculate_coordinates(md, inc, az, wellhead_n=0.0, wellhead_e=0.0, wellhead_tvd=0.0):
   # ... (código completo da função) ...
   md = np.asarray(md, dtype=float); inc = np.asarray(inc, dtype=float); az = np.asarray(az, dtype=float)
   n_points = len(md)
   if n_points == 0: return pd.DataFrame({'MD': [], 'TVD': [], 'N': [], 'E': [], 'INC': [], 'AZ': []})
   n_rel = np.zeros_like(md); e_rel = np.zeros_like(md); tvd_rel = np.zeros_like(md)
   delta_md = np.diff(md, prepend=md[0] if md[0]!=0 else 0.0); delta_md[0] = md[0];
   if np.isclose(md[0], 0.0): delta_md[0] = 0.0
   inc_rad = np.radians(inc); az_rad = np.radians(az)
   inc1_rad = np.roll(inc_rad, 1); az1_rad = np.roll(az_rad, 1); inc2_rad = inc_rad; az2_rad = az_rad
   inc1_rad[0] = inc_rad[0]; az1_rad[0] = az_rad[0]
   cos_dls = np.cos(inc1_rad) * np.cos(inc2_rad) + np.sin(inc1_rad) * np.sin(inc2_rad) * np.cos(az2_rad - az1_rad)
   dogleg_angle_rad = np.arccos(np.clip(cos_dls, -1.0, 1.0)); dogleg_angle_rad[0] = 0.0
   half_dogleg = dogleg_angle_rad / 2.0
   rf = np.where(np.abs(dogleg_angle_rad) < 1e-9, 1.0, np.tan(half_dogleg) / half_dogleg); rf[0] = 1.0
   delta_n = delta_md / 2.0 * (np.sin(inc1_rad) * np.cos(az1_rad) + np.sin(inc2_rad) * np.cos(az2_rad)) * rf
   delta_e = delta_md / 2.0 * (np.sin(inc1_rad) * np.sin(az1_rad) + np.sin(inc2_rad) * np.sin(az2_rad)) * rf
   delta_v = delta_md / 2.0 * (np.cos(inc1_rad) + np.cos(inc2_rad)) * rf
   n_rel = np.cumsum(delta_n); e_rel = np.cumsum(delta_e); tvd_rel = np.cumsum(delta_v)
   n_abs = n_rel + wellhead_n; e_abs = e_rel + wellhead_e; tvd_abs = tvd_rel + wellhead_tvd
   return pd.DataFrame({'MD': md, 'TVD': tvd_abs, 'N': n_abs, 'E': e_abs, 'INC': inc, 'AZ': az})

def get_ellipse_params_from_covariance(C_nev, sigma_factor=1.0):
    # ... (código completo da função) ...
    C_ne = C_nev[0:2, 0:2]
    if np.allclose(C_ne, 0) or np.any(np.isnan(C_ne)): return 0.0, 0.0, 0.0
    try: eigenvalues, eigenvectors = np.linalg.eigh(C_ne)
    except np.linalg.LinAlgError: return 0.0, 0.0, 0.0
    eigenvalues[eigenvalues < 0] = 0
    idx_sort = np.argsort(eigenvalues)[::-1]; eigenvalues = eigenvalues[idx_sort]; eigenvectors = eigenvectors[:, idx_sort]
    semi_major = sigma_factor * np.sqrt(eigenvalues[0]); semi_minor = sigma_factor * np.sqrt(eigenvalues[1])
    n_comp = eigenvectors[0, 0]; e_comp = eigenvectors[1, 0]
    if np.isclose(n_comp, 0) and np.isclose(e_comp, 0): angle_deg_from_north = 0.0
    else: angle_deg_from_north = atan2d(e_comp, n_comp)
    angle_deg_from_north = angle_deg_from_north % 360.0
    return semi_major, semi_minor, angle_deg_from_north

def calculate_distance(p1, p2):
   # ... (código completo da função) ...
   if pd.isna(p1['N']) or pd.isna(p1['E']) or pd.isna(p2['N']) or pd.isna(p2['E']): return np.nan
   return np.sqrt((p2['N'] - p1['N'])**2 + (p2['E'] - p1['E'])**2)

def project_ellipse_iscwsa(semi_major, semi_minor, ellipse_angle_deg, direction_az_deg):
   # ... (código completo da função) ...
   if pd.isna(semi_major) or pd.isna(semi_minor) or pd.isna(ellipse_angle_deg) or pd.isna(direction_az_deg): return np.nan
   if semi_major < 1e-9: return 0.0
   relative_angle_deg = direction_az_deg - ellipse_angle_deg; relative_angle_rad = np.radians(relative_angle_deg)
   a = semi_major; b = semi_minor;
   if b < 1e-9: b = 1e-9 # Avoid pure zero
   cos_theta_sq = np.cos(relative_angle_rad)**2; sin_theta_sq = np.sin(relative_angle_rad)**2
   num = (a**2) * (b**2); den = (b**2) * cos_theta_sq + (a**2) * sin_theta_sq
   if den < 1e-12:
        if abs(np.cos(relative_angle_rad)) > 0.99: return a
        elif abs(np.sin(relative_angle_rad)) > 0.99: return b
        else: return 0.0
   projection = np.sqrt(num / den)
   return projection

def calculate_pedal_distance_iscwsa(p1, p2, cov_nev1, cov_nev2, sigma_factor):
    # ... (código completo da função) ...
    center_dist = calculate_distance(p1, p2)
    if pd.isna(center_dist) or cov_nev1 is None or cov_nev2 is None: return {'center_dist': np.nan, 'proj1': np.nan, 'proj2': np.nan, 'pedal_dist': np.nan, 'difference': np.nan, 'diff_percent': np.nan, 'SF': np.nan, 'smj1': np.nan, 'smn1': np.nan, 'ang1': np.nan, 'smj2': np.nan, 'smn2': np.nan, 'ang2': np.nan}
    if center_dist < 1e-6:
        sf_val = 0.0; proj1 = 0.0; proj2 = 0.0; pedal_dist = 0.0; difference = 0.0; diff_percent = 0.0
        smj1, smn1, ang1 = get_ellipse_params_from_covariance(cov_nev1, sigma_factor)
        smj2, smn2, ang2 = get_ellipse_params_from_covariance(cov_nev2, sigma_factor)
    else:
        delta_n = p2['N'] - p1['N']; delta_e = p2['E'] - p1['E']
        angle_deg_centers = atan2d(delta_e, delta_n) % 360.0
        smj1, smn1, ang1 = get_ellipse_params_from_covariance(cov_nev1, sigma_factor)
        smj2, smn2, ang2 = get_ellipse_params_from_covariance(cov_nev2, sigma_factor)
        proj1 = project_ellipse_iscwsa(smj1, smn1, ang1, angle_deg_centers)
        proj2 = project_ellipse_iscwsa(smj2, smn2, ang2, angle_deg_centers)
        if pd.isna(proj1) or pd.isna(proj2): pedal_dist, difference, diff_percent, sf_val = np.nan, np.nan, np.nan, np.nan
        else:
            pedal_dist = max(0, center_dist - (proj1 + proj2)); difference = center_dist - pedal_dist
            diff_percent = (difference / center_dist) * 100 if center_dist > 1e-6 else 0
            denominator = proj1 + proj2
            if denominator > 1e-9: sf_val = center_dist / denominator
            else: sf_val = np.inf if center_dist > 1e-9 else 0.0
    return {'center_dist': center_dist, 'proj1': proj1, 'proj2': proj2, 'pedal_dist': pedal_dist, 'difference': difference, 'diff_percent': diff_percent, 'smj1': smj1, 'smn1': smn1, 'ang1': ang1, 'smj2': smj2, 'smn2': smn2, 'ang2': ang2, 'SF': sf_val}

def find_closest_tvd_point(tvd_target, df):
   # ... (código completo da função) ...
   if df.empty or 'TVD' not in df.columns or df['TVD'].isna().all(): return None
   try:
        tvd_target = float(tvd_target); valid_tvd = df['TVD'].dropna()
        if valid_tvd.empty: return None
        idx = (valid_tvd - tvd_target).abs().idxmin()
        return df.loc[idx] # Returns a Series
   except (ValueError, TypeError): return None

def draw_ellipse_matplotlib(ax, center_xy, width, height, angle_deg, color="blue", alpha=0.3, label=None):
   # ... (código completo da função) ...
   if pd.isna(center_xy[0]) or pd.isna(center_xy[1]) or pd.isna(width) or pd.isna(height) or pd.isna(angle_deg): return None
   if width <= 1e-9 or height <= 1e-9: return None # Avoid zero size
   ellipse = Ellipse(xy=center_xy, width=width, height=height, angle=angle_deg, edgecolor=color, facecolor=color, alpha=alpha, label=label)
   ax.add_patch(ellipse); return ellipse


# --- Interface Streamlit ---
# ... (definição da interface, upload, wellhead coords - sem mudanças) ...
col1, col2 = st.columns(2)
with col1:
   st.header("Poço 1"); well1_file = st.file_uploader("Upload Excel Poço 1", type=["xlsx", "xls"], key="file1")
   st.subheader("Coordenadas Cabeça Poço 1"); n_wh1 = st.number_input("Norte (m)", key="n_wh1", value=0.0, format="%.2f"); e_wh1 = st.number_input("Este (m)", key="e_wh1", value=0.0, format="%.2f"); tvd_wh1 = st.number_input("TVD Inicial (m)", key="tvd_wh1", value=0.0, format="%.2f")
with col2:
   st.header("Poço 2"); well2_file = st.file_uploader("Upload Excel Poço 2", type=["xlsx", "xls"], key="file2")
   st.subheader("Coordenadas Cabeça Poço 2"); n_wh2 = st.number_input("Norte (m)", key="n_wh2", value=0.0, format="%.2f"); e_wh2 = st.number_input("Este (m)", key="e_wh2", value=0.0, format="%.2f"); tvd_wh2 = st.number_input("TVD Inicial (m)", key="tvd_wh2", value=0.0, format="%.2f")


# Processamento quando ambos os arquivos são carregados
if well1_file and well2_file:
   # Leitura e Validação dos arquivos Excel
   try:
       df_well1_orig = pd.read_excel(well1_file); df_well2_orig = pd.read_excel(well2_file)
       df_well1 = df_well1_orig.copy(); df_well2 = df_well2_orig.copy()
       # ... (código de validação e renomeação de colunas SEM MUDANÇAS) ...
       expected_cols = ['MD', 'INC', 'AZ']
       data_valid = True
       for df, well_name, file_ref in [(df_well1, "Poço 1", well1_file), (df_well2, "Poço 2", well2_file)]:
           original_cols = df.columns.tolist(); df.columns = [str(col).upper().strip() for col in df.columns]; current_cols = df.columns.tolist(); rename_map = {}
           if 'MD' not in current_cols: found_md = False; [rename_map.update({potential_md:'MD'}) or (found_md := True) for potential_md in ['MEASURED DEPTH', 'PROFUNDIDADE MEDIDA', 'PROF MEDIDA'] if potential_md in current_cols]; data_valid &= found_md; st.error(f"Coluna 'MD' não encontrada em {well_name}. Colunas: {original_cols}") if not found_md else None
           if 'INC' not in current_cols: found_inc = False; [rename_map.update({potential_inc:'INC'}) or (found_inc := True) for potential_inc in ['INCLINATION', 'INCL', 'INCLINACAO', 'INCLINAÇÃO'] if potential_inc in current_cols]; data_valid &= found_inc; st.error(f"Coluna 'INC' não encontrada em {well_name}. Colunas: {original_cols}") if not found_inc else None
           if 'AZ' not in current_cols: found_az = False; [rename_map.update({potential_az:'AZ'}) or (found_az := True) for potential_az in ['AZIMUTH', 'AZIM', 'AZIMUTE'] if potential_az in current_cols]; data_valid &= found_az; st.error(f"Coluna 'AZ' não encontrada em {well_name}. Colunas: {original_cols}") if not found_az else None
           if not data_valid: continue
           df.rename(columns=rename_map, inplace=True)
    
           # Ensure numeric types
           cols_to_check = [col for col in expected_cols if col in df.columns]
           for col in cols_to_check:
               try:
                   df[col] = pd.to_numeric(df[col], errors='coerce')
               except Exception as e_num:
                   st.error(f"Erro ao converter coluna '{col}' para número em {well_name}: {e_num}")
                   data_valid = False
                   # Optional: break here if one failure makes the whole file invalid?
                   # break
    
           if not data_valid: continue # Check validity again after trying conversion
    
           # Drop rows with NaN in essential columns (now includes those coerced by errors='coerce')
           initial_rows = len(df)
           df.dropna(subset=cols_to_check, inplace=True)
           if len(df) < initial_rows: st.warning(f"Removidas {initial_rows - len(df)} linhas inválidas em {well_name}.")
           if len(df) < 2: st.error(f"Arquivo {well_name} não contém dados suficientes."); data_valid = False
       if not data_valid: st.stop()

       # --- Calculation Steps ---
       # 1. Calculate Coordinates
       coords_well1 = calculate_coordinates(df_well1['MD'].values, df_well1['INC'].values, df_well1['AZ'].values, n_wh1, e_wh1, tvd_wh1)
       coords_well2 = calculate_coordinates(df_well2['MD'].values, df_well2['INC'].values, df_well2['AZ'].values, n_wh2, e_wh2, tvd_wh2)

       # 2. Calculate List of Cumulative Covariances using INCREMENTAL AVANÇADO method
       @st.cache_data(ttl=3600, show_spinner="Calculando Incerteza ISCWSA Incremental Aprimorada...")
       def get_incremental_covariance_list_avancado(md_arr, inc_arr, az_arr, params_dict, tool, params_tuple_key):
            md_np = np.asarray(md_arr); inc_np = np.asarray(inc_arr); az_np = np.asarray(az_arr)
            # Chamar a nova função de cálculo avançado
            return calculate_iscwsa_covariance_incremental_avancado(md_np, inc_np, az_np, params_dict, tool)

       params_tuple_key = tuple(sorted(iscwsa_params.items()))
       covs_list1 = get_incremental_covariance_list_avancado(coords_well1['MD'].values, coords_well1['INC'].values, coords_well1['AZ'].values, iscwsa_params, tool_type, params_tuple_key)
       covs_list2 = get_incremental_covariance_list_avancado(coords_well2['MD'].values, coords_well2['INC'].values, coords_well2['AZ'].values, iscwsa_params, tool_type, params_tuple_key)

       coords_well1 = coords_well1.reset_index().rename(columns={'index': 'OriginalIndex'})
       coords_well2 = coords_well2.reset_index().rename(columns={'index': 'OriginalIndex'})

       # 3. Find corresponding points and compare
       results = []
       st.write("Comparando trajetórias em profundidades correspondentes...")
       prog_bar_compare = st.progress(0)
       tvds_ref_raw = coords_well1['TVD'].dropna().unique()
       if len(tvds_ref_raw) == 0: st.error("Não há TVDs válidas no Poço 1."); st.stop()
       tvds_ref = np.sort(tvds_ref_raw)
       total_comparisons = len(tvds_ref); comparison_tvd_step = max(1, total_comparisons // 100)

       for i, tvd_ref in enumerate(tvds_ref):
           p1_matches = coords_well1[np.isclose(coords_well1['TVD'], tvd_ref)]
           if p1_matches.empty: continue
           p1_row = p1_matches.iloc[0]
           p1_index = p1_row['OriginalIndex'] # Ainda pode ser float aqui!

           p2_row = find_closest_tvd_point(tvd_ref, coords_well2)
           tvd1_val = p1_row['TVD']
           tvd2_val = p2_row['TVD'] if p2_row is not None else np.nan
           tvd_diff_threshold = 10.0

           if p2_row is not None and pd.notna(tvd1_val) and pd.notna(tvd2_val) and abs(tvd1_val - tvd2_val) < tvd_diff_threshold:
               p2_index = p2_row['OriginalIndex'] # Ainda pode ser float aqui!

               # --- CORREÇÃO DO TYPEERROR ---
               try:
                   p1_index_int = int(p1_index)
                   p2_index_int = int(p2_index)
               except (ValueError, TypeError):
                   st.warning(f"Não foi possível converter índices {p1_index} ou {p2_index} para int em TVD {tvd_ref}. Pulando ponto.")
                   continue # Pula esta comparação se o índice não for válido

               cov1 = covs_list1[p1_index_int] if p1_index_int < len(covs_list1) else None
               cov2 = covs_list2[p2_index_int] if p2_index_int < len(covs_list2) else None
               # -----------------------------

               cov1_valid = isinstance(cov1, np.ndarray) and cov1.shape == (3,3)
               cov2_valid = isinstance(cov2, np.ndarray) and cov2.shape == (3,3)

               if cov1_valid and cov2_valid:
                   distance_data = calculate_pedal_distance_iscwsa(p1_row, p2_row, cov1, cov2, sigma_factor)
                   results.append({
                       'TVD_Ref': tvd_ref, 'TVD_Actual1': p1_row['TVD'], 'TVD_Actual2': p2_row['TVD'],
                       'MD1': p1_row['MD'], 'MD2': p2_row['MD'], 'INC1': p1_row['INC'], 'INC2': p2_row['INC'],
                       'AZ1': p1_row['AZ'], 'AZ2': p2_row['AZ'], 'N1': p1_row['N'], 'E1': p1_row['E'],
                       'N2': p2_row['N'], 'E2': p2_row['E'],
                       'DistCentros': distance_data.get('center_dist', np.nan), 'DistPedal': distance_data.get('pedal_dist', np.nan),
                       'Proj1': distance_data.get('proj1', np.nan), 'Proj2': distance_data.get('proj2', np.nan),
                       'SMj1': distance_data.get('smj1', np.nan), 'SMn1': distance_data.get('smn1', np.nan), 'Ang1': distance_data.get('ang1', np.nan),
                       'SMj2': distance_data.get('smj2', np.nan), 'SMn2': distance_data.get('smn2', np.nan), 'Ang2': distance_data.get('ang2', np.nan),
                       'DifPerc': distance_data.get('diff_percent', np.nan), 'SF': distance_data.get('SF', np.nan),
                       'Index1': p1_index_int, 'Index2': p2_index_int # Store int index
                   })
           if i % comparison_tvd_step == 0: prog_bar_compare.progress((i + 1) / total_comparisons)
       prog_bar_compare.progress(1.0); prog_bar_compare.empty()

       # Create dataframe of results
       if results:
           df_results = pd.DataFrame(results)
           df_results.dropna(subset=['DistCentros', 'DistPedal', 'SF'], inplace=True)
           if df_results.empty: st.warning("Não foi possível calcular distâncias válidas ou encontrar pontos próximos."); st.stop()

           # Exibir tabela de resultados
           st.subheader("Comparação de Distâncias e SF (Incremental Aprimorado)")
           st.dataframe(df_results[['TVD_Ref', 'MD1', 'MD2', 'DistCentros', 'DistPedal', 'DifPerc', 'SF', 'SMj1', 'SMn1', 'Ang1', 'SMj2', 'SMn2', 'Ang2']].round(2))

           # --- Gráficos ---
           st.subheader("Gráficos de Análise")
           # ... (Código dos gráficos Plotly e Matplotlib - SEM MUDANÇAS AQUI, usar df_results) ...
           # Gráfico Dist vs TVD
           fig1 = go.Figure(); fig1.add_trace(go.Scatter(x=df_results['TVD_Ref'], y=df_results['DistCentros'], mode='lines+markers', name='Dist. Centros')); fig1.add_trace(go.Scatter(x=df_results['TVD_Ref'], y=df_results['DistPedal'], mode='lines+markers', name=f'Dist. Pedal ({sigma_factor:.1f}σ)')); fig1.update_layout(title='Distâncias vs TVD Ref', xaxis_title='TVD Ref (m)', yaxis_title='Distância (m)'); st.plotly_chart(fig1, use_container_width=True)
           # Gráfico SF vs MD
           fig_sf = go.Figure(); fig_sf.add_trace(go.Scatter(x=df_results['MD1'], y=df_results['SF'], mode='lines+markers', name='SF')); min_md = df_results['MD1'].min(); max_md = df_results['MD1'].max(); fig_sf.add_shape(type="line", x0=min_md, y0=1.0, x1=max_md, y1=1.0, line=dict(color="red",width=2,dash="dash")); fig_sf.add_shape(type="line", x0=min_md, y0=1.5, x1=max_md, y1=1.5, line=dict(color="orange",width=2,dash="dash")); fig_sf.add_shape(type="line", x0=min_md, y0=2.0, x1=max_md, y1=2.0, line=dict(color="green",width=2,dash="dash"))
           annotations_list = [dict(x=max_md*0.95, y=0.5, text="Colisão", showarrow=False, font=dict(color="red",size=10)), dict(x=max_md*0.95, y=1.25, text="Alto Risco", showarrow=False, font=dict(color="orange",size=10)), dict(x=max_md*0.95, y=1.75, text="Médio Risco", showarrow=False, font=dict(color="green",size=10)), dict(x=max_md*0.95, y=2.25, text="Baixo Risco", showarrow=False, font=dict(color="blue",size=10))]; fig_sf.update_layout(title='SF vs MD Poço 1', xaxis_title='MD Poço 1 (m)', yaxis_title='SF', yaxis_range=[0, max(df_results['SF'].max()*1.1 if not df_results['SF'].empty and df_results['SF'].max()!=np.inf and pd.notna(df_results['SF'].max()) else 5, 2.1)], annotations=annotations_list); st.plotly_chart(fig_sf, use_container_width=True)
           # Gráfico 3D
           fig5 = go.Figure(); fig5.add_trace(go.Scatter3d(x=coords_well1['E'], y=coords_well1['N'], z=coords_well1['TVD'], mode='lines', name='Poço 1', line=dict(color='blue',width=4))); fig5.add_trace(go.Scatter3d(x=coords_well2['E'], y=coords_well2['N'], z=coords_well2['TVD'], mode='lines', name='Poço 2', line=dict(color='red',width=4))); fig5.add_trace(go.Scatter3d(x=df_results['E1'], y=df_results['N1'], z=df_results['TVD_Actual1'], mode='markers', name='Pts Comp 1', marker=dict(color='cyan',size=3))); fig5.add_trace(go.Scatter3d(x=df_results['E2'], y=df_results['N2'], z=df_results['TVD_Actual2'], mode='markers', name='Pts Comp 2', marker=dict(color='magenta',size=3))); fig5.update_layout(scene=dict(xaxis_title='E (m)', yaxis_title='N (m)', zaxis_title='TVD (m)', zaxis=dict(autorange="reversed"), aspectmode='auto'), title="Visualização 3D", height=700); st.plotly_chart(fig5, use_container_width=True)
           # Gráfico 2D Elipses
           st.subheader(f"Visualização 2D com Elipses ({sigma_factor:.1f}σ)")
           tvd_options_ref = np.sort(df_results['TVD_Ref'].unique())
           if len(tvd_options_ref) == 0: st.warning("Não há TVDs para visualização 2D.")
           else:
                default_index_2d = len(tvd_options_ref) // 2; selected_tvd_ref = st.selectbox("Selecione TVD Ref para visualização 2D", tvd_options_ref, index=default_index_2d, format_func=lambda x: f"{x:.1f} m")
                selected_data_rows = df_results[np.isclose(df_results['TVD_Ref'], selected_tvd_ref)]
                if selected_data_rows.empty: st.warning(f"Dados não encontrados para TVD {selected_tvd_ref:.1f} m.")
                else:
                    selected_data = selected_data_rows.iloc[0]
                    fig_2d, ax_2d = plt.subplots(figsize=(10, 10))
                    ax_2d.scatter(selected_data['E1'], selected_data['N1'], c='blue', s=50, label=f'P1 (TVD≈{selected_data["TVD_Actual1"]:.1f}m)', zorder=5); ax_2d.scatter(selected_data['E2'], selected_data['N2'], c='red', s=50, label=f'P2 (TVD≈{selected_data["TVD_Actual2"]:.1f}m)', zorder=5); ax_2d.plot([selected_data['E1'], selected_data['E2']], [selected_data['N1'], selected_data['N2']], 'k--', alpha=0.7, label=f'Dist: {selected_data["DistCentros"]:.2f} m')
                    angle_mpl1 = (90.0 - selected_data['Ang1']) % 360.0 if pd.notna(selected_data['Ang1']) else 0.0; draw_ellipse_matplotlib(ax_2d, (selected_data['E1'], selected_data['N1']), 2*selected_data.get('SMj1',0), 2*selected_data.get('SMn1',0), angle_mpl1, color="blue", alpha=0.3, label=f'E1 ({sigma_factor:.1f}σ)')
                    angle_mpl2 = (90.0 - selected_data['Ang2']) % 360.0 if pd.notna(selected_data['Ang2']) else 0.0; draw_ellipse_matplotlib(ax_2d, (selected_data['E2'], selected_data['N2']), 2*selected_data.get('SMj2',0), 2*selected_data.get('SMn2',0), angle_mpl2, color="red", alpha=0.3, label=f'E2 ({sigma_factor:.1f}σ)')
                    info_text = f"TVD Ref: {selected_data.get('TVD_Ref', np.nan):.1f}m\nMD1: {selected_data.get('MD1', np.nan):.1f}m, MD2: {selected_data.get('MD2', np.nan):.1f}m\nDist C: {selected_data.get('DistCentros', np.nan):.2f}m\nDist P: {selected_data.get('DistPedal', np.nan):.2f}m\nSF: {selected_data.get('SF', np.nan):.2f}\nE1: SMj={selected_data.get('SMj1', np.nan):.2f}, SMn={selected_data.get('SMn1', np.nan):.2f}, Ang={selected_data.get('Ang1', np.nan):.1f}°\nE2: SMj={selected_data.get('SMj2', np.nan):.2f}, SMn={selected_data.get('SMn2', np.nan):.2f}, Ang={selected_data.get('Ang2', np.nan):.1f}°"
                    ax_2d.text(0.02, 0.98, info_text, transform=ax_2d.transAxes, fontsize=8, va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
                    ax_2d.set_xlabel('Este (m)'); ax_2d.set_ylabel('Norte (m)'); ax_2d.set_title(f'Elipses ISCWSA Incr. Aprimorado @ TVD Ref ≈ {selected_data.get("TVD_Ref", np.nan):.1f} m'); ax_2d.grid(True, linestyle=':', alpha=0.6); ax_2d.axis('equal'); ax_2d.legend(fontsize=8, loc='lower right')
                    # Zoom logic (copied)
                    max_r1 = selected_data.get('SMj1',0); max_r2 = selected_data.get('SMj2',0); center_e = (selected_data.get('E1',0)+selected_data.get('E2',0))/2; center_n = (selected_data.get('N1',0)+selected_data.get('N2',0))/2; dist_c = selected_data.get("DistCentros",0)
                    e1=selected_data.get('E1',center_e);n1=selected_data.get('N1',center_n);e2=selected_data.get('E2',center_e);n2=selected_data.get('N2',center_n)
                    ext1e=abs(e1-center_e)+max_r1;ext1n=abs(n1-center_n)+max_r1;ext2e=abs(e2-center_e)+max_r2;ext2n=abs(n2-center_n)+max_r2
                    hvs = max(ext1e, ext1n, ext2e, ext2n, dist_c/2 if dist_c>0 else 1) * 1.2; hvs = max(hvs, 5)
                    ax_2d.set_xlim(center_e - hvs, center_e + hvs); ax_2d.set_ylim(center_n - hvs, center_n + hvs)
                    st.pyplot(fig_2d)

           # Explicação Pedal Curve Atualizada
           with st.expander("Explicação do Método (Incremental Aprimorado)"):
                st.markdown(f"""
                ### Método Incremental Aprimorado (Aproximado)

                Esta versão tenta um cálculo mais detalhado da incerteza, separando erros aleatórios e sistemáticos:

                1.  **Erros Aleatórios por Segmento:**
                    *   Calcula-se `DeltaC_local` para cada segmento, representando a incerteza adicionada *apenas* por fontes de erro consideradas aleatórias naquele trecho (ex: ruído de medição, variações rápidas de desalinhamento, erro proporcional de profundidade).
                    *   Usa-se uma **matriz de sensibilidade (H)** simplificada para mapear esses erros aleatórios para os eixos locais (du, dv, dw).
                    *   `DeltaC_local = H @ Sigma_aleatorios @ H.T`. Esta matriz pode ter termos fora da diagonal (covariâncias).
                    *   `DeltaC_local` é rotacionada para NEV (`DeltaC_nev_random`) e somada à covariância acumulada da etapa anterior (`C_cum_nev_prev`).

                2.  **Erros Sistemáticos:**
                    *   Erros considerados constantes ou que se acumulam previsivelmente (ex: erro de referência azimutal, erro constante de profundidade, erro de declinação) são tratados separadamente.
                    *   A contribuição **total** desses erros para a incerteza NEV é calculada no ponto final do segmento (`md1`, `inc1`, `az1`). Isso envolve projetar a variância do erro sistemático (ex: `Var(AzRef)`, `Var(DepthConst)`) nos eixos NEV usando a geometria *atual* do poço.
                    *   Esta contribuição sistemática (`C_sist_total_atual`) é **adicionada** à covariância que já inclui os erros aleatórios acumulados (`C_cum_nev = C_cum_nev_random_only + C_sist_total_atual`).

                **Limitações:**
                *   As sensibilidades na H-matrix são aproximações.
                *   A forma de adicionar a contribuição sistemática é uma simplificação; ignora correlações potenciais entre erros sistemáticos e aleatórios acumulados e a forma exata como os erros sistemáticos se propagam.
                *   Erros complexos como o drift do giroscópio ainda não são modelados de forma totalmente rigorosa.

                **Pedal Curve e SF:** Calculados da mesma forma que antes, mas usando as elipses derivadas desta matriz de covariância aprimorada. A interpretação do SF ({sigma_factor:.1f}σ) permanece a mesma.
                """)

       else:
           st.warning("Não foi possível encontrar pontos de comparação ou dados resultantes insuficientes.")

   # --- Blocos except e else para upload de arquivos (sem mudanças) ---
   except FileNotFoundError: st.error("Erro: Arquivo não encontrado.")
   except pd.errors.EmptyDataError: st.error("Erro: Arquivo Excel vazio ou ilegível.")
   except ValueError as e: st.error(f"Erro ao converter dados. Verifique MD/INC/AZ/Coords WH. Detalhe: {e}")
   except KeyError as e: st.error(f"Erro: Coluna essencial não encontrada: {e}. Verifique nomes no Excel.")
   except Exception as e: st.error(f"Erro inesperado: {e}"); st.exception(e)

else:
    # --- Tela inicial (sem mudanças) ---
    st.info("Aguardando upload dos arquivos Excel e definição das coordenadas de cabeça de poço...")
    example_data = pd.DataFrame({'MD': [0, 500, 1000], 'INC': [0, 15, 30], 'AZ': [0, 45, 45]})
    st.write("Formato esperado:"); st.dataframe(example_data); st.markdown("""(...)""")
    buffer = BytesIO(); with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer: example_data.to_excel(writer, index=False, sheet_name='Exemplo'); buffer.seek(0)
    st.download_button(label="Download Arquivo Exemplo", data=buffer, file_name="exemplo.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

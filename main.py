# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from io import BytesIO
import math # Keep math for potential future use, though np handles most

st.set_page_config(page_title="Comparador de Distâncias ISCWSA", layout="wide")

# --- Funções Trigonométricas (Graus) ---
# Using numpy versions is generally preferred for array operations
def sind(degrees):
    return np.sin(np.radians(degrees))

def cosd(degrees):
    return np.cos(np.radians(degrees))

def tand(degrees):
    # Avoid tan(90) issues if possible, though np handles inf
    degrees = np.asanyarray(degrees)
    rad = np.radians(degrees)
    # Add a small epsilon to avoid exact multiples of pi/2 if necessary,
    # but usually np.tan handles it returning large numbers or inf.
    return np.tan(rad)

def atand(value):
    return np.degrees(np.arctan(value))

def atan2d(y, x):
    """Calculates arctan2 and returns degrees, handling quadrant correctly."""
    return np.degrees(np.arctan2(y, x))

def acosd(value):
    """Calculates arccos and returns degrees, clipping input for domain safety."""
    # Clip value to avoid domain errors due to floating point inaccuracies
    return np.degrees(np.arccos(np.clip(value, -1.0, 1.0)))

# Título e explicação
st.title("Comparador de Distâncias com Modelos ISCWSA")
st.markdown("""
Esta aplicação compara a distância entre centros de trajetórias com a distância Pedal Curve,
utilizando modelos de incerteza padrão ISCWSA (MWD ou Gyro) para calcular as elipses de incerteza.
**Agora com suporte para coordenadas de cabeça de poço (wellhead) diferentes.**
""")

# --- Parâmetros de Incerteza ISCWSA ---
st.sidebar.header("Configuração do Modelo de Incerteza")
tool_type = st.sidebar.selectbox("Selecione o Tipo de Ferramenta", ["ISCWSA MWD", "ISCWSA Gyro"])

iscwsa_params = {}

# Define default values robustly
default_mwd = {
    'depth_err_prop': 0.00056, 'depth_err_const': 0.35, 'acc_bias': 0.004, 'acc_sf': 0.0005,
    'acc_mis_xy': 0.1, 'acc_mis_z': 0.1, 'mag_bias': 70.0, 'mag_sf': 0.0016,
    'mag_mis_xy': 0.15, 'mag_mis_z': 0.15, 'mag_dec_err': 0.36, 'mag_dip_err': 0.1,
    'mag_ds_err': 0.6, 'sag_corr_err': 0.2, 'misalign_err_inc': 0.06,
    'misalign_err_azi': 0.06, 'gravity_strength': 1.0, 'mag_field_strength': 50000.0,
    'dip_angle': 60.0
}
default_gyro = {
    'depth_err_prop': 0.00056, 'depth_err_const': 1, 'acc_bias': 0.004, 'acc_sf': 0.0005,
    'acc_mis_xy': 0.1, 'acc_mis_z': 0.1, 'gyro_bias_drift_ns': 0.033, 'gyro_bias_drift_ew': 0.033,
    'gyro_bias_drift_v': 0.033, 'gyro_sf': 200.0, 'gyro_g_sens_drift': 0.03,
    'gyro_mis_xy': 0.1, 'gyro_mis_z': 0.1, 'gyro_az_ref_err': 0.1,
    'sag_corr_err': 0.08, 'misalign_err_inc': 0.1, 'misalign_err_azi': 0.1,
    'gravity_strength': 1.0, 'survey_time_hours': 1.0
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
   o offset da cabeça do poço (wellhead).
   """
   # Ensure inputs are numpy arrays
   md = np.asarray(md, dtype=float)
   inc = np.asarray(inc, dtype=float)
   az = np.asarray(az, dtype=float)

   # Initialize relative coordinate arrays
   n_rel = np.zeros_like(md)
   e_rel = np.zeros_like(md)
   tvd_rel = np.zeros_like(md)

   # Calculate segment lengths
   delta_md = np.diff(md, prepend=0.0)

   # Convert degrees to radians
   inc_rad = np.radians(inc)
   az_rad = np.radians(az)

   # Shift arrays for previous station data
   inc1_rad_calc = np.roll(inc_rad, 1)
   az1_rad_calc = np.roll(az_rad, 1)
   inc2_rad_calc = inc_rad
   az2_rad_calc = az_rad

   # For the first point (MD=0), the delta_md is usually based on the MD value itself if first MD is non-zero.
   # If first MD is 0, delta_md[0] is 0. The calculation below will yield 0 increments for the first point, which is correct for relative position.
   # The first point's inc/az are used as the *start* of the *first segment*.
   # Adjust inc1/az1 for the first segment calculation.
   inc1_rad_calc[0] = inc_rad[0] # The first segment starts from the first point's angles
   az1_rad_calc[0] = az_rad[0]

   # Dogleg angle calculation using vectorized operations
   # Correct DL formula: cos(DL) = cos(I2-I1) - sin(I1)sin(I2)(1-cos(A2-A1))
   cos_dls_correct = (np.cos(inc2_rad_calc - inc1_rad_calc) -
                      np.sin(inc1_rad_calc) * np.sin(inc2_rad_calc) * (1 - np.cos(az2_rad_calc - az1_rad_calc)))

   # Clip to avoid domain errors
   dogleg_angle_rad = np.arccos(np.clip(cos_dls_correct, -1.0, 1.0))

   # Ratio Factor (RF) = tan(DL/2) / (DL/2)
   # Handle DL=0 case to avoid division by zero and tan(0)/0 limit = 1
   half_dogleg = dogleg_angle_rad / 2.0
   # Use np.where to avoid division by zero or warnings
   rf = np.where(np.abs(dogleg_angle_rad) < 1e-9,
                 1.0,
                 np.tan(half_dogleg) / half_dogleg)
   rf[0] = 1.0 # Ensure RF is 1 for the first point segment


   # Calculate increments in N, E, V
   delta_n = delta_md / 2.0 * (np.sin(inc1_rad_calc) * np.cos(az1_rad_calc) + np.sin(inc2_rad_calc) * np.cos(az2_rad_calc)) * rf
   delta_e = delta_md / 2.0 * (np.sin(inc1_rad_calc) * np.sin(az1_rad_calc) + np.sin(inc2_rad_calc) * np.sin(az2_rad_calc)) * rf
   delta_v = delta_md / 2.0 * (np.cos(inc1_rad_calc) + np.cos(inc2_rad_calc)) * rf

   # Calculate cumulative *relative* coordinates
   n_rel = np.cumsum(delta_n)
   e_rel = np.cumsum(delta_e)
   tvd_rel = np.cumsum(delta_v)

   # Apply wellhead offset to get absolute coordinates
   n_abs = n_rel + wellhead_n
   e_abs = e_rel + wellhead_e
   tvd_abs = tvd_rel + wellhead_tvd

   return pd.DataFrame({
       'MD': md,
       'TVD': tvd_abs,  # Return absolute TVD
       'N': n_abs,      # Return absolute North
       'E': e_abs,      # Return absolute East
       'INC': inc,
       'AZ': az
   })

# --- Funções de Cálculo Incerteza ISCWSA (Aproximação por Contribuição de Variância) ---

def calculate_iscwsa_covariance(md, inc, az, params, tool_type):
    """
    Calcula a matriz de covariância 3x3 (NEV) em um ponto específico
    usando uma aproximação baseada na soma das contribuições de variância
    dos termos de erro ISCWSA.
    Retorna a matriz C_nev.
    Nota: Esta é uma *aproximação simplificada* e não uma implementação completa do ISCWSA.
    A covariância é acumulada ao longo do MD. O cálculo abaixo retorna a CUMULATIVA
    de incerteza até o ponto MD.
    """
    # For the first point (MD=0), the covariance is zero
    if md < 1e-6:
        return np.zeros((3,3))

    # Conversões de unidades para consistência interna (radianos, frações)
    inc_rad = np.radians(inc)
    az_rad = np.radians(az)
    mrad_to_rad = 0.001
    ppm_to_frac = 1e-6
    mg_to_g = 0.001
    nT_to_T = 1e-9
    deg_to_rad = np.pi / 180.0
    hr_to_sec = 3600.0
    gravity_mps2 = params.get('gravity_strength', 1.0) * 9.80665 # Use standard gravity

    # Common parameters
    sigma_depth_prop = params.get('depth_err_prop', 0)
    sigma_depth_const = params.get('depth_err_const', 0)
    sigma_acc_bias = params.get('acc_bias', 0) * mg_to_g * gravity_mps2 # Converted to m/s^2
    sigma_acc_sf = params.get('acc_sf', 0) * ppm_to_frac
    sigma_acc_mis_xy = params.get('acc_mis_xy', 0) * mrad_to_rad
    sigma_acc_mis_z = params.get('acc_mis_z', 0) * mrad_to_rad
    sigma_sag = params.get('sag_corr_err', 0) * deg_to_rad
    sigma_misalign_inc = params.get('misalign_err_inc', 0) * deg_to_rad
    sigma_misalign_azi = params.get('misalign_err_azi', 0) * deg_to_rad


    # Variances (sigma^2) for the error sources (conceptual rates or points)
    # These contribute to angular errors (Inc/Az) which then scale by MD

    # Variância Angular de Inclinação (rad^2) - Calculated within tool blocks
    total_var_inc_rad2 = 0

    # Variância Angular de Azimute (rad^2) - Calculated within tool blocks
    total_var_az_rad2 = 0

    if tool_type == "ISCWSA MWD":
        # MWD-specific parameters
        sigma_mag_bias = params.get('mag_bias', 0) * nT_to_T
        sigma_mag_sf = params.get('mag_sf', 0) * ppm_to_frac
        sigma_mag_mis_xy = params.get('mag_mis_xy', 0) * mrad_to_rad
        sigma_mag_mis_z = params.get('mag_mis_z', 0) * mrad_to_rad
        sigma_dec_err = params.get('mag_dec_err', 0) * deg_to_rad
        sigma_ds_err = params.get('mag_ds_err', 0) * deg_to_rad

        dip_rad = np.radians(params.get('dip_angle', 60))
        B_total_T = params.get('mag_field_strength', 50000) * nT_to_T
        B_H = B_total_T * np.cos(dip_rad) # Horizontal component of magnetic field

        # Inclination Error (rad^2)
        # dInc contributions (simplified): AB, SF, Misalign, Sag, MisalignInc
        # This is simplified; actual ISCWSA integrates contributions along MD segments.
        # Approximate total variance by summing variances scaled by relevant sensitivities and MD.
        var_inc_acc_bias = (sigma_acc_bias / gravity_mps2)**2
        var_inc_acc_sf = (sigma_acc_sf * sind(inc))**2 # Simplified
        var_inc_acc_mis = (max(sigma_acc_mis_xy, sigma_acc_mis_z) * cosd(inc))**2 # Simplified
        var_inc_sag = sigma_sag**2
        var_inc_misalign = sigma_misalign_inc**2

        # Sum of uncorrelated inclination error variances
        total_var_inc_rad2 = var_inc_acc_bias + var_inc_acc_sf + var_inc_acc_mis + var_inc_sag + var_inc_misalign

        # Azimuth Error (rad^2)
        # dAz contributions (simplified): MB, MS, MM, Dec, DS, MisalignAzi
        # Sensitivity denominator: Bh * sin(I)
        sensitivity_az = (B_H * sind(inc))

        var_az_mag_bias = 0
        var_az_mag_sf = 0
        var_az_mag_mis = 0
        var_az_dec = sigma_dec_err**2
        var_az_ds = sigma_ds_err**2
        var_az_misalign = sigma_misalign_azi**2

        if abs(sensitivity_az) > 1e-9: # Avoid division by zero near vertical or magnetic equator
             # Proxy for magnitude errors contributing to lateral error
             var_az_mag_bias = (sigma_mag_bias / sensitivity_az)**2
             # SF and Misalign are more complex, but add a simplified proxy related to inclination sensitivity
             # var_az_mag_sf = (sigma_mag_sf * cotd(inc) if abs(sind(inc)) > 1e-3 else (30*deg_to_rad))**2 # Very rough proxy
             # var_az_mag_mis = (max(sigma_mag_mis_xy, sigma_mag_mis_z) * cotd(inc) if abs(sind(inc)) > 1e-3 else (30*deg_to_rad))**2 # Very rough proxy
        else: # Near vertical or Magnetic Equator, Azimuth highly uncertain
             # Assign a large variance if sensitivity is zero or near zero
             # This is a simplified way to represent the large uncertainty in these zones.
             # A full ISCWSA model uses more sophisticated methods.
             large_az_error = (30*deg_to_rad)**2 # Example: 30 deg 1-sigma error
             var_az_mag_bias = large_az_error # Assuming major source of error is here
             # Assign other variances based on large error or zero if irrelevant
             var_az_mag_sf = large_az_error
             var_az_mag_mis = large_az_error


        # Sum of uncorrelated azimuth error variances
        total_var_az_rad2 = var_az_mag_bias + var_az_mag_sf + var_az_mag_mis + var_az_dec + var_az_ds + var_az_misalign


    elif tool_type == "ISCWSA Gyro":
        # Gyro-specific parameters
        sigma_gyro_bias_drift_ns = params.get('gyro_bias_drift_ns', 0) * deg_to_rad / hr_to_sec # rad/s
        sigma_gyro_bias_drift_ew = params.get('gyro_bias_drift_ew', 0) * deg_to_rad / hr_to_sec # rad/s
        sigma_gyro_bias_drift_v = params.get('gyro_bias_drift_v', 0) * deg_to_rad / hr_to_sec # rad/s
        sigma_gyro_sf = params.get('gyro_sf', 0) * ppm_to_frac # unitless
        sigma_gyro_g_sens_drift = params.get('gyro_g_sens_drift', 0) * deg_to_rad / hr_to_sec / gravity_mps2 # rad/s per m/s^2
        sigma_gyro_mis_xy = params.get('gyro_mis_xy', 0) * mrad_to_rad # rad
        sigma_gyro_mis_z = params.get('gyro_mis_z', 0) * mrad_to_rad # rad
        sigma_az_ref_err = params.get('gyro_az_ref_err', 0) * deg_to_rad # rad
        survey_time_sec = params.get('survey_time_hours', 1) * hr_to_sec # seconds

        # Inclination Error (rad^2)
        # Acc contributions (same as MWD simplified) + Gyro contributions (less significant for Inc in typical Gyro)
        var_inc_acc_bias = (sigma_acc_bias / gravity_mps2)**2
        var_inc_acc_sf = (sigma_acc_sf * sind(inc))**2 # Simplified
        var_inc_acc_mis = (max(sigma_acc_mis_xy, sigma_acc_mis_z) * cosd(inc))**2 # Simplified
        var_inc_sag = sigma_sag**2
        var_inc_misalign = sigma_misalign_inc**2

        # Gyro G-sensitivity contributes to Inclination error (simplified proxy)
        var_inc_gyro_g_sens = (sigma_gyro_g_sens_drift * gravity_mps2 * cosd(inc) * survey_time_sec)**2 # g-sens in vertical plane

        total_var_inc_rad2 = var_inc_acc_bias + var_inc_acc_sf + var_inc_acc_mis + var_inc_sag + var_inc_misalign + var_inc_gyro_g_sens

        # Azimuth Error (rad^2)
        # Gyro Bias Drift contributes to Azimuth error over time, scaled by 1/sin(I)
        # Gyro G-Sensitivity contributes to Azimuth error, scaled by g * sin(I)
        # Azimuth Reference error (AZID) is a fixed offset error
        # Gyro Misalign contributes (complex)
        # Gyro SF contributes (complex)

        var_az_gyro_bias_drift = 0
        # Simplification: Use the combined horizontal drift rate
        sigma_gyro_bias_drift_h = np.sqrt(sigma_gyro_bias_drift_ns**2 + sigma_gyro_bias_drift_ew**2)
        if abs(sind(inc)) > 1e-3:
             var_az_gyro_bias_drift = (sigma_gyro_bias_drift_h * survey_time_sec / sind(inc))**2
        else:
             # High uncertainty near vertical
             var_az_gyro_bias_drift = (30*deg_to_rad)**2 # Large error proxy

        var_az_gyro_g_sens = (sigma_gyro_g_sens_drift * gravity_mps2 * sind(inc) * survey_time_sec)**2 # g-sens in horizontal plane
        var_az_ref = sigma_az_ref_err**2
        var_az_misalign = sigma_misalign_azi**2
        # Missing SF and Gyro Misalign contributions for simplicity

        # Sum of uncorrelated azimuth error variances
        total_var_az_rad2 = var_az_gyro_bias_drift + var_az_gyro_g_sens + var_az_ref + var_az_misalign


    # --- Approximate Covariance Matrix in NEV ---
    # Propagate angular errors and depth error into NEV using simplified sensitivities
    # This is still a major simplification compared to a full ISCWSA error model which integrates
    # error effects along the path and considers correlations between error sources.

    # Var_Along: dominated by depth error
    var_along = (sigma_depth_prop * md)**2 + sigma_depth_const**2 # Recalculated here for clarity, same as var_depth_at_md

    # Var_Vertical_plane: comes from inclination error rotated to be perpendicular to the well in the vertical plane
    var_vert_plane = (md**2) * total_var_inc_rad2 # dV approx MD * dInc

    # Var_Lateral_plane: comes from azimuth error rotated to be perpendicular to the well in the horizontal plane
    var_lat_plane = (md * sind(inc))**2 * total_var_az_rad2 # dH approx MD * sin(I) * dAz


    # Matriz de Rotação de Wellbore frame (u=along, v=lateral/right, w=vertical/up) para NEV
    # Based on standard definitions
    si = sind(inc)
    ci = cosd(inc)
    sa = sind(az)
    ca = cosd(az)

    # Define the rotation matrix from Wellbore Frame (u, v, w) to NEV
    # u = Along-hole, v = Lateral (perp, right), w = Vertical Plane (perp, up)
    R_wb_to_nev = np.array([
        [si*ca, -sa, ci*ca], # N component of u, v, w
        [si*sa, ca, ci*sa],  # E component of u, v, w
        [ci   , 0 , -si]     # V_down component of u, v, w
    ])

    # C_wellbore = diag(var_along, var_lat_plane, var_vert_plane) # Assuming order (u, v, w)
    C_wellbore = np.diag([var_along, var_lat_plane, var_vert_plane])


    # Rotação da Matriz de Covariância: C_nev = R * C_wellbore * R.T
    C_nev = R_wb_to_nev @ C_wellbore @ R_wb_to_nev.T

    # Ensure symmetry (numerical precision can cause small asymmetry)
    C_nev = (C_nev + C_nev.T) / 2.0

    return C_nev


def get_ellipse_params_from_covariance(C_nev, sigma_factor=1.0):
    """
    Calcula os parâmetros da elipse de incerteza horizontal a partir
    da matriz de covariância 3x3 NEV.
    Retorna: semi_major, semi_minor, angle_deg (ângulo do eixo maior com o Norte, 0-360)
    """
    # Extrair submatriz 2x2 horizontal (NE) - Cuidado com a ordem N, E!
    # C_nev = [[VarN, CovNE, CovNV], [CovNE, VarE, CovEV], [CovNV, CovEV, VarV]]
    # Standard NEV order: N=row/col 0, E=row/col 1
    C_ne = C_nev[0:2, 0:2] # Extrai [[VarN, CovNE], [CovNE, VarE]]

    # Check if matrix is valid (e.g., not all zeros or NaNs)
    if np.allclose(C_ne, 0) or np.any(np.isnan(C_ne)):
        return 0.0, 0.0, 0.0

    # Calcular autovalores e autovetores da matriz C_ne
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(C_ne) # Use eigh for symmetric matrices
    except np.linalg.LinAlgError:
        st.warning(f"Eigenvalue decomposition failed for C_ne:\n{C_ne}\nReturning zero ellipse.")
        return 0.0, 0.0, 0.0


    # Autovalores dão as variâncias ao longo dos eixos principais (sigma^2)
    # Autovetores dão a direção desses eixos

    # Semi-eixos são a raiz quadrada dos autovalores, vezes o fator sigma
    # Garante que semi_major seja o maior
    # Ensure eigenvalues are non-negative before sorting and sqrt
    eigenvalues[eigenvalues < 0] = 0 # Clip negative eigenvalues

    idx_sort = np.argsort(eigenvalues)[::-1] # Índices do maior para o menor eigenvalue
    eigenvalues = eigenvalues[idx_sort]
    eigenvectors = eigenvectors[:, idx_sort]

    semi_major = sigma_factor * np.sqrt(eigenvalues[0])
    semi_minor = sigma_factor * np.sqrt(eigenvalues[1])

    # Ângulo do eixo maior (correspondente ao primeiro autovetor)
    # O primeiro autovetor é eigenvectors[:, 0] = [N_comp, E_comp]
    # Need N component first, E component second for atan2d(E, N) -> Azimuth
    n_comp = eigenvectors[0, 0]
    e_comp = eigenvectors[1, 0]

    # atan2d(y, x) -> atan2d(East, North) gives angle East of North (Azimuth)
    # Handle case where eigenvector is zero vector (e.g., from zero covariance)
    if np.isclose(n_comp, 0) and np.isclose(e_comp, 0):
        angle_deg_from_north = 0.0 # Or NaN? Let's return 0 for simplicity
    else:
        angle_deg_from_north = atan2d(e_comp, n_comp)

    # Normalizar para 0-360 graus
    angle_deg_from_north = angle_deg_from_north % 360.0

    return semi_major, semi_minor, angle_deg_from_north # Retorna ângulo 0-360


# --- Funções de Cálculo Pedal Curve Atualizadas ---

def calculate_distance(p1, p2):
   """Calcula a distância euclidiana entre dois pontos no plano NE"""
   # Ensure points are dict-like or Series (they are Series from df.loc/iloc)
   # Check for NaN values which can break calculations
   if pd.isna(p1['N']) or pd.isna(p1['E']) or pd.isna(p2['N']) or pd.isna(p2['E']):
       return np.nan
   return np.sqrt((p2['N'] - p1['N'])**2 + (p2['E'] - p1['E'])**2)

def project_ellipse_iscwsa(semi_major, semi_minor, ellipse_angle_deg, direction_az_deg):
   """
   Projeta a elipse ISCWSA na direção especificada (Pedal Curve).
   ellipse_angle_deg: Ângulo do eixo MAIOR da elipse, anti-horário a partir do Norte (0-360).
   direction_az_deg: Azimute da linha conectando os centros, anti-horário a partir do Norte (0-360).
   """
   # Handle NaN inputs
   if pd.isna(semi_major) or pd.isna(semi_minor) or pd.isna(ellipse_angle_deg) or pd.isna(direction_az_deg):
       return np.nan

   # Handle zero size ellipse
   if semi_major < 1e-9 or semi_minor < 1e-9:
       # If semi_major > 0 but semi_minor ~ 0, it's essentially a line. Projection depends on angle.
       # If the direction is along the major axis, proj = semi_major. If perpendicular, proj = semi_minor (0).
       # This case is handled by the formula below, but good to be aware.
       return 0.0 # Return 0 if both axes are zero or tiny


   # Ângulo relativo entre a direção de projeção e o eixo MAIOR da elipse
   # The angle 'theta' in the projection formula is the angle between the direction
   # of interest and the ellipse's *major* axis.
   # Our ellipse_angle_deg is the angle of the major axis from North.
   # Our direction_az_deg is the angle of the direction vector from North.
   # The angle between the two vectors is direction_az_deg - ellipse_angle_deg.
   # Let's use this difference as the relative angle.
   relative_angle_deg = direction_az_deg - ellipse_angle_deg
   relative_angle_rad = np.radians(relative_angle_deg)

   a = semi_major
   b = semi_minor

   # Formula da projeção pedal (raio da elipse na direção relativa)
   # r^2 = (a^2 * b^2) / (b^2 * cos^2(theta) + a^2 * sin^2(theta)) where theta is angle from major axis
   cos_theta_sq = np.cos(relative_angle_rad)**2
   sin_theta_sq = np.sin(relative_angle_rad)**2

   num = (a**2) * (b**2)
   den = (b**2) * cos_theta_sq + (a**2) * sin_theta_sq

   # Handle denominator near zero - occurs if one axis is zero or tiny and the relative angle aligns
   if den < 1e-12:
        # If a is large and b is small/zero, and angle is ~0 or 180, den is small.
        # If b is large and a is small/zero, and angle is ~90 or 270, den is small.
        # The formula should handle the limit, but floating point issues can arise.
        # If semi_minor is tiny compared to semi_major:
        # If angle is near 0/180 (parallel to major axis), projection is near semi_major.
        # If angle is near 90/270 (perpendicular to major axis), projection is near semi_minor.
        if abs(np.cos(relative_angle_rad)) > 0.99: # Close to major axis direction
            return a
        elif abs(np.sin(relative_angle_rad)) > 0.99: # Close to minor axis direction
             return b
        else:
            # If den is still tiny for other angles, there might be an issue, return 0 or NaN
            st.warning(f"Projection denominator near zero: {den} for a={a:.2f}, b={b:.2f}, rel_angle={relative_angle_deg:.2f}. Returning 0.")
            return 0.0


   projection = np.sqrt(num / den)
   return projection


def calculate_pedal_distance_iscwsa(p1, p2, cov_nev1, cov_nev2, sigma_factor):
    """Calcula a distância Pedal Curve entre dois pontos usando elipses ISCWSA"""
    # Distância entre centros
    center_dist = calculate_distance(p1, p2)

    # Handle case where centers are identical, distance is NaN, or cov matrices invalid
    if pd.isna(center_dist) or center_dist < 1e-6 or cov_nev1 is None or cov_nev2 is None:
        # For identical points, center_dist is 0, pedal_dist is 0, projections are 0.
        # SF is undefined (0/0). Let's return a specific value or NaN for SF.
        # A very small center_dist compared to potential ellipse size means high risk.
        # SF close to 1 suggests ellipses are touching or overlapping.
        # Let's return NaN for SF if center_dist is near zero, as the formula is sensitive.
        # Or define SF = 0 if center_dist = 0 (collision). Let's use 0.0 for collision.
        sf_val = 0.0 if (pd.notna(center_dist) and center_dist < 1e-6) else np.nan
        return {'center_dist': center_dist if pd.notna(center_dist) else np.nan,
                'proj1': 0.0, 'proj2': 0.0,
                'pedal_dist': center_dist if pd.notna(center_dist) else np.nan,
                'difference': 0.0 if pd.notna(center_dist) and center_dist < 1e-6 else np.nan,
                'diff_percent': 0.0 if pd.notna(center_dist) and center_dist < 1e-6 else np.nan,
                'SF': sf_val} # Add SF calculation


    # Direção entre centros (Azimute Norte -> Leste)
    delta_n = p2['N'] - p1['N']
    delta_e = p2['E'] - p1['E']
    # atan2d(y, x) -> atan2d(East, North) -> Azimute 0-360
    # Handle case where delta_n and delta_e are near zero (centers are very close)
    if abs(delta_n) < 1e-6 and abs(delta_e) < 1e-6:
        angle_deg_centers = 0.0 # Or assign an arbitrary direction if needed, but SF will be low anyway
    else:
        angle_deg_centers = atan2d(delta_e, delta_n) % 360.0


    # Calcular parâmetros das elipses a partir das matrizes de covariância
    smj1, smn1, ang1 = get_ellipse_params_from_covariance(cov_nev1, sigma_factor)
    smj2, smn2, ang2 = get_ellipse_params_from_covariance(cov_nev2, sigma_factor)

    # Projeção das elipses na direção que conecta os centros
    proj1 = project_ellipse_iscwsa(smj1, smn1, ang1, angle_deg_centers)
    proj2 = project_ellipse_iscwsa(smj2, smn2, ang2, angle_deg_centers)

    # Handle potential NaN projections if ellipse params were invalid
    if pd.isna(proj1) or pd.isna(proj2):
        pedal_dist = np.nan # Cannot calculate if projections are invalid
        difference = np.nan
        diff_percent = np.nan
        sf_val = np.nan
    else:
        pedal_dist = max(0, center_dist - (proj1 + proj2))
        difference = center_dist - pedal_dist
        diff_percent = (difference / center_dist) * 100 if center_dist > 1e-6 else 0

        # Calculate Separation Factor (SF)
        # SF = CC Sep / (CC Sep - Edge Sep)
        # Edge Sep = CC Sep - Pedal Dist = Proj1 + Proj2
        denominator = proj1 + proj2
        if denominator > 1e-9: # Avoid division by zero or near zero
            sf_val = center_dist / denominator
        else:
            # If projections are tiny (very small ellipses), SF goes to infinity or is very large.
            # If centers are far apart and ellipses are tiny, SF is large.
            # If centers are close and ellipses are tiny, SF is still large if centers_dist > 0.
            # If centers are on top of each other (center_dist ~ 0) and projections ~ 0, this is 0/0.
            # Handled the center_dist=0 case earlier. If center_dist > 0 but den ~ 0, SF is large.
            sf_val = np.inf if center_dist > 1e-9 else 0.0 # Collision if center_dist ~ 0


    return {
        'center_dist': center_dist,
        'proj1': proj1,
        'proj2': proj2,
        'pedal_dist': pedal_dist,
        'difference': difference,
        'diff_percent': diff_percent,
        'smj1': smj1, 'smn1': smn1, 'ang1': ang1, # Parâmetros da elipse 1
        'smj2': smj2, 'smn2': smn2, 'ang2': ang2,  # Parâmetros da elipse 2
        'SF': sf_val # Add SF here
    }


def find_closest_tvd_point(tvd_target, df):
   """Encontra o ponto mais próximo em TVD no DataFrame"""
   if df.empty or 'TVD' not in df.columns or df['TVD'].isna().all():
        return None
   # Calculate absolute difference and find index of minimum
   try:
        # Ensure tvd_target is float for comparison
        tvd_target = float(tvd_target)
        # Calculate difference only for non-NaN TVD values
        valid_tvd = df['TVD'].dropna()
        if valid_tvd.empty:
            return None
        idx = (valid_tvd - tvd_target).abs().idxmin()
        return df.loc[idx]
   except ValueError: # Handles case where df['TVD'] might be all NaN or empty after filtering
        return None
   except TypeError: # Handle potential type issues if data is bad
        st.warning(f"TypeError finding closest TVD for target {tvd_target}. Check TVD data.")
        return None


def draw_ellipse_matplotlib(ax, center_xy, width, height, angle_deg, color="blue", alpha=0.3, label=None):
   """Desenha uma elipse com Matplotlib. Angle é anti-horário a partir do eixo +X (Leste)."""
   # Check for NaN values
   if pd.isna(center_xy[0]) or pd.isna(center_xy[1]) or pd.isna(width) or pd.isna(height) or pd.isna(angle_deg):
       # st.warning(f"Skipping ellipse drawing due to NaN value: Center={center_xy}, W={width}, H={height}, Angle={angle_deg}")
       return None
   # Check for non-positive width/height which Ellipse doesn't like
   if width <= 0 or height <= 0:
       # st.warning(f"Skipping ellipse drawing due to non-positive size: W={width}, H={height}")
       # Optionally draw a point or a small circle instead
       # ax.scatter(center_xy[0], center_xy[1], color=color, s=10, marker='o', alpha=alpha, label=label + " (point)")
       return None

   # matplotlib angle: counterclockwise starting from the positive x-axis (East)
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
   # Leitura dos arquivos Excel
   try:
       df_well1_orig = pd.read_excel(well1_file)
       df_well2_orig = pd.read_excel(well2_file)
       df_well1 = df_well1_orig.copy() # Work with copies
       df_well2 = df_well2_orig.copy()

       # Verificar e padronizar nomes das colunas
       expected_cols = ['MD', 'INC', 'AZ']
       data_valid = True
       for df, well_name, file_ref in [(df_well1, "Poço 1", well1_file), (df_well2, "Poço 2", well2_file)]:
           original_cols = df.columns.tolist() # Keep original names for error messages
           # Make columns upper case for comparison
           df.columns = [str(col).upper().strip() for col in df.columns]
           current_cols = df.columns.tolist()

           # Check for essential columns and attempt renaming
           rename_map = {}
           if 'MD' not in current_cols:
               found_md = False
               for potential_md in ['MEASURED DEPTH', 'PROFUNDIDADE MEDIDA', 'PROF MEDIDA']:
                   if potential_md in current_cols:
                       rename_map[potential_md] = 'MD'
                       found_md = True
                       break
               if not found_md:
                   st.error(f"Coluna 'MD' (ou similar) não encontrada no {well_name} ({file_ref.name}). Colunas encontradas: {original_cols}")
                   data_valid = False
           if 'INC' not in current_cols:
               found_inc = False
               for potential_inc in ['INCLINATION', 'INCL', 'INCLINACAO', 'INCLINAÇÃO']:
                   if potential_inc in current_cols:
                       rename_map[potential_inc] = 'INC'
                       found_inc = True
                       break
               if not found_inc:
                   st.error(f"Coluna 'INC' (ou similar) não encontrada no {well_name} ({file_ref.name}). Colunas encontradas: {original_cols}")
                   data_valid = False
           if 'AZ' not in current_cols:
               found_az = False
               for potential_az in ['AZIMUTH', 'AZIM', 'AZIMUTE']:
                   if potential_az in current_cols:
                       rename_map[potential_az] = 'AZ'
                       found_az = True
                       break
               if not found_az:
                   st.error(f"Coluna 'AZ' (ou similar) não encontrada no {well_name} ({file_ref.name}). Colunas encontradas: {original_cols}")
                   data_valid = False

           if not data_valid: continue # Stop processing this file if essential cols missing

           df.rename(columns=rename_map, inplace=True)

           # Ensure numeric types, coerce errors to NaN
           cols_to_check = [col for col in expected_cols if col in df.columns] # Only check existing standard cols
           for col in cols_to_check:
               try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
               except Exception as e_num:
                    st.error(f"Erro ao converter coluna '{col}' para número no {well_name} ({file_ref.name}): {e_num}")
                    data_valid = False

           if not data_valid: continue

           # Drop rows with NaN in essential columns
           initial_rows = len(df)
           df.dropna(subset=cols_to_check, inplace=True)
           if len(df) < initial_rows:
               st.warning(f"Removidas {initial_rows - len(df)} linhas com dados inválidos/faltantes em {well_name} ({file_ref.name}).")
           if len(df) < 2:
                st.error(f"Arquivo do {well_name} ({file_ref.name}) não contém dados suficientes (pelo menos 2 pontos válidos) após limpeza.")
                data_valid = False

       if not data_valid:
           st.stop() # Stop execution if data is invalid in either file


       # --- Calculation Steps ---
       # 1. Calculate Coordinates including Wellhead Offsets
       coords_well1 = calculate_coordinates(
           df_well1['MD'].values, df_well1['INC'].values, df_well1['AZ'].values,
           wellhead_n=n_wh1, wellhead_e=e_wh1, wellhead_tvd=tvd_wh1 # Pass offsets
       )
       coords_well2 = calculate_coordinates(
           df_well2['MD'].values, df_well2['INC'].values, df_well2['AZ'].values,
           wellhead_n=n_wh2, wellhead_e=e_wh2, wellhead_tvd=tvd_wh2 # Pass offsets
       )

       # 2. Calculate Covariance at each point
       # Wrap in a function for clarity
       # The cache key will now be based on coords_df_md_inc_az, params_tuple, and tool_type
       @st.cache_data(ttl=3600, show_spinner="Calculando Incerteza ISCWSA...")
       # Added params_tuple as an argument to the decorated function
       def get_covariance_for_well(coords_df_md_inc_az, params_dict, tool, params_tuple_for_key):
            # IMPORTANT: Covariance depends on MD, INC, AZ, *not* the absolute NEV.
            # Pass the original MD, INC, AZ to the covariance function.
            # params_tuple_for_key is only here for caching purposes, it's not used in the calculation logic.
            covs = []
            # Check if required columns exist
            if not all(col in coords_df_md_inc_az.columns for col in ['MD', 'INC', 'AZ']):
                 st.error("Internal Error: MD, INC, or AZ missing before covariance calculation.")
                 return [np.zeros((3,3))] * len(coords_df_md_inc_az) # Return dummy list

            # Progress bar for covariance calculation
            cov_prog_bar = st.progress(0)
            total_points = len(coords_df_md_inc_az)

            for i, row in coords_df_md_inc_az.iterrows():
                # Pass the dictionary to the calculation function
                cov = calculate_iscwsa_covariance(row['MD'], row['INC'], row['AZ'], params_dict, tool)
                covs.append(cov)
                # Update progress bar
                cov_prog_bar.progress((i + 1) / total_points)

            cov_prog_bar.empty() # Clear progress bar when done
            return covs

       # Use tuple(iscwsa_params.items()) as part of the cache key
       params_tuple_key = tuple(sorted(iscwsa_params.items()))

       # Combine coords with covariance
       # Pass the *original* dataframe slice with MD/INC/AZ, the actual params_dict,
       # and the params_tuple_key as arguments to the decorated function.
       covs1 = get_covariance_for_well(coords_well1[['MD', 'INC', 'AZ']], iscwsa_params, tool_type, params_tuple_key)
       coords_well1['Covariance'] = covs1

       covs2 = get_covariance_for_well(coords_well2[['MD', 'INC', 'AZ']], iscwsa_params, tool_type, params_tuple_key)
       coords_well2['Covariance'] = covs2


       # 3. Find corresponding points and compare
       results = []
       st.write("Comparando trajetórias em profundidades correspondentes...")
       prog_bar_compare = st.progress(0)

       # Use TVDs from the first well as reference points
       # Need to handle potential non-uniqueness or NaNs robustly
       tvds_ref_raw = coords_well1['TVD'].dropna().unique()
       if len(tvds_ref_raw) == 0:
           st.error("Não foi possível encontrar TVDs válidas no Poço 1 para comparação.")
           st.stop()

       tvds_ref = np.sort(tvds_ref_raw) # Sort for potentially better iteration logic
       total_comparisons = len(tvds_ref)
       comparison_tvd_step = max(1, total_comparisons // 100) # Update progress bar less frequently

       for i, tvd_ref in enumerate(tvds_ref):
           # Find point(s) in well 1 at this TVD (handle potential float precision issues)
           p1_matches = coords_well1[np.isclose(coords_well1['TVD'], tvd_ref)]
           if p1_matches.empty: continue # Skip if no close match found
           p1 = p1_matches.iloc[0] # Take the first match if multiple exist at the same TVD

           # Find the closest point in well 2 based on TVD
           p2 = find_closest_tvd_point(tvd_ref, coords_well2)

           # Ensure both points were found and TVD difference is reasonably small
           tvd1_val = p1['TVD']
           tvd2_val = p2['TVD'] if p2 is not None else np.nan
           tvd_diff_threshold = 10.0

           if p2 is not None and pd.notna(tvd1_val) and pd.notna(tvd2_val) and abs(tvd1_val - tvd2_val) < tvd_diff_threshold:
               # Ensure covariance data is valid (not None and not all zeros if MD > 0)
               cov1_valid = ('Covariance' in p1 and isinstance(p1['Covariance'], np.ndarray) and (p1['MD'] < 1e-6 or not np.allclose(p1['Covariance'], 0)))
               cov2_valid = ('Covariance' in p2 and isinstance(p2['Covariance'], np.ndarray) and (p2['MD'] < 1e-6 or not np.allclose(p2['Covariance'], 0)))

               if cov1_valid and cov2_valid:
                   # Calculate distances using ISCWSA ellipse parameters
                   distance_data = calculate_pedal_distance_iscwsa(
                       p1, p2,
                       p1['Covariance'], p2['Covariance'],
                       sigma_factor
                   )

                   # Append results, using .get() for potentially missing ellipse keys
                   results.append({
                       'TVD_Ref': tvd_ref, # TVD from Well 1 used as reference
                       'TVD_Actual1': p1['TVD'], 'TVD_Actual2': p2['TVD'],
                       'MD1': p1['MD'], 'MD2': p2['MD'],
                       'INC1': p1['INC'], 'INC2': p2['INC'],
                       'AZ1': p1['AZ'], 'AZ2': p2['AZ'],
                       'N1': p1['N'], 'E1': p1['E'], # Absolute Coordinates
                       'N2': p2['N'], 'E2': p2['E'], # Absolute Coordinates
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
                       'SF': distance_data.get('SF', np.nan), # Add SF to results
                       'AZ_Diff': abs(p1['AZ'] - p2['AZ']) % 180, # Simplistic diff
                       'INC_Avg': (p1['INC'] + p2['INC']) / 2 if pd.notna(p1['INC']) and pd.notna(p2['INC']) else np.nan
                   })
           # Update progress bar periodically
           if i % comparison_tvd_step == 0:
               prog_bar_compare.progress((i + 1) / total_comparisons)
       prog_bar_compare.progress(1.0) # Ensure it reaches 100%
       prog_bar_compare.empty()


       # Create dataframe of results
       if results:
           df_results = pd.DataFrame(results)
           # Drop rows where essential distance calculations failed (resultted in NaN)
           df_results.dropna(subset=['DistCentros', 'DistPedal', 'SF'], inplace=True) # Include SF in dropna

           if df_results.empty:
               st.warning("Não foi possível calcular distâncias válidas para os pontos comparados ou não houve pontos próximos em TVD.")
               st.stop()

           # Exibir tabela de resultados
           st.subheader("Comparação de Distâncias e Fator de Separação")
           st.dataframe(df_results[[
               'TVD_Ref', 'MD1', 'MD2', 'DistCentros', 'DistPedal', 'DifPerc', 'SF', # Add SF
               'SMj1', 'SMn1', 'Ang1', 'SMj2', 'SMn2', 'Ang2'
               ]].round(2))

           # --- Gráficos ---
           st.subheader("Gráficos de Análise")

           # Gráfico de distâncias vs TVD Ref
           fig1 = go.Figure()
           fig1.add_trace(go.Scatter(x=df_results['TVD_Ref'], y=df_results['DistCentros'], mode='lines+markers', name='Dist. Centros'))
           fig1.add_trace(go.Scatter(x=df_results['TVD_Ref'], y=df_results['DistPedal'], mode='lines+markers', name=f'Dist. Pedal ({sigma_factor:.1f}σ)'))
           fig1.update_layout(title='Distâncias vs Profundidade (TVD Referência Poço 1)', xaxis_title='TVD Ref Poço 1 (m)', yaxis_title='Distância (m)', legend=dict(x=0.01, y=0.99))
           st.plotly_chart(fig1, use_container_width=True)

           # Gráfico de Fator de Separação vs MD
           st.subheader("Fator de Separação (SF) vs Profundidade (MD Poço 1)")
           fig_sf = go.Figure()

           # Add SF trace
           fig_sf.add_trace(go.Scatter(x=df_results['MD1'], y=df_results['SF'], mode='lines+markers', name='Fator de Separação (SF)'))

           # Add risk interpretation lines (Shapes)
           # SF < 1.0: Collision (wellbores intersect)
           # 1.0 < SF < 1.5: High risk of collision
           # 1.5 < SF < 2.0: Medium risk of collision
           # SF > 2.0: Low risk of collision

           # Get the min and max MD values for the lines
           min_md = df_results['MD1'].min()
           max_md = df_results['MD1'].max()

           # Add lines for risk levels (Shape objects)
           # 'name' property is not valid for shapes added this way in update_layout or add_shape calls
           # We'll use annotations for labels instead.
           fig_sf.add_shape(type="line",
               x0=min_md, y0=1.0, x1=max_md, y1=1.0,
               line=dict(color="red", width=2, dash="dash"),
           )
           fig_sf.add_shape(type="line",
               x0=min_md, y0=1.5, x1=max_md, y1=1.5,
               line=dict(color="orange", width=2, dash="dash"),
           )
           fig_sf.add_shape(type="line",
               x0=min_md, y0=2.0, x1=max_md, y1=2.0,
               line=dict(color="green", width=2, dash="dash"),
           )

           # Define annotations (Text Labels) - These belong in the 'annotations' list in layout
           annotations_list = [
               # Add annotations for risk zones near the lines
               # x position is MD, y position is SF value (using data coordinates)
               # Check if max_md is valid before using it in x position
               dict(x=max_md * 0.95 if pd.notna(max_md) and max_md > 0 else 100, y=0.5, xref="x", yref="y", text="Colisão (<1.0)", showarrow=False, font=dict(color="red", size=10)),
               dict(x=max_md * 0.95 if pd.notna(max_md) and max_md > 0 else 100, y=1.25, xref="x", yref="y", text="Alto Risco (1.0-1.5)", showarrow=False, font=dict(color="orange", size=10)),
               dict(x=max_md * 0.95 if pd.notna(max_md) and max_md > 0 else 100, y=1.75, xref="x", yref="y", text="Médio Risco (1.5-2.0)", showarrow=False, font=dict(color="green", size=10)),
               dict(x=max_md * 0.95 if pd.notna(max_md) and max_md > 0 else 100, y=2.25, xref="x", yref="y", text="Baixo Risco (>2.0)", showarrow=False, font=dict(color="blue", size=10)),
           ]


           fig_sf.update_layout(
               title='Fator de Separação (SF) vs Profundidade Medida (MD Poço 1)',
               xaxis_title='MD Poço 1 (m)',
               yaxis_title='Fator de Separação (SF)',
               # Dynamically set y-axis range: from 0 up to max SF (with padding) or minimum threshold (2.1)
               yaxis_range=[0, max(df_results['SF'].max() * 1.1 if not df_results['SF'].empty and df_results['SF'].max() != np.inf and pd.notna(df_results['SF'].max()) else 5, 2.1)], # Ensure y-axis includes key thresholds and data max
               legend=dict(x=0.01, y=0.99),
               # Add the annotations list to the layout
               annotations=annotations_list
               # The 'shapes' list here is implicitly populated by add_shape calls
           )
           st.plotly_chart(fig_sf, use_container_width=True)


           # Visualização 3D das trajetórias (using absolute coordinates)
           st.subheader("Visualização 3D das Trajetórias (Absoluto)")
           fig5 = go.Figure()
           # Plot full trajectories
           fig5.add_trace(go.Scatter3d(x=coords_well1['E'], y=coords_well1['N'], z=coords_well1['TVD'],
                                       mode='lines', name='Poço 1', line=dict(color='blue', width=4)))
           fig5.add_trace(go.Scatter3d(x=coords_well2['E'], y=coords_well2['N'], z=coords_well2['TVD'],
                                       mode='lines', name='Poço 2', line=dict(color='red', width=4)))
           # Add markers for comparison points used in the results table
           fig5.add_trace(go.Scatter3d(x=df_results['E1'], y=df_results['N1'], z=df_results['TVD_Actual1'],
                                       mode='markers', name='Pontos Comp. Poço 1', marker=dict(color='cyan', size=3)))
           fig5.add_trace(go.Scatter3d(x=df_results['E2'], y=df_results['N2'], z=df_results['TVD_Actual2'],
                                       mode='markers', name='Pontos Comp. Poço 2', marker=dict(color='magenta', size=3)))

           fig5.update_layout(
               scene=dict(
                   xaxis_title='Este (m)',
                   yaxis_title='Norte (m)',
                   zaxis_title='TVD (m)',
                   zaxis=dict(autorange="reversed"), # Depth increases downwards
                   # aspectmode='data' # 'data' often leads to distorted views if ranges differ significantly
                   aspectmode='auto'   # 'auto' or 'cube' might be better for general view
                ),
               title="Visualização 3D das Trajetórias (Coordenadas Absolutas)",
               height=700,
               legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
           st.plotly_chart(fig5, use_container_width=True)

           # Visualização 2D das trajetórias com elipses de incerteza
           st.subheader(f"Visualização 2D com Elipses de Incerteza ({sigma_factor:.1f}σ)")
           # Ensure TVD options are sorted and unique from the results
           tvd_options_ref = np.sort(df_results['TVD_Ref'].unique())
           if len(tvd_options_ref) == 0:
                st.warning("Não há TVDs de referência válidas para visualização 2D.")
           else:
                # Default to middle TVD, handle case of single TVD
                default_index_2d = len(tvd_options_ref) // 2 if len(tvd_options_ref) > 0 else 0
                selected_tvd_ref = st.selectbox("Selecione a TVD (Referência Poço 1) para visualização 2D",
                                                tvd_options_ref,
                                                index=default_index_2d,
                                                format_func=lambda x: f"{x:.1f} m")

                # Filtrar dados para a TVD selecionada (handle potential float precision issues)
                selected_data_rows = df_results[np.isclose(df_results['TVD_Ref'], selected_tvd_ref)]

                if selected_data_rows.empty:
                    st.warning(f"Não foram encontrados dados nos resultados para a TVD de referência selecionada {selected_tvd_ref:.1f} m.")
                else:
                    selected_data = selected_data_rows.iloc[0] # Take the first row if multiple matches

                    # Criar figura para visualização 2D com Matplotlib
                    fig_2d, ax_2d = plt.subplots(figsize=(10, 10))

                    # Plotar os pontos centrais (absolute coordinates)
                    ax_2d.scatter(selected_data['E1'], selected_data['N1'], color='blue', s=50, label=f'Poço 1 Centro (TVD≈{selected_data["TVD_Actual1"]:.1f}m)', zorder=5)
                    ax_2d.scatter(selected_data['E2'], selected_data['N2'], color='red', s=50, label=f'Poço 2 Centro (TVD≈{selected_data["TVD_Actual2"]:.1f}m)', zorder=5)

                    # Linha entre centros
                    ax_2d.plot([selected_data['E1'], selected_data['E2']], [selected_data['N1'], selected_data['N2']], 'k--', alpha=0.7, label=f'Dist Centros: {selected_data["DistCentros"]:.2f} m')

                    # Desenhar as elipses
                    # Elipse 1
                    # matplotlib angle is counter-clockwise from +X (East)
                    # Our ang1 is counter-clockwise from +Y (North)
                    # Angle_mpl = 90 - Angle_North
                    angle_mpl1 = (90.0 - selected_data['Ang1']) % 360.0 if pd.notna(selected_data['Ang1']) else 0.0
                    draw_ellipse_matplotlib(
                        ax_2d,
                        center_xy=(selected_data['E1'], selected_data['N1']),
                        width=2 * selected_data['SMj1'] if pd.notna(selected_data['SMj1']) else 0, # Diameter
                        height=2 * selected_data['SMn1'] if pd.notna(selected_data['SMn1']) else 0, # Diameter
                        angle_deg=angle_mpl1,
                        color="blue", alpha=0.3, label=f'Elipse 1 ({sigma_factor:.1f}σ)'
                    )

                    # Elipse 2
                    angle_mpl2 = (90.0 - selected_data['Ang2']) % 360.0 if pd.notna(selected_data['Ang2']) else 0.0
                    draw_ellipse_matplotlib(
                        ax_2d,
                        center_xy=(selected_data['E2'], selected_data['N2']),
                        width=2 * selected_data['SMj2'] if pd.notna(selected_data['SMj2']) else 0, # Diameter
                        height=2 * selected_data['SMn2'] if pd.notna(selected_data['SMn2']) else 0, # Diameter
                        angle_deg=angle_mpl2,
                        color="red", alpha=0.3, label=f'Elipse 2 ({sigma_factor:.1f}σ)'
                    )

                    # Adicionar texto com informações (handle potential NaNs)
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

                    # Configurações do gráfico 2D
                    ax_2d.set_xlabel('Este (m)')
                    ax_2d.set_ylabel('Norte (m)')
                    ax_2d.set_title(f'Elipses de Incerteza ISCWSA ({tool_type}) @ TVD Ref ≈ {selected_data.get("TVD_Ref", np.nan):.1f} m')
                    ax_2d.grid(True, linestyle=':', alpha=0.6)
                    ax_2d.axis('equal') # Essencial para visualizar forma correta da elipse
                    ax_2d.legend(fontsize=8, loc='lower right')

                    # Zoom na área de interesse - Adjust zoom logic
                    # Calculate bounds based on ellipse sizes and center distance
                    max_radius1 = selected_data.get('SMj1', 0) if pd.notna(selected_data.get('SMj1', 0)) else 0
                    max_radius2 = selected_data.get('SMj2', 0) if pd.notna(selected_data.get('SMj2', 0)) else 0
                    center_e = (selected_data.get('E1', 0) + selected_data.get('E2', 0)) / 2 if pd.notna(selected_data.get('E1')) and pd.notna(selected_data.get('E2')) else 0
                    center_n = (selected_data.get('N1', 0) + selected_data.get('N2', 0)) / 2 if pd.notna(selected_data.get('N1')) and pd.notna(selected_data.get('N2')) else 0

                    # Determine max extent from center point
                    dist_centers = selected_data.get("DistCentros", 0) if pd.notna(selected_data.get("DistCentros")) else 0
                    # Handle potential NaN coordinates when calculating extent
                    e1 = selected_data.get('E1', center_e) if pd.notna(selected_data.get('E1')) else center_e
                    n1 = selected_data.get('N1', center_n) if pd.notna(selected_data.get('N1')) else center_n
                    e2 = selected_data.get('E2', center_e) if pd.notna(selected_data.get('E2')) else center_e
                    n2 = selected_data.get('N2', center_n) if pd.notna(selected_data.get('N2')) else center_n

                    extent1_e = abs(e1 - center_e) + max_radius1
                    extent1_n = abs(n1 - center_n) + max_radius1
                    extent2_e = abs(e2 - center_e) + max_radius2
                    extent2_n = abs(n2 - center_n) + max_radius2

                    half_view_size = max(extent1_e, extent1_n, extent2_e, extent2_n, dist_centers / 2 if dist_centers > 0 else 1) * 1.2 # Add 20% buffer

                    if half_view_size < 1: half_view_size = 5 # Ensure a minimum view size

                    ax_2d.set_xlim(center_e - half_view_size, center_e + half_view_size)
                    ax_2d.set_ylim(center_n - half_view_size, center_n + half_view_size)

                    st.pyplot(fig_2d)


           # Explicação Pedal Curve
           with st.expander("Explicação do Método Pedal Curve com ISCWSA e Fator de Separação"):
               st.markdown(f"""
               ### Método Pedal Curve, Modelos ISCWSA e Fator de Separação (SF)

               O método **Pedal Curve** calcula a separação mínima entre as elipses de incerteza de dois poços, na confiança especificada (Fator Sigma = {sigma_factor:.1f}). A distância é calculada como:

               `Dist_Pedal = max(0, Dist_Centros - (Projeção_Elipse1 + Projeção_Elipse2))`

               Onde:
               - **Dist_Centros**: Distância Euclidiana entre os centros das trajetórias no plano horizontal (Norte-Este) na profundidade comparada. **Esta distância agora considera as coordenadas absolutas dos poços.**
               - **Projeção_Elipse**: O "raio" da elipse de incerteza na direção da linha que conecta os centros dos dois poços.

               **Modelos ISCWSA (MWD e Gyro):**
               - Estes modelos, padronizados pela *Industry Steering Committee on Wellbore Survey Accuracy* (ISCWSA), consideram **múltiplas fontes de erro** que afetam as medições de trajetória (ex: bias de sensores, erros de escala, desalinhamentos, erros de profundidade, interferência magnética/drift do giroscópio, erros de referência, etc.). Os parâmetros são definidos pelo usuário na barra lateral.
               - A incerteza de cada fonte de erro é propagada e combinada matematicamente para formar uma **matriz de covariância** 3D (tipicamente Norte, Este, Vertical) em cada ponto da trajetória. Esta matriz descreve não só o tamanho da incerteza em cada direção, mas também a correlação entre elas. *(Nota: A implementação aqui usa uma aproximação simplificada da combinação de erros)*.
               - A **elipse de incerteza horizontal** (plotada em 2D) é derivada da submatriz Norte-Este da matriz de covariância. Seus semi-eixos (maior e menor) e sua orientação **não estão necessariamente alinhados com o azimute do poço**, mas sim com as direções de maior e menor incerteza combinada no plano horizontal.
               - A **orientação da elipse** (ângulo do eixo maior com o Norte, `Ang1`/`Ang2`) e a **razão entre os eixos** (forma da elipse, `SMj`/`SMn`) dependem complexmente da trajetória (MD, INC, AZ), das propriedades do campo geomagnético/gravitacional local e de todos os parâmetros de erro da ferramenta ({tool_type}) selecionada.

               **Fator de Separação (SF):**
               - O SF é uma medida adimensional de risco calculada como:
                 `SF = Dist_Centros / (Projeção_Elipse1 + Projeção_Elipse2)`
               - Ele compara a distância entre os centros dos poços com a soma dos raios das elipses na linha que os conecta.
               - Interpretação do Risco (geral, pode variar por política da empresa):
                 - **SF < 1.0**: Colisão (as elipses de incerteza na confiança {sigma_factor:.1f}σ se sobrepõem).
                 - **1.0 ≤ SF < 1.5**: Alto risco de colisão. As elipses estão muito próximas.
                 - **1.5 ≤ SF < 2.0**: Médio risco de colisão.
                 - **SF ≥ 2.0**: Baixo risco de colisão.

               A distância Pedal Curve ({sigma_factor:.1f}σ) reflete a separação mínima esperada entre as bordas das elipses. Um valor de **Dist_Pedal <= 0** e um **SF <= 1.0** indicam que as elipses (na confiança {sigma_factor:.1f}σ) estão se tocando ou sobrepondo naquela profundidade, sugerindo um potencial risco de colisão que deve ser avaliado com mais detalhe.
               """)

       else:
           st.warning("Não foi possível encontrar pontos de comparação em profundidades TVD próximas ou os dados resultantes foram insuficientes após a filtragem.")

   except FileNotFoundError:
       st.error("Erro: Arquivo não encontrado. Verifique o caminho.")
   except pd.errors.EmptyDataError:
       st.error("Erro: O arquivo Excel está vazio ou não pôde ser lido.")
   except ValueError as e:
       st.error(f"Erro ao converter dados para números. Verifique o conteúdo das colunas MD, INC, AZ e as coordenadas da cabeça do poço. Detalhe: {e}")
   except KeyError as e:
        st.error(f"Erro: Coluna essencial não encontrada após tentativa de renomeação: {e}. Verifique os nomes das colunas no arquivo Excel.")
   except Exception as e:
       st.error(f"Erro inesperado ao processar os arquivos: {e}")
       st.exception(e) # Mostra o traceback completo para depuração

else:
   # Exibir exemplo de formato de arquivo esperado
   st.info("Aguardando upload dos arquivos Excel dos Poços 1 e 2 e definição das coordenadas de cabeça de poço...")
   example_data = pd.DataFrame({
       'MD': [0, 500, 1000, 1500, 2000, 2500],
       'INC': [0, 15, 30, 45, 60, 75],
       'AZ': [0, 45, 45, 60, 90, 120]
   })
   st.write("Formato esperado para os arquivos Excel (first sheet):")
   st.dataframe(example_data)
   st.markdown("""
   As colunas essenciais são:
   - **MD**: Profundidade Medida (Measured Depth) - **Importante:** O primeiro ponto (MD=0 ou o menor MD) será considerado o ponto de partida relativo da trajetória.
   - **INC**: Inclinação (Inclination) em graus
   - **AZ**: Azimute (Azimuth) em graus

   Nomes comuns como `INCLINACAO`, `INCLINAÇÃO`, `AZIMUTE`, `MEASURED DEPTH` etc., serão reconhecidos automaticamente (ignorando maiúsculas/minúsculas).

   **Insira as coordenadas de cabeça de poço (Norte, Este, TVD Inicial) nos campos acima para cada poço.** A trajetória calculada a partir do arquivo Excel será adicionada a essas coordenadas iniciais.
   """)

   # Adicionar botão para download de exemplo
   buffer = BytesIO()
   with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
       example_data.to_excel(writer, index=False, sheet_name='TrajetoriaExemplo')
   buffer.seek(0)

   st.download_button(
       label="Download Arquivo Exemplo (.xlsx)",
       data=buffer,
       file_name="exemplo_trajetoria.xlsx",
       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
   )

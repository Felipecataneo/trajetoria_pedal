import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from mpl_toolkits.mplot3d import Axes3D
import math

# Set page config
st.set_page_config(page_title="Separation Factor Calculator", layout="wide")

# Add title and description
st.title("Separation Factor Calculator")
st.markdown("""
This application calculates the Separation Factor between two well trajectories. Upload the trajectory data (MD, Inclination, Azimuth)
for two wells and provide the wellhead coordinates to analyze the separation between the wellpaths.
           
**How it works:**
1. Enter wellhead coordinates for both wells
2. Upload trajectory files for both wells (Excel files with MD, Inclination, and Azimuth columns)
3. The app will calculate and display the Separation Factor along the wellpaths
""")

# Create columns for wellhead coordinates
col1, col2 = st.columns(2)

# Well 1 coordinates input
with col1:
    st.subheader("Well 1 - Wellhead Coordinates")
    north1 = st.number_input("North (m)", value=0.0, key="north1")
    east1 = st.number_input("East (m)", value=0.0, key="east1")
    tvd1 = st.number_input("TVD (m)", value=0.0, key="tvd1")

# Well 2 coordinates input
with col2:
    st.subheader("Well 2 - Wellhead Coordinates")
    north2 = st.number_input("North (m)", value=100.0, key="north2")
    east2 = st.number_input("East (m)", value=100.0, key="east2")
    tvd2 = st.number_input("TVD (m)", value=0.0, key="tvd2")

# Create columns for file uploads
col1, col2 = st.columns(2)

# Well 1 trajectory file upload
with col1:
    st.subheader("Well 1 - Trajectory Data")
    traj_file1 = st.file_uploader("Upload Excel file with MD, Inc, Azimuth", type=["xlsx", "xls"], key="traj1")
   
    # Preview column mapping
    if traj_file1:
        try:
            df1 = pd.read_excel(traj_file1)
            st.write("Preview of Well 1 data:")
            st.dataframe(df1.head())
           
            # Column mapping for Well 1
            st.subheader("Map columns for Well 1")
            md_col1 = st.selectbox("Select MD column", df1.columns, key="md_col1")
            inc_col1 = st.selectbox("Select Inclination column", df1.columns, key="inc_col1")
            az_col1 = st.selectbox("Select Azimuth column", df1.columns, key="az_col1")
        except Exception as e:
            st.error(f"Error reading Well 1 file: {e}")
            df1 = None

# Well 2 trajectory file upload
with col2:
    st.subheader("Well 2 - Trajectory Data")
    traj_file2 = st.file_uploader("Upload Excel file with MD, Inc, Azimuth", type=["xlsx", "xls"], key="traj2")
   
    # Preview column mapping
    if traj_file2:
        try:
            df2 = pd.read_excel(traj_file2)
            st.write("Preview of Well 2 data:")
            st.dataframe(df2.head())
           
            # Column mapping for Well 2
            st.subheader("Map columns for Well 2")
            md_col2 = st.selectbox("Select MD column", df2.columns, key="md_col2")
            inc_col2 = st.selectbox("Select Inclination column", df2.columns, key="inc_col2")
            az_col2 = st.selectbox("Select Azimuth column", df2.columns, key="az_col2")
        except Exception as e:
            st.error(f"Error reading Well 2 file: {e}")
            df2 = None

# Function to calculate minimum curvature trajectory
def min_curve_calc(md, inc, az, north_surface, east_surface, tvd_surface):
    """
    Calculate 3D wellpath coordinates using minimum curvature method
   
    Args:
        md: Measured depth array
        inc: Inclination array (degrees)
        az: Azimuth array (degrees)
        north_surface: Surface north coordinate
        east_surface: Surface east coordinate
        tvd_surface: Surface TVD coordinate
       
    Returns:
        Tuple of North, East, TVD arrays
    """
    # Convert to radians
    inc_rad = np.radians(inc)
    az_rad = np.radians(az)
   
    # Initialize arrays
    north = np.zeros_like(md)
    east = np.zeros_like(md)
    tvd = np.zeros_like(md)
   
    # Set surface coordinates
    north[0] = north_surface
    east[0] = east_surface
    tvd[0] = tvd_surface
   
    # Calculate wellpath
    for i in range(1, len(md)):
        dmd = md[i] - md[i-1]
       
        if abs(inc_rad[i] - inc_rad[i-1]) < 0.001 and abs(az_rad[i] - az_rad[i-1]) < 0.001:
            # Straight section
            north[i] = north[i-1] + dmd * np.sin(inc_rad[i]) * np.cos(az_rad[i])
            east[i] = east[i-1] + dmd * np.sin(inc_rad[i]) * np.sin(az_rad[i])
            tvd[i] = tvd[i-1] + dmd * np.cos(inc_rad[i])
        else:
            # Curved section
            inc1, inc2 = inc_rad[i-1], inc_rad[i]
            az1, az2 = az_rad[i-1], az_rad[i]
           
            # Calculate dogleg angle
            dogleg = np.arccos(np.cos(inc2 - inc1) - np.sin(inc1) * np.sin(inc2) * (1 - np.cos(az2 - az1)))
           
            # Calculate ratio factor
            if dogleg < 0.001:
                rf = 1.0
            else:
                rf = 2 * np.tan(dogleg/2) / dogleg
               
            # Calculate increments
            dnorth = dmd / 2 * (np.sin(inc1) * np.cos(az1) + np.sin(inc2) * np.cos(az2)) * rf
            deast = dmd / 2 * (np.sin(inc1) * np.sin(az1) + np.sin(inc2) * np.sin(az2)) * rf
            dtvd = dmd / 2 * (np.cos(inc1) + np.cos(inc2)) * rf
           
            # Update coordinates
            north[i] = north[i-1] + dnorth
            east[i] = east[i-1] + deast
            tvd[i] = tvd[i-1] + dtvd
   
    return north, east, tvd

# Function to calculate pedal curve (elliptical conic) and separation factor
def calculate_separation_factor(md1, north1, east1, tvd1, md2, north2, east2, tvd2):
    """
    Calculate separation factor between two wellpaths
   
    Args:
        md1, north1, east1, tvd1: Arrays for well 1
        md2, north2, east2, tvd2: Arrays for well 2
       
    Returns:
        DataFrame with separation factor calculation
    """
    # Initialize results
    results = []
   
    # Set range for calculation - use shortest common MD range
    max_md1 = max(md1)
    max_md2 = max(md2)
   
    # Interpolate to common MD steps
    step = 10  # 10m steps
    max_md = min(max_md1, max_md2)
    md_common = np.arange(100, max_md, step)  # Start from 100m to avoid surface section
   
    # Function to interpolate coordinates
    def interp_coords(md, md_vals, north_vals, east_vals, tvd_vals):
        north_interp = np.interp(md, md_vals, north_vals)
        east_interp = np.interp(md, md_vals, east_vals)
        tvd_interp = np.interp(md, md_vals, tvd_vals)
        return north_interp, east_interp, tvd_interp
   
    # Calculate separation factor for each MD step
    for md in md_common:
        # Interpolate coordinates for well 1
        n1, e1, t1 = interp_coords(md, md1, north1, east1, tvd1)
       
        # Interpolate coordinates for well 2
        n2, e2, t2 = interp_coords(md, md2, north2, east2, tvd2)
       
        # Calculate center-to-center distance (CC Sep)
        cc_sep = np.sqrt((n2 - n1)**2 + (e2 - e1)**2)
       
        # Calculate vectors for well 1
        if md < md1[-1] - step:
            n1_next, e1_next, t1_next = interp_coords(md + step, md1, north1, east1, tvd1)
            vec1 = np.array([n1_next - n1, e1_next - e1, t1_next - t1])
            vec1 = vec1 / np.linalg.norm(vec1)
        else:
            # Use previous vector if we're at the end
            n1_prev, e1_prev, t1_prev = interp_coords(md - step, md1, north1, east1, tvd1)
            vec1 = np.array([n1 - n1_prev, e1 - e1_prev, t1 - t1_prev])
            vec1 = vec1 / np.linalg.norm(vec1)
       
        # Calculate vectors for well 2
        if md < md2[-1] - step:
            n2_next, e2_next, t2_next = interp_coords(md + step, md2, north2, east2, tvd2)
            vec2 = np.array([n2_next - n2, e2_next - e2, t2_next - t2])
            vec2 = vec2 / np.linalg.norm(vec2)
        else:
            # Use previous vector if we're at the end
            n2_prev, e2_prev, t2_prev = interp_coords(md - step, md2, north2, east2, tvd2)
            vec2 = np.array([n2 - n2_prev, e2 - e2_prev, t2 - t2_prev])
            vec2 = vec2 / np.linalg.norm(vec2)
       
        # Vector between centers
        center_vec = np.array([n2 - n1, e2 - e1, t2 - t1])
        center_vec_norm = center_vec / np.linalg.norm(center_vec)
       
        # Calculate edge separations (perpendicular distances)
        # Plane normal to trajectory vectors
        normal1 = np.cross(vec1, np.cross(center_vec_norm, vec1))
        normal1 = normal1 / np.linalg.norm(normal1)
       
        normal2 = np.cross(vec2, np.cross(-center_vec_norm, vec2))
        normal2 = normal2 / np.linalg.norm(normal2)
       
        # Perpendicular distances (Edge Sep)
        edge_sep = cc_sep
       
        # Calculate separation factor
        if edge_sep > cc_sep:
            sf = 0  # This is a theoretical impossibility
        else:
            sf = cc_sep / (cc_sep - edge_sep) if edge_sep < cc_sep else float('inf')
       
        # Calculate common measured depth at TVD point (average)
        tvd_point = (t1 + t2) / 2
       
        # Append results
        results.append({
            'MD': md,
            'TVD': tvd_point,
            'North1': n1,
            'East1': e1,
            'TVD1': t1,
            'North2': n2,
            'East2': e2,
            'TVD2': t2,
            'CC_Sep': cc_sep,
            'Edge_Sep': edge_sep,
            'SF': sf
        })
   
    return pd.DataFrame(results)

# Calculate button
if st.button("Calculate Separation Factor"):
    if 'df1' in locals() and 'df2' in locals() and df1 is not None and df2 is not None:
        try:
            # Extract data from dataframes
            md1 = df1[md_col1].values
            inc1 = df1[inc_col1].values
            az1 = df1[az_col1].values
           
            md2 = df2[md_col2].values
            inc2 = df2[inc_col2].values
            az2 = df2[az_col2].values
           
            # Calculate 3D trajectories
            n1, e1, t1 = min_curve_calc(md1, inc1, az1, north1, east1, tvd1)
            n2, e2, t2 = min_curve_calc(md2, inc2, az2, north2, east2, tvd2)
           
            # Calculate separation factor
            sf_results = calculate_separation_factor(md1, n1, e1, t1, md2, n2, e2, t2)
           
            # Display results
            st.subheader("Separation Factor Results")
            st.dataframe(sf_results)
           
            # Plot separation factor vs depth
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(sf_results['TVD'], sf_results['SF'], 'r-', linewidth=2)
           
            # Add risk level lines
            ax.axhline(y=1.0, color='y', linestyle='-', label='Collision')
            ax.axhline(y=1.5, color='r', linestyle='-', label='High Risk')
            ax.axhline(y=2.0, color='g', linestyle='-', label='Medium Risk')
           
            # Add labels and title
            ax.set_xlabel('Measured Depth [m]')
            ax.set_ylabel('Separation Factor')
            ax.set_title('Separation Factor vs Depth')
            ax.grid(True)
            ax.legend()
           
            # Set y-axis limits
            ax.set_ylim(0, 10)
           
            # Display plot
            st.pyplot(fig)
           
            # 3D Wellpath visualization
            fig3d = plt.figure(figsize=(10, 8))
            ax3d = fig3d.add_subplot(111, projection='3d')
           
            # Plot wellpaths
            ax3d.plot(e1, n1, t1, label='Well 1')
            ax3d.plot(e2, n2, t2, label='Well 2')
           
            # Set equal aspect ratio
            max_range = np.array([
                max(e1.max(), e2.max()) - min(e1.min(), e2.min()),
                max(n1.max(), n2.max()) - min(n1.min(), n2.min()),
                max(t1.max(), t2.max()) - min(t1.min(), t2.min())
            ]).max() / 2.0
           
            mid_x = (max(e1.max(), e2.max()) + min(e1.min(), e2.min())) * 0.5
            mid_y = (max(n1.max(), n2.max()) + min(n1.min(), n2.min())) * 0.5
            mid_z = (max(t1.max(), t2.max()) + min(t1.min(), t2.min())) * 0.5
           
            ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
            ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
            ax3d.set_zlim(mid_z - max_range, mid_z + max_range)
           
            # Invert z-axis for TVD
            ax3d.invert_zaxis()
           
            # Add labels
            ax3d.set_xlabel('East [m]')
            ax3d.set_ylabel('North [m]')
            ax3d.set_zlabel('TVD [m]')
            ax3d.legend()
           
            st.pyplot(fig3d)
           
            # Save button for results
            csv = sf_results.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="separation_factor_results.csv",
                mime="text/csv",
            )
           
        except Exception as e:
            st.error(f"Error in calculation: {e}")
            st.error("Please check your input data and column mappings.")
    else:
        st.error("Please upload trajectory files for both wells.")

# Add explanation of Separation Factor
with st.expander("About Separation Factor"):
    st.markdown("""
    ### Separation Factor Calculation
   
    The Separation Factor (SF) is a measure used in the oil and gas industry to assess the risk of collision between two wellbores. It is calculated as:
   
    SF = CC Sep / (CC Sep - Edge Sep)
   
    Where:
    - **CC Sep** is the center-to-center separation between two wellbores
    - **Edge Sep** is the edge-to-edge separation (perpendicular distance)
   
    ### Risk Interpretation:
    - SF < 1.0: Collision (wellbores intersect)
    - 1.0 < SF < 1.5: High risk of collision
    - 1.5 < SF < 2.0: Medium risk of collision
    - SF > 2.0: Low risk of collision
   
    The Pedal Curve method (Elliptical Conic) is commonly used to calculate the SF. This approach uses tangents to the vector between wellbore centers.
    """)
   
    # Add the reference image
    st.image("https://i.imgur.com/xYZ123.jpg", caption="Pedal Curve Visualization")

# Footer
st.markdown("---")
st.markdown("© 2025 Separation Factor Calculator App")

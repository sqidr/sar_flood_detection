import os
import subprocess
import rasterio
import rioxarray
import numpy as np
import xarray as xr  # <--- Make sure this line is added!

# ==================== CONFIGURATION ====================
GPT_PATH = "/home/st-juho/esa-snap/bin/gpt"  # If 'gpt' isn't in your system PATH, use absolute path (e.g., "/opt/snap/bin/gpt")
INPUT_SAFE = "/home/st-juho/code_testing/S1B_IW_SLC__1SDV_20170725T122204_20170725T122234_006644_00BAFB_0D34.SAFE"  # Can be .zip or unzipped .SAFE folder
DEM_PATH = "/home/st-juho/code_testing/dem_10m_wgs84.tif"
OUTPUT_DIR = "./sar_output"
SERVER_RAM_GB = 64  # Amount of RAM to allocate to SNAP (e.g., 32, 64)
# =======================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
snap_output_xml = os.path.join(OUTPUT_DIR, "pipeline_graph.xml")
snap_output_tif = os.path.join(OUTPUT_DIR, "snap_temp_output.tif")

# Step 1: Automatically read the exact resolution of your custom DEM
print("--- Step 1: Reading Custom DEM Spatial Resolution ---")
with rasterio.open(DEM_PATH) as src:
    dem_resolution = abs(src.res[0])
    print(f"Detected Custom DEM Resolution: {dem_resolution} meters/degrees")

# Step 2: Generate the Headless SNAP XML Processing Graph
# This processes the full image, removes noise, calibrates, debursts, and applies the DEM
xml_graph = f"""<graph id="Graph">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>{INPUT_SAFE}</file>
    </parameters>
  </node>
  <node id="ThermalNoiseRemoval">
    <operator>ThermalNoiseRemoval</operator>
    <sources><sourceProduct>Read</sourceProduct></sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <selectedPolarisations>VV,VH</selectedPolarisations>
    </parameters>
  </node>
  <node id="Calibration">
    <operator>Calibration</operator>
    <sources><sourceProduct>ThermalNoiseRemoval</sourceProduct></sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <outputSigmaBand>true</outputSigmaBand>
      <outputBetaBand>false</outputBetaBand>
      <outputGammaBand>false</outputGammaBand>
      <outputImageInComplex>false</outputImageInComplex>
    </parameters>
  </node>
  <node id="TOPSAR-Deburst">
    <operator>TOPSAR-Deburst</operator>
    <sources><sourceProduct>Calibration</sourceProduct></sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <selectedPolarisations>VV,VH</selectedPolarisations>
    </parameters>
  </node>
  <node id="Multilook">
    <operator>Multilook</operator>
    <sources><sourceProduct>TOPSAR-Deburst</sourceProduct></sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <nRgLooks>4</nRgLooks>
      <nAzLooks>1</nAzLooks>
      <grSquarePixel>false</grSquarePixel>
    </parameters>
  </node>
  <node id="Terrain-Correction">
    <operator>Terrain-Correction</operator>
    <sources><sourceProduct>Multilook</sourceProduct></sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <demName>External DEM</demName>
      <externalDEMFile>{DEM_PATH}</externalDEMFile>
      <externalDEMNoDataValue>-9999</externalDEMNoDataValue>
      <demResamplingMethod>BILINEAR_INTERPOLATION</demResamplingMethod>
      <imgResamplingMethod>BILINEAR_INTERPOLATION</imgResamplingMethod>
      <pixelSpacingInMeter>0.0</pixelSpacingInMeter>
      <pixelSpacingInDegree>{dem_resolution}</pixelSpacingInDegree>
      <mapProjection>GEOGCS["WGS84(DD)", DATUM["WGS84", SPHEROID["WGS84", 6378137.0, 298.257223563]], PRIMEM["Greenwich", 0.0], UNIT["degree", 0.017453292519943295]]</mapProjection>
      <saveSelectedSourceBand>true</saveSelectedSourceBand>
    </parameters>
  </node>
  <node id="Write">
    <operator>Write</operator>
    <sources><sourceProduct>Terrain-Correction</sourceProduct></sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>{snap_output_tif}</file>
      <formatName>GeoTIFF-BigTIFF</formatName>
    </parameters>
  </node>
</graph>
"""

with open(snap_output_xml, "w") as f:
    f.write(xml_graph)

# Step 3: Run SNAP headless engine via Subprocess
print("--- Step 2: Executing SNAP Pipeline on Server (This may take a few minutes) ---")
# We pass the server ram variable directly to Java execution here
cmd = [GPT_PATH, snap_output_xml, f"-J-Xmx{SERVER_RAM_GB}G", "-c", f"{int(SERVER_RAM_GB*0.75)}G"]

process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
if process.returncode != 0:
    print("SNAP Processing Failed! Error log:")
    print(process.stderr)
    exit(1)
print("SNAP Processing finished successfully.")

# Step 4: Coregistration & Pixel-for-Pixel matching using Python
print("--- Step 3: Forcing Exact Pixel Grid and Resolution Alignment with DEM ---")
target_dem = rioxarray.open_rasterio(DEM_PATH)
processed_sar = rioxarray.open_rasterio(snap_output_tif)

# reproject_match forces the SAR image to adopt the exact coordinates, grid shape, and resolution of your DEM
aligned_sar = processed_sar.rio.reproject_match(target_dem)

# Step 5: Convert Linear Magnitude to Decibels (dB) and save individual files
print("--- Step 4: Converting to Decibels (dB) and Exporting Final Tifs ---")
band_names = ["VV", "VH"]  # Match the output order from the XML script

for i, pol in enumerate(band_names):
    linear_data = aligned_sar.isel(band=i).values
    
    # Safely convert to dB handling 0 values
    with np.errstate(divide='ignore', invalid='ignore'):
        db_data = np.where(linear_data > 0, 10 * np.log10(linear_data), -9999)
        
    # FIX: Clone the exact spatial template from the perfectly matched SAR data
    output_raster = aligned_sar.isel(band=i).copy()
    
    # Inject the new dB math into the clone
    output_raster.values = db_data 
    
    # Set NoData value and export
    output_raster.rio.write_nodata(-9999, inplace=True)
    
    final_output_path = os.path.join(OUTPUT_DIR, f"Sentinel1_Final_{pol}_dB.tif")
    output_raster.rio.to_raster(final_output_path)
    print(f"Saved: {final_output_path}")

# Cleanup temporary intermediate files
if os.path.exists(snap_output_tif): os.remove(snap_output_tif)
if os.path.exists(snap_output_xml): os.remove(snap_output_xml)

print("--- PIPELINE COMPLETE ---")
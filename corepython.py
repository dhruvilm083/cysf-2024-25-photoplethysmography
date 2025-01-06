import streamlit as st
import numpy as np
import cv2
import tempfile
import pandas as pd
from scipy.signal import find_peaks
import pylance 

def mark_outliers(signal, z_threshold=3.0):
   
    mean_val = np.mean(signal)
    std_val  = np.std(signal, ddof=1) if len(signal) > 1 else 0.0

    outliers = []
    for val in signal:
        if std_val == 0:
            
            outliers.append(False)
        else:
            z_score = (val - mean_val) / std_val
            outliers.append(abs(z_score) > z_threshold)
    return outliers

def smooth_signal(signal, window_size=5):
    
    if window_size < 2:
        return signal
    smoothed = np.convolve(signal, np.ones(window_size)/window_size, mode='valid')
    return np.concatenate((
        np.full(window_size-1, smoothed[0]),
        smoothed
    ))

#revamped the code here w/ zscore outliers
def compute_bpm_ignoring_outliers(red_intensity, outlier_flags, fps):
    

    
    filtered_values = []
    keep_indices = []
    for i, val in enumerate(red_intensity):
        if not outlier_flags[i]:
            filtered_values.append(val)
            keep_indices.append(i)

   
    if len(filtered_values) == 0:
        return None, None, None, []  # We'll handle it outside

    
    if all(v == 0 for v in filtered_values):
        return "ALL_ZERO", None, None, []

    
    smoothed_signal_filtered = smooth_signal(filtered_values, window_size=5)

   
    peak_indices_filtered, _ = find_peaks(
        smoothed_signal_filtered,
        distance=fps * 0.3,   
        prominence=0.5       
    )

    
    if len(peak_indices_filtered) < 2:
        
        return 60.0, 0.0, smoothed_signal_filtered, []

    
    distances = np.diff(peak_indices_filtered)
    if len(distances) == 0:
        
        return 60.0, 0.0, smoothed_signal_filtered, []

    avg_distance = np.mean(distances)
    bpm = (fps / avg_distance) * 60

    
    distances_bpm = (fps / distances) * 60
    error_bpm = np.std(distances_bpm)

    return bpm, error_bpm, smoothed_signal_filtered, peak_indices_filtered

#sine wave generation for math
def generate_sine_wave_ignoring_outliers(bpm, fps, n_frames, outlier_flags):
   
    freq_hz = bpm / 60.0
    sine_wave = []
    for i in range(n_frames):
        if outlier_flags[i]:
            
            sine_wave.append(np.nan)
        else:
            t = i / fps
            sine_val = np.sin(2.0 * np.pi * freq_hz * t)
            sine_wave.append(sine_val)
    return sine_wave, freq_hz

def is_video_too_dark(red_intensity, brightness_threshold=30):
    return np.mean(red_intensity) < brightness_threshold

#reverse csv function
def main():
    

    input_method = st.radio("input type", ["video", "csv"])

    if input_method == "video":
        uploaded_video = st.file_uploader("video file", type=["mp4", "mov", "avi", "mkv"])
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)

            if not cap.isOpened():
                st.error("bad video")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 1:
                st.warning("unknown fps, default to 30")
                fps = 30.0

            
            list_avg_red = []
            frame_count = 0
            max_red_vals = []
            max_red_x = []
            max_red_y = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                red_channel = frame[:, :, 2]
                avg_val = np.mean(red_channel)
                list_avg_red.append(avg_val)
                _, max_val, _, max_loc = cv2.minMaxLoc(red_channel)
                max_red_vals.append(max_val)
                max_red_x.append(max_loc[0])
                max_red_y.append(max_loc[1])

            cap.release()

            if len(list_avg_red) == 0:
                st.warning("no frames")
                return

            if is_video_too_dark(list_avg_red):
                st.warning("too dark")
                return

            
            outlier_bool = mark_outliers(list_avg_red, z_threshold=3.0)
            
            outlier_flags_str = ["Yes" if b else "No" for b in outlier_bool]

            
            result = compute_bpm_ignoring_outliers(list_avg_red, outlier_bool, fps)
            if isinstance(result[0], str) and result[0] == "ALL_ZERO":
                st.warning("no red available")
                return
            bpm, error_bpm, smoothed_filtered, peak_indices_filtered = result

            if bpm is None:
                st.warning("no frames left, you probably weren't recording a heartbeat")
                return
            sine_wave, freq_hz = generate_sine_wave_ignoring_outliers(bpm, fps, len(list_avg_red), outlier_bool)

            st.markdown(f"<h2 style='text-align: center; font-size: 3em;'>{bpm:.1f} BPM</h2>", 
                        unsafe_allow_html=True)
            st.write(f"*error (±)**: {error_bpm:.2f} BPM")
            st.write(f"**frequency**: {freq_hz:.4f} Hz")

            
            df_summary = pd.DataFrame({
                "file_name": [uploaded_video.name],
                "heart_rate_bpm": [bpm],
                "error_bpm": [error_bpm],
                "freq_hz": [freq_hz]
            })
            st.dataframe(df_summary)

            # csv making
            df_detailed = pd.DataFrame({
                "frame_index": range(len(list_avg_red)),
                "average_red_intensity": list_avg_red,
                "outlier_frame": outlier_flags_str,       
                "max_red_value": max_red_vals,
                "max_red_x": max_red_x,
                "max_red_y": max_red_y,
               
            })

 
            smoothed_full = smooth_signal(list_avg_red, window_size=5)
            df_detailed["smoothed_intensity_full"] = smoothed_full

            
            df_detailed["sine_wave"] = sine_wave

        
            df_detailed["peak_detected"] = 0  
            df_detailed["heart_rate_bpm"] = bpm
            df_detailed["error_bpm"] = error_bpm
            df_detailed["sine_freq_hz"] = freq_hz

            #download
            csv_bytes = df_detailed.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Detailed CSV",
                data=csv_bytes,
                file_name="detailed_ppg_analysis_outliers_forced.csv",
                mime="text/csv"
            )

    else:
        #upload csv instead

        uploaded_csv = st.file_uploader("csv with `average_red_intensity`", type=["csv"])
        if uploaded_csv is not None:
            df_input = pd.read_csv(uploaded_csv)
            if "average_red_intensity" not in df_input.columns:
                st.error("no column 'average_red_intensity'")
                return

            fps = st.number_input(
                "fps for csv",
                min_value=1.0,
                value=30.0,
                help="fps of data"
            )

            list_avg_red = df_input["average_red_intensity"].values.tolist()
            if len(list_avg_red) == 0:
                st.warning("no rows in csv add data")
                return

            outlier_bool = mark_outliers(list_avg_red, z_threshold=3.0)
            outlier_flags_str = ["Yes" if b else "No" for b in outlier_bool]
            if is_video_too_dark(list_avg_red):
                st.warning("brightness to low as per data")
                return

            result = compute_bpm_ignoring_outliers(list_avg_red, outlier_bool, fps)
            if isinstance(result[0], str) and result[0] == "ALL_ZERO":
                st.warning("bad frames cant bpm.")
                return
            bpm, error_bpm, smoothed_filtered, peak_indices_filtered = result

            if bpm is None:
                st.warning("no good frames")
                return

            sine_wave, freq_hz = generate_sine_wave_ignoring_outliers(bpm, fps, len(list_avg_red), outlier_bool)

            st.markdown(f"<h2 style='text-align: center; font-size: 3em;'>{bpm:.1f} BPM</h2>", 
                        unsafe_allow_html=True)
            st.write(f"**error (±)**: {error_bpm:.2f} BPM")
            st.write(f"**frequency sin wave**: {freq_hz:.4f} Hz")

            df_summary = pd.DataFrame({
                "csv_file_name": [uploaded_csv.name],
                "heart_rate_bpm": [bpm],
                "error_bpm": [error_bpm],
                "freq_hz": [freq_hz]
            })
            st.dataframe(df_summary)

            df_detailed = df_input.copy()
            df_detailed["outlier_frame"] = outlier_flags_str


            smoothed_full = smooth_signal(list_avg_red, window_size=5)
            df_detailed["smoothed_intensity_full"] = smoothed_full


            df_detailed["sine_wave"] = sine_wave

           
            df_detailed["peak_detected"] = 0

            df_detailed["heart_rate_bpm"] = bpm
            df_detailed["error_bpm"] = error_bpm
            df_detailed["sine_freq_hz"] = freq_hz

            csv_bytes = df_detailed.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Detailed CSV",
                data=csv_bytes,
                file_name="detailed_ppg_analysis_csv_outliers_forced.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()

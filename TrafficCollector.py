from io import BytesIO
from pathlib import Path
import tempfile
import threading
import time
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import xmltodict
import folium
import m3u8
import imageio_ffmpeg
from PIL import Image, ImageDraw
from ultralytics import YOLO
from tqdm.auto import tqdm
from IPython.display import display, Video

import traffic_utils as tutils

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(levelname)s:%(name)s:  %(message)s')

HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://511.vdot.virginia.gov/"}
OUTPUT_DIR = Path("~/scr10/trafficcheck_outputs").expanduser()
OUTPUT_DIR.mkdir(exist_ok=True)

SENSOR_URL = 'https://data.511-atis-ttrip-prod.iteriscloud.com/smarterRoads/trafficSensor/detectorTMDD/current/trafficSensor_tmdd.xml?token=%242b%2410%24N%2FYVSDe.cOJFuEhCXEV4UuZogMpl7P.0s34ySk%2Fx40HfYIxI.CpH6'
CAMERA_URL = "https://511.vdot.virginia.gov/services/511/map/layers/map/cams"
WEST, SOUTH, EAST, NORTH = -77.5940, 37.8982, -77.4026, 38.3724

if __name__ == '__main__':
    """
    Collect station and camera metadata
    """
    logger.info("Getting station and camera metadata")
    station_df, camera_df, station_by_id = tutils.get_station_camera_df(
        sensor_url=SENSOR_URL,
        camera_url=CAMERA_URL,
        bbox = [WEST, SOUTH, EAST, NORTH]
    )
        
    logger.info("Station dataframe")
    logger.info(station_df.head())
    logger.info("Camera dataframe")
    logger.info(camera_df.head())

    """
    Downselect to specific camera IDs and sensor IDs
    """
    CAMERA_IDS = ["4363", "4364", "4365"]
    CAMERA_ID = "4363"
    SENSOR_ID = "OpenTMS-Detector4941987957"
    
    # Video cropping parameters
    LINE_1_LEFT_Y = 0.20
    LINE_1_RIGHT_Y = 0.45
    LINE_2_LEFT_Y = 0.50
    LINE_2_RIGHT_Y = 0.25
    CROP_LINES = [LINE_1_LEFT_Y, LINE_1_RIGHT_Y, LINE_2_LEFT_Y, LINE_2_RIGHT_Y]
    
    chosen_cameras = camera_df.set_index("id").loc[CAMERA_IDS].reset_index()
    chosen_sensor = station_df.loc[station_df.id == SENSOR_ID]
    if chosen_sensor.empty:
        raise ValueError("Selected sensor is missing from the study-area inventory.")
    if not chosen_cameras.direction.eq("NB").all():
        raise ValueError("One of the selected cameras is no longer labeled NB.")
    camera = chosen_cameras.loc[chosen_cameras.id == CAMERA_ID].iloc[0]
    selected_station = station_by_id[SENSOR_ID]
    mainline_ids = {
        detector["detector-inventory-header"]["device-id"]
        for detector in tutils.as_list(selected_station["inventory"]["detector-list"]["detector"])
        if tutils.direction_label(detector["detector-inventory-header"].get("link-direction")) == "NB"
        and str(detector["detector-inventory-header"].get("device-name", "")).lower().startswith("lane")
    }
    if not mainline_ids:
        raise ValueError("No NB mainline lane detectors were found for the selected sensor.")

    logger.info(f"Comparison uses camera {CAMERA_ID} and {len(mainline_ids)} NB sensor lanes.")

    """ 
    Video and sensor capture
    """
    CAPTURE_MINUTES = 1
    SENSOR_POLL_SECONDS = 60
    
    CAPTURE_ID = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    OUTPUT_DIR = OUTPUT_DIR / CAPTURE_ID
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving all data in {OUTPUT_DIR}")
    
    RAW_VIDEO_DIR = OUTPUT_DIR / f"camera_{CAMERA_ID}_raw_segments"
    RAW_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    
    sensor_samples, sensor_errors = [], []
    stop_sensor = threading.Event()

    sensor_thread = threading.Thread(
        target=tutils.poll_sensor, 
        kwargs=dict(
            sensor_id=SENSOR_ID,
            sensor_url=SENSOR_URL,
            sensor_samples=sensor_samples,
            sensor_errors=sensor_errors,
            stop_sensor=stop_sensor,
            SENSOR_POLL_SECONDS=SENSOR_POLL_SECONDS,
        ),
        daemon=True
    )
    sensor_thread.start()
    try:
        captured_segments = tutils.capture_video(
            url=camera["https_url"],
            seconds=CAPTURE_MINUTES * 60,
        )
    finally:
        stop_sensor.set()
        sensor_thread.join(timeout=25)

    logger.info(sensor_samples)
    logger.info(sensor_errors)
    
    # Save sensor polls immediately so they remain available even if later video
    # decoding or YOLO processing needs to be rerun.
    if sensor_errors:
        logger.info(f"Sensor poll failures: {len(sensor_errors)}")
    if sensor_samples:
        sensor_polls = pd.concat(sensor_samples, ignore_index=True)
        sensor_polls_file = f"sensor_{SENSOR_ID}_raw_polls.csv"
        sensor_polls.to_csv(OUTPUT_DIR / sensor_polls_file, index=False)
        logger.info(f"Sensor source sample periods (seconds): {sorted(sensor_polls.sample_seconds.dropna().unique())}")
        logger.info(f"Saved raw sensor polls: {sensor_polls_file}")
    else:
        raise RuntimeError(f"No sensor readings captured. Errors: {sensor_errors}")

    # Save each original transport-stream segment and a timestamp manifest for
    # future reprocessing with a different crop, model, or aggregation interval.
    segment_manifest = []
    for segment_number, segment in enumerate(captured_segments):
        segment_path = RAW_VIDEO_DIR / f"segment_{segment_number:06d}.ts"
        segment_path.write_bytes(segment["bytes"])
        segment_manifest.append({
            "segment_number": segment_number,
            "file": segment_path.name,
            "start_utc": segment["start"],
            "duration_seconds": segment["duration"],
            "estimated_time": segment["estimated_time"],
            "bytes": len(segment["bytes"]),
        })
    logger.info(f"Saved original video segments: {RAW_VIDEO_DIR}")

    segment_manifest = pd.DataFrame(segment_manifest)
    segment_manifest_file = f"camera_{CAMERA_ID}_segment_manifest.csv"
    segment_manifest.to_csv(OUTPUT_DIR /segment_manifest_file, index=False)
    logger.info(f"Saved segment timestamps: f{segment_manifest_file}")

    """
    YOLO on captured data
    """
    model = YOLO("yolo11s.pt")
    vehicle_ids = [class_id for class_id, name in model.names.items() if name in {"car", "truck"}]
    if not vehicle_ids:
        raise ValueError("The model has no car or truck classes.")
    
    counts = []
    for segment in tqdm(captured_segments, desc="YOLO on captured video", unit="segment"):
        frames, source_fps = tutils.read_segment(segment)
        cropped = [tutils.crop_below_lines(frame, *CROP_LINES) for frame in frames]
        for index, result in enumerate(tutils.detect_vehicles(cropped, model, vehicle_ids)):
            timestamp = segment["start"] + pd.Timedelta(seconds=(index + 0.5) * segment["duration"] / len(frames))
            counts.append({"time_utc": timestamp, "car_boxes": len(result.boxes)})
    frame_counts = pd.DataFrame(counts)
    frame_counts_file = f"camera_{CAMERA_ID}_raw_frame_counts.csv"
    frame_counts.to_csv(OUTPUT_DIR / frame_counts_file, index=False)
    logger.info(f"Saved raw per-frame YOLO counts: {frame_counts_file}")

    logger.info(f"Analyzed {len(frame_counts)} frames; collected {sensor_polls.polled_at.nunique()} sensor polls.")
    if any(segment["estimated_time"] for segment in captured_segments):
        logger.info("Camera UTC times are estimated from HLS publication; a small timing offset may remain.")
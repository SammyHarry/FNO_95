from io import BytesIO
from pathlib import Path
import tempfile
import threading
import time

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

HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://511.vdot.virginia.gov/"}


"""
--------------------------------------------------
Helper functions to get sensor and camera metadata
--------------------------------------------------
"""

def as_list(value):
    return value if isinstance(value, list) else ([] if value is None else [value])

def get_stations(sensor_url):
    """ Parse stations from sensor url
    """
    response = requests.get(sensor_url, timeout=20)
    response.raise_for_status()
    parsed = xmltodict.parse(response.content, process_namespaces=True, namespaces={
        "http://www.openroadsconsulting.org/detector": None,
        "http://www.qfree.com/common": None,
    })
    return as_list(parsed["stationList"]["station"])

def direction_label(value):
    """ Standardize directions """
    return {"n": "NB", "s": "SB", "e": "EB", "w": "WB"}.get(
        str(value).lower(), str(value or "Unknown").upper()
    )


def get_station_camera_df(sensor_url, camera_url, bbox, headers=HEADERS):
    """ Get dataframes containing sensor station and camera metadata
        located within a lat/lon bounding box
    """
    stations = get_stations(sensor_url)
    station_by_id = {}
    station_rows = []
    for station in stations:
        inventory = station["inventory"]
        header = inventory["detector-station-inventory-header"]
        station_by_id[header["device-id"]] = station
        directions = {
            direction_label(detector["detector-inventory-header"].get("link-direction"))
            for detector in as_list(inventory["detector-list"].get("detector"))
        }
        station_rows.append({
            "id": header["device-id"], "name": header.get("device-name"),
            "route": header.get("link-name"), "direction": " / ".join(sorted(directions)),
            "latitude": float(header["device-location"]["latitude"]) / 1e6,
            "longitude": float(header["device-location"]["longitude"]) / 1e6,
        })
    station_df = pd.DataFrame(station_rows)
    
    response = requests.get(camera_url, headers=headers, timeout=30)
    response.raise_for_status()
    camera_df = pd.DataFrame([
        dict(feature["properties"], longitude=feature["geometry"]["coordinates"][0],
             latitude=feature["geometry"]["coordinates"][1])
        for feature in response.json()["features"]
    ])
    camera_df["id"] = camera_df["id"].astype(str)

    WEST, SOUTH, EAST, NORTH = bbox
    
    station_df = station_df[
        station_df.longitude.between(WEST, EAST) & station_df.latitude.between(SOUTH, NORTH)
    ].copy()
    camera_df = camera_df[
        camera_df.longitude.between(WEST, EAST) & camera_df.latitude.between(SOUTH, NORTH)
    ].copy()
    print(f"Study area: {len(station_df)} sensor sites and {len(camera_df)} cameras")
    
    return station_df, camera_df, station_by_id

"""
-----------------------------------
Helper functions to get sensor data
-----------------------------------
"""

def sensor_readings(station_list):
    """ Collect the most recent sensor readings from a list of stations
    """
    rows = []
    for station in station_list:
        inventory = station["inventory"]
        station_id = inventory["detector-station-inventory-header"]["device-id"]
        detector_info = {
            detector["detector-inventory-header"]["device-id"]: detector["detector-inventory-header"]
            for detector in as_list(inventory["detector-list"].get("detector"))
        }
        for data in as_list(station.get("data", {}).get("detector-list", {}).get("detector-data-detail")):
            info = detector_info.get(data["detector-id"], {})
            def timestamp(field):
                value = data.get(field, {}).get("dateTimeHolder", {}).get("iso-8601")
                return pd.to_datetime(value, utc=True, errors="coerce")
            rows.append({
                "station_id": station_id, "detector_id": data["detector-id"],
                "name": info.get("device-name"), "direction": direction_label(info.get("link-direction")),
                "interval_start": timestamp("start-time"), "interval_end": timestamp("end-time"),
                "sample_seconds": pd.to_numeric(data.get("sample-period"), errors="coerce"),
                "vehicle_count": pd.to_numeric(data.get("vehicle-count"), errors="coerce"),
                "occupancy": pd.to_numeric(data.get("vehicle-occupancy"), errors="coerce"),
                "speed": pd.to_numeric(data.get("vehicle-speed"), errors="coerce"),
            })
    return pd.DataFrame(rows)

def poll_sensor(sensor_id, sensor_url, sensor_samples, sensor_errors, stop_sensor, SENSOR_POLL_SECONDS):
    while not stop_sensor.is_set():
        try:
            current = next(s for s in get_stations(sensor_url)
                           if s["inventory"]["detector-station-inventory-header"]["device-id"] == sensor_id)
            readings = sensor_readings([current])
            if readings.empty:
                raise ValueError("Selected sensor returned no detector readings.")
            readings["polled_at"] = pd.Timestamp.now(tz="UTC")
            sensor_samples.append(readings)
        except Exception as exc:
            sensor_errors.append(type(exc).__name__ + ' ' + str(exc))
        stop_sensor.wait(SENSOR_POLL_SECONDS)

"""
------------------------------
Helper functions to get video
------------------------------
"""

def get_snapshot(camera_row, headers=HEADERS):
    """ Get the current snapshot from a specific camera
    """
    response = requests.get(camera_row["image_url"], headers=headers, timeout=20)
    response.raise_for_status()
    return np.asarray(Image.open(BytesIO(response.content)).convert("RGB"))

def crop_below_lines(frame_rgb, 
                     LINE_1_LEFT_Y, 
                     LINE_1_RIGHT_Y, 
                     LINE_2_LEFT_Y, 
                     LINE_2_RIGHT_Y):
    """ Black out everything above either line; retain pixels below both
    """
    height, width = frame_rgb.shape[:2]
    x_fraction = np.linspace(0, 1, width)
    first_line = LINE_1_LEFT_Y + (LINE_1_RIGHT_Y - LINE_1_LEFT_Y) * x_fraction
    second_line = LINE_2_LEFT_Y + (LINE_2_RIGHT_Y - LINE_2_LEFT_Y) * x_fraction
    cutoff_y = np.maximum(first_line, second_line) * (height - 1)
    keep = np.arange(height)[:, None] >= cutoff_y[None, :]
    cropped = frame_rgb.copy()
    cropped[~keep] = 0
    return cropped

def get_playlist(url, headers=HEADERS):
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    if not response.text.lstrip().startswith("#EXTM3U"):
        raise ValueError("Camera endpoint did not return an HLS playlist.")
    return m3u8.loads(response.text, uri=response.url)


def capture_video(url, seconds, headers=HEADERS):
    playlist = get_playlist(url)
    if playlist.is_variant:
        variant = max(playlist.playlists, key=lambda p: p.stream_info.bandwidth)
        url = variant.absolute_uri
        playlist = get_playlist(url)
    seen = {segment.absolute_uri for segment in playlist.segments}
    captured, errors = [], []
    last_end, last_sequence = None, None
    started = time.monotonic()
    with tqdm(total=seconds, desc="Capturing live video", unit="s") as progress:
        while time.monotonic() - started < seconds:
            try:
                playlist = get_playlist(url)
                for index, segment in enumerate(playlist.segments):
                    if segment.absolute_uri in seen:
                        continue
                    response = requests.get(segment.absolute_uri, headers=headers, timeout=15)
                    response.raise_for_status()
                    sequence = playlist.media_sequence + index
                    source_time = segment.program_date_time or segment.current_program_date_time
                    if source_time is not None:
                        start = pd.Timestamp(source_time).tz_convert("UTC")
                    elif last_end is not None and sequence == last_sequence + 1:
                        start = last_end
                    else:
                        remaining = sum(s.duration for s in playlist.segments[index:])
                        start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=remaining)
                    captured.append({"bytes": response.content, "start": start,
                                     "duration": segment.duration, "estimated_time": source_time is None})
                    seen.add(segment.absolute_uri)
                    last_end = start + pd.Timedelta(seconds=segment.duration)
                    last_sequence = sequence
            except requests.RequestException as exc:
                errors.append(type(exc).__name__)
            progress.update(min(seconds, time.monotonic() - started) - progress.n)
            time.sleep(1)
        progress.update(seconds - progress.n)
    if not captured:
        raise RuntimeError(f"No video segments arrived. Network errors: {errors[-3:]}")
    if errors:
        print(f"Video requests retried: {len(errors)}; missing frames affect window coverage.")
    return captured


def read_segment(segment):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "segment.ts"
        path.write_bytes(segment["bytes"])
        reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
        try:
            metadata = next(reader)
            width, height = metadata["size"]
            frames = [np.frombuffer(raw, np.uint8).reshape(height, width, 3) for raw in reader]
        finally:
            reader.close()
    return frames, float(metadata["fps"])


def write_clip(frames, path, fps):
    height, width = frames[0].shape[:2]
    writer = imageio_ffmpeg.write_frames(str(path), size=(width, height), fps=fps,
                                       codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                                       macro_block_size=1)
    writer.send(None)
    try:
        for frame in frames:
            writer.send(np.ascontiguousarray(frame).tobytes())
    finally:
        writer.close()

"""
-------------------------
Helper functions for YOLO
-------------------------
"""
def detect_vehicles(frames, model, vehicle_ids, CONFIDENCE=0.05):
    for index in range(0, len(frames), 16):
        # NumPy inputs to YOLO use BGR; our images and video writer use RGB.
        batch = [frame[:, :, ::-1].copy() for frame in frames[index:index + 16]]
        yield from model.predict(batch, classes=vehicle_ids, conf=CONFIDENCE, imgsz=320, verbose=False)


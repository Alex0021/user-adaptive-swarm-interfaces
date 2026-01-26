sm_metadata: dict[str, int] = {
    "stream_ready": 1,
    "calibration_ok": 1,
    "active_data_cnt": 1
} 
"""
Metadata fields byte sizes present in the shared memory with Unity.
"""

sm_gaze_data: dict[str, int] = {
    "timestamp": 8,
    "left_gaze_point": 4 * 3,
    "right_gaze_point": 4 * 3,
    "left_point_screen": 4 * 2,
    "right_point_screen": 4 * 2,
    "left_validity": 1,
    "right_validity": 1,
    "left_pupil_diameter": 4,
    "right_pupil_diameter": 4,
}
"""
Gaze data fields byte sizes present in the shared memory with Unity.
"""
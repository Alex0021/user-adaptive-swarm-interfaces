import os
from pathlib import Path
import subprocess
import tobii_research as tr
import logging
from eye_tracker.constants import LOGS_DIR, DATA_DIR

CALL_MODE = "usercalibration"
SCREEN = "1"    
EXIT_ERROR_CALIBRATION_CANCELLED = 30

logger = logging.getLogger(__name__)

def invoke_eyetracker_calibration_manager(eyetracker):
    if not eyetracker:
        logger.info("No eyetracker provided, using the first available eyetracker.")
        eyetrackers = tr.find_all_eyetrackers()
        if len(eyetrackers) == 0:
            logger.error("No eyetrackers found.")
            return
        eyetracker = eyetrackers[0]

    logger.info("Starting eyetracker calibration with serial number %s on screen %s", eyetracker.serial_number, SCREEN)
    try:
        etm_path = Path(os.environ['LOCALAPPDATA']) / "Programs/TobiiProEyeTrackerManager/TobiiProEyeTrackerManager.exe"

        log_file = open(LOGS_DIR / "etm_calibration.log", "w", encoding="utf-8")

        etm_process = subprocess.Popen([
            etm_path,
            f"--mode={CALL_MODE}",
            f"--device-sn={eyetracker.serial_number}",
            f"--screen={SCREEN}"
        ],
        stdout=log_file)

        stdout, stderr = etm_process.communicate()
        log_file.close()
        if etm_process.returncode != 0:
            for l in stdout.splitlines():
                if l.startswith("ETM Error:"):
                    logger.error(l)
        elif etm_process.returncode == EXIT_ERROR_CALIBRATION_CANCELLED:
            logger.warning("Calibration was cancelled by the user.")
        else:
            logger.info("Calibration exited successfully.")

    except Exception as e:
        logger.error("An error occurred: %s", e)

def save_latest_calibration(eyetracker: tr.EyeTracker):
    calib = eyetracker.retrieve_calibration_data()
    if not calib:
        logger.error("No calibration data found.")
        return

    file_path = DATA_DIR / f"calibration_{eyetracker.serial_number}.bin"
    with open(file_path, "wb") as f:
        f.write(calib)

    logger.info("Calibration data saved to %s", file_path)

def apply_saved_calibration(eyetracker: tr.EyeTracker):
    file_path = DATA_DIR / f"calibration_{eyetracker.serial_number}.bin"
    if not file_path.exists():
        logger.error("Calibration file %s does not exist.", file_path)
        return

    with open(file_path, "rb") as f:
        calib_data = f.read()

    eyetracker.apply_calibration_data(calib_data)
    logger.info("Applied calibration data from %s", file_path)


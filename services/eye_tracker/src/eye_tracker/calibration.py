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


def invoke_eyetracker_calibration_manager(eyetracker: tr.EyeTracker | None):
    """
    Launch the Tobii Eye Tracker Manager for calibration. (External application)

    :param eyetracker (tr.EyeTracker): The eye tracker device to calibrate. If None, the first available eye tracker will be used.
    """
    if not eyetracker:
        logger.info("No eyetracker provided, using the first available eyetracker.")
        eyetrackers = tr.find_all_eyetrackers()
        if len(eyetrackers) == 0:
            logger.error("No eyetrackers found.")
            return
        eyetracker = eyetrackers[0]

    logger.info(
        "Starting eyetracker calibration with serial number %s on screen %s",
        eyetracker.serial_number,
        SCREEN,
    )
    try:
        etm_path = (
            Path(os.environ["LOCALAPPDATA"])
            / "Programs/TobiiProEyeTrackerManager/TobiiProEyeTrackerManager.exe"
        )

        log_file = open(LOGS_DIR / "etm_calibration.log", "w", encoding="utf-8")

        etm_process = subprocess.Popen(
            [
                etm_path,
                f"--mode={CALL_MODE}",
                f"--device-sn={eyetracker.serial_number}",
                f"--screen={SCREEN}",
            ],
            stdout=log_file,
        )

        stdout, stderr = etm_process.communicate()
        log_file.close()
        if etm_process.returncode != 0:
            for line in stdout.splitlines():
                if line.startswith(bytes("ETM Error:", "utf-8")):
                    logger.error(line)
        elif etm_process.returncode == EXIT_ERROR_CALIBRATION_CANCELLED:
            logger.warning("Calibration was cancelled by the user.")
        else:
            logger.info("Calibration exited successfully.")

    except Exception as e:
        logger.error("An error occurred: %s", e, exc_info=True)


def save_latest_calibration(eyetracker: tr.EyeTracker):
    """
    Save the latest calibration data from the eye tracker to a file.

    :param eyetracker (tr.EyeTracker): The eye tracker device from which to save calibration data.
    """
    calib = eyetracker.retrieve_calibration_data()
    if not calib:
        logger.error("No calibration data found.")
        return

    file_path = DATA_DIR / f"calibration_{eyetracker.serial_number}.bin"
    with open(file_path, "wb") as f:
        f.write(calib)

    logger.info("Calibration data saved to %s", file_path)


def apply_saved_calibration(eyetracker: tr.EyeTracker):
    """
    Apply saved calibration data to the eye tracker.

    :param eyetracker (tr.EyeTracker): The eye tracker device to which to apply calibration data.
    """
    file_path = DATA_DIR / f"calibration_{eyetracker.serial_number}.bin"
    if not file_path.exists():
        logger.error("Calibration file %s does not exist.", file_path)
        return

    with open(file_path, "rb") as f:
        calib_data = f.read()

    eyetracker.apply_calibration_data(calib_data)
    logger.info("Applied calibration data from %s", file_path)

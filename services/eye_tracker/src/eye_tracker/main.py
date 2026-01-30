from eye_tracker.constants import EYETRACKER_SN
import tobii_research as tr
import logging
import time
from eye_tracker.stream import EyeTrackerStream

ANIMATION_FRAMES = ["|", "/", "-", "\\"]


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s - %(name)s] %(levelname)s::%(message)s"
    )


def init_eyetracker(sn: str):
    eyetracker_list = tr.find_all_eyetrackers()
    for et in eyetracker_list:
        if et.serial_number == sn:
            return et


def main():
    setup_logging()
    logger = logging.getLogger()
    logger.info("Eye Tracker Service started.")

    eyetracker = init_eyetracker(EYETRACKER_SN)
    if not eyetracker:
        logger.error("Eyetracker with serial number %s not found.", EYETRACKER_SN)
        return

    streamer = EyeTrackerStream()

    eyetracker.subscribe_to(
        tr.EYETRACKER_GAZE_DATA, streamer.gaze_data_callback, as_dictionary=True
    )
    streamer.start_stream()

    while True:
        try:
            # Little animation in terminal to show service is running
            for frame in ANIMATION_FRAMES:
                print(
                    f"\r{frame} Publishing gaze data @ {streamer.monitor.get_frequency():.1f} Hz | Latency {streamer.monitor.get_avg_queue_cnt():.1f} | Total {streamer.monitor.get_total_messages()}",
                    end="",
                )
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("-" * 30)
            logger.info("Eye Tracker Service stopping.")
            streamer.stop_stream()
            eyetracker.unsubscribe_from(
                tr.EYETRACKER_GAZE_DATA, streamer.gaze_data_callback
            )
            break


if __name__ == "__main__":
    main()

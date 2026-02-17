from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, TypeVar

import numpy as np


# Type aliases
class DataclassLike(Protocol):
    @classmethod
    def size(cls) -> int: ...
    @classmethod
    def from_buffer(cls, data: bytes) -> DataclassLike: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataclassLike: ...


T = TypeVar("T", bound=DataclassLike)


class Listener(Protocol[T]):
    def __call__(self, datas: Sequence[T], batch_update: bool = False) -> None: ...


@dataclass
class Metadata:
    is_sender_ready: np.uint8
    calibration_ok: np.uint8
    is_receiver_ready: np.uint8
    head: np.int32
    tail: np.int32

    def get_conversion_str(self) -> str:
        return "BBBII"

    def __len__(self) -> int:
        return self.size()

    @classmethod
    def size(cls) -> int:
        return 1 + 1 + 1 + 4 + 4

    @classmethod
    def from_buffer(cls, buffer: bytes) -> Metadata:
        return Metadata(
            is_sender_ready=np.frombuffer(buffer[0:1], dtype=np.uint8)[0],
            calibration_ok=np.frombuffer(buffer[1:2], dtype=np.uint8)[0],
            is_receiver_ready=np.frombuffer(buffer[2:3], dtype=np.uint8)[0],
            head=np.frombuffer(buffer[3:7], dtype=np.int32)[0],
            tail=np.frombuffer(buffer[7:11], dtype=np.int32)[0],
        )


@dataclass
class NBackData:
    timestamp: np.int64
    response_timestamp: np.int64
    nback_level: np.int8
    stimulus: np.int8
    participant_response: np.int8
    is_correct: np.int8

    def get_conversion_str(self) -> str:
        return "<2q4B"

    def __len__(self) -> int:
        return self.size()

    @classmethod
    def size(cls) -> int:
        return 8 + 8 + 1 + 1 + 1 + 1

    @classmethod
    def from_buffer(cls, buffer: bytes) -> NBackData:
        if len(buffer) < cls.size():
            raise ValueError(
                f"Buffer size {len(buffer)} is smaller than expected size {cls.size()}."
            )
        return NBackData(
            timestamp=np.frombuffer(buffer[0:8], dtype=np.int64)[0],
            response_timestamp=np.frombuffer(buffer[8:16], dtype=np.int64)[0],
            nback_level=np.frombuffer(buffer[16:17], dtype=np.int8)[0],
            stimulus=np.frombuffer(buffer[17:18], dtype=np.int8)[0],
            participant_response=np.frombuffer(buffer[18:19], dtype=np.int8)[0],
            is_correct=np.frombuffer(buffer[19:20], dtype=np.int8)[0],
        )

    @classmethod
    def from_dict(cls, data: dict) -> NBackData:
        try:
            return NBackData(
                timestamp=np.int64(data["timestamp"]),
                response_timestamp=np.int64(data["response_timestamp"]),
                nback_level=np.int8(data["nback_level"]),
                stimulus=np.int8(data["stimulus"]),
                participant_response=np.int8(data["participant_response"]),
                is_correct=np.int8(data["is_correct"]),
            )
        except KeyError as e:
            raise ValueError(f"Missing key in data dictionary: {e}") from e


@dataclass
class DroneData:
    timestamp: np.int64
    id: np.int8
    position_x: np.float32
    position_y: np.float32
    position_z: np.float32
    orientation_x: np.float32
    orientation_y: np.float32
    orientation_z: np.float32
    velocity_x: np.float32
    velocity_y: np.float32
    velocity_z: np.float32
    angular_velocity_x: np.float32
    angular_velocity_y: np.float32
    angular_velocity_z: np.float32
    acceleration_x: np.float32
    acceleration_y: np.float32
    acceleration_z: np.float32

    def get_conversion_str(self) -> str:
        return "<q3f3f3f3f3f3f"

    def __len__(self) -> int:
        return self.size()

    @classmethod
    def size(cls) -> int:
        return 8 + 1 + 3 * 4 + 3 * 4 + 3 * 4 + 3 * 4 + 3 * 4

    @classmethod
    def from_buffer(cls, buffer: bytes) -> DroneData:
        if len(buffer) < cls.size():
            raise ValueError(
                f"Buffer size {len(buffer)} is smaller than expected size {cls.size()}."
            )
        return DroneData(
            timestamp=np.frombuffer(buffer[0:8], dtype=np.int64)[0],
            id=np.frombuffer(buffer[8:9], dtype=np.int8)[0],
            position_x=np.frombuffer(buffer[9:13], dtype=np.float32)[0],
            position_y=np.frombuffer(buffer[13:17], dtype=np.float32)[0],
            position_z=np.frombuffer(buffer[17:21], dtype=np.float32)[0],
            orientation_x=np.frombuffer(buffer[21:25], dtype=np.float32)[0],
            orientation_y=np.frombuffer(buffer[25:29], dtype=np.float32)[0],
            orientation_z=np.frombuffer(buffer[29:33], dtype=np.float32)[0],
            velocity_x=np.frombuffer(buffer[33:37], dtype=np.float32)[0],
            velocity_y=np.frombuffer(buffer[37:41], dtype=np.float32)[0],
            velocity_z=np.frombuffer(buffer[41:45], dtype=np.float32)[0],
            angular_velocity_x=np.frombuffer(buffer[45:49], dtype=np.float32)[0],
            angular_velocity_y=np.frombuffer(buffer[49:53], dtype=np.float32)[0],
            angular_velocity_z=np.frombuffer(buffer[53:57], dtype=np.float32)[0],
            acceleration_x=np.frombuffer(buffer[57:61], dtype=np.float32)[0],
            acceleration_y=np.frombuffer(buffer[61:65], dtype=np.float32)[0],
            acceleration_z=np.frombuffer(buffer[65:69], dtype=np.float32)[0],
        )

    @classmethod
    def from_dict(cls, data: dict) -> DroneData:
        try:
            return DroneData(
                timestamp=np.int64(data["timestamp"]),
                id=np.int8(data["id"]),
                position_x=np.float32(data["position"][0]),
                position_y=np.float32(data["position"][1]),
                position_z=np.float32(data["position"][2]),
                orientation_x=np.float32(data["orientation"][0]),
                orientation_y=np.float32(data["orientation"][1]),
                orientation_z=np.float32(data["orientation"][2]),
                velocity_x=np.float32(data["velocity"][0]),
                velocity_y=np.float32(data["velocity"][1]),
                velocity_z=np.float32(data["velocity"][2]),
                angular_velocity_x=np.float32(data["angular_velocity"][0]),
                angular_velocity_y=np.float32(data["angular_velocity"][1]),
                angular_velocity_z=np.float32(data["angular_velocity"][2]),
                acceleration_x=np.float32(data["acceleration"][0]),
                acceleration_y=np.float32(data["acceleration"][1]),
                acceleration_z=np.float32(data["acceleration"][2]),
            )
        except KeyError as e:
            raise ValueError(f"Missing key in data dictionary: {e}") from e

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DroneData):
            return NotImplemented
        return self.timestamp == other.timestamp and self.id == other.id


@dataclass
class GazeData:
    timestamp: np.int64
    left_gaze_point_x: np.float32
    left_gaze_point_y: np.float32
    left_gaze_point_z: np.float32
    right_gaze_point_x: np.float32
    right_gaze_point_y: np.float32
    right_gaze_point_z: np.float32
    left_point_screen_x: np.float32
    left_point_screen_y: np.float32
    right_point_screen_x: np.float32
    right_point_screen_y: np.float32
    left_validity: np.int8
    right_validity: np.int8
    left_pupil_diameter: np.float32
    right_pupil_diameter: np.float32
    left_openness_validity: np.int8 = np.int8(0)
    right_openness_validity: np.int8 = np.int8(0)
    left_openness: np.float32 = np.float32(0.0)
    right_openness: np.float32 = np.float32(0.0)

    def get_conversion_str(self) -> str:
        return "<q10f2B2f2B2f"

    def __len__(self) -> int:
        return self.size()

    @classmethod
    def size(cls) -> int:
        return 8 + 3 * 4 + 3 * 4 + 2 * 4 + 2 * 4 + 2 * 1 + 2 * 4 + 2 * 1 + 2 * 4

    @classmethod
    def from_buffer(cls, buffer: bytes) -> GazeData:
        return GazeData(
            timestamp=np.frombuffer(buffer[0:8], dtype=np.int64)[0],
            left_gaze_point_x=np.frombuffer(buffer[8:12], dtype=np.float32)[0],
            left_gaze_point_y=np.frombuffer(buffer[12:16], dtype=np.float32)[0],
            left_gaze_point_z=np.frombuffer(buffer[16:20], dtype=np.float32)[0],
            right_gaze_point_x=np.frombuffer(buffer[20:24], dtype=np.float32)[0],
            right_gaze_point_y=np.frombuffer(buffer[24:28], dtype=np.float32)[0],
            right_gaze_point_z=np.frombuffer(buffer[28:32], dtype=np.float32)[0],
            left_point_screen_x=np.frombuffer(buffer[32:36], dtype=np.float32)[0],
            left_point_screen_y=np.frombuffer(buffer[36:40], dtype=np.float32)[0],
            right_point_screen_x=np.frombuffer(buffer[40:44], dtype=np.float32)[0],
            right_point_screen_y=np.frombuffer(buffer[44:48], dtype=np.float32)[0],
            left_validity=np.frombuffer(buffer[48:49], dtype=np.int8)[0],
            right_validity=np.frombuffer(buffer[49:50], dtype=np.int8)[0],
            left_pupil_diameter=np.frombuffer(buffer[50:54], dtype=np.float32)[0],
            right_pupil_diameter=np.frombuffer(buffer[54:58], dtype=np.float32)[0],
            left_openness_validity=np.frombuffer(buffer[58:59], dtype=np.int8)[0],
            right_openness_validity=np.frombuffer(buffer[59:60], dtype=np.int8)[0],
            left_openness=np.frombuffer(buffer[60:64], dtype=np.float32)[0],
            right_openness=np.frombuffer(buffer[64:68], dtype=np.float32)[0],
        )

    @classmethod
    def from_dict(cls, data: dict) -> GazeData:
        try:
            return GazeData(
                timestamp=np.int64(data["system_time_stamp"]),
                left_gaze_point_x=np.float32(
                    data["left_gaze_origin_in_user_coordinate_system"][0]
                ),
                left_gaze_point_y=np.float32(
                    data["left_gaze_origin_in_user_coordinate_system"][1]
                ),
                left_gaze_point_z=np.float32(
                    data["left_gaze_origin_in_user_coordinate_system"][2]
                ),
                right_gaze_point_x=np.float32(
                    data["right_gaze_origin_in_user_coordinate_system"][0]
                ),
                right_gaze_point_y=np.float32(
                    data["right_gaze_origin_in_user_coordinate_system"][1]
                ),
                right_gaze_point_z=np.float32(
                    data["right_gaze_origin_in_user_coordinate_system"][2]
                ),
                left_point_screen_x=np.float32(
                    data["left_gaze_point_on_display_area"][0]
                ),
                left_point_screen_y=np.float32(
                    data["left_gaze_point_on_display_area"][1]
                ),
                right_point_screen_x=np.float32(
                    data["right_gaze_point_on_display_area"][0]
                ),
                right_point_screen_y=np.float32(
                    data["right_gaze_point_on_display_area"][1]
                ),
                left_validity=np.int8(data["left_pupil_validity"]),
                right_validity=np.int8(data["right_pupil_validity"]),
                left_pupil_diameter=np.float32(data["left_pupil_diameter"]),
                right_pupil_diameter=np.float32(data["right_pupil_diameter"]),
                left_openness_validity=np.int8(
                    data.get("left_eye_openness_validity", 0)
                ),
                right_openness_validity=np.int8(
                    data.get("right_eye_openness_validity", 0)
                ),
                left_openness=np.float32(data.get("left_eye_openness", 0.0)),
                right_openness=np.float32(data.get("right_eye_openness", 0.0)),
            )
        except KeyError as e:
            raise ValueError(f"Missing key in data dictionary: {e}") from e


class ExperimentState(Enum):
    Idle = 0
    IdleSilent = 1
    Wait = 2
    WaitForUser = 3
    Welcome = 4
    RcControls = 5
    Calibration = 6
    FlyingInstructions = 7
    FlyingPractice = 8
    NBackInstructions = 9
    NBackPractice = 10
    ExperimentBegin = 11
    Task = 12
    Countdown = 13
    Trial = 14
    Finished = 15


@dataclass
class ExperimentStatus:
    previous_state: ExperimentState
    current_state: ExperimentState
    next_state: ExperimentState
    current_task: int
    current_trial: int
    nback_levels_order: list[int]
    current_nback_level: int
    state_enter_timestamp: np.int64

    @classmethod
    def from_dict(cls, data: dict) -> ExperimentStatus:
        try:
            return ExperimentStatus(
                previous_state=ExperimentState[data["previousState"]],
                current_state=ExperimentState[data["state"]],
                next_state=ExperimentState[data["nextState"]],
                current_task=data["currentTask"],
                current_trial=data["currentTrial"],
                nback_levels_order=data["nbackLevelsOrder"],
                state_enter_timestamp=np.int64(data["stateEnterTimestamp"]),
                current_nback_level=data.get(
                    "currentNBackLevel", -1
                ),  # Optional field, default to -1 if not present
            )
        except KeyError as e:
            raise ValueError(f"Missing key in data dictionary: {e}") from e
        except ValueError as e:
            raise ValueError(f"Invalid value in data dictionary: {e}") from e

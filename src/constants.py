"""Static facts about the PAMAP2 dataset (column layout, activity IDs, sensor ranges).

Everything here comes from the dataset's ``readme.pdf`` shipped inside the UCI archive.
Configurable choices (which activities/sensors to use) live in ``configs/params.yaml``.
"""

from __future__ import annotations

GRAVITY: float = 9.80665
"""Standard gravity in m/s^2, used to express accelerometer ranges."""

IMU_LOCATIONS: tuple[str, ...] = ("hand", "chest", "ankle")
"""IMU placements in file order. ``hand`` is the wrist of the dominant arm."""

IMU_FIELDS: tuple[str, ...] = (
    "temp",
    "acc16_x",
    "acc16_y",
    "acc16_z",
    "acc6_x",
    "acc6_y",
    "acc6_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "mag_x",
    "mag_y",
    "mag_z",
    "orient_0",
    "orient_1",
    "orient_2",
    "orient_3",
)
"""The 17 columns of each IMU block, in file order."""


def imu_columns(location: str) -> list[str]:
    """Return the 17 column names of one IMU block.

    Args:
        location: One of :data:`IMU_LOCATIONS`.

    Returns:
        Column names such as ``hand_acc16_x``.
    """
    return [f"{location}_{field}" for field in IMU_FIELDS]


RAW_SENSOR_COLUMNS: list[str] = ["heart_rate"] + [col for loc in IMU_LOCATIONS for col in imu_columns(loc)]
"""All sensor columns of a ``.dat`` file (everything except timestamp and label)."""

DAT_COLUMNS: list[str] = ["timestamp", "activity_id", *RAW_SENSOR_COLUMNS]
"""The 54 whitespace-separated columns of every ``.dat`` file, in order."""

META_COLUMNS: list[str] = ["subject_id", "session"]
"""Columns added by the converter to identify the source file."""

ACTIVITY_NAMES: dict[int, str] = {
    0: "transient",
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    9: "watching_tv",
    10: "computer_work",
    11: "car_driving",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    18: "folding_laundry",
    19: "house_cleaning",
    20: "playing_soccer",
    24: "rope_jumping",
}
"""All activity IDs that may appear in the raw files (0 = transient, not a real activity)."""

PROTOCOL_ACTIVITIES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24)
"""The 12 activities of the data-collection protocol."""

STATIC_ACTIVITIES: tuple[int, ...] = (1, 2, 3)
"""Lying, sitting, standing: postures without locomotion."""

HIGH_INTENSITY_ACTIVITIES: tuple[int, ...] = (5, 24)
"""Running and rope jumping: the most vigorous protocol activities."""

SUBJECT_IDS: tuple[int, ...] = tuple(range(101, 110))
"""Subjects 101-109."""

SESSIONS: tuple[str, ...] = ("protocol", "optional")
"""``Protocol`` and ``Optional`` sub-folders of the dataset."""

IMU_RATE_HZ: float = 100.0
"""Sampling rate of the IMUs (and of the ``.dat`` rows)."""

ACC16_LIMIT: float = 16 * GRAVITY
"""Measurement range of the +-16 g accelerometer in m/s^2."""

ACC6_LIMIT: float = 6 * GRAVITY
"""Measurement range of the +-6 g accelerometer in m/s^2 (saturates during impacts)."""

GYRO_LIMIT: float = 2000.0 * 3.141592653589793 / 180.0
"""Upper bound for angular rate in rad/s (+-2000 deg/s, generous for this sensor)."""

HR_RANGE: tuple[float, float] = (30.0, 230.0)
"""Physiologically plausible heart-rate range in bpm."""

SENSOR_GROUPS: dict[str, tuple[str, ...]] = {
    "all": ("hand", "chest", "ankle", "hr"),
    "wrist_hr": ("hand", "hr"),
    "wrist": ("hand",),
    "chest_hr": ("chest", "hr"),
}
"""Named sensor subsets. ``hr`` is the heart-rate monitor, the rest are IMU locations."""

# config.py

# EAR Thresholds for drowsiness
EAR_THRESHOLD = 0.25
DROWSY_FRAMES = 48

# Blink settings
BLINK_CONSEC_FRAMES = 3

# Head movement
MAX_HEAD_TILT = 0.35  # to detect looking away/down

# Absence detection
ABSENCE_TIME_LIMIT = 10  # seconds

# Warning thresholds
MAX_WARNINGS = 3

# Paths
LOG_PATH = "backend/logs/activity_log.csv"

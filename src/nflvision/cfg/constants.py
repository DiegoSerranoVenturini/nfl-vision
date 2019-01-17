import os
from nflvision.cfg import meta


class ImageFormatConstants:

    IMG_SIZE = 256


class TrackingConstants:

    ARTIFACT_LOCATION = meta.PROJECT_HOME + "/mlflow/artifacts"
    TRACKING_URI = meta.PROJECT_HOME + "/mlflow/mlruns"

    NET_PATH = "net"


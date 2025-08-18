from datetime import datetime
from zoneinfo import ZoneInfo


def get_time_string():
    return datetime.now(tz=ZoneInfo("Asia/Seoul")).strftime("%Y_%m_%d_%H_%M_%S")

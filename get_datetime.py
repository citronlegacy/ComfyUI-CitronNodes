import datetime

class GetDateTime:
    """
    ComfyUI Node: Get DateTime
    Outputs the current date, time, and datetime as separate strings.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {}
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("Date", "Time", "DateTime")
    FUNCTION = "get_datetime"
    CATEGORY = "Utility"

    def get_datetime(self):
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        datetime_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        return (date_str, time_str, datetime_str)

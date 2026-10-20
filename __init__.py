from .get_datetime import GetDateTime
from .citron_image_saver import NODE_CLASS_MAPPINGS as IMAGE_SAVER_NODE_CLASS_MAPPINGS

NODE_CLASS_MAPPINGS = {
    "GetDateTime": GetDateTime,
    **IMAGE_SAVER_NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GetDateTime": "Get DateTime",
}

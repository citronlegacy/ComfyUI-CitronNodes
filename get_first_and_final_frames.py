
# Based on code from https://github.com/DoctorDiffusion/ComfyUI-MediaMixer

import torch

class GetFirstAndFinalFrames:
    """
    ComfyUI Node: Get First and Final Frames
    Outputs the first and final frames of a batch of images.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("First Frame", "Final Frame")
    FUNCTION = "process_images"
    CATEGORY = "Utility"

    def process_images(self, images):
        # Handle multiple images
        if images.dim() == 4:
            first_image = images[0]
            final_image = images[-1]

            # Ensure both are in batch format
            if first_image.dim() == 3:
                first_image = first_image.unsqueeze(0)
            if final_image.dim() == 3:
                final_image = final_image.unsqueeze(0)

            return (first_image, final_image)

        # Handle single image
        elif images.dim() == 3:
            img = images.unsqueeze(0)
            return (img, img)

        else:
            # Unexpected input
            return (None, None)

# Node export mappings
NODE_CLASS_MAPPINGS = {
    "get_first_and_final_frames": GetFirstAndFinalFrames
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "get_first_and_final_frames": "Get First and Final Frames"
}

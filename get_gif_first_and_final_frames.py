from PIL import Image
import numpy as np
import torch

class GetGifFirstAndFinalFrames:
    """
    ComfyUI Node: Get GIF First and Final Frames
    Accepts a path to a .gif file and outputs the first and final frames as images.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gif_path": ("STRING", {"multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("First Frame", "Final Frame")
    FUNCTION = "process_gif"
    CATEGORY = "Utility"

    def process_gif(self, gif_path):
        try:
            with Image.open(gif_path) as im:
                frames = []
                for frame in range(im.n_frames):
                    im.seek(frame)
                    frame_img = im.convert("RGB")
                    frames.append(np.array(frame_img))
                if len(frames) == 0:
                    return (None, None)
                first_frame = frames[0]
                final_frame = frames[-1]
                # Add batch axis for compatibility
                first_frame = np.expand_dims(first_frame, axis=0)
                final_frame = np.expand_dims(final_frame, axis=0)
                # Convert to torch tensors and scale to [0,1] float
                first_frame_tensor = torch.from_numpy(first_frame).float() / 255.0
                final_frame_tensor = torch.from_numpy(final_frame).float() / 255.0
                return (first_frame_tensor, final_frame_tensor)
        except Exception as e:
            print(f"Error processing GIF: {e}")
            return (None, None)

# Node export mappings
NODE_CLASS_MAPPINGS = {
    "get_gif_first_and_final_frames": GetGifFirstAndFinalFrames
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "get_gif_first_and_final_frames": "Get GIF First and Final Frames"
}

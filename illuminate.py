import os
import sys
import wave

import numpy as np
import sox
from PIL import Image as PILImage


class Audio(object):
    SAMPLE_WIDTH_TO_DTYPE = {
            1: np.int8,
            2: np.int16,
        }

    def __init__(self, path):
        self.path = path
        self.dtype = np.int16
        self.num_channels = 2
        self.sample_width = 2
        self.sample_rate = 44100
        self.audio_array = None

        self.setup_wav_data(self.path)

    def convert_to_16(self):
        new_path = 'temp16_' + os.path.basename(self.path)
        trans = sox.Transformer()
        trans.convert(bitdepth=16)
        trans.build(self.path, new_path)
        return new_path

    def setup_wav_data(self, path):
        try:
            wav_file = wave.open(path, 'r')
        except:
            # Can be due to the fact that the wave libray cannot support samples with a 32 bit-depth
            raise Exception("Unsupported file.")
        self.sample_width = wav_file.getsampwidth()

        # We unfortunately need to handle audio files with a sample width of 3
        # bytes in a special way because numpy doesn't have the 24 bit int data type.
        # For that reason, we downgrade the wave file to 16 bits using the SOX library.
        if self.sample_width == 3:
            self.path = self.convert_to_16()
            wav_file.close()
            wav_file = wave.open(self.path)
            self.sample_width = wav_file.getsampwidth()

        self.dtype = self.SAMPLE_WIDTH_TO_DTYPE[self.sample_width]
        self.sample_rate = wav_file.getframerate()
        self.num_channels = wav_file.getnchannels()
        self.audio_array = np.frombuffer(wav_file.readframes(-1), dtype=self.dtype)

        if len(self.audio_array) <= 0:
            print("Empty sample.")
        wav_file.close()

    def convert_array_to_wav(self, audio_array):
        if 'temp16' in self.path:
            self.path = self.path.strip('temp16_')
        with wave.open('new_' + os.path.basename(self.path), "w") as wav_file:
            wav_file.setnchannels(self.num_channels)
            wav_file.setsampwidth(self.sample_width)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframesraw(audio_array)


class Image(object):

    def __init__(self, path):
        self.path = path
        self.dtype = np.uint8
        self.image_array = None

        self.setup_image_data()

    def setup_image_data(self):
        with PILImage.open(self.path) as im:
            self.image_array = np.asarray(im, dtype=self.dtype)

    def convert_array_to_image(self, image_array):
        PILImage.fromarray(image_array.astype(np.uint8)).save(
            'new_' + os.path.basename(self.path)
        )


def apply_effect(np_array, factor, dtype):
    treated_array = np_array.astype(np.float16) * factor
    return treated_array.astype(dtype)


if __name__ == "__main__":
    image = Image(sys.argv[1])
    image_factor = sys.argv[2]
    try:
        image_factor = float(image_factor)
    except TypeError:
        print("Factor type must be a number.")
    treated_image_array = apply_effect(image.image_array, image_factor, image.dtype)
    image.convert_array_to_image(treated_image_array)

    a = Audio(sys.argv[3])
    audio_factor = sys.argv[4]
    try:
        audio_factor = float(audio_factor)
    except TypeError:
        print("Factor type must be a number.")
    treated_audio_array = apply_effect(a.audio_array, audio_factor, a.dtype)
    a.convert_array_to_wav(treated_audio_array)



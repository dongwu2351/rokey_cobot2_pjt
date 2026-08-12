from __future__ import annotations

import argparse

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pyaudio


def list_devices() -> None:
    audio = pyaudio.PyAudio()
    try:
        for index in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(index)
            if int(info.get("maxInputChannels", 0)) > 0:
                print(
                    f"{index}: {info.get('name')} "
                    f"(inputs={int(info['maxInputChannels'])}, "
                    f"default_rate={int(info['defaultSampleRate'])})"
                )
    finally:
        audio.terminate()


def plot_microphone(*, device_index: int | None, rate: int = 16_000) -> None:
    chunk = rate * 30 // 1_000
    audio = pyaudio.PyAudio()
    stream_args = {
        "format": pyaudio.paInt16,
        "channels": 1,
        "rate": rate,
        "input": True,
        "frames_per_buffer": chunk,
    }
    if device_index is not None:
        stream_args["input_device_index"] = device_index
    stream = audio.open(**stream_args)

    figure, axis = plt.subplots(figsize=(10, 6))
    (line,) = axis.plot(np.arange(chunk), np.zeros(chunk))
    axis.set_title("Real-Time Audio Waveform")
    axis.set_xlabel("Samples")
    axis.set_ylabel("Amplitude")
    axis.set_ylim(-(2**15), 2**15)

    def update_plot(_frame):
        data = stream.read(chunk, exception_on_overflow=False)
        line.set_ydata(np.frombuffer(data, dtype=np.int16))
        return (line,)

    _animation = animation.FuncAnimation(
        figure,
        update_plot,
        blit=True,
        interval=30,
        cache_frame_data=False,
    )
    try:
        plt.show()
    finally:
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        audio.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual microphone checker")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    args = parser.parse_args()
    if args.list_devices:
        list_devices()
        return
    plot_microphone(device_index=args.device_index, rate=args.sample_rate)


if __name__ == "__main__":
    main()

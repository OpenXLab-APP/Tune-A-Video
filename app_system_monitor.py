#!/usr/bin/env python

from __future__ import annotations

import collections

import gradio as gr
import nvitop
import pandas as pd
import plotly.express as px
import psutil


class SystemMonitor:
    MAX_SIZE = 61

    def __init__(self):
        self.devices = nvitop.Device.all()
        self.cpu_memory_usage = collections.deque(
            [0 for _ in range(self.MAX_SIZE)], maxlen=self.MAX_SIZE)
        self.cpu_memory_usage_str = ''
        self.gpu_memory_usage = collections.deque(
            [0 for _ in range(self.MAX_SIZE)], maxlen=self.MAX_SIZE)
        self.gpu_util = collections.deque([0 for _ in range(self.MAX_SIZE)],
                                          maxlen=self.MAX_SIZE)
        self.gpu_memory_usage_str = ''
        self.gpu_util_str = ''

    def update(self) -> None:
        self.update_cpu()
        self.update_gpu()

    def update_cpu(self) -> None:
        memory = psutil.virtual_memory()
        self.cpu_memory_usage.append(memory.percent)
        self.cpu_memory_usage_str = f'{memory.used / 1024**3:0.2f}GiB / {memory.total / 1024**3:0.2f}GiB ({memory.percent}%)'

    def update_gpu(self) -> None:
        if not self.devices:
            return
        device = self.devices[0]
        self.gpu_memory_usage.append(device.memory_percent())
        self.gpu_util.append(device.gpu_utilization())
        self.gpu_memory_usage_str = f'{device.memory_usage()} ({device.memory_percent()}%)'
        self.gpu_util_str = f'{device.gpu_utilization()}%'

    def get_json(self) -> dict[str, str]:
        return {
            'CPU memory usage': self.cpu_memory_usage_str,
            'GPU memory usage': self.gpu_memory_usage_str,
            'GPU Util': self.gpu_util_str,
        }

    def get_graph_data(self) -> dict[str, list[int | float]]:
        return {
            'index': list(range(-self.MAX_SIZE + 1, 1)),
            'CPU memory usage': self.cpu_memory_usage,
            'GPU memory usage': self.gpu_memory_usage,
            'GPU Util': self.gpu_util,
        }

    def get_graph(self):
        df = pd.DataFrame(self.get_graph_data())
        return px.line(df,
                       x='index',
                       y=[
                           'CPU memory usage',
                           'GPU memory usage',
                           'GPU Util',
                       ],
                       range_y=[-5,
                                105]).update_layout(xaxis_title='Time',
                                                    yaxis_title='Percentage')


def create_monitor_demo() -> gr.Blocks:
    monitor = SystemMonitor()
    with gr.Blocks() as demo:
        gr.JSON(value=monitor.update, every=1, visible=False)
        gr.JSON(value=monitor.get_json, show_label=False, every=1)
        gr.Plot(value=monitor.get_graph, show_label=False, every=1)
    return demo


if __name__ == '__main__':
    demo = create_monitor_demo()
    demo.queue(api_open=False).launch()

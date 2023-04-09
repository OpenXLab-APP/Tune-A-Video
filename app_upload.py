#!/usr/bin/env python

from __future__ import annotations

import os

import gradio as gr

from constants import MODEL_LIBRARY_ORG_NAME, UploadTarget
from uploader import upload
from utils import find_exp_dirs


def load_local_model_list() -> dict:
    choices = find_exp_dirs()
    return gr.update(choices=choices, value=choices[0] if choices else None)


def create_upload_demo() -> gr.Blocks:
    model_dirs = find_exp_dirs()

    with gr.Blocks() as demo:
        with gr.Box():
            gr.Markdown('Local Models')
            reload_button = gr.Button('Reload Model List')
            model_dir = gr.Dropdown(
                label='Model names',
                choices=model_dirs,
                value=model_dirs[0] if model_dirs else None)
        with gr.Box():
            gr.Markdown('Upload Settings')
            with gr.Row():
                use_private_repo = gr.Checkbox(label='Private', value=True)
                delete_existing_repo = gr.Checkbox(
                    label='Delete existing repo of the same name', value=False)
            upload_to = gr.Radio(label='Upload to',
                                 choices=[_.value for _ in UploadTarget],
                                 value=UploadTarget.MODEL_LIBRARY.value)
            model_name = gr.Textbox(label='Model Name')
            hf_token = gr.Text(label='Hugging Face Write Token',
                               visible=os.getenv('HF_TOKEN') is None)
        upload_button = gr.Button('Upload')
        gr.Markdown(f'''
            - You can upload your trained model to your personal profile (i.e. https://huggingface.co/{{your_username}}/{{model_name}}) or to the public [Tune-A-Video Library](https://huggingface.co/{MODEL_LIBRARY_ORG_NAME}) (i.e. https://huggingface.co/{MODEL_LIBRARY_ORG_NAME}/{{model_name}}).
            ''')
        with gr.Box():
            gr.Markdown('Output message')
            output_message = gr.Markdown()

        reload_button.click(fn=load_local_model_list,
                            inputs=None,
                            outputs=model_dir)
        upload_button.click(fn=upload,
                            inputs=[
                                model_dir,
                                model_name,
                                upload_to,
                                use_private_repo,
                                delete_existing_repo,
                                hf_token,
                            ],
                            outputs=output_message)
    return demo


if __name__ == '__main__':
    demo = create_upload_demo()
    demo.queue(api_open=False, max_size=1).launch()

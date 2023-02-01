#!/usr/bin/env python

from __future__ import annotations

import os

import gradio as gr

from constants import MODEL_LIBRARY_ORG_NAME, SAMPLE_MODEL_REPO, UploadTarget
from inference import InferencePipeline
from trainer import Trainer


def create_training_demo(trainer: Trainer,
                         pipe: InferencePipeline | None = None) -> gr.Blocks:
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown('Training Data')
                    training_video = gr.File(label='Training video')
                    training_prompt = gr.Textbox(
                        label='Training prompt',
                        max_lines=1,
                        placeholder='A man is surfing')
                    gr.Markdown('''
                        - Upload a video and write a prompt describing the video.
                        ''')
                with gr.Box():
                    gr.Markdown('Output Model')
                    output_model_name = gr.Text(label='Name of your model',
                                                max_lines=1)
                    delete_existing_model = gr.Checkbox(
                        label='Delete existing model of the same name',
                        value=False)
                    validation_prompt = gr.Text(label='Validation Prompt')
                with gr.Box():
                    gr.Markdown('Upload Settings')
                    with gr.Row():
                        upload_to_hub = gr.Checkbox(
                            label='Upload model to Hub', value=True)
                        use_private_repo = gr.Checkbox(label='Private',
                                                       value=True)
                        delete_existing_repo = gr.Checkbox(
                            label='Delete existing repo of the same name',
                            value=False)
                    upload_to = gr.Radio(
                        label='Upload to',
                        choices=[_.value for _ in UploadTarget],
                        value=UploadTarget.MODEL_LIBRARY.value)
                    gr.Markdown(f'''
                    - By default, trained models will be uploaded to [Tune-A-Video Library](https://huggingface.co/{MODEL_LIBRARY_ORG_NAME}) (see [this example model](https://huggingface.co/{SAMPLE_MODEL_REPO})).
                    - You can also choose "Personal Profile", in which case, the model will be uploaded to https://huggingface.co/{{your_username}}/{{model_name}}.
                    ''')

            with gr.Box():
                gr.Markdown('Training Parameters')
                with gr.Row():
                    base_model = gr.Text(label='Base Model',
                                         value='CompVis/stable-diffusion-v1-4',
                                         max_lines=1)
                    resolution = gr.Dropdown(choices=['512', '768'],
                                             value='512',
                                             label='Resolution',
                                             visible=False)
                num_training_steps = gr.Number(
                    label='Number of Training Steps', value=300, precision=0)
                learning_rate = gr.Number(label='Learning Rate',
                                          value=0.000035)
                gradient_accumulation = gr.Number(
                    label='Number of Gradient Accumulation',
                    value=1,
                    precision=0)
                seed = gr.Slider(label='Seed',
                                 minimum=0,
                                 maximum=100000,
                                 step=1,
                                 value=0)
                fp16 = gr.Checkbox(label='FP16', value=True)
                use_8bit_adam = gr.Checkbox(label='Use 8bit Adam', value=False)
                checkpointing_steps = gr.Number(label='Checkpointing Steps',
                                                value=1000,
                                                precision=0)
                validation_epochs = gr.Number(label='Validation Epochs',
                                              value=100,
                                              precision=0)
                gr.Markdown('''
                    - The base model must be a model that is compatible with [diffusers](https://github.com/huggingface/diffusers) library.
                    - It takes a few minutes to download the base model first.
                    - Expected time to train a model for 300 steps: 20 minutes with T4, 8 minutes with A10G, (4 minutes with A100)
                    - It takes a few minutes to upload your trained model.
                    - You may want to try a small number of steps first, like 1, to see if everything works fine in your environment.
                    - You can check the training status by pressing the "Open logs" button if you are running this on your Space.
                    ''')

        remove_gpu_after_training = gr.Checkbox(
            label='Remove GPU after training',
            value=False,
            interactive=bool(os.getenv('SPACE_ID')),
            visible=False)
        run_button = gr.Button('Start Training')

        with gr.Box():
            gr.Markdown('Output message')
            output_message = gr.Markdown()

        if pipe is not None:
            run_button.click(fn=pipe.clear)
        run_button.click(fn=trainer.run,
                         inputs=[
                             training_video,
                             training_prompt,
                             output_model_name,
                             delete_existing_model,
                             validation_prompt,
                             base_model,
                             resolution,
                             num_training_steps,
                             learning_rate,
                             gradient_accumulation,
                             seed,
                             fp16,
                             use_8bit_adam,
                             checkpointing_steps,
                             validation_epochs,
                             upload_to_hub,
                             use_private_repo,
                             delete_existing_repo,
                             upload_to,
                             remove_gpu_after_training,
                         ],
                         outputs=output_message)
    return demo


if __name__ == '__main__':
    hf_token = os.getenv('HF_TOKEN')
    trainer = Trainer(hf_token)
    demo = create_training_demo(trainer)
    demo.queue(max_size=1).launch(share=False)

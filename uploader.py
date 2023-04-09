from __future__ import annotations

import os
import pathlib
import shlex
import subprocess

import slugify
from huggingface_hub import HfApi

from constants import (MODEL_LIBRARY_ORG_NAME, URL_TO_JOIN_MODEL_LIBRARY_ORG,
                       UploadTarget)


def join_model_library_org(hf_token: str) -> None:
    subprocess.run(
        shlex.split(
            f'curl -X POST -H "Authorization: Bearer {hf_token}" -H "Content-Type: application/json" {URL_TO_JOIN_MODEL_LIBRARY_ORG}'
        ))


def upload(local_folder_path: str,
           target_repo_name: str,
           upload_to: str,
           private: bool = True,
           delete_existing_repo: bool = False,
           hf_token: str = '') -> str:
    hf_token = os.getenv('HF_TOKEN') or hf_token
    if not hf_token:
        raise ValueError
    api = HfApi(token=hf_token)

    if not local_folder_path:
        raise ValueError
    if not target_repo_name:
        target_repo_name = pathlib.Path(local_folder_path).name
    target_repo_name = slugify.slugify(target_repo_name)

    if upload_to == UploadTarget.PERSONAL_PROFILE.value:
        organization = api.whoami()['name']
    elif upload_to == UploadTarget.MODEL_LIBRARY.value:
        organization = MODEL_LIBRARY_ORG_NAME
        join_model_library_org(hf_token)
    else:
        raise ValueError

    repo_id = f'{organization}/{target_repo_name}'
    if delete_existing_repo:
        try:
            api.delete_repo(repo_id, repo_type='model')
        except Exception:
            pass
    try:
        api.create_repo(repo_id, repo_type='model', private=private)
        api.upload_folder(repo_id=repo_id,
                          folder_path=local_folder_path,
                          path_in_repo='.',
                          repo_type='model')
        url = f'https://huggingface.co/{repo_id}'
        message = f'Your model was successfully uploaded to {url}.'
    except Exception as e:
        message = str(e)
    return message

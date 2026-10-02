"""Hugging Face Hub integration for models, datasets, and benchmark publishing."""

import os
from pathlib import Path

from huggingface_hub import HfApi, login

from hcs_image_evolution.utils.logging import logger


class HuggingFaceStorage:
    """Manages secure communication with the Hugging Face Hub."""

    def __init__(self, token: str | None = None):
        self.token = token or os.environ.get("HF_TOKEN")
        if self.token:
            try:
                login(token=self.token, write_permission=True)
            except Exception as e:
                logger.warning("HF login note: %s", e)
        self.api = HfApi(token=self.token)

    def upload_model_artifact(
        self,
        repo_id: str,
        folder_path: Path,
        commit_message: str,
        private: bool = False,
    ) -> str:
        """Uploads trained model weights, GGUF files, or configs to a HF model repo."""
        self.api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        url = self.api.upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
        logger.info("Uploaded model artifact to HF Hub: %s", url)
        return str(url)

    def upload_dataset_shard(
        self,
        repo_id: str,
        file_path: Path,
        path_in_repo: str,
        commit_message: str,
        private: bool = True,
    ) -> str:
        """Uploads curated WebDataset shards or Parquet metadata to a HF dataset repo."""
        self.api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        url = self.api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )
        logger.info("Uploaded dataset file to HF Hub: %s", url)
        return str(url)

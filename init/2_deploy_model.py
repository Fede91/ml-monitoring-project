import os
import argparse
from huggingface_hub import HfApi

if __name__ == "__main__":
  # Parse the arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("--path", default="../model", help="Model path")
  args = parser.parse_args()

  # Upload the model to the Hugging Face hub
  api = HfApi(token=os.getenv("HF_TOKEN"))
  api.upload_folder(
    folder_path=args.path,
    repo_id=os.getenv("HF_REPO_ID"),
    repo_type="model",
  )
from huggingface_hub import snapshot_download
repo_id = "BAAI/Bunny-v1.0-3B" # "yzd-v/DWPose"
local_dir = "/mnt/d/HuggingFaceModels/" # "./yzd-v/DWPose/"
local_dir_use_symlinks = False                               
snapshot_download(repo_id=repo_id, local_dir=local_dir,local_dir_use_symlinks=local_dir_use_symlinks)
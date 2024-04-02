# Download the pre-trained weights of DOFA

from huggingface_hub import hf_hub_download
import os
file_path = os.path.dirname(os.path.abspath(__file__))

hf_hub_download(repo_id="XShadow/DOFA", filename="DOFA_ViT_base_e100.pth", local_dir=file_path)

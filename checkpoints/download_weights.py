# Download the pre-trained weights of DOFA

from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="XShadow/DOFA", filename="DOFA_ViT_base_e100.pth", local_dir="./")

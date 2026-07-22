MODEL = "https://huggingface.co/earthflow/DOFA/resolve/ef59b0580b0093c9bd342e0438efcee37f7b369e/DOFA_ViT_base_e100.pth"

# hubconf.py

dependencies = ["torch", "timm"]
import torch

def vit_base_dofa(pretrained=True, strict=False, **kwargs):
    # Import inside the entrypoint so Torch Hub can report a missing dependency
    # before this timm-dependent import runs.
    from dofa_v1 import vit_base_patch16

    model = vit_base_patch16(**kwargs)

    if pretrained:
        # Direct URL to your checkpoint on Hugging Face
        url = MODEL

        # Download & load
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        model.load_state_dict(state_dict, strict=strict)

    return model


#TODO: add support to more model types

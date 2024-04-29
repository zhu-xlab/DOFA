#Run the following cmd once in the shell
export PYTHONPATH=`pwd`:$PYTHONPATH

mim train mmsegmentation configs/dofa_vit_seg.py --launcher pytorch --gpus 4

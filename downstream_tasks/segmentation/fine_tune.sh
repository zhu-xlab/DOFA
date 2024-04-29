export PYTHONPATH=`pwd`:$PYTHONPATH

mim train mmpretrain configs/dofa_base_resisc45.py --launcher pytorch --gpus 4

To use the NTK and FLIR tensorboard:

1) If coming from another mmdetection env note that flir.txt has been added to the requirments and must be installed
2) Neural-Toolkit must be cloned on to your computer. Add the ROOT_DIR for neural toolkit to your PYTHONPATH env variable. Also add {NTK_ROOT_DIR}/Dataset-toolkit to your PYTHONPATH.
3) In your config file change the variable associated with `dataset_type` to 'FlirNtk'. 
4) Change the _base_ file from `../_base_/default_runtime.py` to `../_base_/default_runtime_FLIR.py` the import change to this file is the `log_config[hooks]` arg (must be `TensorboardLoggerHookFlir`)

Several customizations are allowed for the tensorboard logger. See `{mmdet_root}/datasets/flir_ntk.py` for details. These args can be altered in your config by adding them to the `evaluation` variable. The function parameter name you want to alter should the dict key and the parameter value desired should be the dict value. 

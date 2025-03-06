# Naplab 

This project contains code for parsing and structuring data collected by the naplab car. The structure integrates the data into models utilizing the nuScenes dataset. The naplab folder contains code implementations, in addition to jupyter-notebook examples on how to use.  

Modules for integrating f-theta camera model into StreamMapNet (and similar Maptrack) is provided in "hd_map_naplab_modules". 


### How to run

Clone project
```
git clone git@github.com:farup/naplab.git
```

Create env with python build in venv module:

```
python -m venv naplab
```

or env with anacoda: 

```
conda create --name naplab python
```


install requirments file

```
pip install -r naplab/requirements.txt
```


Navigate to the notebooks folder in the naplab folder.


### To convert

HD map modules with OpenMMLab utilize an earlier version numpy. To be able to correctly read the converted pickle file, we need to use the same numpy version to create it. For me this is numpy 1.24.4.


```
python3.8 -m venv naplab_conv
```

```
pip install numpy==1.24.4
```


### StreamMapNet Predicitons


![StreamMapNet](figs/gif/streammapnet.gif)


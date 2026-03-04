
CPU-only branch ...

On ubuntu 24: install venv

```
cd ~/thesis/sogmm_registration$ source ../.venv/bin/activate
sudo apt install python3.12-dev python3.12-venv
python3.12 -m venv .venv
pip install gdown tqdm rosbags pyqt6 pyridescence matplotlib scipy opencv-python future scikit-image open3d nanobind
sudo apt install libpcl-dev
```

sudo apt install gcc-9 g++-9

bash build_dry --safe --clean
bash build_wet --safe --cleam

https://robots.engin.umich.edu/nclt/

------

❯ i wanna make it work with relative paths too: ~/runs/ just because for convivent short path                                                                                
                                                                                              
● Simplest fix — just create a symlink once:                                                                                                                                 
                                         
  ln -s /home/sonieth3/thesis/sogmm_registration/wet/src/gmm_d2d_registration_examples/python/create_gmm/runs ~/runs                                                         
               
Then ~/runs/nclt_2013-01-10/... will work everywhere — not just in this script but in any tool you use.                                                                    

ln -s /home/sonieth3/thesis/sogmm_registration/wet/src/gmm_d2d_registration_examples/python/create_gmm/runs ~/runs
Create ~/runs symlink to GMM runs directory

python debug_registration.py ~/runs/nclt_2013-01-10/150_components_04030142/3880.gmm  ~/runs/nclt_2013-01-10/150_components_04030142/3881.gmm --init com
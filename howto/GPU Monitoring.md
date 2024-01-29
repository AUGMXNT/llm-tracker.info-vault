# nvtop
Despite the name, this also works with AMD GPUs and is generally the best tool for tracking what your GPUs are up to on your system 

# btop
btop now has GPU monitoring, which is convenient if you have 1 or 2 GPUs.

If you're distro doesn't have support w/ 1.3.0 you will need to compile it yourself. Luckily, it's pretty easy:
```
# https://github.com/aristocratos/btop/#compilation-linux
git clone https://github.com/aristocratos/btop
cd btop
make
sudo make install
```
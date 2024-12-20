# mamba
Use [mamba](https://github.com/conda-forge/miniforge) instead of conda. It's significantly faster.

Always create a new environment 

Currently I have ~60 mamba envs on my main dev machines. >300GB on disk, but tough early lessons that overloading envs would always lead to dll/version hell. One recently learned pro-tip. You can use a template env to speed this up: `create -n [new-env] --clone [baseml-env]`
# Tools I Depend On
## mamba
Use [mamba](https://github.com/conda-forge/miniforge) instead of conda. It's significantly faster.
### Cloning envs
Currently I have ~60 mamba envs on my main dev machines. >300GB on disk, but tough early lessons that overloading envs would always lead to dll/version hell. One recently learned pro-tip. You can use a template env to speed this up: `create -n [new-env] --clone [baseml-env]`

On my local workstation that has a reasonably fast PCIe 4.0 NVMe SSD (5GB/s+ sequential writes) it takes about **28 s** of wall time to create a clone (18GB env, 17GB of it is PyTorch):

```shell
🐟 ❯ time mamba create -n test --clone baseml
Source:      /home/lhl/mambaforge/envs/baseml
Destination: /home/lhl/mambaforge/envs/test
Packages: 25
Files: 26407

Downloading and Extracting Packages:


Downloading and Extracting Packages:

Preparing transaction: done
Verifying transaction: done
Executing transaction: done

To activate this environment, use

     $ mamba activate test

To deactivate an active environment, use

     $ mamba deactivate


________________________________________________________
Executed in   27.91 secs    fish           external
   usr time    4.18 secs  934.00 micros    4.18 secs
   sys time   13.33 secs  126.00 micros   13.33 secs
```

Wait, that's not so fast you might say, however, even with all libs cached and using `uv` it takes almost **80 s** (of just runtime mind you, no time running commands) to load even the most basic packages you'll use almost everywhere:

```shell
🐟 ❯ time mamba create -n test2 python=3.12 -y

Looking for: ['python=3.12']

conda-forge/linux-64                                        Using cache
conda-forge/noarch                                          Using cache
pkgs/main/linux-64                                          Using cache
pkgs/main/noarch                                            Using cache
pkgs/r/linux-64                                             Using cache
pkgs/r/noarch                                               Using cache
Transaction

  Prefix: /home/lhl/mambaforge/envs/test2

  Updating specs:

   - python=3.12


  Package                Version  Build               Channel           Size
──────────────────────────────────────────────────────────────────────────────
  Install:
──────────────────────────────────────────────────────────────────────────────

  + ld_impl_linux-64        2.43  h712a8e2_2          conda-forge     Cached
  + _libgcc_mutex            0.1  conda_forge         conda-forge     Cached
  + ca-certificates   2024.12.14  hbcca054_0          conda-forge     Cached
  + libgomp               14.2.0  h77fa898_1          conda-forge     Cached
  + _openmp_mutex            4.5  2_gnu               conda-forge     Cached
  + libgcc                14.2.0  h77fa898_1          conda-forge     Cached
  + libzlib                1.3.1  hb9d3cd8_2          conda-forge     Cached
  + liblzma                5.6.3  hb9d3cd8_1          conda-forge     Cached
  + libgcc-ng             14.2.0  h69a702a_1          conda-forge     Cached
  + openssl                3.4.0  hb9d3cd8_0          conda-forge     Cached
  + libexpat               2.6.4  h5888daf_0          conda-forge     Cached
  + libsqlite             3.47.2  hee588c1_0          conda-forge     Cached
  + tk                    8.6.13  noxft_h4845f30_101  conda-forge     Cached
  + ncurses                  6.5  he02047a_1          conda-forge     Cached
  + libxcrypt             4.4.36  hd590300_1          conda-forge     Cached
  + libffi                 3.4.2  h7f98852_5          conda-forge     Cached
  + bzip2                  1.0.8  h4bc722e_7          conda-forge     Cached
  + libuuid               2.38.1  h0b41bf4_0          conda-forge     Cached
  + libnsl                 2.0.1  hd590300_0          conda-forge     Cached
  + readline                 8.2  h8228510_1          conda-forge     Cached
  + tzdata                 2024b  hc8b5060_0          conda-forge     Cached
  + python                3.12.8  h9e4cc4f_1_cpython  conda-forge     Cached
  + wheel                 0.45.1  pyhd8ed1ab_1        conda-forge     Cached
  + setuptools            75.6.0  pyhff2d567_1        conda-forge     Cached
  + pip                   24.3.1  pyh8b19718_2        conda-forge        1MB

  Summary:

  Install: 25 packages

  Total download: 1MB

──────────────────────────────────────────────────────────────────────────────


pip                                                  1.2MB @  12.3MB/s  0.1s

Downloading and Extracting Packages:

Preparing transaction: done
Verifying transaction: done
Executing transaction: done

To activate this environment, use

     $ mamba activate test2

To deactivate an active environment, use

     $ mamba deactivate


________________________________________________________
Executed in    5.49 secs    fish           external
   usr time    4.11 secs  253.00 micros    4.11 secs
   sys time    0.85 secs  957.00 micros    0.84 secs

🐟 ❯ mamba activate test2

🐟 ❯ time pip install uv
Collecting uv
  Using cached uv-0.5.11-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (11 kB)
Using cached uv-0.5.11-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (15.0 MB)
Installing collected packages: uv
Successfully installed uv-0.5.11

________________________________________________________
Executed in  602.39 millis    fish           external
   usr time  453.32 millis  316.00 micros  453.00 millis
   sys time   92.58 millis   43.00 micros   92.54 millis

🐟 ❯ time uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
Using Python 3.12.8 environment at: mambaforge/envs/test2
Resolved 15 packages in 4.24s
Prepared 14 packages in 1m 02s
Installed 14 packages in 267ms
 + filelock==3.13.1
 + fsspec==2024.2.0
 + jinja2==3.1.3
 + markupsafe==2.1.5
 + mpmath==1.3.0
 + networkx==3.2.1
 + numpy==1.26.3
 + pillow==10.2.0
 + pytorch-triton-rocm==3.1.0
 + sympy==1.13.1
 + torch==2.5.1+rocm6.2
 + torchaudio==2.5.1+rocm6.2
 + torchvision==0.20.1+rocm6.2
 + typing-extensions==4.9.0

________________________________________________________
Executed in   66.65 secs    fish           external
   usr time   32.43 secs  279.00 micros   32.43 secs
   sys time   12.96 secs   32.00 micros   12.96 secs

🐟 ❯ time uv pip install transformers
Using Python 3.12.8 environment at: mambaforge/envs/test2
Resolved 17 packages in 13ms
Prepared 8 packages in 441ms
Installed 13 packages in 46ms
 + certifi==2024.12.14
 + charset-normalizer==3.4.0
 + huggingface-hub==0.27.0
 + idna==3.10
 + packaging==24.2
 + pyyaml==6.0.2
 + regex==2024.11.6
 + requests==2.32.3
 + safetensors==0.4.5
 + tokenizers==0.21.0
 + tqdm==4.67.1
 + transformers==4.47.1
 + urllib3==2.2.3

________________________________________________________
Executed in  527.05 millis    fish           external
   usr time  235.67 millis  518.00 micros  235.16 millis
   sys time  365.12 millis    0.00 micros  365.12 millis

🐟 ❯ time uv pip install xformers
Using Python 3.12.8 environment at: mambaforge/envs/test2
Resolved 13 packages in 108ms
Prepared 1 package in 1.34s
Installed 1 package in 15ms
 + xformers==0.0.28.post3

________________________________________________________
Executed in    1.48 secs      fish           external
   usr time  231.86 millis  364.00 micros  231.49 millis
   sys time  111.66 millis   42.00 micros  111.61 millis

🐟 ❯ time uv pip install triton
Using Python 3.12.8 environment at: mambaforge/envs/test2
Resolved 2 packages in 43ms
Prepared 1 package in 3.19s
Installed 1 package in 514ms
 + triton==3.1.0

________________________________________________________
Executed in    3.76 secs    fish           external
   usr time    1.43 secs    0.00 micros    1.43 secs
   sys time    0.75 secs  576.00 micros    0.75 secs
```

**NOTE:** One important note is that when you `--clone` a new env, it doesn't move over your environment variables. If you assign GPU specific stuff for example, you might want to either:

```shell
# export env to file, do what you want
conda env export -n baseml > environment.yml

# pipe template env direct to new_env
conda env export -n baseml | mamba env update -n new_env -f -
```
## uv
Implicit in that first example, but basically, I use mamba/conda since it will manage CUDA/GCC and other system libraries as well, but for Python libs, I basically stick completely to `pip` from within the mamba env. Or more specifically, nowadays I always first have [uv](https://github.com/astral-sh/uv) as the first thing installed (`pip install uv` if it's not in your `--clone`) and then run `uv pip` almost anytime I'd normally run `pip`.

I haven't aliased that yet since it will probably still get me into trouble (I have `upip` though), but you can broadly think of it as on the order of how `mamba` improves `conda` performance.

## byobu
I'm a long-time user of [byobu](https://www.byobu.org/), a wrapper against traditional [tmux](https://github.com/tmux/tmux)/[screen](https://www.gnu.org/software/screen/) terminal multiplexers. These are a **must** IMO for allowing easy attachment/detachment of long-running adhoc sessions.

Byobu has a bunch of extra quality of life additions, but I also add a few things to help. I have a custom function `b` that will also try to load up the mamba env if I'm trying to startup/reconnect to a named session. In `~/.bashrc`:

```bash
# byobu
function b() {
  if [ -z "$1" ]; then
    byobu list-session
    return 1
  fi
  byobu list-sessions | grep -q "$1" && byobu attach-session -t "$1" || byobu new-session -s "$1"
}

# Auto mamba env in tmux
if [ -n "$TMUX" ]; then
  # Get the current tmux session name
  tmux_session_name=$(tmux display-message -p '#S')
  
  # Attempt to activate the Mamba environment with the same name as the tmux session
  mamba activate $tmux_session_name 2>/dev/null
fi
```

or in `~/.config/fish/config.fish`:
```fish
function b
    if test -z "$argv[1]"
        byobu list-session
        return 1
    end

    if byobu list-sessions | grep -q "$argv[1]"
        byobu attach-session -t "$argv[1]"
    else
        byobu new-session -s "$argv[1]"
    end
end

# Try to start mamba env of same name if in tmux
# Check if inside a TMUX session
if test -n "$TMUX"
    # Get the current tmux session name
    set tmux_session_name (tmux display-message -p '#S')

    # Attempt to activate the Mamba environment with the same name as the tmux session
    # Redirect stderr to /dev/null to suppress any error messages
    mamba activate $tmux_session_name[1]
end
```
## Starship
I use [Starship](https://starship.rs/), a simple/fast cross-shell prompt. For the past few years I've mostly used `fish` as my shell, but often need to hop into `bash` for POSIX compatibility (I will probably see if I can switch to `zsh` and get everything I'm used to in `fish` at some point).

I'm not really a power user or a ricer, but it has a lot of nice stuff built in. One customization I use in `~/.config/starship.toml` to help me tell what shell I'm in:

```
[shell]
fish_indicator = "🐟"
zsh_indicator = "ℤ"
bash_indicator = "\\$"
```

## Atuin
 [Atuin](https://atuin.sh/) is a better history manager that I've been using for a few years now, and despite having to do a lot of configuration to get/keep it working how I want, I couldn't imagine going back to anything that has *less* functionality. It really pains me when I am on a terminal system without this.

In my `~/.config/atuin/config.toml` the major changes I make from defaults:

```
# I almost always want to find the last thing I typed *in each session*, not globally
filter_mode_shell_up_key_binding = "session"

# when I search I usually want to see my current context
inline_height = 16

# Maybe some people want to just execute, but I like to treat the picker like a picker and either execute or edit
enter_accept = false
```

I don't use the cloud syncing nor have I ever bothered to setup a [my own server](https://docs.atuin.sh/self-hosting/server-setup/) - at some point maybe there will be a smart SSH-based lazy-sync or something but for now, I've been OK with not having the sync

# Tools I Want to Integrate

## llm
There are a bunch of command-line LLM tools but simonw's [llm](https://github.com/simonw/llm) is probably the most flexible/mature. I've yet to really integrate it (or any related tools into my workflow)
## wut
This is still a little underbaked atm but still a great idea:
https://github.com/shobrook/wut
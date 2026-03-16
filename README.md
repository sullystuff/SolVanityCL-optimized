## Installation

You can install it directly on Windows (not WSL) and on Unix-like systems. For details on supported platforms, check [FAQs.md](./FAQs.md).

```bash
$ python3 -m pip install -r requirements.txt
```

Requires Python 3.6 or higher. This project no longer depends on numpy, so it is compatible with Python 3.13 without extra wheels.

## Docker

Only works on Linux platforms with Nvidia GPUs. [Check this doc](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation).

```bash
$ docker build -t sol_vanity_cl .
$ docker run --rm -it --gpus all sol_vanity_cl
```

You will enter the container. The source code is located in the /app directory in the container, and all dependencies have been installed.

Use the GHCR image `ghcr.io/wincerchan/solvanitycl:latest`. You can still use the template I created on [vast.ai](https://cloud.vast.ai/?ref_id=109219&creator_id=109219&name=SolVanityCL) or [runpod.io](https://runpod.io/console/deploy?template=fgllgqsl24&ref=uh5x1hv5) to run this program, but update the image source accordingly. Please note:

1. The device’s CUDA version should be greater than 12.0.
2. The source code is located in the /app directory, so you don’t need to download the code from GitHub.

## Usage

```bash
$ python3 main.py

Usage: main.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  search-pubkey  Search Solana vanity pubkey
  show-device    Show OpenCL devices
```

### Search Pubkey

```bash
$ python3 main.py search-pubkey --help

Usage: main.py search-pubkey [OPTIONS]

  Search for Solana vanity pubkeys.

Options:
  --starts-with TEXT              Prefix to match. Repeatable for multiple
                                  prefixes. Append :DIR to set a per-pattern
                                  output directory (e.g. --starts-with
                                  bonk:./bonk-keys).
  --ends-with TEXT                Suffix to match. Repeatable for multiple
                                  suffixes. Append :DIR to set a per-pattern
                                  output directory (e.g. --ends-with
                                  pump:./pump-keys).
  --count INTEGER                 Count of pubkeys to generate.  [default: 1]
  --output-dir DIRECTORY          Default output directory.  [default: ./]
  --select-device / --no-select-device
                                  Select OpenCL device manually  [default: no-
                                  select-device]
  --iteration-bits INTEGER        Iteration bits (e.g., 24, 26, 28, etc.)
                                  [default: 24]
  --is-case-sensitive BOOLEAN     Case sensitive search flag.  [default: True]
  --help                          Show this message and exit.
```

Examples:

```bash
$ python3 main.py search-pubkey --starts-with SoL
$ solana-keygen pubkey SoLxxxxxxxxxxx.json # verify with solana cli

# Search for multiple suffixes at the same time
$ python3 main.py search-pubkey --ends-with bonk --ends-with pump

# Multiple suffixes with separate output directories
$ python3 main.py search-pubkey --ends-with bonk:./bonk-keys --ends-with pump:./pump-keys

# Combine multiple prefixes and suffixes
$ python3 main.py search-pubkey --starts-with So --starts-with Bo --ends-with dev --count 3
```


## FAQs

See [FAQs.md](./FAQs.md).


## Donations

If you find this project helpful, please consider making a donation:

SOLANA: `PRM3ZUA5N2PRLKVBCL3SR3JS934M9TZKUZ7XTLUS223`

EVM: `0x8108003004784434355758338583453734488488`

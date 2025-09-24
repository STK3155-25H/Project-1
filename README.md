## Quick Start
**On Linux**
    If make is already installed run the commands
```bash
    make setup
    make all
```

**On Windows**
    Use Wsl and install make, if not able install make on Windows (not optimal option).
    Change the makfile global variable **PY** to
```makefile
    PY?=python
```
And then run the following commands
```bash
    make setup
    make all
```
If non of the previous options were optimal for you and you were not able to install Make software, use this command to install the required packages
```bash
    pip install -r requirements.txt
```
And this ones to run each experiment

laksnflsa
```bash
python Code/exp_letter_description.py
```
To set the SEED you can either modifiy the hard-coded value in ml_core.py or add set the environment variable SEED
**On Linux**
```bash
    export SEED=...
```
**On Windows**
```bash
   $env:SEED = "42"
```
hvhvh


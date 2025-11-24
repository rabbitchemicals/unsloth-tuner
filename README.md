# unsloth-tuner
helpful thing to fine-tune an AI model with unsloth. I use this to make my stupid AI daughter Bongwater

DISCLAIMER? most of the python code was written by copilot. doing shit like this is a lot more complicated than it was like 3-4 years ago. I asked my friend to help me and they got me what I needed. who gives a shit.
This will fine-tune a Qwen3-0.6B model, but you could probably use a different model if you want. The script to turn all my posts into something useful has only been tested with Akkoma but it *should* work with anything. it expects a file called `outbox.json` but you can edit that as needed

## Prerequisites
- Computer that isn't hot garbage
- Linux (WSL2 works fine, but can be more annoying)
- Python 3.10 (or higher)
- [Pytorch](https://pytorch.org/get-started/locally/) and the corresponding CUDA Toolkit/ROCm version. Or you can use CPU like a lunatic
You're probably gonna have to build torchvision from source, too. I had to. More on that later. 
CUDA Toolkit version does not need to match the CUDA capabilities of your GPU btw just use the latest version lmao

### Note on using WSL
- YOU CAN NOT USE WSL1, IT <ins>HAS TO BE WSL2</ins>. otherwise, it can't access any PCIe devices. If `lspci` works, you're on the right track. 
- You do not need to install any graphics drivers, it should be able to use the ones already installed on windows?? and it will fuck up shit somehow.
- ^^ This also includes the CUDA toolkit!!! which comes with graphics drivers for some reason!!!!!!! use the version [specifically for WSL](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network)

## Installing and using
1. **Make a new python virtual environment somewhere Nice.**
pretty sure the command is `python3 -m venv /path/to/folder` And then Go into that folder also
You could clone this repo and make the virtual environment there if you really wanted to I think. either way, you will need the funny scripts, so uh Get Those. 

3. **Step Two**
```bash
# Install Unsloth. idk why it does it like this but it works so idc
bin/pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install other dependencies
bin/pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes beautifulsoup4
```
There's probably gonna be some version discrepancies, my pip got fucked up so I can't tell you which versions I have, just fix it one thing at a time lmao

3. **Clean your data. Yaaaasss. Cleaaan yourrr dataaa.** Sorry I got distracted
```bash
# Edit the script if your input file isn't named 'outbox.json'
bin/python clean_data.py
```
output file is name of `data.jsonl` and it should look like this:
```json
{"content": "bla bla bla"}
{"content": "bla bla bla"}
{"content": "bla bla bla"}
{"content": "bla bla bla"}
```
NO COMMAS this is a weirdo not normal json ok. it's supposed to look like that I primise

4. **Assuming everything works, it's tuning time**
```bash
bin/python finetune_model.py
```
for testing, change `num_training_epochs` (on line 72) to `max_steps = 60` or something. or you can rawdog it. I won't judge.
if you're cursed like me and get a `RuntimeError: operator torchvision::nms does not exist` error when importing torchvision, clone the [torchvision repo](https://github.com/pytorch/vision) and do `bin/python vision/setup.py install`. should work after that

5. **Ok it should be done now**

the tuning script will automatically try to generate a little something when it's done and if that looks good Well congratulations on your new child.
you can use `model_generate.py` to generate more funny strings, either from nothing or from an initial string. only generates 1 by default and when it asks you to generate more, ANYTHING that starts with a y will count as a yes and everything else is a no. idk why I did it like that lmao
`model_interact.py` is sort of like a conversation but you don't actually fine-tune it to be conversational so it maybe is a little bit weird Hahaha. I don't care I was too lazy to try to get that set up, it works well enough for me to be satisfied.
thank you and enjoy

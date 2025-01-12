# Installation
```bash
sudo apt-get install espeak-ng -y
sudo apt update -y
sudo apt-get install git-lfs -y
git lfs install
git clone https://huggingface.co/hexgrad/Kokoro-82M kokoro_swan -y
cd kokoro_swam
/usr/bin/python3.12 -m venv .kokoro_venv
source .kokoro_venv/bin/activate
```
```python
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
#pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
pip install phonemizer transformers scipy munch
```

# Issues
- the `torch` stable `cpu` only doesn't work as it seems like it doesn't implement the `weights_only` feature, that is why i have installed a `nightly` version which has more `experiemental` features and it worked fine. `CHATGPT` has been *'S*****'* on that. I checked Google/Youtube(also Google lol) and found the solution.
- need to install `git lfs` (large file size), I have omitted that and got some issues. so install it `sudo apt install git-lfs -y`


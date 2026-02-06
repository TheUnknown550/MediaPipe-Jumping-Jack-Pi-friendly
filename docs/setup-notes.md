# Setup Notes

## 1) Build deps + pyenv (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install -y build-essential curl git \
  libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
curl https://pyenv.run | bash
```

## 2) Initialize pyenv in your shell
Add to `~/.bashrc` (or `~/.zshrc`):
```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```
Reload your shell: `source ~/.bashrc`

## 3) Install Python 3.12 and create a venv
```bash
pyenv install 3.12.7
pyenv local 3.12.7   # inside your project folder
python -m venv .venv
source .venv/bin/activate
python --version
```

## 4) Install MediaPipe
```bash
pip install --upgrade pip
pip install mediapipe
```

## 5) Quick sanity check
```bash
python -c "import mediapipe as mp; print(mp.__version__)"
```

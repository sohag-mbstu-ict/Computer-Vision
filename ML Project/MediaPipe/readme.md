### 1. Create a Virtual Environment
Create an isolated Python environment to manage dependencies:
```bash
python -m venv env
```
## Dependencies:
```bash
Python==3.12.3
mediapipe==0.10.21
opencv-python==4.12.0.88
```

### 2. Activate the Virtual Environment
```bash
Windows:

.\env\Scripts\activate

Linux / macOS:

source env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Inference
```bash
python3 inference.py --video input.mp4 --output out
```
--video specifies the path to the input video.

--output specifies the path/folder where the output will be saved.

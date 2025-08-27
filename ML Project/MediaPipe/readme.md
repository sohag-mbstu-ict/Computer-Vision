### 1. Create a Virtual Environment
Create an isolated Python environment to manage dependencies:
```bash
python -m venv env
```
### 2. Activate the Virtual Environment

Windows:

.\env\Scripts\activate

Linux / macOS:

source env/bin/activate

### 3. Install Dependencies

pip install -r requirements.txt

### 4. Run Inference

python3 inference.py --video input.mp4 --output out


--video specifies the path to the input video.

--output specifies the path/folder where the output will be saved.

5. Deactivate the Virtual Environment

Once you're done, you can deactivate the environment:

deactivate


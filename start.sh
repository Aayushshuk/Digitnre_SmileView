pip uninstall -y opencv-python opencv-contrib-python || true
pip install --no-cache-dir -r requirements.txt
python app.py

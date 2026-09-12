"""Check the built wheel outside the source checkout using installed dependencies."""

import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    wheel = sorted((root / "dist").glob("sat_clas-*.whl"))[-1]
    with tempfile.TemporaryDirectory() as directory:
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(directory)
        environment = os.environ.copy()
        environment.update(PYTHONPATH=directory, SAT_CLAS_HOME=directory)
        environment.pop("SAT_CLAS_CONFIG", None)
        check = '''
from src.utils.configuration import load_config
from src.interface.app import app, STATIC
from fastapi.testclient import TestClient
client = TestClient(app)
assert client.get("/").status_code == 200
assert client.get("/static/app.js").status_code == 200
assert load_config()["sam3"]["model_id"] == "facebook/sam3"
response = client.post("/api/segment", files={"image": ("broken.png", b"not an image", "image/png")})
assert response.status_code == 400, response.text
print("ISOLATED WHEEL PASS", STATIC)
'''
        subprocess.run([sys.executable, "-c", check], cwd=directory, env=environment, check=True)


if __name__ == "__main__":
    main()

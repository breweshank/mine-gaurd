import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
LICENSE_FILE = os.path.join(BACKEND_DIR, "license.json")

def check_license():

    if not os.path.exists(LICENSE_FILE):
        raise Exception("License file missing")

    with open(LICENSE_FILE) as f:
        data = json.load(f)

    if data.get("valid") != True:
        raise Exception("License invalid")

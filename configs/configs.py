# -*- coding: utf-8 -*-
# """
# configs.py
# Created on March 4, 2025
# @ Author: Mazhar
# """

import logging
import os
from pathlib import Path

import yaml

logger: logging.Logger = logging.getLogger(name="app.logs")

# CFGS: str = os.path.abspath(path="./configs/cfgs.yaml")  # Use absolute path
CFGS: str = os.path.join(Path(__file__).parent, "cfgs.yaml")  # Use absolute path

print(CFGS)

cfgs: dict = {}
try:
    with open(file=CFGS, mode="r") as file:
        cfgs: dict = yaml.safe_load(stream=file)
except FileNotFoundError:
    logger.error(msg=f"{CFGS} not found.")
    cfgs = {}
finally:
    if not cfgs:
        logger.error(msg="No configuration found in the YAML file.")
        raise ValueError("No configuration found in the YAML file. Cannot continue.")

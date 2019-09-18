from configparser import ConfigParser

from puzzlesolver.utils import get_project_root

PROJECT_ROOT = get_project_root()
config = ConfigParser()
CONFIG_FILE = os.path.join(PROJECT_ROOT, "puzzlesolver/classifiers/config.ini")
config.read(CONFIG_FILE)

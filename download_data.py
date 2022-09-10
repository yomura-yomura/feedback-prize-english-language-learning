import pathlib
import kaggle
import zipfile
import tqdm
import sys


root_path = pathlib.Path(__file__).resolve().parent
data_root_path = root_path / "data"


kaggle_api = kaggle.KaggleApi()
kaggle_api.authenticate()
kaggle_api.competition_download_files("feedback-prize-effectiveness", path=data_root_path, quiet=False)
downloaded_path = data_root_path / "feedback-prize-effectiveness.zip"

with zipfile.ZipFile(downloaded_path) as f:
    for member in tqdm.tqdm(f.infolist(), desc="Extracting", file=sys.stdout):
        f.extract(member, data_root_path / "feedback-prize-effectiveness")

downloaded_path.unlink()

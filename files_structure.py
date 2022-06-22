import glob
from pathlib import Path


def get_files_structure(main_dir, extension):
    if not (isinstance(main_dir, str) and Path(main_dir).is_dir()):
        raise Exception("Please specify main_dir in config file")
    data_dir = Path(main_dir)
    files_structure = {}
    all_filepaths = sorted(data_dir.rglob(f"*{extension}"))
    data_rel_dirs = sorted(set(filepath.parent.relative_to(data_dir) for filepath in all_filepaths))
    for directory in data_rel_dirs:
        filepaths = (main_dir / directory).glob(f"*{extension}")
        filenames = [filepath.name for filepath in filepaths]
        files_structure[str(directory)] = sorted(filenames)
    return files_structure

import glob
import os


def get_files_structure(main_dir, extension):
    if not (isinstance(main_dir, str) and os.path.isdir(main_dir)):
        raise Exception("Please specify main_dir in config file")
    files_structure = {}
    all_filepaths = sorted(
        glob.glob(os.path.join(main_dir + f"/**/*{extension}"), recursive=True)
    )
    data_abs_dirs = set(os.path.dirname(filepath) for filepath in all_filepaths)
    data_rel_dirs = [os.path.relpath(dir, main_dir) for dir in data_abs_dirs]
    dirs = sorted(data_rel_dirs)
    for directory in dirs:
        filepaths = glob.glob(os.path.join(main_dir, directory) + f"/*{extension}")
        filenames = [os.path.basename(filepath) for filepath in filepaths]
        files_structure[directory] = sorted(filenames)
    return files_structure

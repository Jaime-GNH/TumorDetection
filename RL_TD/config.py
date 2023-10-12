import os

config = {
    "Data": {
        "Loader": {
            "folder_path": [dir_path for dir_path, dir_name, _ in os.walk(os.getcwd()) if 'resources' in dir_name][0]
        },
    },
    "Model": {}
}

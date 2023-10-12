import os

config = {
    "Data": {
        "Loader": {
            "folder_path": [
                os.path.join(dir_path, 'Dataset_BUSI_with_GT')
                for dir_path, dir_name, _ in os.walk(os.getcwd())
                if 'Dataset_BUSI_with_GT' in dir_name
            ]
        },
    },
    "Model": {}
}

import os

config = {
    "Data": {
        "Loader": {
            "folder_path": [dirpath for dirpath, dirname, _ in os.walk(os.getcwd()) if 'resources' in dirname][0]
        },
    },
    "Model": {}
}

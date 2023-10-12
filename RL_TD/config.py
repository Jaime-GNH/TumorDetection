import os

config = {
    "Loader": {
        "dir_path": [
            os.path.join(dir_path, 'Dataset_BUSI_with_GT')
            for dir_path, dir_name, _ in os.walk(os.getcwd())
            if 'Dataset_BUSI_with_GT' in dir_name
        ][0],
        "init": {
            "imgs_regex": r'\*\*).png'
        },
        "call": {
            "classes": {
                "normal": 0,
                "benign": 1,
                "malignant": 2
            },
            "join_classes": None,  # {['bening','malignant']: 'tumor'}
            "add_masks": True
        }
    },
    "Model": {}
}

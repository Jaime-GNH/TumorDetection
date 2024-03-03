import os


class WorkingDir:
    """
    Working Directory finder and setter.
    """
    default_path = '\\'.join(
        os.path.abspath(os.getcwd()).split('\\')[
            :os.path.abspath(os.getcwd()).split('\\').index('TumorDetection') + 1
        ]
    )

    @classmethod
    def set_wd(cls) -> str:
        """
        Sets the working directory at the top of the project.
        :return: Current working directory.
        """
        current_path = os.path.abspath(os.getcwd())
        os.chdir(
            cls.default_path
        )
        return current_path

    @classmethod
    def getwd_from_path(cls, path: str) -> str:
        """
        Get default working directory from another directory.
        :return: working directory.
        """
        return os.path.abspath(
            os.path.join(os.getcwd(),
                         os.path.relpath(cls.default_path, start=path)
                         )
        )


if __name__ == '__main__':
    print(WorkingDir.getwd_from_path(os.getcwd()))

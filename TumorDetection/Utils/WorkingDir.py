import os


class WorkingDir:
    default_path = '\\'.join(
        os.path.abspath(os.getcwd()).split('\\')[
            :os.path.abspath(os.getcwd()).split('\\').index('TumorDetection') + 1
        ]
    )

    @classmethod
    def set_wd(cls):
        """
        Sets the working directory at the top of the project.
        :return:
        """
        current_path = os.path.abspath(os.getcwd())
        os.chdir(
            cls.default_path
        )
        return current_path

    @classmethod
    def getwd_from_path(cls, path):
        """

        :return:
        """
        return os.path.abspath(
            os.path.join(os.getcwd(),
                         os.path.relpath(cls.default_path, start=path)
                         )
        )


if __name__ == '__main__':
    print(WorkingDir.getwd_from_path(os.getcwd()))

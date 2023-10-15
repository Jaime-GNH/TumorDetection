import cv2

# TODO: Superponer imágenes bien.
#  Generar varias ventanas independientes.
#  Mascaras:
#   Escalar valores a 255.
#   Ver como las sperpone la gente
def show_images(images, win_title='Viewer'):
    """
    Function to show a set of images
    :param images: list(np.ndarray) or (np.ndarray)
        images or image to show
    :param win_title: (str)
        Window title
    """
    if not isinstance(images, (tuple, list)):
        images = [images]

    for image in images:
        cv2.imshow(win_title, image)

    cv2.waitKey(0)
    cv2.destroyWindow(win_title)

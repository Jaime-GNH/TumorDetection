
class PatchDataset(BaseClass):
    """
    Makes a Batched dataset from a set of images and masks
    """
    def __call__(self, images, masks, **kwargs):
        """

        :param images: List(np.array)
        :param masks: List(np.array)
        :param kwargs:
        :return:
        """

        kwargs = self._default_config(PatchDatasetCall, **kwargs)
        assert (kwargs.get('resize_dim')[0] - kwargs.get('patch_dim')[0]) % kwargs.get('patch_step') == 0, \
            f'(Resize dim - patch dim) % patch_step must be 0.'
        assert (kwargs.get('resize_dim')[1] - kwargs.get('patch_dim')[1]) % kwargs.get('patch_step') == 0, \
            f'(Resize dim - patch dim) % patch_step must be 0.'

        if kwargs.get('resize'):
            images = [cv2.resize(image, kwargs.get('resize_dim'),
                                 interpolation=kwargs.get('interpolation_method')) for image in images]
            masks = [cv2.resize(mask, kwargs.get('resize_dim'),
                                interpolation=cv2.INTER_NEAREST_EXACT) for mask in masks]
        # num_classes = np.max(np.array(masks))
        if kwargs.get('normalize'):
            images = [image.astype('float') / 255. for image in images]

        im_tr, im_ts, mask_tr, mask_ts = train_test_split(np.stack(images), np.stack(masks),
                                                          test_size=kwargs.get('test_size'),
                                                          shuffle=kwargs.get('shuffle'),
                                                          random_state=kwargs.get('random_state'))
        if kwargs.get('mode') == 'images':
            return im_tr, im_ts, mask_tr.squeeze(), mask_ts.squeeze()
            # return (im_tr, im_ts,
            #         keras.utils.to_categorical(mask_tr, num_classes=num_classes),
            #         keras.utils.to_categorical(mask_ts, num_classes=num_classes))
        elif kwargs.get('mode') == 'patches':
            x_tr = np.concatenate([self.make_patches(image,
                                                     kwargs.get('patch_dim'),
                                                     kwargs.get('patch_step')).reshape(-1, *kwargs.get('patch_dim'), 1)
                                   for image in im_tr])
            y_tr = np.concatenate([self.make_patches(mask,
                                                     kwargs.get('patch_dim'),
                                                     kwargs.get('patch_step')).reshape(-1, *kwargs.get('patch_dim'), 1)
                                   for mask in mask_tr])

            x_ts = np.concatenate([self.make_patches(image,
                                                     kwargs.get('patch_dim'),
                                                     kwargs.get('patch_step')).reshape(-1, *kwargs.get('patch_dim'), 1)
                                   for image in im_ts])
            y_ts = np.concatenate([self.make_patches(mask,
                                                     kwargs.get('patch_dim'),
                                                     kwargs.get('patch_step')).reshape(-1, *kwargs.get('patch_dim'), 1)
                                   for mask in mask_ts])
            if kwargs.get('filter_empty'):
                pass
                idx = np.where(np.max(y_tr, axis=(1, 2)).flatten() != 0)
                x_tr = x_tr[idx]
                y_tr = y_tr[idx]

            return x_tr, x_ts, y_tr.squeeze(), y_ts.squeeze()
            # return (x_tr, x_ts,
            #         keras.utils.to_categorical(y_tr, num_classes=num_classes+1),
            #         keras.utils.to_categorical(y_ts, num_classes=num_classes+1))
        else:
            raise ValueError(f'mode must be one of: "images", "patches". Got {kwargs.get("mode")}')

    @staticmethod
    def make_patches(image, patch_dim, patch_step):
        """
        (HxW) -> (PHxPWxHxW)
        :return:
        """
        return patchify(image,
                        patch_dim,
                        patch_step)

    @staticmethod
    def unpatch_image(patches, image_dim):
        """
        (PHxPWxHxW) -> (HxW)
        :param patches:
        :param image_dim:
        :return:
        """
        return unpatchify(patches, image_dim)

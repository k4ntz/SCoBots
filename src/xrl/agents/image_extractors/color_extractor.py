import numpy as np
from .utils import find_objects, mark_point, repeat_upsample, load_game_dict, \
                   plot_with_hover, _increment_string
from PIL import Image

n_times = 0


def make_square(im, size, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    new_im = Image.new('RGB', size, fill_color)
    new_im.paste(im, (int((size[0] - x) / 2), int((size[1] - y) / 2)))
    return new_im


class ColorExtractor():
    """
    Color Extractor module that extracts features.

    """
    def __init__(self, objects_colors=None, game=None, *args, **kwargs):
        """
        Constructor.

        Args:

        """
        if objects_colors is not None:
            self.objects_colors = objects_colors
            self.game = "Unknown"
        elif game is not None:
            try:
                game_dict = load_game_dict(game)
                self.objects_colors = game_dict["colors"]
                self.splitted_objects = game_dict["splitted_objects"]
                self.game = game
                if "obj_size" in game_dict:
                    self.obj_size = game_dict["obj_size"]
                else:
                    self.obj_size = None
                self.max_obj = game_dict["max_obj"]
                self.img_shape = (210, 160)
                self.divide = max(*self.img_shape)
                self.obj_shape = None
                self.memory = []
                self.auto_change = True
                self._fill_memory = True
                self._calling_function = self._filling_memory
                self.z_what_size = 32
            except KeyError as err:
                msg = f"\nGame {game} not supported by ColorExtractor\n" + \
                       "please add it to: games_configs.yaml\n"
                print(msg)
                raise err
        else:
            raise ValueError("You need to give objects_colors or game to Color extractor")
        self.show_objects = False

    @property
    def fill_memory(self):
        return self._fill_memory

    @fill_memory.setter
    def fill_memory(self, new_value):
        if new_value:
            self._fill_memory = True
            self._calling_function = self._filling_memory
        else:
            self._fill_memory = False
            self.run_pca_on_memory(plot=False)
            self._calling_function = self.classify

    def _filling_memory(self, image):
        positions, boxes, types = find_objects(image, self.objects_colors, size=self.obj_size,
                                               splitted_objects=self.splitted_objects,
                                               mark_objects=self.show_objects)
        self.memory.extend(boxes)
        global n_times
        n_times += 1
        if self.show_objects and n_times > 200:
            import matplotlib.pyplot as plt
            ax = plt.gca()
            ax.imshow(image, interpolation='nearest', aspect="equal")
            ax.set_axis_off()
            plt.tight_layout()
            plt.show()
        # if n_times > 200 and self.auto_change:
        #     print("Changing from fill memory mode to extract object mode")
        #     self.run_pca_on_memory(plot=True) # plot a 2D PCA of current collected objects
        #     self.fill_memory = False
        ret = {}
        for type, pos in zip(types, positions):
            while type in ret.keys():
                type = _increment_string(type)
            ret[type] = pos
        #print(ret); exit()
        return ret


    def classify(self, image):
        # To be reworked
        all_omage_descriptions = []
        objects_in_image = []
        pos_and_z_whats = []
        positions, boxes, types = find_objects(image, self.objects_colors, size=self.obj_size,
                                        splitted_objects=self.splitted_objects,
                                        mark_objects=self.show_objects)
        for pos, omage in np.array(objects_in_image).T:
            pos_and_z_whats.append((pos, self.pca.predict(omage)))
            all_omage_descriptions.append(pos_and_z_whats)
        return all_omage_descriptions

    def _calling_function(self, images):
        pass

    def run_pca_on_memory(self, plot):
        from operator import itemgetter
        shapes = [om.shape for om in self.memory]
        self.max_size = (max(shapes, key=itemgetter(0))[0],
                    max(shapes, key=itemgetter(1))[1])
        resized_images = [self.flatten_center_image(om) for om in self.memory]
        from sklearn.decomposition import PCA
        if plot:
            self.pca = PCA(n_components=2)
            # self.pca = PCA(0.95)
            self.components = self.pca.fit_transform(resized_images)
            modes = self.get_modes_from_memory()
            pruned = self.prune_pca(self.components, modes, self.memory)
            plot_with_hover(*pruned)
        else:
            self.pca = PCA(n_components=32)
            self.components = self.pca.fit_transform(resized_images)

    def prune_pca(self, components, modes, omages):
        f_comp, f_modes, f_omages = [], [], []
        for com, mod, oma in zip(components, modes, omages):
            if list(com) not in f_comp:
                f_comp.append(list(com))
                f_modes.append(mod)
                f_omages.append(oma)
        return f_comp, f_modes, f_omages


    def flatten_center_image(self, img):
        f = np.zeros((*self.max_size, 3), np.uint8)
        ax, ay = (self.max_size[1] - img.shape[1])//2, (self.max_size[0] - img.shape[0])//2
        f[ay:img.shape[0]+ay, ax:ax+img.shape[1]] = img
        return f.flatten()

    def __call__(self, images):
        if len(images.shape) == 3:
            batch_size = 1
        elif len(images.shape) == 5:
            batch_size = images.shape[0]
            images = images.reshape(-1, 210, 160, 3)
        all_images_objects = self._calling_function(images)
        if batch_size == 1:
            return all_images_objects

        return all_images_objects.reshape(batch_size, -1)

    def get_modes_from_memory(self):
        modes = []
        for om in self.memory:
            values, counts = np.unique(om.reshape(-1, 3), return_counts=True, axis=0)
            modes.append(values[np.argmax(counts)])
        return modes

    @property
    def __name__(self):
        return f"Color Extractor on game {self.game}"


def show_img(img):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()

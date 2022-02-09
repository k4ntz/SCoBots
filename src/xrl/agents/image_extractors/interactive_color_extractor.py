import numpy as np
from .utils import find_objects, mark_point, repeat_upsample, load_game_dict, \
    use_existing
from collections import OrderedDict
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

NEW_CLASSNAME = None

class IColorExtractor():
    """
    Interactive Color Extractor module that extracts features.

    """
    attributes_to_save = ["game", "objects_colors", "splitted_objects",
                          "obj_size", "max_obj", "img_shape", "divide",
                          "obj_features", "imp_objects", "n_times",
                          "show_objects", "interactive"]

    def __init__(self, objects_colors=None, game="Unknown", image=None,
                 path=None, load=False):
        """
        Constructor.

        Args:

        """
        self.show_objects = False
        if objects_colors is not None:
            self.objects_colors = objects_colors
            self.game = "Unknown"
        elif game is not "Unknown":
            if path is None:
                path = os.path.join(os.path.dirname(__file__),
                                    f"ICE/{game}.ice")
            if load:
                self.load(path)
                return
            elif os.path.exists(path) and use_existing(path):
                self.load(path)
                return
            try:
                game_dict = load_game_dict(game)
                self.objects_colors = list(game_dict["colors"].values())
                self.splitted_objects = game_dict["splitted_objects"]
                self.game = game
                self.obj_size = game_dict["obj_size"]
                # self.max_obj = int(input('What is the maximum number of objects ?'))
                self.max_obj = 20
                self.img_shape = (210, 160)
                self.divide = max(*self.img_shape)
                self.obj_features = 2
                self.imp_objects = ObjDict(self.obj_features, self.max_obj)
                self.n_times = 0
            except KeyError as err:
                msg = f"\nGame {game} not supported by ColorExtractor\n" + \
                       "please add it to: games_configs.yaml\n"
                print(msg)
                raise err
        else:
            print("Empty ICE. You need to give objects_colors or game to Color extractor")
        if image is not None:
            self.discarded_zone = discard_zone(image)

        self.interactive = True

    def __call__(self, images):
        if len(images.shape) == 3:
            batch_size = 1
        elif len(images.shape) == 5:
            batch_size = images.shape[0]
            images = images.reshape(-1, 210, 160, 3)
        all_images_objects = self._calling_function(images)
        if batch_size == 1:
            return np.array(all_images_objects)
        return np.array(all_images_objects).reshape(batch_size, -1)

    def _calling_function(self, images):
        pass

    @property
    def interactive(self):
        return self._interactive

    @interactive.setter
    def interactive(self, new_value):
        if new_value:
            self._interactive = True
            self._calling_function = self._interactive_call
        else:
            self._interactive = False
            self._calling_function = self._operating_call

    def _operating_call(self, image):
        img_objects = {}
        positions, boxes = find_objects(image, self.objects_colors, size=self.obj_size,
                                        splitted_objects=self.splitted_objects,
                                        mark_objects=self.show_objects)
        obj_classes = self.imp_objects.find_categories(boxes)
        img_objects = dict(zip(obj_classes, positions))
        return img_objects

    def _interactive_call(self, image):
        positions, boxes = find_objects(image, self.objects_colors,
                                        size=self.obj_size,
                                        splitted_objects=self.splitted_objects,
                                        mark_objects=self.show_objects)
        self.imp_objects.fill(boxes, positions, image)
        self.n_times += 1
        if self.show_objects and self.n_times > 200:  # delays
            plt.imshow(image, interpolation='nearest', aspect="equal")
            plt.tight_layout()
            plt.show()
        return None

    def show(self):
        for name, obj in self.imp_objects.items():
            import ipdb; ipdb.set_trace()

    def save(self, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__),
                                f"ICE/{self.game}.ice")
        if "/" in path:
            dir_path = "/".join(path.split("/")[:-1])
            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)
        config_dictionary = {a: getattr(self, a) for a in self.attributes_to_save}
        with open(path, 'wb') as config_file:
            pickle.dump(config_dictionary, config_file)

    def load(self, path):
        if not os.path.exists(path):
            err_msg = f"Could not find the ICE save file.\nDoes {path} exist ?"
            raise FileNotFoundError(err_msg)
        with open(path, 'rb') as config_file:
            config_dictionary = pickle.load(config_file)
        for attr, val in config_dictionary.items():
            setattr(self, attr, val)
        print(f"Loaded obj config from {path}")
        self.interactive = True

    def __repr__(self):
        mode = "interactive" if self._interactive else "operating"
        return f"Interactive Color Extractor for game {self.game} in {mode} mode"


class ObjDict():
    """
    Object (ordered) Dictionary tracking which object has already been seen
    and assigned a class
    """
    def __init__(self, nb_obj_comp, max_obj):
        self.current_obj = OrderedDict()
        self.nb_obj_comp = nb_obj_comp
        self.max_obj = max_obj
        self._vsize = nb_obj_comp*max_obj
        self.object_samples = {}

    def to_array(self):
        if self.current_obj:
            obj_vect = np.array(list(self.current_obj.values())).flatten()
            return np.pad(obj_vect, (0, self._vsize-len(obj_vect)), mode='constant')
        else:
            return np.zeros(self.max_obj * self.nb_obj_comp)

    def empty(self):
        for key in self.current_obj.keys():
            self.current_obj[key] = (0, 0)

    def fill(self, omages, positions, image):
        for omg_sample, position in zip(omages, positions):
            already_seen = False
            for classname, sample_list in self.object_samples.items():
                if ObjSample(omg_sample) in sample_list:
                    already_seen = True
                    self.current_obj[classname] = position
                    break
            if not already_seen:
                print("Object has not been seen yet !")
                import matplotlib.pyplot as plt
                f, axarr = plt.subplots(1, 2, figsize=(20, 10))
                for ax, img in zip(axarr, [image, omg_sample]):
                    ax.imshow(img)
                    ax.xaxis.set_major_locator(plt.NullLocator())
                    ax.yaxis.set_major_locator(plt.NullLocator())
                    buttons = []
                    objects_fields = list(self.object_samples.keys())
                    objects_fields.append("textbox")
                    for i, el in enumerate(objects_fields):
                        if el != "textbox":
                            axcut = plt.axes([0.2 + 0.05*i, 0.05, 0.04, 0.02], label=el)
                            bcut = Button(axcut, el, hovercolor='green')
                            bcut.on_clicked(_on_button_click)
                            buttons.append(bcut)
                        else:
                            axcut = plt.axes([0.2 + 0.05*i + 0.034, 0.05, 0.04, 0.02], label=el)
                            tcut = TextBox(axcut, label="New class: ")
                            tcut.on_submit(_on_text_submit)
                plt.show()
                global NEW_CLASSNAME
                assert NEW_CLASSNAME is not None
                if NEW_CLASSNAME in self.object_samples.keys():
                    self.object_samples[NEW_CLASSNAME].append(ObjSample(omg_sample))
                else:
                    self.object_samples[NEW_CLASSNAME] = [ObjSample(omg_sample)]
                self.current_obj[NEW_CLASSNAME] = position
                NEW_CLASSNAME = None

    def _categorize(self, obj_im):
        for classname, sample_list in self.object_samples.items():
            if ObjSample(obj_im) in sample_list:
                return classname
            else:
                return "Unknown"

    def find_categories(self, objects_imgs):
        categories = []
        for obj_im in objects_imgs:
            categories.append(self._categorize(obj_im))
        return categories


    def __repr__(self):
        return f"Object Dict containing: {self.object_samples.keys()}"

    def __iter__(self):
        return iter(self.object_samples)

    def items(self):
        for key in sorted(self.object_samples.keys(), key=lambda x:x.lower()):
            yield key, self.object_samples[key]

    def __getitem__(self, key):
         return self.object_samples[key]

    def change_obj_category(self, object_cat, object_pos, new_cat):
        obj = self[object_cat].pop(object_pos)
        if new_cat in self.object_samples.keys():
            self.object_samples[new_cat].append(obj)
        else:
            self.object_samples[new_cat] = [obj]

    def change_category_name(self, old_name, new_name):
        if new_name in self.object_samples.keys():
             print(f"Merging category {old_name} inside {new_name}")
             self.object_samples[new_name].extend(self.object_samples[old_name])
             del self.object_samples[old_name]
        else:
            print(f"Changing category name: {old_name} -> {new_name}")
            self.object_samples[new_name] = self.object_samples[old_name]
            del self.object_samples[old_name]


class ObjSample():
    def __init__(self, img):
        self.omg_sample = img
        self.dominant_color = self._dominant_color()

    def __eq__(self, stored_spl):
        if self.dominant_color != stored_spl.dominant_color:
            return False
        h0, w0, _ = stored_spl.shape
        h1, w1, _ = self.shape
        if w0 != w1 and h0 != h1:
            return False
        elif not self.omg_sample.tobytes() == stored_spl.omg_sample.tobytes():
            if w0 < w1:  # current sample is smaller and h0 == h1
                if (stored_spl.omg_sample == self.omg_sample[:, :w0]).all() or \
                   (stored_spl.omg_sample == self.omg_sample[:, -w0:]).all():
                    print(f"Updating image sample : {stored_spl.shape} -> {self.shape}")
                    show_img([stored_spl.omg_sample, self.omg_sample])
                    stored_spl.omg_sample = self.omg_sample
                    stored_spl._dominant_color = self.dominant_color
                    return True
                return False
            elif h0 < h1:  # current sample is smaller and w0 == w1
                if (stored_spl.omg_sample == self.omg_sample[:h0]).all() or \
                   (stored_spl.omg_sample == self.omg_sample[-h0:]).all():
                    print(f"Updating image sample : {stored_spl.shape} -> {self.shape}")
                    show_img([stored_spl.omg_sample, self.omg_sample])
                    stored_spl.omg_sample = self.omg_sample
                    stored_spl._dominant_color = self.dominant_color
                    return True
                return False
            elif w0 > w1:  # current sample is bigger and h0 == h1
                return (stored_spl.omg_sample[:, :w1] == self.omg_sample).all() or \
                   (stored_spl.omg_sample[:, -w1:] == self.omg_sample).all()
            elif h0 > h1:  # current sample is bigger and w0 == w1
                return (stored_spl.omg_sample[:h1] == self.omg_sample).all() or \
                    (stored_spl.omg_sample[-h1:] == self.omg_sample).all()

        # only same size image remaining
        for mod_spl in [self.omg_sample, np.fliplr(self.omg_sample),
                        np.flipud(self.omg_sample)]:
            diff = (mod_spl == stored_spl.omg_sample)
            ratio = np.count_nonzero(diff) / diff.size
            if ratio >= 0.8:
                return True
        return False

    @property
    def shape(self):
        return self.omg_sample.shape

    @property
    def avg_color(self):
        return np.average(self.omg_sample, (0,1))

    def _dominant_color(self):
        colors = {}
        for cell in self.omg_sample:
            for color in cell:
                hex = '#%02x%02x%02x' % tuple(color)
                if hex in colors:
                    colors[hex] += 1
                else:
                    colors[hex] = 1
        return max(colors, key=colors.get)

    def __repr__(self):
        return f"Object sample: size {self.shape}, dominant color {self.dominant_color}"


def discard_zone(image):
    print("You can discard a zone of the environment, the features extractor would then not look at it.")
    input_msg = "To discard a zone, please input:\n",
    "u (to discard a zone on the UPPER side)\n",
    "b (to discard a zone on the BOTTOM side)\n",
    "r (to discard a zone on the RIGHT side)\n",
    "l (to discard a zone on the LEFT side)\n",
    "Just enter to continue"
    inputed = "None"
    while inputed != "":
        if inputed == "u":
            import ipdb; ipdb.set_trace()
            new_image = image.copy()
        print(input_msg)
        inputed = input().lower()


def show_img(imgs):
    import matplotlib.pyplot as plt
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
    f, axarr = plt.subplots(1, len(imgs))
    for ax, img in zip(axarr, imgs):
        ax.imshow(img)
        ax.xaxis.set_major_locator(plt.MaxNLocator(2))
        ax.yaxis.set_major_locator(plt.MaxNLocator(2))
    plt.show()

def _on_text_submit(el):
    global NEW_CLASSNAME
    plt.close('all')
    NEW_CLASSNAME = el

def _on_button_click(el):
    global NEW_CLASSNAME
    plt.close('all')
    NEW_CLASSNAME = el.inaxes._label

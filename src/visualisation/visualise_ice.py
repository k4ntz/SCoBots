from tkinter import *
from tkinter import filedialog as fd
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

import sys
sys.path.append(".")
sys.path.append("..")
from xrl.agents.image_extractors import IColorExtractor as ColorExtractor


game = "MsPacman"


root = Tk()
root.title("ICE Viewer")
FILEPATH = None
entries = []
feature_extractor = None


def repeat_upsample(rgb_array, k=8, l=8, err=[]):
    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)


def on_press_update_button():
    for entry in entries:
        if entry.old_name != entry.val.get():
            feature_extractor.imp_objects.change_category_name(entry.old_name, entry.val.get())
    feature_extractor.save(FILEPATH)
    print(f"Saved config in {FILEPATH}")
    display_feat_extr(feature_extractor)


def change_category(button, category):
    feature_extractor.imp_objects.change_obj_category(button.name, button.number, category["text"])
    display_feat_extr(feature_extractor)
    print("Category changed")


def move_image(button):
    catSelector = Toplevel(root)
    catSelector.title("Select new Category")
    Label(catSelector, text ="To what Category do you want to move ").grid(row=0, column=0, columnspan=5)
    Label(catSelector, image=button.image).grid(row=0, column=5)
    Label(catSelector, text=" ?").grid(row=0, column=6)
    global feature_extractor
    for j, name in enumerate(feature_extractor.imp_objects):
        category = Button(catSelector, text=name)
        category.configure(command=lambda button=button, category=category: change_category(button, category))
        category.grid(row=1, column=j, padx=(0, 5))


def display_feat_extr(feature_extractor):
    for label in root.grid_slaves():
        label.grid_forget()
    entries.clear()
    for i, (name, obj_samples) in enumerate(feature_extractor.imp_objects.items()):
        val_label = StringVar()
        entry = ttk.Entry(root, textvariable=val_label, width=15)
        entry.delete(0, END)
        entry.insert(0, name)
        entry.grid(row=i, column=0, padx=(0, 8))
        entry.val = val_label
        entry.old_name = name
        entries.append(entry)
        for j, obj_sample in enumerate(obj_samples):
            array = obj_sample.omg_sample
            # if name[0] == '_':
            #     zoom = 4
            zoom = 8
            img = ImageTk.PhotoImage(image=Image.fromarray(repeat_upsample(array, zoom, zoom)))
            button = Button(root, image=img)
            button.configure(command=lambda button = button: move_image(button))
            button.image = img
            button.name = name
            button.number = j
            button.grid(row=i, column=j+1, padx=(0, 5))

    import_button.grid(row=i+1, column=1, columnspan=5, sticky=W)
    update_button = ttk.Button(root, text="Update ICE", width=12,
                               command=on_press_update_button)
    update_button.grid(row=i+1, column=0)


def on_press_import_button(filepath=None):
    if filepath is None:
        global FILEPATH
        FILEPATH = fd.askopenfilename(filetypes=[
                    ("ICE save format", ".ice"),
                ])
        filepath = FILEPATH
    global feature_extractor
    feature_extractor = ColorExtractor()
    feature_extractor.load(path=filepath)
    display_feat_extr(feature_extractor)


import_button = ttk.Button(root, text="Import ICE", width=12,
                           command=on_press_import_button)
import_button.grid(row=1, column=0)

root.mainloop()

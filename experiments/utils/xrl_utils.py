import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import json
import os
INTERACTIVE = False
if INTERACTIVE:
    mpl.use("TkAgg")
else:
    mpl.use("Agg")
mpl.rcParams['toolbar'] = 'None' 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 




figure_output_folder = "figures"
if not os.path.exists(figure_output_folder):
    os.makedirs(figure_output_folder)
else:
    for f in os.listdir(figure_output_folder):
        os.remove(os.path.join(figure_output_folder, f))


def to_rgba(color):
    return np.concatenate([np.array(color)/255, [.5]])

class Drawer:
    def __init__(self, obs_raw):
        # plotting
        fps = 30
        self.frame_delta = 1.0 / fps
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        zoom = 8
        fsize = (obs_raw.shape[1]+0) * px * zoom, (obs_raw.shape[0]+ 0) * px * zoom
        self.fig, axes = plt.subplots(2, 2, figsize=fsize, gridspec_kw={'height_ratios': [1, 5]})
        for ax in axes.flatten():
            ax.axis("off")
        rows, cells, colors = [], [], []
        columns = ["X, Y", "W, H", "R, G, B"]
        for obj in range(9):
            rows.append("category")
            cells.append(["xy", "wh", "rgb"])
            colors.append([1, 1, 1, 1])
        self.my_table = axes[0][0].table(cellText=cells,
                                    rowLabels=rows,
                                    rowColours=colors,
                                    colLabels=columns,
                                    colWidths=[.2,.2,.3],
                                    cellLoc ='center',
                                    loc='center')
        self.my_table.set_fontsize(80)
        self.my_table.scale(1, 2.3)
        self.img = axes[1][0].imshow(obs_raw, interpolation='none')
        self.features_text = axes[0][1].text(.2, 0., '', fontsize=22)
        self.img2 = axes[1][1].imshow(obs_raw, interpolation='none')
        plt.ion()
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.tight_layout()
        self.max_nb_row = 0
        self.pause = False

    def update(self, run, step):
        if INTERACTIVE:
            self.update_tk_canvas()
        else:
            plt.savefig(os.path.join(figure_output_folder, f"explanations_run{run}_step{step}.png"))

    def update_tk_canvas(self):
        self.fig.canvas.get_tk_widget().update()

    def draw_play(self, action, env,):
        self.img.set_data(env._obj_obs)
        self.img2.set_data(env._rel_obs)
        # table
        self.update_table(env)
        # rightings
        to_draw = "\n".join(env._top_features)
        to_draw +="\n\n"
        if action:
            to_draw += f" --> {env.action_space_description[action]}"
        self.features_text.set_text(to_draw)
        plt.pause(self.frame_delta)

    def draw_explain(self, rule, action, env, normalize = True, denorm_dict = None):
        to_draw = "IF"
        premise = rule.premise.pop()
        terms_set = premise.terms
        for term in terms_set:
            key_str = term.variable
            # insert space after comma
            #key_str = key_str.replace(",", ", ") #TODO: added but might not always be correct
            if normalize:
                m, s = denorm_dict[key_str]
                denorm_value = int((term.threshold * s) + m)
            else:
                denorm_value = term.threshold
            to_draw += f"\n{term.variable} {term.operator} {denorm_value}"
        #import ipdb; ipdb.set_trace()
        self.img.set_data(env._obj_obs)
        self.img2.set_data(env._rel_obs)
        # table
        self.update_table(env)
        # rightings
        to_draw += "\n\n"
        if action:
            to_draw += f" --> {env.action_space_description[action]}"
        self.features_text.set_text(to_draw)
        plt.pause(self.frame_delta)

    def update_table(self, env):
        # table
        nb_row = len(self.my_table.get_celld()) // 4
        if self.max_nb_row < nb_row:
            self.max_nb_row = nb_row
            plt.tight_layout()
        for i, obj in enumerate(env.oc_env.objects):
            if i+1 > nb_row:
                height = self.my_table.get_celld()[(1, -1)].get_height()
                self.my_table.add_cell(i+1, -1, width=0.2, height=height, text=obj.category, loc="center", facecolor=to_rgba(obj.rgb))
                self.my_table.add_cell(i+1, 0, width=0.2, height=height, text=obj.xy, loc="center")
                self.my_table.add_cell(i+1, 1, width=0.2, height=height, text=obj.wh, loc="center")
                self.my_table.add_cell(i+1, 2, width=0.3, height=height, text=obj.rgb, loc="center")
            else:
                self.my_table.get_celld()[(i+1, -1)].get_text().set_text(obj.category)
                self.my_table.get_celld()[(i+1, -1)].set_color(to_rgba(obj.rgb))
                self.my_table.get_celld()[(i+1, 0)].get_text().set_text(obj.xy)
                self.my_table.get_celld()[(i+1, 1)].get_text().set_text(obj.wh)
                self.my_table.get_celld()[(i+1, 2)].get_text().set_text(obj.rgb)
        if len(env.oc_env.objects) > 1:
            while nb_row > i+1:
                del self.my_table._cells[(nb_row, -1)]
                del self.my_table._cells[(nb_row, 0)]
                del self.my_table._cells[(nb_row, 1)]
                del self.my_table._cells[(nb_row, 2)]
                nb_row -= 1

    def onclick(self, event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        if event.button == 1:
            self.pause = not self.pause
        elif self.pause and event.button == 3:
            i = 0
            while True:
                savename = f"explanations_{i}.svg"
                if not os.path.exists(savename):
                    break
                i += 1
            plt.savefig(savename)
            print(f"Saving current image in {savename}")

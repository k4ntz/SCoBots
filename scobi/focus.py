import yaml
import numpy as np
import math
from pathlib import Path
from itertools import permutations
from scobi.concepts import init as concept_init
from scobi.utils.decorators import FUNCTIONS, PROPERTIES
from termcolor import colored
from collections.abc import Iterable

class Focus():
    def __init__(self, env_name, interactive, reward, hide_properties, fodir, fofile, raw_features, actions, refresh_yaml, l):
        concept_init()
        self.PROPERTY_LIST = []
        self.FUNCTION_LIST = []
        self.OBJECTS = raw_features
        self.OBJECT_NAMES = [x.name for x in self.OBJECTS]
        self.ACTIONS = actions
        self.ENV_NAME = env_name
        self.FOCUSFILEPATH = None
        self.PARSED_OBJECTS = []
        self.PARSED_ACTIONS = []
        self.PARSED_PROPERTIES = []
        self.PARSED_FUNCTIONS = []
        self.FEATURE_VECTOR_BACKMAP = []

        self.PROPERTY_COMPUTE_LAYER = []
        self.FUNC_COMPUTE_LAYER = []
        self.PROPERTY_COMPUTE_LAYER_SIZE = 0
        self.FUNC_COMPUTE_LAYER_SIZE = 0
        self.CURRENT_PROPERTY_COMPUTE_LAYER = []
        self.CURRENT_FUNC_COMPUTE_LAYER = []
        
        self.FEATURE_VECTOR_SIZE = 0
        self.CURRENT_FEATURE_VECTOR = []
        self.FEATURE_VECTOR_PROPS_SIZE = 0
        self.CURRENT_FEATURE_VECTOR_PROPS = []
        self.FEATURE_VECTOR_FUNCS_SIZE = 0
        self.CURRENT_FEATURE_VECTOR_FUNCS = []
        self.CURRENT_FREEZE_MASK = []

        self.REWARD_SHAPING = reward
        self.REWARD_FUNC = None
        self.reward_history = [0, 0]
        self.reward_threshold = -1
        self.HIDE_PROPERTIES = hide_properties

        self.running_stats = []
        self.l = l
        self.generate_property_set()
        self.generate_function_set()
        self.last_obs_vector = []
        self.first_pass = True

        # rework the control flow here, keep interactive mode or not?
        fdir = Path.cwd() / Path(fodir)
        fdir.mkdir(exist_ok=True)
        
        if interactive == True:
            l.GeneralInfo("Interactive Mode")
            if fofile:
                fpath = fdir / Path(fofile)
                if fpath.exists():
                    l.GeneralInfo("Focus file %s found." % colored(fpath.name, "light_green"))
                    self.load_focus_file(fpath)
                    l.GeneralInfo("File is valid. Imported.")
                    self.FOCUSFILEPATH = fpath
                else:
                    l.GeneralError("Specified focus file %s not found!" %  colored(fpath.name, "light_green"))
            else:
                fpath = Path.cwd() / Path(fodir) / Path("default_focus_" + env_name + ".yaml")
                if fpath.exists():
                    self.FOCUSFILEPATH = fpath
                    l.GeneralError("No focus file specified, but found an auto-generated default. Edit %s and pass it to continue." % colored(fpath.name, "light_green"))
                else:
                    self.generate_fresh_yaml(fpath)
                    l.GeneralError("No focus file specified! Auto-generated a default focus file. Edit %s and pass it to continue." % colored(fpath.name, "light_green"))
        else:
            l.GeneralInfo("Non-Interactive Mode")
            if fofile:
                l.GeneralWarning("Specified focus file ignored, because in non-interactive scobi mode. Using default.")
            fpath = fdir / Path("default_focus_" + env_name + ".yaml")
            if not fpath.exists():
                self.generate_fresh_yaml(fpath)
                l.GeneralWarning("No default focus file found. Auto-generated %s." % colored(fpath.name, "light_green"))
            l.GeneralInfo("Focus file %s found." % colored(fpath.name, "light_green"))
            if refresh_yaml:
                l.GeneralInfo("Refreshing yaml to make sure it's up-to-date.")
                self.generate_fresh_yaml(fpath)
            self.load_focus_file(fpath)
            l.GeneralInfo("File is valid. Imported.")
            self.FOCUSFILEPATH = fpath
        
        if self.REWARD_SHAPING != 0:
            if self.REWARD_SHAPING == 1:
                rewstring = "scobi" 
            elif self.REWARD_SHAPING == 2:
                rewstring = "env + scobi"
            else:
                rewstring = "unknown"
            l.GeneralInfo("Reward Shaping: %s." % colored(rewstring, "light_green"))
            self.REWARD_FUNC = self.get_reward_func(self.ENV_NAME)
            if self.REWARD_FUNC:
                l.GeneralInfo("Reward function is valid. Bound.")
            else:
                l.GeneralError("Reward function for %s not implemented!" % colored(env_name, "light_green"))
        else:
            l.GeneralInfo("Reward Shaping: %s." % colored("disabled", "light_yellow"))

        if self.HIDE_PROPERTIES == True:
            l.GeneralInfo("Object properties are %s from the observation vector." % colored("excluded", "light_yellow"))
        else:
            l.GeneralInfo("Object properties are %s in the observation vector." % colored("included", "light_green"))


    def generate_property_set(self):
        for k, v in PROPERTIES.items():
            for o in self.OBJECTS:
                if type(o) == v["expects"][0][0].annotation: #assume only one input from property
                    e = [k, o.name]
                    self.PROPERTY_LIST.append(e)

    def generate_function_set(self):
        for k, v in FUNCTIONS.items():
            para_len = len(v["expects"])
            property_combis = permutations(self.PROPERTY_LIST, para_len)
            for combi in property_combis:
                combi_sig = [PROPERTIES[x[0]]["returns"][0] for x in combi]
                function_sig = [x[0].annotation for x in v["expects"]]
                if combi_sig == function_sig:
                    self.FUNCTION_LIST.append([k, list(combi)])

    def get_object_by_name(self, name, objs):
        if type(objs) == dict:
            for o in objs.values():
                if o.name == name:
                    return o
            return None
        else:
            for o in objs:
                if o.name == name:
                    return o
            return None


    def print_state(self):
        print("---OBJECTS---")
        for o in self.OBJECTS:
            print(o.name)
        print("---PROPERTIES---")
        for p in self.PROPERTY_LIST:
            print(p)
        print("---FUNCTIONS---")
        for f in self.FUNCTION_LIST:
            print(f)


    def avail_to_yaml_dict(self, k, v):
        out_dict = {
            k : {
                "in": [x[1] for x in v["expects"]],
                "description": v["returns"][1] 
            }
        }
        return out_dict

    def proplist_to_yaml_dict(self, x):
        out_dict = {
            x[0] : x[1]
        }
        return out_dict

    def funclist_to_yaml_dict(self, x):
        instance_arg_list = []

        for i in x[1]:
            instance_arg_list.append({i[0] : i[1]})

        out_dict = {
            x[0] : instance_arg_list
        }
        return out_dict


    def generate_fresh_yaml(self, fpath):
        yaml_dict = {
            "ENVIRONMENT" : "",
            "AVAILABLE_CONCEPTS" : {
                "objects" : [],
                "actions" : [],
                "properties" : [],
                "functions" : []
            },
            "SELECTION": {
                "objects" : [],
                "actions" : [],
                "properties" : [],
                "functions" : []
            }
        }

        yaml_dict["ENVIRONMENT"] = self.ENV_NAME
        avail = yaml_dict["AVAILABLE_CONCEPTS"]
        avail["objects"] = [x.name for x in self.OBJECTS]
        avail["actions"] = [x for x in self.ACTIONS]
        avail["properties"] = [self.avail_to_yaml_dict(k, v) for k, v in PROPERTIES.items()]
        avail["functions"] =  [self.avail_to_yaml_dict(k, v) for k, v in FUNCTIONS.items()]

        use = yaml_dict["SELECTION"]
        use["objects"] = [x.name for x in self.OBJECTS]
        use["actions"] = [x for x in self.ACTIONS]
        use["properties"] = [self.proplist_to_yaml_dict(x) for x in self.PROPERTY_LIST]
        use["functions"] = [self.funclist_to_yaml_dict(x) for x in self.FUNCTION_LIST]

        with open(fpath, "w") as f:
            yaml.dump(yaml_dict, f, sort_keys=False)


    def validate_objects(self, objs):
        if not objs:
            return False
        for o in objs:
            if o not in self.OBJECT_NAMES:
                return False
        return True


    def validate_actions(self, acts):
        if not acts:
            return False
        for o in acts:
            if o not in self.ACTIONS:
                return False
        return True


    def validate_properties(self, props):
        for p in props:
            if p not in PROPERTIES.keys():
                return False
        return True


    def validate_functions(self, funcs):
        for f in funcs:
            if f not in FUNCTIONS.keys():
                return False
        return True


    def validate_properties_signatures(self, propslist):
        for p in propslist:
            if p[1] not in self.OBJECT_NAMES:
                self.l.FocusFileParserError("Unknown object in properties selection: %s" % p[1])
            if p[0] not in PROPERTIES.keys():
                self.l.FocusFileParserError("Unknown object in properties selection: %s" % p[0])
            prop_definition = PROPERTIES[p[0]]
            o = self.get_object_by_name(p[1], self.OBJECTS)
            prop_sig = prop_definition["expects"][0][0].annotation
            if type(o) != prop_sig:
                 self.l.GeneralError("Signature mismatch. Property '%s' expects '%s'" % (p[0], prop_sig))
        return True


    def validate_functions_signatures(self, funclist):
        for f in funclist:
            parsed_para_sig = []
            if f[0] not in FUNCTIONS.keys():
                self.l.FocusFileParserError("Unknown function in function selection: %s" % f[0])
            for para in f[1]:
                if para[0] not in PROPERTIES.keys():
                    self.l.FocusFileParserError("Unknown property in functions selection: %s" % para[0])
                if para[1] not in self.OBJECT_NAMES:
                    self.l.FocusFileParserError("Unknown object in functions selection: %s" % para[1])
                prop_definition = PROPERTIES[para[0]]
                o = self.get_object_by_name(para[1], self.OBJECTS)
                prop_sig = prop_definition["expects"][0][0].annotation
                parsed_para_sig.append(prop_definition["returns"][0])
                if type(o) != prop_sig:
                    self.l.FocusFileParserError("Signature mismatch in functions selection. Property '%s' expects '%s'" % (para[0], prop_sig))
            func_definition = FUNCTIONS[f[0]]
            function_sig = [x[0].annotation for x in func_definition["expects"]]
            sig_desc = [x[1] for x in func_definition["expects"]]
            if function_sig != parsed_para_sig:
                self.l.FocusFileParserError("Signature mismatch in functions selection. Function '%s' expects '%s'" % (f[0], sig_desc))
        return True


    def import_objects(self, objs):
        if self.validate_objects(objs):
            return objs
        else:
            self.l.FocusFileParserError("Invalid objects specified in objects selection!")

    def import_actions(self, acts):
        if self.validate_actions(acts):
            return acts
        else:
            self.l.FocusFileParserError("Invalid actions specified in actions selection!")

    def import_properties(self, props):
        out = []
        for p in props:
            out.append(list(p.items())[0])
        out_list =  list(map(list, out))
        if self.validate_properties_signatures(out_list):
            return out_list
        else:
            return None

    def import_functions(self ,funcs):
        out = []
        funcs_to_vali = []
        properties_to_vali = []
        objects_to_vali = []
        for p in funcs:
            item = list(p.items())[0]
            fname = item[0]
            fparas = item[1]
            funcs_to_vali.append(fname)
            para_list = [fname, []]
            for p in fparas:
                para_tuple = list(p.items())[0]
                properties_to_vali.append(para_tuple[0]) # not used ?
                objects_to_vali.append(para_tuple[1]) # not used ?
                para_list[1].append(list(para_tuple))
            out.append(para_list)
        if self.validate_functions_signatures(out):
            return out
        else:
            return None


    def load_focus_file(self, fpath):
        with open(fpath, "r") as f:
            in_dict = yaml.safe_load(f)
        parsed_env_name = in_dict["ENVIRONMENT"]
        if self.ENV_NAME != parsed_env_name:
            self.l.FocusFileParserError("Env and focus file env do not match: %s, %s" % (self.ENV_NAME, parsed_env_name))
        sdict = in_dict["SELECTION"]
        self.PARSED_OBJECTS = self.import_objects(sdict["objects"])
        self.PARSED_ACTIONS = self.import_actions(sdict["actions"])
        self.PARSED_PROPERTIES = self.import_properties(sdict["properties"])
        self.PARSED_FUNCTIONS = self.import_functions(sdict["functions"])
        # based on the focus file selection,
        # construct a 2 layer computation graph for the feature vector:
        # 1     PROPERTY_COMPUTE_LAYER
        # 2     FUNC_COMPUTE_LAYER
        prop_name_obj_name_pairs = []
        parsed_fv_index = 0
        for p in self.PARSED_PROPERTIES:
            property_name = p[0]
            object_name = p[1]
            prop_name_obj_name_pairs.append((property_name, object_name))
            prop_func = PROPERTIES[property_name]["object"]
            return_len = len(PROPERTIES[property_name]["returns"][0].__args__)
            for _ in range(return_len):
                self.FEATURE_VECTOR_BACKMAP.append(parsed_fv_index)
            parsed_fv_index += 1
            def prop(input_dict, prop_func=prop_func, object_name=object_name):
                func = prop_func
                return func(input_dict[object_name])
            self.PROPERTY_COMPUTE_LAYER.append(prop)
        for f in self.PARSED_FUNCTIONS:
            func_name = f[0]
            input_props = f[1]
            property_result_idxs = []
            for p in input_props:
                property_name = p[0]
                object_name = p[1]
                property_result_idxs.append(prop_name_obj_name_pairs.index((property_name, object_name)))
            f = FUNCTIONS[func_name]["object"]
            return_len = len(FUNCTIONS[func_name]["returns"][0].__args__)
            for _ in range(return_len):
                self.FEATURE_VECTOR_BACKMAP.append(parsed_fv_index)
            parsed_fv_index += 1
            ol = [0 for _ in range(len(property_result_idxs))]
            def func(prop_results, f=f, idxs=property_result_idxs, outlist=ol):
                f_in = outlist
                for i, j in enumerate(idxs):
                    f_in[i] = prop_results[j]
                return f(*f_in)
            self.FUNC_COMPUTE_LAYER.append(func)
        # init compute layer lists
        self.PROPERTY_COMPUTE_LAYER_SIZE = len(self.PROPERTY_COMPUTE_LAYER)
        self.FUNC_COMPUTE_LAYER_SIZE = len(self.FUNC_COMPUTE_LAYER)
        self.CURRENT_PROPERTY_COMPUTE_LAYER = [0 for _ in range(self.PROPERTY_COMPUTE_LAYER_SIZE)]
        self.CURRENT_FUNC_COMPUTE_LAYER = [0 for _ in range(self.FUNC_COMPUTE_LAYER_SIZE)]

    def get_feature_vector(self, inc_objects_list):
        # evaluate a 2 layer computation graph for the feature vector:
        # IN    object_dict
        # 1     PROPERTY_COMPUTE_LAYER 
        #       property_values
        # 2     FUNC_COMPUTE_LAYER
        #       function_values
        # OUT   HSTACK(CONCAT(property_values, function_values))

        # fill missing objects as None
        input_dict = {}
        for obj in inc_objects_list:
            input_dict[obj.name] = obj 
        for name in self.OBJECT_NAMES:
            if not name in input_dict.keys():
                input_dict[name] = None

        # calc property layer
        for i in range(self.PROPERTY_COMPUTE_LAYER_SIZE):
            f = self.PROPERTY_COMPUTE_LAYER[i]
            self.CURRENT_PROPERTY_COMPUTE_LAYER[i] = f(input_dict)

        # calc function layer
        for i in range(self.FUNC_COMPUTE_LAYER_SIZE):
            f = self.FUNC_COMPUTE_LAYER[i]
            self.CURRENT_FUNC_COMPUTE_LAYER[i] = f(self.CURRENT_PROPERTY_COMPUTE_LAYER)

        if self.first_pass:
            self.first_pass = False
            props = [i for e in self.CURRENT_PROPERTY_COMPUTE_LAYER for i in e]
            funcs = [i for e in self.CURRENT_FUNC_COMPUTE_LAYER for i in e]
            out = props + funcs
            self.FEATURE_VECTOR_SIZE = len(out)
            self.FEATURE_VECTOR_PROPS_SIZE = len(props)
            self.FEATURE_VECTOR_FUNCS_SIZE = len(funcs)
            self.CURRENT_FEATURE_VECTOR_PROPS = [0 for _ in range(self.FEATURE_VECTOR_PROPS_SIZE)]
            self.CURRENT_FEATURE_VECTOR_FUNCS = [0 for _ in range(self.FEATURE_VECTOR_FUNCS_SIZE)]
            self.CURRENT_FREEZE_MASK = [1 for _ in range(self.FEATURE_VECTOR_SIZE)]

            # unpack property layer
            idx = 0
            for f in self.CURRENT_PROPERTY_COMPUTE_LAYER:
                for ff in f:
                    self.CURRENT_FEATURE_VECTOR_PROPS[idx] = ff
                    idx += 1
            
            # unpack function layer
            idx = 0
            for f in self.CURRENT_FUNC_COMPUTE_LAYER:
                for ff in f:
                    self.CURRENT_FEATURE_VECTOR_FUNCS[idx] = ff
                    idx += 1

            out = self.CURRENT_FEATURE_VECTOR_PROPS + self.CURRENT_FEATURE_VECTOR_FUNCS
            for e in out:
                item = 0.0 if e is None else e
                self.last_obs_vector.append(item)

            if self.REWARD_SHAPING != 0:
                reward = self.REWARD_FUNC(self.last_obs_vector)
            else:
                reward = 0
            
            out = self.last_obs_vector
            if self.HIDE_PROPERTIES:
                out = out[self.FEATURE_VECTOR_PROPS_SIZE:]
            return out, reward

        # unpack property layer
        idx = 0
        for f in self.CURRENT_PROPERTY_COMPUTE_LAYER:
            for ff in f:
                self.CURRENT_FEATURE_VECTOR_PROPS[idx] = ff
                idx += 1
        
        # unpack function layer
        idx = 0
        for f in self.CURRENT_FUNC_COMPUTE_LAYER:
            for ff in f:
                self.CURRENT_FEATURE_VECTOR_FUNCS[idx] = ff
                idx += 1

        out = self.CURRENT_FEATURE_VECTOR_PROPS + self.CURRENT_FEATURE_VECTOR_FUNCS
        # freeze feature entries that are derived from invisible objects
        # objects are distinguished by order, not id
        # if object id=1 on position 1 becomes invisible, and obj id=2, pos=2 remains visible
        # obj with id=2 will be pos=1 and objc id=1 will be first position of hidden objects
        for i in range(self.FEATURE_VECTOR_SIZE): 
            if out[i] is None:
                out[i] = self.last_obs_vector[i]
                self.CURRENT_FREEZE_MASK[i] = 0
            else:
                self.CURRENT_FREEZE_MASK[i] = 1
        self.last_obs_vector = out
        
        if self.REWARD_SHAPING != 0:
            reward = self.REWARD_FUNC(out)
        else:
            reward = 0
        if self.HIDE_PROPERTIES:
            out = out[self.FEATURE_VECTOR_PROPS_SIZE:]
        return out, reward
    
    def get_feature_vector_description(self):
        fv = self.PARSED_PROPERTIES + self.PARSED_FUNCTIONS
        return (fv, np.array(self.FEATURE_VECTOR_BACKMAP))
    
    def get_current_freeze_mask(self):
        return self.CURRENT_FREEZE_MASK
    
    def get_reward_func(self, env):
        fv_description, fv_backmap = self.get_feature_vector_description()
        i = 0
        if "Pong" in env:
            # pong reward function
            for feature in fv_description:
                i += 1
                feature_name = feature[0]
                feature_signature = feature[1]
                if feature_name == "DISTANCE":
                    input1 = feature_signature[0]
                    input2 = feature_signature[1]
                    if input1[0] == "POSITION" and input1[1] == "Player1" and input2[0] == "POSITION" and input2[1] == "Ball1":
                        idxs = np.where(fv_backmap == i-1)[0]
            # reward when player decreases y-distance to ball
            def reward(fv, idxs=idxs):
                v_entries = fv[idxs[0]:idxs[-1]+1]
                self.reward_history[0] = self.reward_history[1]
                self.reward_history[1] = abs(v_entries[1]) # absolute distance on y-axis
                delta = self.reward_history[0] - self.reward_history[1] #decrease in distance: positive sign
                return delta * 0.1
            return reward
        elif "Kangaroo" in env:
            # kangaroo reward function
            player_idxs = []
            distance_idxs = []
            for feature in fv_description:
                i += 1
                feature_name = feature[0]
                feature_signature = feature[1]
                if feature_name == "POSITION":
                    if feature_signature == "Player1":
                        player_idxs = np.where(fv_backmap == i-1)[0]
                if feature_name == "DISTANCE":
                    input1 = feature_signature[0]
                    input2 = feature_signature[1]
                    if input1[0] == "POSITION" and input1[1] == "Player1" and input2[0] == "POSITION" and input2[1] == "Scale1":
                        distance_idxs = np.where(fv_backmap == i-1)[0]
            # reward when player achieves new y-coord low an
            def reward(fv, p_idxs=player_idxs, d_idxs=distance_idxs):
                p_entries = fv[p_idxs[0]:p_idxs[-1]+1]
                y_coord_reward = 0
                if self.reward_threshold == -1:
                    self.reward_threshold = p_entries[1]
                    y_coord_reward = 0
                else:
                    delta = self.reward_threshold - abs(p_entries[1])
                    if delta > 0:
                        self.reward_threshold = abs(p_entries[1])
                        y_coord_reward = delta # reward when player achieves new y-coord low an
                
                d_entries = fv[d_idxs[0]:d_idxs[-1]+1]
                self.reward_history[0] = self.reward_history[1]
                self.reward_history[1] = abs(d_entries[0]) # x-dist
                delta = self.reward_history[0] - self.reward_history[1]
                distance_reward = delta if delta > 0 else 0 # reward when decreasing x-distance to Scale1
                return y_coord_reward + 0.1 * distance_reward
            return reward
        elif "Skiing" in env:
            # skiing reward function
            player_position_idxs = []
            flag_center_idxs = []
            flag_velocity_idxs = []
            for feature in fv_description:
                i += 1
                feature_name = feature[0]
                feature_signature = feature[1]
                if feature_name == "CENTER":
                    input1 = feature_signature[0]
                    input2 = feature_signature[1]
                    if input1[0] == "POSITION" and input1[1] == "Flag1" and input2[0] == "POSITION" and input2[1] == "Flag2":
                        flag_center_idxs = np.where(fv_backmap == i-1)[0]
                if feature_name == "POSITION":
                    if feature_signature == "Player1":
                        player_position_idxs = np.where(fv_backmap == i-1)[0]
                if feature_name == "DIR_VELOCITY":
                    input = feature_signature[0]
                    if input[0] == "POSITION_HISTORY" and input[1] == "Flag1":
                        flag_velocity_idxs = np.where(fv_backmap == i-1)[0]
            # reward for high player velocity and player decreases euc-distance to center of flag1 and flag2
            def reward(fv, c_idxs=flag_center_idxs, p_idxs=player_position_idxs, v_idxs=flag_velocity_idxs):
                p_entries = fv[p_idxs[0]:p_idxs[-1]+1]
                c_entries = fv[c_idxs[0]:c_idxs[-1]+1]
                v_entries = fv[v_idxs[0]:v_idxs[-1]+1]
                euc_dist = FUNCTIONS["EUCLIDEAN_DISTANCE"]["object"]
                player_flag_distance = euc_dist(p_entries, c_entries)[0]
                self.reward_history[0] = self.reward_history[1]
                self.reward_history[1] = player_flag_distance
                delta = self.reward_history[0] - self.reward_history[1] #decrease in distance: positive sign
                player_flag_distance_delta = delta if delta > 0 else 0 #only give positives
                euc_velocity_flag = np.clip(math.sqrt((v_entries[0])**2 + (v_entries[1])**2), 0, 10) #clip to 10
                return euc_velocity_flag + 4 * player_flag_distance_delta
            return reward


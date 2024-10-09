import yaml
import numpy as np
import math
from pathlib import Path
from itertools import permutations
from scobi.concepts import init as concept_init
from scobi.utils.decorators import FUNCTIONS
from termcolor import colored

class Focus():
    def __init__(self, env_name, reward, hide_properties, fofiles_dir_name, fofile, raw_features, max_obj_dict, actions, refresh_yaml, logger):
        concept_init()
        self.FUNCTION_LIST = []
        self.MAX_NB_OBJECTS = max_obj_dict
        self.INIT_OBJECTS = raw_features
        self.INIT_OBJECT_NAMES = [type(x).__name__ for x in self.INIT_OBJECTS] #Can get this differently
        self.NS_REPR_LIST = []
        self.NS_REPR_TYPES = []
        self.OBJECT_NAMES = []

        self.ACTIONS = actions
        self.ENV_NAME = env_name.split("/")[-1] # handle v5 namespace case
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
        self.OBSERVATION_SIZE = 0
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
        self.reward_subgoals = 0
        self.reward_helper_var = False
        self.HIDE_PROPERTIES = hide_properties

        self.running_stats = []
        self.logger = logger
        # self.generate_property_set()
        self.generate_ns_repr_set()
        self.generate_function_set()
        self.last_obs_vector = []
        self.first_pass = True

        fofiles_dir_path = Path.cwd() / Path(fofiles_dir_name)
        fofiles_dir_path.mkdir(exist_ok=True)
        logger.GeneralInfo("Focus file directory: %s." % colored(fofiles_dir_name, "light_green"))
        if fofile: # pruned focus file passed
            fofile_path = fofiles_dir_path / Path(fofile)
            if fofile_path.exists(): # if it exists, try to load it
                logger.GeneralInfo("Specified Focus file %s found." % colored(fofile_path.name, "light_green"))
                self.load_focus_file(fofile_path)
                logger.GeneralInfo("Specified Focus File is valid. Imported.")
                self.FOCUSFILEPATH = fofile_path
            else: # if passed focus file doesnt exist, exit
                logger.GeneralError("Specified Focus File %s not found!" %  colored(fofile_path.name, "light_green"))
        else: # no pruned focus file passed
            fofile_path = fofiles_dir_path / Path("default_focus_" + self.ENV_NAME + ".yaml")
            if not fofile_path.exists(): # default focus file does not exist
                self.generate_fresh_yaml(fofile_path)
                logger.GeneralWarning("No Default Focus File found. Auto-generated %s." % colored(fofile_path.name, "light_green"))
            else:
                logger.GeneralInfo("Default Focus file %s found." % colored(fofile_path.name, "light_green"))
                if refresh_yaml:
                    logger.GeneralInfo("Rebuilding it to make sure it's up-to-date.")
                    self.generate_fresh_yaml(fofile_path)
            self.load_focus_file(fofile_path)
            logger.GeneralInfo("Default Focus File is valid. Imported.")
            self.FOCUSFILEPATH = fofile_path
        
        if self.REWARD_SHAPING != 0: # set respective reward shaping
            if self.REWARD_SHAPING == 1:
                rewstring = "scobi" 
            elif self.REWARD_SHAPING == 2:
                rewstring = "env + scobi"
            else:
                rewstring = "unknown"
            logger.GeneralInfo("Reward Shaping: %s." % colored(rewstring, "light_green"))
            self.REWARD_FUNC = self.get_reward_func(self.ENV_NAME)
            if self.REWARD_FUNC is not None:
                if self.REWARD_FUNC == "norew":
                    logger.GeneralError("Reward function for %s not implemented!" % colored(self.ENV_NAME, "light_green"))
                else:
                    logger.GeneralInfo("Reward function is valid. Bound.")
            else:
                logger.GeneralError("Reward function for %s is expecting properties/concepts that are missing in the focus file!" % colored(self.ENV_NAME, "light_green"))
        else:
            logger.GeneralInfo("Reward Shaping: %s." % colored("disabled", "light_yellow"))

        if self.HIDE_PROPERTIES: # hide properties from observation or not
            logger.GeneralInfo("Object properties are %s from the observation vector." % colored("excluded", "light_yellow"))
        else:
            logger.GeneralInfo("Object properties are %s in the observation vector." % colored("included", "light_green"))
    
    def generate_ns_repr_set(self):
        for k,v in self.MAX_NB_OBJECTS.items():
            # match with first of the object
            obj = self.INIT_OBJECTS[self.INIT_OBJECT_NAMES.index(k)]
            # extract meanings and types of ns_repr
            # iterate over number of objects of current kind 
            for i in range( v):
                self.OBJECT_NAMES.append(k+str(i+1))
                ns_meanings = [[meaning, k+str(i+1)] for meaning in obj._ns_meaning]
                self.NS_REPR_LIST += ns_meanings
                self.NS_REPR_TYPES += obj._ns_types

    # def generate_property_set(self):
    #     print(PROPERTIES)
    #     for k, v in PROPERTIES.items():
    #         for o in self.OBJECTS:
    #             if type(o) == v["expects"][0][0].annotation: #assume only one input from property
    #                 e = [k, o.name]
    #                 self.PROPERTY_LIST.append(e)
    #     print(self.PROPERTY_LIST)
    #     exit()

    def generate_function_set(self):
        for k, v in FUNCTIONS.items():
            para_len = len(v["expects"])
            # property_combis = permutations(self.PROPERTY_LIST, para_len)
            ns_repr_combis = permutations(self.NS_REPR_LIST, para_len)
            function_sig = [x[0].annotation for x in v["expects"]]
            for combi in ns_repr_combis:
                # combi_sig_orig = [PROPERTIES[x[0]]["returns"][0] for x in combi]
                combi_sig = []
                for c in combi:
                    idx = self.NS_REPR_LIST.index(c)
                    combi_sig.append(self.NS_REPR_TYPES[idx])
                function_sig = [x[0].annotation for x in v["expects"]]
                if combi_sig == function_sig:
                    self.FUNCTION_LIST.append([k, list(combi)])

    # def get_object_by_name(self, name, objs):
    #     if type(objs) is dict:
    #         for o in objs.values():
    #             if o.name == name:
    #                 return o
    #         return None
    #     else:
    #         for o in objs:
    #             if o.name == name:
    #                 return o
    #         return None
        
    # def get_initial_object_by_name(self, name, objs):
    #     initial_name = name[:-1] + "1"
    #     for o in objs:
    #         if o.name == initial_name:
    #             return o
    #     return None


    def print_state(self):
        print("---NEUROSYMBOLIC---")
        for p in self.NS_REPR_LIST:
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
                # "properties" : [],
                "functions" : []
            },
            "SELECTION": {
                "objects" : [],
                "actions" : [],
                # "properties" : [],
                "functions" : []
            }
        }

        yaml_dict["ENVIRONMENT"] = self.ENV_NAME
        avail = yaml_dict["AVAILABLE_CONCEPTS"]
        avail["objects"] = self.OBJECT_NAMES #[x.name for x in self.OBJECTS]
        avail["actions"] = [x for x in self.ACTIONS]
        # avail["properties"] = [self.avail_to_yaml_dict(k, v) for k, v in PROPERTIES.items()]
        avail["functions"] =  [self.avail_to_yaml_dict(k, v) for k, v in FUNCTIONS.items()]

        use = yaml_dict["SELECTION"]
        use["objects"] = self.OBJECT_NAMES#[x.name for x in self.OBJECTS]
        use["actions"] = [x for x in self.ACTIONS]
        # use["properties"] = [self.proplist_to_yaml_dict(x) for x in self.PROPERTY_LIST]
        use["functions"] = [self.funclist_to_yaml_dict(x) for x in self.FUNCTION_LIST]

        with open(fpath, "w") as f:
            yaml.dump(yaml_dict, f, sort_keys=False)


    def validate_objects(self, objs):
        if not objs:
            return False
        for o in objs:
            if o not in self.OBJECT_NAMES:
                return False, o
        return True, None


    def validate_actions(self, acts):
        if not acts:
            return False
        for o in acts:
            if o not in self.ACTIONS:
                return False
        return True


    def validate_functions(self, funcs):
        for f in funcs:
            if f not in FUNCTIONS.keys():
                return False
        return True


    def validate_functions_signatures(self, funclist):
        for f in funclist:
            parsed_para_sig = []
            if f[0] not in FUNCTIONS.keys():
                self.logger.FocusFileParserError("Unknown function in function selection: %s" % f[0])
            for para in f[1]:
                if para not in self.NS_REPR_LIST:
                    self.logger.FocusFileParserError("Unknown property in functions selection: %s" % para[0])
                if para[1] not in self.OBJECT_NAMES:
                    self.logger.FocusFileParserError("Unknown object in functions selection: %s" % para[1])
                para_idx = self.NS_REPR_LIST.index(para)
                parsed_para_sig.append(self.NS_REPR_TYPES[para_idx])
            func_definition = FUNCTIONS[f[0]]
            function_sig = [x[0].annotation for x in func_definition["expects"]]
            sig_desc = [x[1] for x in func_definition["expects"]]
            if function_sig != parsed_para_sig:
                self.logger.FocusFileParserError("Signature mismatch in functions selection. Function '%s' expects '%s'" % (f[0], sig_desc))
        return True


    def import_objects(self, objs):
        passed, obj = self.validate_objects(objs)
        if passed:
            return objs
        else:
            self.logger.FocusFileParserError("Invalid objects specified in objects selection: "+obj+" !")

    def import_actions(self, acts):
        if self.validate_actions(acts):
            return acts
        else:
            self.logger.FocusFileParserError("Invalid actions specified in actions selection!")


    def import_functions(self ,funcs):
        out = []
        funcs_to_vali = []
        properties_to_vali = []
        objects_to_vali = []
        if not funcs:
            return []
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
            self.logger.FocusFileParserError("Env and focus file env do not match: %s, %s" % (self.ENV_NAME, parsed_env_name))
        sdict = in_dict["SELECTION"]
        self.PARSED_OBJECTS = self.import_objects(sdict["objects"])
        self.PARSED_ACTIONS = self.import_actions(sdict["actions"])
        self.PARSED_FUNCTIONS = self.import_functions(sdict["functions"])
        # based on the focus file selection,
        # construct a single layer computation graph for the feature vector:
        # 1     FUNC_COMPUTE_LAYER
        parsed_fv_index = 0

        for ns_repr_type in self.NS_REPR_TYPES:
            arg_len = len(str(ns_repr_type).split('[')[1][:-1].split(','))
            for _ in range(arg_len):
                self.FEATURE_VECTOR_BACKMAP.append(parsed_fv_index)
            parsed_fv_index += 1

        #TODO: ADD NS_REPR to FEATURE_VECTOR_BACKMAP
        # for p in self.PARSED_PROPERTIES:
        #     property_name = p[0]
        #     object_name = p[1]
        #     prop_name_obj_name_pairs.append((property_name, object_name))
        #     prop_func = PROPERTIES[property_name]["object"]
        #     return_len = len(PROPERTIES[property_name]["returns"][0].__args__)
        #     for _ in range(return_len):
        #         self.FEATURE_VECTOR_BACKMAP.append(parsed_fv_index)
        #     parsed_fv_index += 1for p in self.PARSED_PROPERTIES:
        #     property_name = p[0]
        #     object_name = p[1]
        #     prop_name_obj_name_pairs.append((property_name, object_name))
        #     prop_func = PROPERTIES[property_name]["object"]
        #     return_len = len(PROPERTIES[property_name]["returns"][0].__args__)
        #     for _ in range(return_len):
        #         self.FEATURE_VECTOR_BACKMAP.append(parsed_fv_index)
        #     parsed_fv_index += 1

        for f in self.PARSED_FUNCTIONS:
            func_name = f[0]
            input_props = f[1]
            property_result_idxs = []
            for p in input_props:
                property_name = p[0]
                object_name = p[1]
                # property_result_idxs.append(prop_name_obj_name_pairs.index((property_name, object_name)))
                property_result_idxs.append(self.NS_REPR_LIST.index([property_name, object_name]))
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
        # self.PROPERTY_COMPUTE_LAYER_SIZE = len(self.PROPERTY_COMPUTE_LAYER)
        self.FUNC_COMPUTE_LAYER_SIZE = len(self.FUNC_COMPUTE_LAYER)
        # self.CURRENT_PROPERTY_COMPUTE_LAYER = [0 for _ in range(self.PROPERTY_COMPUTE_LAYER_SIZE)]
        self.CURRENT_FUNC_COMPUTE_LAYER = [0 for _ in range(self.FUNC_COMPUTE_LAYER_SIZE)]
    
    def ns_repr_list_to_func_input(self, ns_repr_list):
        # might be slow
        out_list = []
        # This assumes that all types are tuples
        idx = 0
        for t in self.NS_REPR_TYPES:
            # get argument length from tuple string (Tuple[int, int] -> 2)
            # TODO: Could move this computation to initialization, s.t. its only done once
            arg_len = len(str(t).split('[')[1][:-1].split(','))
            out_list.append(tuple(ns_repr_list[idx:idx+arg_len]))
            idx += arg_len
        return out_list

    def get_feature_vector(self, inc_ns_repr_list):
        # evaluate a 2 layer computation graph for the feature vector:
        # compute the functions given the properties from the neurosymbolic repres. of OCAtari
        # IN   ns_repres (==property_values)
        # 1     FUNC_COMPUTE_LAYER
        #       function_values
        # OUT   HSTACK(CONCAT(property_values, function_values))
        # Instead of having to compute the properties, we get them from OC_Atari directly

        self.CURRENT_PROPERTY_COMPUTE_LAYER = self.ns_repr_list_to_func_input(inc_ns_repr_list)

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
            if self.HIDE_PROPERTIES:
                self.OBSERVATION_SIZE = len(funcs)
            else:
                self.OBSERVATION_SIZE = len(out)
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
            return np.asarray(out, dtype=np.float32), reward

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
                out[i] = 0 #dont freeze. turns out feezing was very bad
                #out[i] = self.last_obs_vector[i]
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
        return np.asarray(out, dtype=np.float32), reward
    
    def get_feature_vector_description(self):
        # fv = self.PARSED_PROPERTIES + self.PARSED_FUNCTIONS
        fv = self.NS_REPR_LIST + self.PARSED_FUNCTIONS
        return (fv, np.array(self.FEATURE_VECTOR_BACKMAP))
    
    def get_current_freeze_mask(self):
        return self.CURRENT_FREEZE_MASK
    
    def get_reward_func(self, env):
        fv_description, fv_backmap = self.get_feature_vector_description()
        i = 0
        if "Pong" in env:
            # pong reward function
            idxs = np.empty(0)
            for feature in fv_description:
                i += 1
                feature_name = feature[0]
                feature_signature = feature[1]
                if feature_name == "DISTANCE":
                    input1 = feature_signature[0]
                    input2 = feature_signature[1]
                    if input1[0] == "POSITION" and input1[1] == "Player1" and input2[0] == "POSITION" and input2[1] == "Ball1":
                        idxs = np.where(fv_backmap == i-1)[0]
            if not idxs.any():
                return None
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
            player_idxs = np.empty(0)
            distance_idxs = np.empty(0)
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
                    if input1[0] == "POSITION" and input1[1] == "Player1" and input2[0] == "POSITION" and input2[1] == "Ladder1":
                        distance_idxs = np.where(fv_backmap == i-1)[0]
            
            if not (player_idxs.any() and distance_idxs.any()):
                return None
            # reward when player achieves new y-coord low and goes to ladder
            def reward(fv, p_idxs=player_idxs, d_idxs=distance_idxs):
                p_entries = fv[p_idxs[0]:p_idxs[-1]+1]
                y_coord_reward = 0
                if self.reward_threshold == -1: #set starting y
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
                delta = self.reward_history[0] - self.reward_history[1] # decreasing x-distance to Scale1
                distance_reward = delta if 100 > delta else 0 # ignore 100+ spikes
                return y_coord_reward + 5 * distance_reward
            return reward
        elif "Skiing" in env:
            # skiing reward function
            dscale = 1
            player_position_idxs = np.empty(0)
            flag_center_idxs = np.empty(0)
            flag_velocity_idxs = np.empty(0)
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
            if not (player_position_idxs.any() and flag_center_idxs.any() and flag_velocity_idxs.any()):
                return None
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
                player_flag_distance_delta = delta * dscale if p_entries[1] < c_entries[1] else 0 #only scale and send if next flag is ahead not behind
                player_flag_distance_delta = player_flag_distance_delta if abs(player_flag_distance_delta) < 20 * dscale else 0 #omit bad delta spikes when new flag in focus
                euc_velocity_flag = np.clip(math.sqrt((v_entries[0])**2 + (v_entries[1])**2), 0, 10) #clip to 10
                # emit subgoal reward when player passes through 20x10px area around flag center
                if p_entries[0] > (c_entries[0] -10) and p_entries[0] < (c_entries[0] + 10)and p_entries[1] > (c_entries[1] -5) and p_entries[1] < (c_entries[1] + 5):
                    if not self.reward_helper_var:
                        self.reward_subgoals = 100
                        self.reward_helper_var = True
                else:
                    self.reward_helper_var = False
                    self.reward_subgoals = 0
                return self.reward_subgoals + euc_velocity_flag + player_flag_distance_delta
            return reward
        else:
            return "norew"

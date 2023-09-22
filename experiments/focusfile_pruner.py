import yaml
from pathlib import Path
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("-e", "--env", type=str, help="environment", default="Pong")
parser.add_argument("-a", "--actions", type=str, help="actions to prune", default=None)
parser.add_argument("-o", "--objects", type=str, help="objects to prune", default=None)
parser.add_argument("-p", "--properties", type=str, help="properties to prune", default=None)
parser.add_argument("-r", "--relations", type=str, help="relations to prune", default=None)
opts = parser.parse_args()


name = "default_focus_" + opts.env + "-v5.yaml"
actions_to_delete = opts.actions.split(",") if opts.actions is not None else []
properties_to_delete = opts.properties.split(",") if opts.properties is not None else []
relations_to_delete = opts.relations.split(",") if opts.relations is not None else []
objects_to_delete = opts.objects.split(",") if opts.objects is not None else []

focusdir = Path("baselines", "focusfiles")
fdir = Path.cwd() / Path(focusdir)
fpath = fdir / Path(name)
with open(fpath, "r") as f:
    in_dict = yaml.safe_load(f)

parsed_env_name = in_dict["ENVIRONMENT"]
sdict = in_dict["SELECTION"]
objects = sdict["objects"]
actions = sdict["actions"]
properties = sdict["properties"]
relations = sdict["functions"]


# actions
for a in actions_to_delete:
    actions.remove(a)

# properties
pruned_properties = properties
for prop in properties_to_delete:
    pruned_properties = [p for p in pruned_properties if prop not in p.keys()]
sdict["properties"] = pruned_properties

# relations
pruned_relations = relations
for rel in relations_to_delete:
    pruned_relations = [p for p in pruned_relations if rel not in p.keys()]
sdict["functions"] = pruned_relations


# objects
for o in objects_to_delete:
    objects.remove(o)

properties = sdict["properties"]
for o in objects_to_delete:
    properties = [p for p in properties if o not in p.values()]
sdict["properties"] = properties

relations = sdict["functions"]
out = []
#print(relations)
hit = False
for relation in relations:
    hit = False
    for o in objects_to_delete:
        arguments = list(relation.values())[0]
        for a in arguments:
            if o in a.values():
                hit = True
    if not hit:
        out.append(relation)

sdict["functions"] = out

outpath = fdir / Path("pruned_"+opts.env.lower()+".yaml")
with open(outpath, "w") as f:
    yaml.dump(in_dict, f, sort_keys=False)
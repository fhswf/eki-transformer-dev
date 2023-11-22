from torch import nn

gpt = nn.ModuleDict(dict(
            conv1 = nn.Conv2d(1, 32, 3, 1),
            relu1 = nn.ReLU(),
            conv2 = nn.Conv2d(32, 64, 3, 1),
            relu2 = nn.ReLU(),
            maxpool2d = nn.MaxPool2d(kernel_size=2),
            dropout1 = nn.Dropout(0.25),
            flatten = nn.Flatten(),
            fc1 = nn.Linear(9216, 128),
            relu3 = nn.ReLU(),
            dropout2 = nn.Dropout(0.5),
            fc2 = nn.Linear(128, 10),
            nested_layer = nn.ModuleList([nn.GELU() for x in range(10)]),
            #results in: 'deeply_nested_layer.deeply_nested_layer4.6' -> dotted notation directly applicable from root model
            deeply_nested_layer = nn.ModuleDict(dict({"deeply_nested_layer" + str(x):nn.ModuleList([nn.BatchNorm1d(100) for y in range(7)]) for x in range(5)}))
        ))
from typing import List
def concat_strings(strings: List[str]) -> str:
    return ''.join(strings)

#print(gpt["conv2"])
#layers get replaced if they are within the model
gpt.add_module("conv2", nn.Conv2d(34, 70, 8, 3))

#print(gpt["conv2"])
import logging
log = logging.getLogger(__name__)
from re import search, split, match, compile, findall
#makes search go through all sublayers regardless of depth
#gonna be dangerous as the regex needs to be applied to the according layer
#-> split regex patterns by index and search only sublayer
regex_pattern = r'.+\.'

#save all layers that are applicable to the regex in a dict
#idea: name of previous layer is key, children are values
#if highest name is regex: use name of model
#e.g: r'.+': config for model gpt -> all immediate children of model
layers: dict = dict()
name_of_model: str = "gpt"
#-> 2 regex, depth of layer is three
#structure of dotted layers: layer -> layer is directly at root level of model
# layer.layer2 -> layer is at root level, layer2 at layer level

REGEX_SEARCH_PATTERN = r'(r\'[^\']+\'|[^\.]+)'
def test_regex_layer_filter(layer_dotted: str, model: nn.Module):
    #problem is '.' is specified in regex
    #quotation marks cannot appear within python names -> splitter for regex strings
    layers_to_be_added = list()
    #IDEA: create list of re.match objects and filter corresponding layers of model by iterating through each match object
    #if result is zero, log an error that no layer could be found and stop iteration
    #after iteration, add all found layers as layerquantargs
    #'deeply_nested_layer.deeply_nested_layer3.3'
    all_layers_within_model = list(dict(model.named_modules()).keys())
    for layers in list([layer_dotted]):
        regex_layers_to_be_added = list()#dict(gpt.named_modules()).keys()
        #all regular expressions within layer config
        #structure of items: (layer_name, if regex: True, else False)
        sublayers = findall(REGEX_SEARCH_PATTERN, layers)
        if len(sublayers) == 0:
            print(f'Layer config {layers} is an empty string.')
            raise ValueError
        filtered_layer_string = ""
        for sublayer in sublayers:
            print(f'going through layer: {sublayer}')
            is_regex = match(r'r\'([^\']+)\'', sublayer)
            if is_regex:
                #extract actual_regex from r'actual_regex' as search is going to be done with previously iterated layers
                #end of regex ($) cannot be used, otherwise checking stops from that regex
                filtered_layer_string = concat_strings([filtered_layer_string, is_regex.groups()[0].replace("$", ""), "\."])
            elif not sublayer.replace('_', '').isalnum():
                print(f'Sublayer {sublayer} for layer {layers} contains special characters without being encapsulated in a regex term.')
                raise KeyError
            else:
                filtered_layer_string = concat_strings([filtered_layer_string, sublayer, "\."])
        #iteration has added one layer seperator (\.) too much
        filtered_layer_string = filtered_layer_string[:-2]
        print(f'Regex pattern to be compiled: {filtered_layer_string}')
        search_filter = compile(filtered_layer_string)
        regex_layers_to_be_added = list(filter(lambda string: search(search_filter, string), all_layers_within_model))
        layers_to_be_added.extend(regex_layers_to_be_added)
    return layers_to_be_added

#'deeply_nested_layer.deeply_nested_layer3.3' deeply_nested_layer.deeply_nested_layer3.3
test = test_regex_layer_filter(layer_dotted = "deeply_nested_layer.r'deeply_nested_layer[0-2]'.3", model=gpt)
print(test)
#print(list(filter(lambda string: search(regex_pattern, string), list(dict(gpt.named_modules()).keys()))))
#nested_layer within gpt makes names of nested_layer have dotted form -> nested_layer.0 to nested_layer.n
#print(list(dict(gpt.named_modules()).keys()))

#deeply nested getter works
#print(gpt.get_submodule('deeply_nested_layer.deeply_nested_layer3.3'))
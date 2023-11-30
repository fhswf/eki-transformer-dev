from dataclasses import dataclass
from typing import Dict, List
from torch.nn import Module
from re import search, subn, compile, findall, match, Pattern
from logging import getLogger
from qtransform.utils.introspection import concat_strings

log = getLogger(__name__)

LAYER_SEPERATOR_STRING = r'(r\'[^\']+\'|[^\.]+)' #regular expressions within config to apply config for multiple layers should be notated as a python raw string
REGEX_SEARCH_PATTERN = r'r\'([^\']+)\'' #a regex within a layer string should have the structure: "r'regex-term'"

def search_layers_from_module(layer_dotted: str, model: Module) -> Dict[str, Module]:
    """
        Returns the name of all layers that appear inside of a model which have the pattern layer_dotted. 
        layer_dotted is the format a layer is returned by torch.nn.Module.named_modules(), meaning:
        sublayer.sublayer2.sublayer3.final_layer with each layer being seperated by a dot. Each sublayer can also
        be a regex term encapsulated in python raw string notation (r'<regex-to-be-applied>'). If layer_dotted ends with a 
        dot (sublayer.sublayer2.), the last dot is ignored and all layers of a model under sublayer.sublayer2 are found.
    """
    #IDEA: create list of re.match objects and filter corresponding layers of model by iterating through each match object
    #if result is zero, log an error that no layer could be found and stop iteration
    #after iteration, add all found layers as layerquantargs
    #'deeply_nested_layer.deeply_nested_layer3.3'
    all_layers_within_model = list(dict(model.named_modules()).keys())
    #model itself is included in named_modules as ''
    #model config currently only works for sublayers, not the entire model
    all_layers_within_model.pop(0)
    found_layers = search_layers_from_strings(layer_dotted, all_layers_within_model)
    return {x: model.get_submodule(x) for x in found_layers}

#TODO: maybe replace exceptions with property "valid" which is then set to False within SearchResult object
def search_layers_from_strings(layer_dotted: str, layers: List[str]) -> List[str]:
    """
        Constructs a regular expression from layer_dotted and filters all entries within layers with 
        it. layer_dotted needs to be in form of <layer1>.layer2>.etc, each layer can also be a regex in form
        of a raw string r'.+'.etc
    """
    #search all layers for pattern
    search_filter = compile_pattern_from_layerstring(layer_dotted).pattern
    found_layers= list(filter(lambda string: search(search_filter, string), layers))
    return found_layers

@dataclass
class CompileResult():
    regex_index: List[int]
    number_of_sublayers: int
    pattern: Pattern
    

def compile_pattern_from_layerstring(layer_dotted: str, log_errors: bool = True) -> CompileResult:
    """
        Compiles a nested layer string, possibly containing regular expressions, into a Pattern object. The Pattern object
        can be used in order to find layers within a model or to simply test the correctness of a nested layer string.

    """
    sublayers = findall(LAYER_SEPERATOR_STRING, layer_dotted)
    if len(sublayers) == 0:
        print(f'Layer config {layers} is an empty string.')
        raise ValueError
    #the string which is going to be used to filter the model's layers
    filtered_layer_string = ""
    regex_index = list() #note down which layers actually are regex
    for i, sublayer in enumerate(sublayers):
        log.debug(f'going through layer: {sublayer}')
        is_regex = match(REGEX_SEARCH_PATTERN, sublayer)
        if is_regex:
            #extract actual_regex from r'actual_regex' as search is going to be done with previously iterated layers
            #end of regex ($) cannot be used, otherwise checking stops from that regex
            filtered_layer_string = concat_strings([filtered_layer_string, is_regex.groups()[0].replace("$", ""), "\."])
            regex_index.append(i)
        #problem when model for some reason has characters that variable names usually canno have, for example when
        #creating layer names from a string
        elif not sublayer.replace('_', '').isalnum():
            if log_errors: log.error(f'Sublayer \"{sublayer} for layer \"{layer_dotted}\" contains special characters without being encapsulated in a regex term.')
            raise ValueError
        else:
            filtered_layer_string = concat_strings([filtered_layer_string, sublayer, "\."])
    #iteration has added one layer seperator (\.) too much
    filtered_layer_string = filtered_layer_string[:-2]
    search_filter = compile(filtered_layer_string + "$")
    if hasattr(log, "trace"): log.trace(f'Pattern to be compiled: {filtered_layer_string}. Compiled: {search_filter}')
    return CompileResult(regex_index=regex_index, number_of_sublayers=len(sublayers), pattern=search_filter)
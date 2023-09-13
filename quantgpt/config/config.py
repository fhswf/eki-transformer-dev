import yaml
import argparse

class merge_config(argparse.Action):
    """ get config yaml file from command line """
    def __call__(self, parser, namespace, values:str) -> None:
        if not type(values) == str:
            parser.error("{0} should be an string.".format(option_string))
        if not os.path.isdir(values) and not os.path.isfile(values):
            parser.error("Specified path ({0}) not found.".format(values))

        
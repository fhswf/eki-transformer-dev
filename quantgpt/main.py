import argparse

config = {}

class merge_config(argparse.Action):
    def __call__(self, parser, namespace, values) -> None:
        if not type(values) == str:
            parser.error("{0} should be an string.".format(option_string))
        if not os.path.isdir(values) and not os.path.isfile(values):
            parser.error("Specified path ({0}) not found.".format(values))

        setattr(namespace, self.dest, values)

def parse_args(args) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="""
        EKI Quant GPT
    """)

    parser.add_argument('--verbose', action='store_true',
                        help="Enable more explicit output and stack traces")

    return parser.parse_args(args)

def main():
    pass


if __name__ == "__main__": 
    main()
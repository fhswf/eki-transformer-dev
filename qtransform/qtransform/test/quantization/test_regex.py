import unittest
class QuantizationRegexTest(unittest.TestCase):
    """
        Testclass to verify that the correct layers containing regular expressions within the quantization config are found.
        Since the regular expressions are up to the user to create, only basic regex strings are tested and checked if they are
        recognized correctly.
    """
    REGEX_EVERYTHING = 'r\'.+\''
    INVALID_REGEX = '.+'
    pass


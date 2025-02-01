#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import logging

""""""

element_name_vs_Z = {1: "h",
    2: "he",
    3: "li",
    4: "be",
    5: "b" ,
    6: "c" ,
    7: "n" ,
    8: "o" ,
    9: "f" ,
    10: "ne" ,
    11: "na" ,
    12: "mg" ,
    13: "al" ,
    14: "si" ,
    15: "p" ,
    16: "s" ,
    17: "cl" ,
    18: "ar" ,
    19: "k" ,
    20: "ca" ,
    21: "sc" ,
    22: "ti" ,
    23: "v" ,
    24: "cr" ,
    25: "mn" ,
    26: "fe" ,
    27: "co" ,
    28: "ni" ,
    29: "cu" ,
    30: "zn" ,
    31: "ga" ,
    32: "ge" ,
    33: "as" ,
    34: "se" ,
    35: "br" ,
    36: "kr" ,
    37: "rb" ,
    38: "sr" ,
    39: "y" ,
    40: "zr" ,
    41: "nb" ,
    42: "mo" ,
    43: "tc" ,
    44: "ru" ,
    45: "rh" ,
    46: "pd" ,
    47: "ag" ,
    48: "cd" ,
    49: "in" ,
    50: "sn" ,
    51: "sb" ,
    52: "te" ,
    53: "i" ,
    54: "xe" ,
    55: "cs" ,
    56: "ba" ,
    57: "la" ,
    58: "ce" ,
    59: "pr" ,
    60: "nd" ,
    61: "pm" ,
    62: "sm" ,
    63: "eu" ,
    64: "gd" ,
    65: "tb" ,
    66: "dy" ,
    67: "ho" ,
    68: "er" ,
    69: "tm" ,
    70: "yb" ,
    71: "lu" ,
    72: "hf" ,
    73: "ta" ,
    74: "w" ,
    75: "re" ,
    76: "os" ,
    77: "ir" ,
    78: "pt" ,
    79: "au" ,
    80: "hg" ,
    81: "tl" ,
    82: "pb" ,
    83: "bi" ,
    84: "po" ,
    85: "at" ,
    86: "rn" ,
    87: "fr" ,
    88: "ra" ,
    89: "ac" ,
    90: "th" ,
    91: "pa" ,
    92: "u" ,
    93: "np" ,
    94: "pu" ,
    95: "am" ,
    96: "cm" ,
    97: "bk" ,
    98: "cf" ,
    99: "es" ,
    100: "fm" ,
    101: "md" ,
    102: "no" ,
    103: "lr" ,
    104: "rf" ,
    105: "db" ,
    106: "sg" ,
    107: "bh" ,
    108: "hs" ,
    109: "mt" ,
    110: "ds" ,
    111: "rg" ,
    112: "cn" ,
    113: "uut" ,
    114: "uuq" ,
    115: "uup" ,
    116: "uuh",
    117: "uus",
    118: "uuo"}

element_Z_vs_name = {name: Z for Z, name in element_name_vs_Z.items()}

"""
what do I want for input options?

choose nuclei - either list in file or separate path to nuclear list

nuclear masses - must be specified together with nuclear list (add DZ option in the program? no?)

mass unit (MeV, u, mass excess as well, binding energy...)

reaction type(s)

nbins (default maximum)



"""


# create dictionary of all keywords and their corresponding handling function
keyword_dict = {"" : ""}



def read_input(input_path):
    """Reads the input from input_path.
    input_path : path to input file"""

    calculation_args = []

    with open(input_path, "utf8", encoding="utf8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = f.readline()


    return calculation_args





def main():
    args = sys.argv[1:3]
    try:
        input_path = args[0]
        output_path = args[1]
    except IndexError:
        logging.error("Insufficient arguments supplied. Have you given both input and output paths?")
        return
    calculation_args = read_input(input_path)
    




if __name__ == "__main__":
    main()
from data_generate import execute

TALYS_PATH = '~/REACLIB-SOLNAR/talys'

# path to xml file for the baseline masses to be used
XML_PATH = "input_data/webnucleo-nuc2021.xml"

# number of Q-value steps (must be odd)
NUM_QS = 21

# binding energy per nucleon fractional step
BE_STEP = 0.0005

nuclei_lst = [[86, 50], [86, 49]]

execute(nuclei_lst, TALYS_PATH, XML_PATH, NUM_QS, BE_STEP)

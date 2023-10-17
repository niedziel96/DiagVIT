from pathlib import Path

IMAGENET_IMAGE_MEAN = [123.68, 116.779, 103.939]

PROJECT_ROOT_PATH = Path(__file__).parents[1]
DATA_PATH = 'E:/data/patch/DiagSet_patches/data'

DIAGSET_ROOT_PATH = DATA_PATH + '/DiagSet'
DIAGSET_SCANS_PATH = DIAGSET_ROOT_PATH + '/scans'
DIAGSET_ANNOTATIONS_PATH = DIAGSET_ROOT_PATH + '/annotations'

DIAGSET_BLOBS_PATH = {'J': 'E:/data/patch/DiagSet/blobs/J/40x/256x256/128x128/0.75',
                      'S': 'E:/data/patch/DiagSet/blobs/S/40x/256x256/128x128/0.75'}

DIAGSET_POSITIONS_PATH = DIAGSET_ROOT_PATH + '/positions'

DIAGSET_DISTRIBUTIONS_PATH = {'J': 'E:/data/patch/DiagSet/distributions/J/40x/256x256/128x128/0.75',
                              'S': 'E:/data/patch/DiagSet/distributions/S/40x/256x256/128x128/0.75'}

DIAGSET_METADATA_PATH = DIAGSET_ROOT_PATH + '/metadata'

DIAGSET_PARTITIONS_PATH = {'J': 'E:/data/patch/DiagSet/partitions/J',
                           'S': 'E:/data/patch/DiagSet/partitions/S'}


DIAGSET_DEBUG_PATH = DIAGSET_ROOT_PATH + '/debug'
DIAGSET_SCAN_INFO_FILE_PATH = DIAGSET_ROOT_PATH + '/scan_info.xlsx'

TISSUE_TAGS = ['J', 'S', 'P']

EXTRACTED_LABELS = {
    'J': ['BG', 'P=BG', 'T', 'P=T', 'N', 'P=N', 'A', 'P=A', 'R', 'P=R', 'RS', 'P=RS', 'X', 'P=X'],
    'S': ['BG', 'P=BG', 'T', 'P=T', 'N', 'P=N', 'A', 'P=A', 'R1', 'P=R1', 'R2', 'P=R2', 'R3', 'P=R3', 'R4', 'P=R4', 'R5', 'P=R5'],
    'P': ['BG', 'P=BG', 'T', 'P=T', 'N', 'P=N', 'A', 'P=A', 'R1', 'P=R1', 'R2', 'P=R2']
}

IGNORED_LABELS = {
    'J': ['P'],
    'S': ['P'],
    'P': ['P']
}

LABEL_TRANSLATIONS = {
    'J': {'P=BG': 'BG', 'P=T': 'T', 'P=N': 'N', 'P=A': 'A', 'P=R': 'R', 'P=RS': 'RS', 'P=X': 'X'},
    'S': {'P=BG': 'BG', 'P=T': 'T', 'P=N': 'N', 'P=A': 'A', 'P=R1': 'R1', 'P=R2': 'R2', 'P=R3': 'R3', 'P=R4': 'R4', 'P=R5': 'R5'},
    'P': {'P=BG': 'BG', 'P=T': 'T', 'P=N': 'N', 'P=A': 'A', 'P=R1': 'R1', 'P=R2': 'R2'}
}

USABLE_LABELS = {
    tag: [
        label for label in EXTRACTED_LABELS[tag] if label not in LABEL_TRANSLATIONS[tag].keys()
    ] for tag in TISSUE_TAGS
}

LABEL_ORDER = {
    'J': ['R', 'RS', 'X', 'A', 'N', 'T', 'BG'],
    'S': ['R5', 'R4', 'R3', 'R2', 'R1', 'A', 'N', 'T', 'BG'],
    'P': ['R2', 'R1', 'A', 'N', 'T', 'BG']
}

LABEL_DICTIONARIES = {
    'J': {
        'BG': 0, 'T': 1, 'N': 2, 'A': 3, 'R': 4, 'RS': 5, 'X': 6
    },
    'S': {
        'BG': 0, 'T': 1, 'N': 2, 'A': 3, 'R1': 4, 'R2': 5, 'R3': 6, 'R4': 7, 'R5': 8
    },
    'P': {
        'BG': 0, 'T': 1, 'N': 2, 'A': 3, 'R1': 4, 'R2': 5
    }
}

ORGAN_TO_TAG_MAPPING = {
    u'stercz': 'S',
    u'prostata': 'S',
    u'gruczoł krokowy': 'S',
    u'jelito grube': 'J',
    u'esica': 'J',
    u'kątnica': 'J',
    u'odbytnica': 'J',
    u'okrężnica': 'J',
    u'poprzecznica': 'J',
    u'wstępnica': 'J',
    u'zstępnica': 'J',
    u'płuco': 'P',
    u'oskrzele': 'P'
}

ENSEMBLE_PARAMETERS = {
    'J': None,
    'S': None,
    'P': None
}

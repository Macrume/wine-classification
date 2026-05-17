from pathlib import Path

def load_feature_names(pathToNames : Path):
     with open(pathToNames) as names:
            return [str(name).strip() for line in names.readlines() for name in line[:-1].split(',')] 
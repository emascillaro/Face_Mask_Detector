from pathlib import Path
from google_drive_downloader import GoogleDriveDownloader as gdd
from tqdm import tqdm

import pandas as pd

# Assign Paths to Datasets
datasetPath = Path('dataset/self-built-masked-face-recognition-dataset')
maskPath = datasetPath/'AFDB_masked_face_dataset'
nonMaskPath = datasetPath/'AFDB_face_dataset'

# Create Dataframe
maskDF = pd.DataFrame()

# Assign 0 to all images w/o mask
for subject in tqdm(list(nonMaskPath.iterdir()), desc='non mask photos'):
    for imgPath in subject.iterdir():
        image = cv2.imread(str(imgPath))
        maskDF = maskDF.append({
            'image': image,
            'mask': 0
        }, ignore_index=True)

# Assign 1 to all images w/ mask
for subject in tqdm(list(maskPath.iterdir()), desc='mask photos'):
    for imgPath in subject.iterdir():
        image = cv2.imread(str(imgPath))
        maskDF = maskDF.append({
            'image': image,
            'mask': 1
        }, ignore_index=True)

maskDF.to_pickle('data/mask_df.pickle')
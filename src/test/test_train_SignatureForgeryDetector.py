from  ..models import train_model_SignatureForgeryDetection_Resnet50 as fD
import pandas as pd

def test_categorizeImages():
    #given a dataframe of genuine and forged signatures it returns combination of anchor, pos & negative image paths.
    inputData = {"personId": [1,1,1,1,1], "fileName": ["Genuine_01.PNG","Genuine_02.PNG","Forge_01.png","Forge_02.png","Forge_03.png"],
 "relPath": ["001","001","001_forge","002_forge","003_forge"], "Genuine":[1,1,0,0,0] }

    
    expectedData = (
     ['/notebooks/capstone/dataset/dataset2/sign_data/test/001/Genuine_01.PNG',
      '/notebooks/capstone/dataset/dataset2/sign_data/test/001/Genuine_01.PNG',
      '/notebooks/capstone/dataset/dataset2/sign_data/test/001/Genuine_01.PNG'],
     ['/notebooks/capstone/dataset/dataset2/sign_data/test/001/Genuine_02.PNG',
      '/notebooks/capstone/dataset/dataset2/sign_data/test/001/Genuine_02.PNG',
      '/notebooks/capstone/dataset/dataset2/sign_data/test/001/Genuine_02.PNG'],
     ['/notebooks/capstone/dataset/dataset2/sign_data/test/001_forge/Forge_01.png',
      '/notebooks/capstone/dataset/dataset2/sign_data/test/002_forge/Forge_02.png',
      '/notebooks/capstone/dataset/dataset2/sign_data/test/003_forge/Forge_03.png'])
    
    df = pd.DataFrame(inputData)
    
    assert(expectedData == fD.categorizeImages(df, "test"))

    
def test_categorizeTestImages():
    #given a dataframe of genuine and forged signatures it returns combination of anchor, pos & negative image paths such that all pos are compared against all other pos and all negs are compared with all the other postives.
    inputData = {"personId": [1,1,1,1,1,1]
                 , "fileName": ["Genuine_01.PNG","Genuine_02.PNG","Genuine_03.PNG", "Forge_01.png","Forge_02.png","Forge_03.png"]
                 , "relPath": ["001","001","001","001_forge","002_forge","003_forge"], "Genuine":[1,1,1,0,0,0] }
    expectedResult = ([1, 1, 1, 1, 1, 1, 1, 1],
         ['/notebooks/capstone/dataset/dataset2/sign_data/test/001/Genuine_01.PNG',
          '/notebooks/capstone/dataset/dataset2/sign_data/test/001/Genuine_01.PNG',
          '/notebooks/capstone/dataset/dataset2/sign_data/test/001/Genuine_01.PNG',
          '/notebooks/capstone/dataset/dataset2/sign_data/test/001/Genuine_01.PNG',
          '/notebooks/capstone/dataset/dataset2/sign_data/test/001/Genuine_01.PNG',
          '/notebooks/capstone/dataset/dataset2/sign_data/test/001/Genuine_01.PNG',
          '/notebooks/capstone/dataset/dataset2/sign_data/test/001/Genuine_01.PNG',
          '/notebooks/capstone/dataset/dataset2/sign_data/test/001/Genuine_01.PNG'],
         ['/notebooks/capstone/dataset/dataset2/sign_data/test/001/Genuine_02.PNG',
          '/notebooks/capstone/dataset/dataset2/sign_data/test/001/Genuine_02.PNG',
          '/notebooks/capstone/dataset/dataset2/sign_data/test/001/Genuine_02.PNG',
          '/notebooks/capstone/dataset/dataset2/sign_data/test/001/Genuine_02.PNG',
          '/notebooks/capstone/dataset/dataset2/sign_data/test/001/Genuine_03.PNG',
          '/notebooks/capstone/dataset/dataset2/sign_data/test/001/Genuine_03.PNG',
          '/notebooks/capstone/dataset/dataset2/sign_data/test/001/Genuine_03.PNG',
          '/notebooks/capstone/dataset/dataset2/sign_data/test/001/Genuine_03.PNG'],
         ['/notebooks/capstone/dataset/dataset2/sign_data/test/001_forge/Forge_01.png',
          '/notebooks/capstone/dataset/dataset2/sign_data/test/002_forge/Forge_02.png',
          '/notebooks/capstone/dataset/dataset2/sign_data/test/003_forge/Forge_03.png',
          '/notebooks/capstone/dataset/dataset2/sign_data/test/001/Genuine_03.PNG',
          '/notebooks/capstone/dataset/dataset2/sign_data/test/001_forge/Forge_01.png',
          '/notebooks/capstone/dataset/dataset2/sign_data/test/002_forge/Forge_02.png',
          '/notebooks/capstone/dataset/dataset2/sign_data/test/003_forge/Forge_03.png',
          '/notebooks/capstone/dataset/dataset2/sign_data/test/001/Genuine_02.PNG'],
         [False, False, False, True, False, False, False, True])
    
    df = pd.DataFrame(inputData)
    
    assert(expectedResult == fD.categorizeTestImages(df, "test"))
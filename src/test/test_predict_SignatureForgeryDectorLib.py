## Test to confirm a good signature
from  ..models.predict_SignatureForgeryDetectorLib import isForgery as fD


def testGoodSignature():
    assert (fD(51, '/notebooks/capstone/dataset/dataset2/sign_data/test/051/06_051.png') == "Probablity of forgery 0.0%")

def testForgerySignature():
    assert (fD(51, '/notebooks/capstone/dataset/dataset2/sign_data/test/049_forg/01_0114049.PNG') == "Probablity of forgery 100.0%")

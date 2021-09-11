import SignerVerifier as s

import io

from unittest.mock import Mock
from unittest.mock import patch

app = s.create_app()
tc = app.test_client()

def testAllowed_file():
    assert(s.allowed_file("x.pdf"))

def testPerformSignatureVerification():
    assert (s.performSignatureVerification(49, '/notebooks/capstone/dataset/dataset2/sign_data/test/049_forg/01_0114049.PNG') == "PersonId: 49: Probablity of forgery 100.0%")

def testIndexPageImpl():
    rsp = tc.get("/")
    expectedRsp = b'<!DOCTYPE html>\n<html>\n<head>\n    <title>Index</title>\n</head>\n<body>\n\n\t<!doctype html>\n\t<title>Upload new File</title>\n\t<h1>Uploads new File</h1>\n\t<form method=post enctype=multipart/form-data action="/upload">\n\t\tPersonId: <input type=text name=personId><br>\n\t\tSignature:<input type=file name=file><br><br>\n\t\t<input type=submit value=Submit>\n\t</form>\n\n    <img src="" alt="User Image" style="max-height: 20em;max-width: 20em;"><b>None </b>\n</body>\n</html>'
    print("###" + str(rsp.get_data()) + "###");
    assert(200 == rsp.status_code)
    assert(expectedRsp == rsp.get_data())

def testIndexPage():
    rsp = tc.get("/")
    expectedRsp = b'<!DOCTYPE html>\n<html>\n<head>\n    <title>Index</title>\n</head>\n<body>\n\n\t<!doctype html>\n\t<title>Upload new File</title>\n\t<h1>Uploads new File</h1>\n\t<form method=post enctype=multipart/form-data action="/upload">\n\t\tPersonId: <input type=text name=personId><br>\n\t\tSignature:<input type=file name=file><br><br>\n\t\t<input type=submit value=Submit>\n\t</form>\n\n    <img src="" alt="User Image" style="max-height: 20em;max-width: 20em;"><b>None </b>\n</body>\n</html>'
    #print("###" + str(rsp.get_data()) + "###");
    assert(expectedRsp == rsp.get_data())

def testIndexPage():
    rsp = tc.get('/?result=UT_Result&filePath=UT_Filepath', follow_redirects=True)

    expectedRsp =b'<!DOCTYPE html>\n<html>\n<head>\n    <title>Index</title>\n</head>\n<body>\n\n\t<!doctype html>\n\t<title>Upload new File</title>\n\t<h1>Uploads new File</h1>\n\t<form method=post enctype=multipart/form-data action="/upload">\n\t\tPersonId: <input type=text name=personId><br>\n\t\tSignature:<input type=file name=file><br><br>\n\t\t<input type=submit value=Submit>\n\t</form>\n\n    <img src="UT_Filepath" alt="User Image" style="max-height: 20em;max-width: 20em;"><b>UT_Result </b>\n</body>\n</html>'
    print("###" + str(rsp.get_data()) + "###");
    assert(expectedRsp == rsp.get_data())

def testUploadFile():
    rsp = tc.get("/upload")
    expectedRsp = b'''<!doctype html>
 <title>Upload new File</title>
 <h1>Upload new File</h1>
 <form method=post enctype=multipart/form-data>
   <input type=file name=file>
   <input type=submit value=Upload>
</form>
'''
    #print("###" + str(rsp.get_data()) + "###");
    assert(expectedRsp == rsp.get_data())

def testUploadFile():
    rsp = tc.post("/upload")
    expectedRsp = b'<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">\n<title>Redirecting...</title>\n<h1>Redirecting...</h1>\n<p>You should be redirected automatically to target URL: <a href="http://localhost/upload">http://localhost/upload</a>. If not click the link.'
    print("###" + str(rsp.get_data()) + "###");
    assert(expectedRsp == rsp.get_data())

#@patch.object(s,'performSignatureVerification', return_value='!!Whew!!')
def testUploadFileImpl():

    data = {'personId': '51'}
    with open('/notebooks/capstone/dataset/dataset2/sign_data/test/049_forg/01_0114049.PNG', 'rb') as f:
      data['file'] = (io.BytesIO(f.read()), 'test.png')
    rsp = tc.post(
        '/upload',content_type='multipart/form-data', data=data, follow_redirects=True)

    expectedRsp =b'<!DOCTYPE html>\n<html>\n<head>\n    <title>Index</title>\n</head>\n<body>\n\n\t<!doctype html>\n\t<title>Upload new File</title>\n\t<h1>Uploads new File</h1>\n\t<form method=post enctype=multipart/form-data action="/upload">\n\t\tPersonId: <input type=text name=personId><br>\n\t\tSignature:<input type=file name=file><br><br>\n\t\t<input type=submit value=Submit>\n\t</form>\n\n    <img src="./upload/test.png" alt="User Image" style="max-height: 20em;max-width: 20em;"><b>PersonId: 51: Probablity of forgery 100.0% </b>\n</body>\n</html>'
    #print("###" + str(rsp.get_data()) + "###");
    assert(expectedRsp == rsp.get_data())

def testCreate_app():
    assert(True);


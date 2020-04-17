import urllib.request
import json
import os

def DownloadSingleFile(fileURL, cnt):
    print('Downloading image...')
    DIR = f'./DB/test'
    if not os.path.exists(DIR):
        os.mkdir(DIR)
    fileName = DIR + '/' + str("%03d"%cnt) + '.jpg'
    urllib.request.urlretrieve(fileURL, fileName)
    print('Done. ' + fileName)


if __name__ == '__main__':

    with open('./output_seoulstore.json') as data_file:
        data = json.load(data_file)

    for i in range(0, len(data)):
        instagramURL = data[i]['img_url']
        DownloadSingleFile(instagramURL, i)

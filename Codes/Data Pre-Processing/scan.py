import datetime
import re
import time

import requests
import winsound
from tqdm import tqdm

dpi = 300

headers2 = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/110.0',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Pragma': 'no-cache',
    'Cache-Control': 'no-cache',
}

data = r'<scan:ScanSettings xmlns:scan="http://schemas.hp.com/imaging/escl/2011/05/03" ' \
       r'xmlns:dd="http://www.hp.com/schemas/imaging/con/dictionaries/1.0/" ' \
       r'xmlns:dd3="http://www.hp.com/schemas/imaging/con/dictionaries/2009/04/06" ' \
       r'xmlns:fw="http://www.hp.com/schemas/imaging/con/firewall/2011/01/05" ' \
       r'xmlns:scc="http://schemas.hp.com/imaging/escl/2011/05/03" ' \
       r'xmlns:pwg="http://www.pwg.org/schemas/2010/12/sm"><pwg:Version>2.1</pwg:Version>' \
       r'<scan:Intent>Photo</scan:Intent><pwg:ScanRegions><pwg:ScanRegion><pwg:Height>3507</pwg:Height>' \
       r'<pwg:Width>2481</pwg:Width><pwg:XOffset>0</pwg:XOffset><pwg:YOffset>0</pwg:YOffset>' \
       r'</pwg:ScanRegion></pwg:ScanRegions><pwg:InputSource>Platen</pwg:InputSource>' \
       rf'<scan:DocumentFormatExt>image/jpeg</scan:DocumentFormatExt><scan:XResolution>{dpi}</scan:XResolution>' \
       rf'<scan:YResolution>{dpi}</scan:YResolution><scan:ColorMode>RGB24</scan:ColorMode>' \
       rf'<scan:CompressionFactor>{0}</scan:CompressionFactor><scan:Brightness>1000</scan:Brightness>' \
       r'<scan:Contrast>1000</scan:Contrast></scan:ScanSettings>'


def check_status():
    cookies = {
        'sid': 's04d7fe0b-66619f4c0e2d9a22aebe35a110ee627d',
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/110.0',
        'Accept': 'application/xml, text/xml, */*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Referer': 'http://192.168.0.104/',
        'Pragma': 'no-cache',
        'Cache-Control': 'no-cache',
    }

    response = requests.get('http://192.168.0.104/eSCL/ScannerStatus', cookies=cookies, headers=headers)
    c = response.content
    regex = r"\<pwg\:JobState\>(\S+)\<\/pwg\:JobState\>"
    match = re.findall(regex, c.decode())
    print(match)
    return all(m in ("Completed", "Aborted") for m in match)


# print(check_status())
# exit()


def scan_page():
    global headers1, headers2, data
    response = requests.post(
        'http://192.168.0.104/eSCL/ScanJobs',
        cookies={
            'sid': 's04d7fe0b-66619f4c0e2d9a22aebe35a110ee627d',
        },
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/110.0',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Content-Type': 'text/xml',
            'Origin': 'http://192.168.0.104',
            'Connection': 'keep-alive',
            'Referer': 'http://192.168.0.104/',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache',
        },
        data=data
    )
    if response.status_code == 201:
        print(f'Scanner link: {response.headers["Location"]}')
    else:
        print(f"Invalid response code {response.status_code}")
        raise ValueError
    response.close()
    response = requests.get(
        f'{response.headers["Location"]}/NextDocument',
        cookies={
            'sid': 's04d7fe0b-66619f4c0e2d9a22aebe35a110ee627d',
        },
        headers=headers2,
        stream=True,
    )
    if response.status_code == 200:
        filename = f"{datetime.datetime.now().strftime('%Y-%m-%d %I.%M.%S %p')}.jpg"
        print(filename)

        total_size_in_bytes = int(response.headers.get('content-length', 1274331))  # 2910642))
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

        with open(filename, 'wb') as f:
            for data in response.iter_content(1024 * 4):
                progress_bar.update(len(data))
                f.write(data)

        progress_bar.close()
        winsound.Beep(1200, 700)
        response.close()
    else:
        print(f'Invalid response code: {response.status_code}')


if __name__ == '__main__':
    while not check_status():
        time.sleep(1)
        print("Sleeping")
    scan_page()

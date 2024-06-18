"""
@big_name:data_teacher	
@file_name:set_cookie	
@data:2024/6/2	
@developers:handsome_lxh
"""
import requests
import json
# 不知道怎么用看这里
# https://www.w3cschool.cn/minicourse/play/antpython06?cp=26206&gid=0
cookie_jar = requests.cookies.RequestsCookieJar()
with open("./cookie.txt") as fin:
    cookiejson = json.loads(fin.read())
    print(cookiejson)
    for cookie in cookiejson:
        cookie_jar.set(
            name=cookie["name"],
            value=cookie["value"],
            domain=cookie["domain"],
            path=cookie["path"]
        )
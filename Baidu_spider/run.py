#coding=utf-8
from BaiduImageSearch import BaiduImage
import sys


keyword = " ".join(sys.argv[1:])
save_path = "_".join(sys.argv[1:])

if not keyword:
    print "亲，你忘记带搜索内容了哦~  搜索内容关键字可多个，使用空格分开"
    print "例如：python run.py 男生 头像"
else:
    search = BaiduImage(keyword, save_path=save_path)
    search.search()

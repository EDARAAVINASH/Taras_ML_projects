import pyqrcode

url="welcome to CSE department"
k=pyqrcode.create(url)
k.svg("lbrce.svg",scale=20)

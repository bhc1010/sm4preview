from sys import argv
from sm4preview import SM4File

def run(fpath):
    sm4 = SM4File(fpath)
    sm4.generate_preview()

if __name__ == '__main__':
    run(argv[1])
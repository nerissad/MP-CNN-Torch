import numpy as np
import sys 
from tempfile import TemporaryFile as tFile
import pdb

path = sys.argv[1]
vocabpath = sys.argv[2]
vecpath = sys.argv[3]

print(path)
print(vocabpath)
print(vecpath)

file=open(path, "r")
count=0
dim=len(file.readline().split())-1
buf_size = 1024 * 1024
read_f = file.read
buf = read_f(buf_size)
while buf:
    count += buf.count('\n')
    buf = read_f(buf_size)
print(count)
print(dim)

vocabf=open(vocabpath,"w")
vecs=np.ndarray((count,dim),dtype='float32')
file.seek(0) 
pdb.set_trace()
for i in range(1,count):
    l=file.readline().split()
    vocabf.write(l[0]+"\n")
    vecs[i-1,:]=l[1:len(l)]
file.close()
vocabf.close()
np.save(vecpath,vecs)
            


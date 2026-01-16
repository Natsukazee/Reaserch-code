#输出数据到output文件中
import math
import re

output_file='output-dNC.txt'

#打开输出文件
with open(output_file,'w') as o:

#打开输入文件
  with open('MOVEMENT.txt','r') as f :
    line=f.readline()
    i=1
    j=1
    while line:
        if j==i*294+65-294:
            print(i, end= ' ', file=o)#打印时间，单位fs
            newline=line.replace("\n",' ')
            print(newline,end=' ', file=o)
            pattern=r'\b\d+\.\d+\b'
            numbers1=re.findall(pattern,line)
            fnumbers1=[0,0,0]
            n=0
            for x in numbers1:
                fnumbers1[n]=float(x)
                n+=1
        if j==i*294+101-294:
            newline = line.replace("\n", ' ')
            print(newline,end=' ',file=o)
            i+=1
            pattern=r'\b\d+\.\d+\b'
            numbers2=re.findall(pattern,line)
            fnumbers2 = [0, 0, 0]
            n = 0
            for x in numbers2:
                fnumbers2[n]=float(x)
                n+=1
            distance=math.sqrt((fnumbers1[0]*10.721-fnumbers2[0]*10.721) ** 2 + (fnumbers1[1]*7.147 - fnumbers2[1]*7.147) ** 2 + ((fnumbers1[2]+1)*7.147 - fnumbers2[2]*7.147) ** 2)
            print(distance,file=o)#打印NC之间的距离，单位A
        j+=1
        line=f.readline()
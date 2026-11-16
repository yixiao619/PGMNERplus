"""
该代码是将Twitter2015或者Twitter2017文件中的"IMAGID**"行去掉
"""
import os

def process_file(input_path,output_path):
    fo=open(output_path,"a+")
    with open(input_path,"r") as fi:
        lines=fi.readlines()
        for line in lines:
            if line.startswith("IMGID"):
                continue
            else:
                line=line.split("\t")
            line=" ".join(line)
            fo.write(line)
    fo.close()

with open("../data/twitter2015/new_dev.txt") as f1:
    lines1=f1.readlines()
    print(len(lines1))
    print(lines1[:10])

with open("../data/Twitter/dev.txt") as f2:
    lines2 = f2.readlines()
    print(len(lines2))
    print(lines2[:10])

diffrent_num=0
for index,line in enumerate(lines2[:-1]):
    try: assert lines1[index]==lines2[index]

    except:
        diffrent_num+=1
        print(index)
        print(lines2[index])
        print(lines1[index])
print(diffrent_num)


# if __name__ == '__main__':
#     process_file("../data/twitter2015/dev.txt","../data/twitter2015/new_dev.txt")


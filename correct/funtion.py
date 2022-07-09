import os
from venv import create
from black import out
import numpy as np
import cv2
from pathlib import Path
import json
from sympy import bspline_basis, false, true
from operator import itemgetter
from symspellpy import SymSpell
import jieba

import soundshapecode as sc

# 辅助性代码

def score_of_str(s_,s):
    n = len(s)
    correct = 0
    last_idx = 0
    for c in s_:
        i = s[last_idx:].find(c)
        if i!=-1:
            last_idx = i
            correct+=1
    return n, correct

def is_Chinese(ch):
    if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

def select_chi(s):
    t = ''
    for c in s:
        if is_Chinese(c):
            t+=c
    return t

def read_img_2_list(img_path):
    # 读取图片
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    # 把图片转换为灰度模式
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(-1, 1)
    return [_[0] for _ in img.tolist()]

# 获取所有汉字的向量表示，以dict储存
def get_all_char_vectors():
    image_paths = [_ for _ in os.listdir("./correct/word_img") if _.endswith("png")]

    img_vector_dict = {}
    for image_path in image_paths:
        img_vector_dict[image_path[0]] = read_img_2_list(img_path="./correct/word_img/"+image_path)

    return img_vector_dict

# 计算两个向量之间的余弦相似度
def cosine_similarity(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return dot_product / ((normA**0.5)*(normB**0.5))

# 输入文字，输出余弦相似度
def word_to_cosine(word1,word2):
    image_paths = [_ for _ in os.listdir("./correct/word_img") if _.endswith("png")]

    vector1 = read_img_2_list(img_path="./correct/word_img/"+ word1+".png")
    vector2 = read_img_2_list(img_path="./correct/word_img/"+ word2+".png")

    return cosine_similarity(vector1,vector2)

# 生成印章预测与答案txt
def maketxt():
    pred_uncover_dir = './dataset/data_disposed/data_new/data_uncovered/result_ylh_v3/'
    pred_cover_dir = './dataset/data_disposed/data_new/data_covered/result_ylh_v3/'
    ans_uncover_dir = './dataset/data_disposed/data_new/answer/uncover_json/'
    ans_cover_dir = './dataset/data_disposed/data_new/answer/cover_json/'

    # # 生成 未覆盖预测txt
    # s=[]
    # s_new=[]    
    # for pred_path in os.listdir(pred_uncover_dir):
    #     sfn = os.path.splitext(pred_path)[0]
    #     pred_path = pred_uncover_dir + pred_path

    #     with open(pred_path, 'r', encoding='utf-8') as f:
    #         pred = json.load(f)
    #         s.append(pred['2'])
        

    # with open('./correct/pred_uncover.txt', 'w', encoding='utf-8') as ftxt:
    #     for i in s:
    #         # if i not in s_new :
    #             # if i != '':
    #         for item in i:
    #             s_new.append(item)
    #             ftxt.write(item+'\n')
    

    # # 生成 覆盖预测txt
    # s=[]
    # s_new=[]    
    # for pred_path in os.listdir(pred_cover_dir):
    #     sfn = os.path.splitext(pred_path)[0]
    #     pred_path = pred_cover_dir + pred_path

    #     with open(pred_path, 'r', encoding='utf-8') as f:
    #         pred = json.load(f)
    #         s.append(pred['2'])

    # with open('./correct/pred_cover.txt', 'w', encoding='utf-8') as ftxt:
    #     for i in s:
    #         # if i not in s_new :
    #             # if i != '':
    #         for item in i:
    #             s_new.append(item)
    #             ftxt.write(item+'\n')

    # 生成 不覆盖答案txt
    s=[]
    s_new=[]    
    for ans_path in os.listdir(ans_uncover_dir):
        sfn = os.path.splitext(ans_path)[0]
        ans_path = ans_uncover_dir + ans_path

        with open(ans_path, 'r', encoding='utf-8') as f:
            ans = json.load(f)
            s.append(ans['2'])

    with open('./correct/ans_uncover.txt', 'w', encoding='utf-8') as ftxt:
        for i in s:
            # if i not in s_new :
                # if i != '':
            for item in i:
                s_new.append(item)
                ftxt.write(item+'\n')

    # 生成 覆盖答案txt
    s=[]
    s_new=[]    
    for ans_path in os.listdir(ans_cover_dir):
        sfn = os.path.splitext(ans_path)[0]
        ans_path = ans_cover_dir + ans_path

        with open(ans_path, 'r', encoding='utf-8') as f:
            ans = json.load(f)
            s.append(ans['2'])

    with open('./correct/ans_cover.txt', 'w', encoding='utf-8') as ftxt:
        for i in s:
            # if i not in s_new :
                # if i != '':
            for item in i:
                s_new.append(item)
                ftxt.write(item+'\n')

# 生成印章类型预测与答案txt
def maketxt_stamps():
    pred_dir = './dataset/data_disposed/data_new/data_covered/result_zc/'
    ans_dir = Path('./dataset/data_disposed/data_new/data_covered/answer/')
    s=[]
    s_new=[]
    
    # 打开pred_dir
    for pred_path in os.listdir(pred_dir):
        sfn = os.path.splitext(pred_path)[0]
        pred_path = pred_dir + pred_path

        with open(pred_path, 'r', encoding='utf-8') as f:
            pred = json.load(f)
            if('0' in ans):
                s.append(ans['0'])
        with open((ans_dir/f'{sfn}.json'), 'r', encoding='utf-8') as f:
            ans = json.load(f)
            if('0' in ans):
                s.append(ans['0']) 

    # 打开pred_dir
    for pred_path in os.listdir(pred_dir):
        sfn = os.path.splitext(pred_path)[0]
        pred_path = pred_dir + pred_path

        with open(pred_path, 'r', encoding='utf-8') as f:
            pred = json.load(f)
        with open((ans_dir/f'{sfn}.json'), 'r', encoding='utf-8') as f:
            ans = json.load(f)
            s.append(ans['0']) 

    with open('./correct/stamps_predtxt.txt', 'w', encoding='utf-8') as ftxt:
        for i in s:
            # if i not in s_new :
                # if i != '':
            s_new.append(i)
            ftxt.write(i+'\n')

    with open('./correct/stamps_anstxt.txt', 'w', encoding='utf-8') as ftxt:
        for i in s:
            # if i not in s_new :
                # if i != '':
            s_new.append(i)
            ftxt.write(i+'\n')
   
# 创建文字库
def creat_hanzi():
    import pygame

    pygame.init()
    font = pygame.font.SysFont('Microsoft YaHei', 64)

    # 获取3500个汉字
    with open("./correct/common_data/all_3500_chars.txt", "r", encoding="utf-8") as f:
        chars = f.read().strip()
    
    # 输出图片
    flag=1
    for char in chars:
        if flag==1 & (char!='\n'):
            flag=0
            continue
        if(isinstance(char,str)):
            rtext = font.render(char, True, (0, 0, 0), (255, 255, 255))
            pygame.image.save(rtext, "./correct/word_img/{}.png".format(char))

def sample():

    # sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)

    # dictionary_path = pkg_resources.resource_filename(
    #     "symspellpy", "zh-50k.txt"
    # )

    # print(dictionary_path)

    # sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

    # resout=sym_spell.word_segmentation("杭州师范大学",max_edit_distance=0)
    # print("杭州师范大学","-->",resout.segmented_string)
    # temp=resout.segmented_string.split()
    str="这天云熵区睡"
    temp=jieba.lcut(str, cut_all=False)
    print(temp) 

# jieba 制作词典
def jieba_creat():

    comps= ()
    tempdict={}

    with open('./correct/comps.txt', 'r', encoding='utf-8') as ftxt:
        comps=ftxt.read().splitlines()

    for i in range(len(comps)):
        temp=jieba.lcut_for_search(comps[i])

        for k in temp:
            flag=true
            for key in tempdict.keys():
                if k==key:
                    tempdict[key]+=1
                    flag=false
                    break
            if flag:
                tempdict[k]=1

    with open('./correct/data_word.txt', 'w', encoding='utf-8') as ftxt:
        # for key in tempdict.keys():
        #     if((len(key)==1)&(tempdict[key]<=30)):
        #         tempdict[key]=30
        #     elif((len(key)>=2)&(tempdict[key]<=80)):
        #         tempdict[key]=80
        
        for key in tempdict.keys():
            ftxt.write(key+" "+str(tempdict[key])+"\n")

    with open('./correct/data.txt', 'w', encoding='utf-8') as ftxt:   
        # for key in tempdict.keys():
        #     if((len(key)==1)&(tempdict[key]<=30)):
        #         tempdict[key]=30
        #     elif((len(key)>=2)&(tempdict[key]<=80)):
        #         tempdict[key]=80
        
        for key in tempdict.keys():
            if(len(key)!=1):
                ftxt.write(key+" "+str(tempdict[key])+"\n")

# 纯预先相似度
def cal_similarity():
    img_vector_dict = get_all_char_vectors()

    # 获取最接近的汉字
    similarity_dict = {}
    
    while True:
        match_char = '国'
        match_vector = img_vector_dict[match_char]
        for char, vector in img_vector_dict.items():
            cosine_similar = cosine_similarity(match_vector, vector)
            similarity_dict[char] = cosine_similar
        # 按相似度排序，取前10个
        sorted_similarity = sorted(similarity_dict.items(), key=itemgetter(1), reverse=True)
        print([(char, round(similarity, 4))for char, similarity in sorted_similarity[:10]])

# 纯形码
def cal_similarity2():
    chi_word1 = '国'
    chi_word2 = ''
    sc.ssc.getHanziStrokesDict()
    sc.ssc.getHanziStructureDict()
    #ssc.generateHanziSSCFile()#生成汉子-ssc映射文件
    sc.ssc.getHanziSSCDict()

    with open("./correct/common_data/all_3500_chars.txt", "r", encoding="utf-8") as f:
        chars = f.read().strip()

    sim=[]
    chi_word1_ssc = sc.ssc.getSSC(chi_word1, 'SHAPE')
    # print(chi_word1_ssc)
    
    for char in chars:
        chi_word2_ssc = sc.ssc.getSSC(char, 'SHAPE')
        # print(chi_word2_ssc)
        output=sc.cp.computeShapeCodeSimilarity(chi_word1_ssc[0],chi_word2_ssc[0])
        sim.append((char,output))
    
    sim.sort(key=lambda x: (x[1], x[0]), reverse=True)  # 按列表的第1个元素正序，第二个元素相同的按第0个元素正序
    print(sim[0:5])

# 结合版本
def cal_similarity3():
    # 形码第一步筛选
    chi_word1 = '国'
    chi_word2 = ''
    sc.ssc.getHanziStrokesDict()
    sc.ssc.getHanziStructureDict()
    #ssc.generateHanziSSCFile()#生成汉字-ssc映射文件
    sc.ssc.getHanziSSCDict()

    with open("./correct/common_data/all_3500_chars.txt", "r", encoding="utf-8") as f:
        chars = f.read().strip()

    sim=[]
    chi_word1_ssc = sc.ssc.getSSC(chi_word1, 'SHAPE')
    
    for char in chars:
        chi_word2_ssc = sc.ssc.getSSC(char, 'SHAPE')
        output=sc.cp.computeShapeCodeSimilarity(chi_word1_ssc[0],chi_word2_ssc[0])
        sim.append([char,output])
    
    sim.sort(key=lambda x: (x[1], x[0]), reverse=True)  # 按列表的第1个元素正序，第二个元素相同的按第0个元素正序
    
    # 图像计算
    for i in sim[0:5]:
        i.append(word_to_cosine(chi_word1,i[0]))
    
    print(sim[0:5])

if __name__ == '__main__':
    # maketxt()
    # sample()
    # jieba_creat()
    # maketxt_stamps()
    # creat_hanzi()
    # cal_similarity()
    # cal_similarity2()
    # cal_similarity3()
    # creat_hanzi()
    print(word_to_cosine('大','丰'))

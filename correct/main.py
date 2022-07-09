# import ast
# from distutils.log import error
# from turtle import right
# import pkg_resources
# import difflib

from sympy import false, true
from symspellpy import SymSpell
import re
import jieba
import soundshapecode as sc
from funtion import word_to_cosine
import os, time
import cv2, numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from symspellpy import SymSpell, Verbosity

### --------印章名 纠错--------- ###

# 去除标点符号与数字，英文
def remove(text):
    remove_chars = '[[a-zA-Z!0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    return re.sub(remove_chars, '', text)

# 计算相似度
def sim_word(astr,mydict):
    sim_dict=[]
    # 计算形码
    for word in mydict:
        sim=0;

        if(len(astr)==len(word)):
            for i in range(len(word)):            
                char1_ssc = sc.ssc.getSSC(astr[i], 'SHAPE')
                char2_ssc = sc.ssc.getSSC(word[i], 'SHAPE')
                sim=sim+sc.cp.computeShapeCodeSimilarity(char1_ssc[0],char2_ssc[0])                 
            sim=sim/len(word)
        sim_dict.append([word,sim])
    sim_dict.sort(key=lambda x: (x[1], x[0]), reverse=True) 
    # print(sim_dict[0:3])

    for i in range(1):
        sim_cos=0;        
        for k in range(len(astr)):            
            sim_cos=sim_cos + word_to_cosine(astr[k],sim_dict[i][0][k])                 
        sim_cos=sim_cos/len(astr)
        sim_dict[i].append(sim_cos)
    print(sim_dict[0:1])

    return sim_dict[0:1]

# 印章名 纠错
def seal_correct():
    # 配置文件
    sc.ssc.getHanziStrokesDict()
    sc.ssc.getHanziStructureDict()
    sc.ssc.getHanziSSCDict()
    sym_spell = SymSpell(max_dictionary_edit_distance=4, prefix_length=7)
    sym_spell_en = SymSpell(max_dictionary_edit_distance=4, prefix_length=7)
    sym_spell_full = SymSpell(max_dictionary_edit_distance=4, prefix_length=7)
    sym_spell_region = SymSpell(max_dictionary_edit_distance=4, prefix_length=7)
    dictionary_path = 'E:\code\learn-code\Stamp\correct\data.txt'
    dictionary_path_word= 'E:\code\learn-code\Stamp\correct\data_word.txt'
    corpus_path='E:\code\learn-code\Stamp\correct\comps.txt'
    dic_region_path='E:\code\learn-code\Stamp\correct\common_data\\region.txt'

    # 加载词
    sym_spell.load_dictionary(dictionary_path,term_index=0, count_index=1)
    sym_spell_en.load_dictionary(dictionary_path_word,term_index=0, count_index=1)
    sym_spell_full.create_dictionary(corpus_path)
    sym_spell_region.create_dictionary(dic_region_path)


    # 数据读入
    pred = ()
    ans= ()
    data_word=()
    data=()

    total = 0
    pred_right = 0
    change_right = 0

    # 初始化阈值
    sim_right=0.8
    cos_right=0.94

    with open('./correct/pred_uncover.txt', 'r', encoding='utf-8') as ftxt:
        pred=ftxt.read().splitlines()
    with open('./correct/ans_uncover.txt', 'r', encoding='utf-8') as ftxt:
        ans=ftxt.read().splitlines()
    with open('./correct/comps.txt', 'r', encoding='utf-8') as ftxt:
        comps=ftxt.read().splitlines()
    with open('./correct/data_word.txt', 'r', encoding='utf-8') as ftxt:
        data_word=ftxt.read().splitlines()
        for i in range(len(data_word)):
            data_word[i]=data_word[i].split(' ')[0]
    with open('./correct/data.txt', 'r', encoding='utf-8') as ftxt:
        data=ftxt.read().splitlines()
        for i in range(len(data)):
            data[i]=data[i].split(' ')[0]


    # 如果出现一样的字在词语中改怎么解决
    # 进一步修改，如武汉市..区
    separator=['省','市','区','县','村','旗']


    # main
    for i in range(len(pred)):
    # for i in range(10):
        flag=false
        input=pred[i]
        output=""
        # 移除中文以外字符
        input=remove(input)
        # 进行全词匹配
        full=sym_spell_full.lookup_compound(input,max_edit_distance=4)
        # 进行分词与纠错
        if(pred[i]!=''):
            for ful in full:
                # 如果全词匹配出错（出现两个项），则直接退出(可以去除flag，直接判断full中的元素数量)
                if flag:
                    break
                # 预处理分词
                split_word=[]
                begin_word=[]
                end_word=[]
                middle_word=[]
                # 开头
                last=0
                for k in range(len(input)):
                    if(input[k] in separator):
                        if((k+1-last)>=2):
                            begin_word.append(input[last:k+1])
                            last=k+1
                input=input[last:len(input)]
                # print("begin:",begin_word)
                for k in range(len(begin_word)):
                    sample=sym_spell_region.lookup_compound(begin_word[k],max_edit_distance=3)
                    for sam in sample:
                        begin_word[k]=str(sam).split(',')[0]
                print("change_begin:",begin_word)


                # 剩余部分 jieba分词 
                jieba.load_userdict(dictionary_path)
                jieba_split=jieba.lcut(input, cut_all=false)
                middle_word=jieba_split
                # print("middle",middle_word)

                remove_k=[]
                change_middle=middle_word

                # 处理单字与纠错
                for k in range(len(middle_word)):
                    ischange=false
                    # 找到单字
                    if(len(middle_word[k])==1):
                        # 整体就一个字
                        if(len(middle_word)==1):
                            continue
                        # 开头单字 且 下一个词不在词典中
                        if(k==0 & (len(middle_word)!=1)):
                            if(middle_word[1] not in data):
                                ischange=true
                                change_middle[1]=middle_word[0]+middle_word[1]
                        # 结尾单字 且 上一个词不在词典中
                        elif((k==len(middle_word)-1) & (len(middle_word)!=1)):
                            if(middle_word[k-1] not in data):
                                ischange=true
                                change_middle[k-1]=middle_word[k-1]+middle_word[k]
                        # 其余单字
                        elif(len(middle_word)!=1):
                            # 前一个词不在词典中 后一个词在
                            if((middle_word[k-1] not in data) & (middle_word[k+1] in data)):
                                ischange=true
                                change_middle[k-1]=middle_word[k-1]+middle_word[k]
                            # 后一个词不在词典中 前一个词在
                            if((middle_word[k-1] in data) & (middle_word[k+1] not in data)):
                                ischange=true
                                change_middle[k+1]=middle_word[k]+middle_word[k+1]
                            # 前后都不在词典中
                            if((middle_word[k-1] not in data) & (middle_word[k+1] not in data)):
                                lef_right=sim_word(middle_word[k-1]+middle_word[k],data)[0][1]
                                rig_right=sim_word(middle_word[k]+middle_word[k+1],data)[0][1]
                                if((lef_right>rig_right)&(lef_right>sim_right)):
                                    ischange=true
                                    change_middle[k-1]=middle_word[k-1]+middle_word[k]
                                elif((lef_right<rig_right)&(rig_right>sim_right)):
                                    ischange=true
                                    change_middle[k+1]=middle_word[k]+middle_word[k+1]
                    if ischange:
                        remove_k.append(k)
                
                # print('k',remove_k)
                middle_word=[]
                for j in range(len(change_middle)):
                    if j not in remove_k:
                        middle_word.append(change_middle[j])
                print('con_middle',change_middle)
                change_middle=[]
                for k in range(len(middle_word)):
                    change_word=sim_word(middle_word[k],data)[0]
                    if((change_word[1]>sim_right) & (change_word[2]>cos_right)):
                        change_middle.append(change_word[0])
                    else:
                        change_middle.append(middle_word[k])
                
                print('change_middle',change_middle)
                middle_word=change_middle


                # 输出
                split_word=begin_word+middle_word+end_word
                print(split_word)
                for k in range(len(split_word)):
                    output=output+split_word[k]
                output=output.replace(' ','')
                # print("output",output)

                # 进行输出判断
                total+=1
                if(ans[i]==pred[i]):
                    pred_right+=1
                if(ans[i]==output):
                    change_right+=1
                if((ans[i]==pred[i])&(ans[i]!=output)):
                    print("——————————改错的情况————————————")
                    # print(pred[i],output,ans[i])
                if((ans[i]!=pred[i])&(ans[i]==output)):
                    print("——————————改对的情况————————————")
                    # print(pred[i],output,ans[i])
                print(pred[i],output,ans[i])
                print('')
                flag=true

    # 计算并输出
    pred_rate=pred_right/total
    change_rate=change_right/total

    print(total,pred_right,change_right)
    print(pred_rate,change_rate)

### -------印章类型 纠错---------- ###

# 计算两个字符串中匹配的字符个数
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

# 判断是否是中文字符
def is_Chinese(ch):
    if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

# 去除字符串中的非中文字符
def select_chi(s):
    t = ''
    for c in s:
        if is_Chinese(c):
            t += c
    return t

# 印章类型 纠错
def detect():
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        use_angle_cls=0, lang='ch', use_gpu=False
    )
    img_paths = list(Path('../dataset/data_new/data_covered/flatten').glob('*.jpg'))   # jpg
    pred_dir = Path('../dataset/data_new/data_covered/result_ylh_rec_v3/')
    # pred_dir.mkdir(exist_ok=1)
    print('img amount:', len(img_paths))
    start = time.process_time()

    pred = {}
    for img_path in tqdm(img_paths):
        # {sfn}__TAG__{tag}.jpg
        sfn, tag = img_path.stem.split('__TAG__')
        sfn = sfn[:-2]
        img = cv2.imread(img_path.as_posix())
        pad = np.ones(img.shape, dtype=np.uint8)*255
        img = np.concatenate((pad, img, pad), axis=0)
        result = ocr(img)
        words = ''.join([result[1][i][0] for i in range(len(result[1]))])

        try:
            pred[sfn]
            try:
                pred[sfn][tag] += [words]
            except:
                pred[sfn][tag] = [words]
        except:
            pred[sfn] = {}
            try:
                pred[sfn][tag] += [words]
            except:
                pred[sfn][tag] = [words]
    for sfn, txt_pred in pred.items():
        # {sfn}.jpg

        js_path = pred_dir / f'{sfn}.json'

        with open(js_path.as_posix(), 'w', encoding='utf-8') as f:
            json.dump(txt_pred, f, ensure_ascii=0)
    end = time.process_time()
    print('ocr done in %6.3f' % (end - start))

# 统计完全相同的字符串的个数
def find(prediction, answers, correct):
    for answer in answers:
        print(prediction + "---" + answer + "\n")
        if prediction == answer:
            correct += 1
            answers.remove(answer)
    return [answers, correct]

# 计算准确率
def cal():
    dictionary_path = Path("./correct/common_data/zh_my.txt")
    sym_spell = SymSpell()
    sym_spell.load_dictionary(dictionary_path, 0, 1)

    pred_dir = list(Path('./dataset/data_new/data_covered/result_ylh_v3/').glob('*.json'))
    ans_dir = Path('./dataset/data_new/data_covered/json/')

    total = 0
    correct = 0
    for pred_path in pred_dir:
        sfn = pred_path.stem
        with open(pred_path, 'r', encoding='utf-8') as f:
            pred = json.load(f)
        with open((ans_dir / f'{sfn}.json'), 'r', encoding='utf-8') as f:
            ans = json.load(f)

        for key in ans.keys():
            for content in ans[key]:
                total += 1
        for key in pred.keys():
            for content in pred[key]:
                if key == '0':
                    result = sym_spell.lookup(content, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)
                    content = result[0].term
                if key == '1':
                    if select_chi(content) != '':
                        try:
                            [ans[key], correct] = find(select_chi(content), ans[key], correct)
                        except:
                            continue
                else:
                    try:
                        [ans[key], correct] = find(select_chi(content), ans[key], correct)
                    except:
                        continue
                    
    print(f'ans_class:{total},pred_class:{correct},acc_class: {correct / total * 100:.2f}%')


### ---main----- ###

if __name__ == '__main__':
    # seal_correct()
    detect()
    cal()

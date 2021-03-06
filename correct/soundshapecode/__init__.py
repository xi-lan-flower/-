from soundshapecode import ssc
from soundshapecode.ssc_similarity import compute_ssc_similarity as cp
from soundshapecode.variant_kmp import VatiantKMP


SIMILARITY_THRESHOLD = 0.8
SSC_ENCODE_WAY = 'SHAPE'#'ALL','SOUND','SHAPE'

def common():
    """注意：
    1.声母最多2位，韵母最多3位
    2.我 和 国     楼和有   也和列  可认为只是声母不一样，而韵母分别看成uo和iou和ie，多出来的部分可看成韵母辅音
    3.留和有   留：liu->l iou     有：you->yiou-> y i ou
    """
    chi_word1 = '大学'
    chi_word2 = '大家'
    ssc.getHanziStrokesDict()
    ssc.getHanziStructureDict()
    #ssc.generateHanziSSCFile()#生成汉子-ssc映射文件
    ssc.getHanziSSCDict()
    
    chi_word1_ssc = ssc.getSSC(chi_word1, SSC_ENCODE_WAY)
    print(chi_word1_ssc)
    
    chi_word2_ssc = ssc.getSSC(chi_word2, SSC_ENCODE_WAY)
    print(chi_word2_ssc)
    
    #应用串的模式匹配KMP算法，找变异词。效率比BF算法高
    kmp = VatiantKMP(SIMILARITY_THRESHOLD)
    kmp.indexKMP(chi_word2_ssc, chi_word1_ssc, SSC_ENCODE_WAY)#主串S、模式串T
    print(kmp.startIdxRes)
    
    variabt_word = set()
    for i in kmp.startIdxRes:
        variabt_word.add(chi_word2[i:i+len(chi_word1)])
    print('变异词：', variabt_word)



def atry():
    """注意：
    1.声母最多2位，韵母最多3位
    2.我 和 国     楼和有   也和列  可认为只是声母不一样，而韵母分别看成uo和iou和ie，多出来的部分可看成韵母辅音
    3.留和有   留：liu->l iou     有：you->yiou-> y i ou
    """
    chi_word1 = '大国'
    chi_word2 = '大固'
    ssc.getHanziStrokesDict()
    ssc.getHanziStructureDict()
    #ssc.generateHanziSSCFile()#生成汉子-ssc映射文件
    ssc.getHanziSSCDict()
    
    chi_word1_ssc = ssc.getSSC(chi_word1, SSC_ENCODE_WAY)
    print(chi_word1_ssc)
    
    chi_word2_ssc = ssc.getSSC(chi_word2, SSC_ENCODE_WAY)
    print(chi_word2_ssc)

    for i in range(len(chi_word1_ssc)):

        output=cp.computeShapeCodeSimilarity(chi_word1_ssc[i],chi_word2_ssc[i])
        print(output)

if __name__=="__main__":
    # common()
    atry()
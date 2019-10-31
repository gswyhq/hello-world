#!/usr/bin/python
# -*- coding: utf-8 -*-
 
import sys
 
god_dic={"LOC_S":0,"LOC_B":0, "LOC_I":0, "LOC_E":0}
pre_dic={"LOC_S":0,"LOC_B":0, "LOC_I":0, "LOC_E":0}
correct_dic={"LOC_S":0,"LOC_B":0, "LOC_I":0, "LOC_E":0}

if __name__=="__main__":
    try:
        file = open(sys.argv[1], "r")
    except:
        print("result file is not specified, or open failed!")
        sys.exit()
    wc = 0
    loc_wc = 0
    wc_of_test = 0
    wc_of_gold = 0
    wc_of_correct = 0
    flag = True
    
    for l in file:
        wc  += 1
        if l=='\n': continue
        _,_, g, r = l.strip().split()
        #并不涉及到地点实体识别
        if "LOC" not in g and "LOC" not in r: continue
        loc_wc += 1
        if "LOC" in g: 
            god_dic[g]+= 1
        if "LOC" in r: 
            pre_dic[r]+=1
        if g == r: 
            correct_dic[r]+=1

    print("WordCount from result:", wc)
    print("WordCount of loc_wc  post :", loc_wc)
    print("真实位置标记个数：", god_dic)
    print("预估位置标记个数：", pre_dic)
    print("正确标记个数：", correct_dic)

    res ={"LOC_S":0.0,"LOC_B":0.0, "LOC_I":0.0, "LOC_E":0.0}

    all_gold = 0
    all_correct = 0 
    all_pre = 0
    for k in god_dic:
        print("------ %s -------" % (k))
        R = correct_dic[k]/float(god_dic[k])
        P = correct_dic[k]/float(pre_dic[k])
        print("[%s] P = %f, R = %f, F-score = %f" % (k, P, R, (2 * P * R) / (P + R)))

        all_pre += pre_dic[k]
        all_correct += correct_dic[k]
        all_gold += god_dic[k]
    print("------ All -------")
    all_R = all_correct/float(all_gold)
    all_P = all_correct/float(all_pre)
    print("[%s] P = %f, R = %f, F-score = %f" % ("All", all_P, all_R, (2 * all_P * all_R) / (all_P + all_R)))




    


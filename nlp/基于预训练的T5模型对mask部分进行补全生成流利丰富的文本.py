

# mask补全填充，生成丰富流利的自然语言句子
# 模型来源：
# https://huggingface.co/Maciel/T5_Mask_Completion

import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
USERNAME = os.getenv("USERNAME")
pretrained = f"D:\\Users\\{USERNAME}\\data\\T5_Mask_Completion"

tokenizer = AutoTokenizer.from_pretrained(pretrained)
model = AutoModelForSeq2SeqLM.from_pretrained(pretrained)
model.eval()

sentence = "[mask]疫情[mask]公园[mask]散步[mask]"
max_input_length = 128
input_encodings = tokenizer(sentence, 
                            max_length=max_input_length, 
                            truncation=True, 
                            return_tensors="pt")
if "token_type_ids" in input_encodings.keys():
    input_encodings.pop("token_type_ids")
output = model.generate(**input_encodings, 
                        num_beams=10,
                        no_repeat_ngram_size=5,
                        do_sample=True, 
                        early_stopping=True,
                        min_length=10,
                        max_length=64,
                        return_dict_in_generate=True,
                        output_scores=True)
decoded_output = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0]
completion = decoded_output.strip()
print("\n原始文本：", sentence)
print("补全文本：", completion)

sentence_list = ['今天[mask]篮球[mask]学校[mask]',
 '[mask]疫情[mask]公园[mask]散步[mask]',
 '[mask]感染新冠[mask]身体不舒服[mask]多休息[mask]',
 '[mask]低头[mask]故乡[mask]',
 '[mask]乡愁[mask]海湾[mask]',
 '[mask]小孩[mask]易怒[mask]应该[mask]',
                 ]

for sentence in sentence_list:
    print("\n原始文本：", sentence)
    input_encodings = tokenizer(sentence,
                                max_length=max_input_length,
                                truncation=True,
                                return_tensors="pt")
    if "token_type_ids" in input_encodings.keys():
        input_encodings.pop("token_type_ids")
    output = model.generate(**input_encodings,
                            num_beams=10,
                            no_repeat_ngram_size=5,
                            do_sample=True,
                            early_stopping=True,
                            min_length=10,
                            max_length=64,
                            return_dict_in_generate=True,
                            output_scores=True)
    decoded_output = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0]
    completion = decoded_output.strip()
    print("补全文本：", completion)

# 原始文本： 今天[mask]篮球[mask]学校[mask]
# 补全文本： 今天,篮球联赛在我们学校的操场上举行。
# 原始文本： [mask]疫情[mask]公园[mask]散步[mask]
# 补全文本： 在疫情爆发之前,您可以在公园里散步。
# 原始文本： [mask]感染新冠[mask]身体不舒服[mask]多休息[mask]
# 补全文本： 如果你感染新冠了,身体不舒服,建议你多休息,注意饮食,不要熬夜,不要熬夜。
# 原始文本： [mask]低头[mask]故乡[mask]
# 补全文本： 他低头看向了故乡的那片土地。
# 原始文本： [mask]乡愁[mask]海湾[mask]
# 补全文本： 在乡愁的海湾里,有那么一个地方。
# 原始文本： [mask]小孩[mask]易怒[mask]应该[mask]
# 补全文本： 所以,当小孩变得易怒时,我们就应该先安抚他,让他知道,他为什么会这样,为什么会这样,他为什么要这样,他为什么这样,他为什么会这样。
#
# 原始文本： [mask]疫情[mask]公园[mask]散步[mask]
# 补全文本： 在疫情发生后,在公园里散步的人越来越多。
#
# 原始文本： 今天[mask]篮球[mask]学校[mask]
# 补全文本： 今天,篮球在各个学校都得到了广泛的应用。
#
# 原始文本： [mask]疫情[mask]公园[mask]散步[mask]
# 补全文本： 在疫情发生后,人们开始在公园里散步。
#
# 原始文本： [mask]感染新冠[mask]身体不舒服[mask]多休息[mask]
# 补全文本： 如果你感染新冠了,身体不舒服,建议你多休息,注意休息,不要熬夜,不要熬夜。
#
# 原始文本： [mask]低头[mask]故乡[mask]
# 补全文本： 他低头看向故乡的夕阳。
#
# 原始文本： [mask]乡愁[mask]海湾[mask]
# 补全文本： 在乡愁的海湾里,有一座小岛。
#
# 原始文本： [mask]小孩[mask]易怒[mask]应该[mask]
# 补全文本： 所以,当小孩变得易怒时,我们更应该学会控制自己的情绪。
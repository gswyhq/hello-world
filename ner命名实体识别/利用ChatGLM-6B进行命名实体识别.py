
from transformers import AutoTokenizer, AutoModel
import time

# 模型来源：https://huggingface.co/THUDM/chatglm-6b-int4
model_dir = 'THUDM/chatglm-6b-int4'
model_dir = "/huggingface/chatglm-6b-int4"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).float()
model = model.eval()
model = model.to('cpu')

response, history = model.chat(tokenizer, "你好", history=[])
print(response)
# 你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。
response, history = model.chat(tokenizer, """你好，我将给定一些文本，请帮我抽取其中的实体，需要抽取的实体有：发案单位、发案时间、司法机关名称、协查事由、司法机关人员姓名、司法机关人员职务、司法机关人员证件号码、文书编号、文书名称、被执行人姓名、被执行人证件号码、保单号、财产类型""", history=history)
print(response)

response, history = model.chat(tokenizer, """浙江省苍南县人民法院协助执行通知书（2023）浙0327执269号之二中国****股份有限公司：关于中请执行人苍南县少华轮胎经营部与被执行人高银良买卖合同纠纷一案，本院已冻结被执行人高银良在你公司的投保合同及相对应的退保金额和账户价值，因被执行人拒不履行生效判决文书所确定的义务，根据《中华人民共和国民事诉讼法》的规定，请协助执行以下事项：办理保单（保单编号P250000188888016）退保手续，核算高银良可获得的保单的现金价值或可退的保险费，强制扣留、提取被执行人名下保单的现金价值和生存金及其他权益，并在人民币12800元范围内将款项划入本院账户（户名：苍南县人民法院执行款专户：卡号：6228580399919888888：开户行：浙江苍南农村商业银行股份有限公司）ANG640南第年四月人日联系人：陈尔德电话：0577-68638888地址：浙江谷州市苍南县玉苍路166666号邮编：325800""", history=history)
print(response)

# 修改提示语
response, history = model.chat(tokenizer, """你好，请根据给定文本，抽取其中的发案单位、发案时间、司法机关名称、协查事由、司法机关人员姓名、司法机关人员职务、司法机关人员证件号码、文书编号、文书名称、被执行人姓名、被执行人证件号码、保单号、财产类型""", history=[])
print(response)


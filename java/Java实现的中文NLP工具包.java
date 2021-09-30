package nlp.demo;

import com.mayabot.nlp.fasttext.FastText;
import com.mayabot.nlp.fasttext.ScoreLabelPair;
import com.mayabot.nlp.fasttext.args.InputArgs;
import com.mayabot.nlp.fasttext.loss.LossName;
import com.mayabot.nlp.module.pinyin.PinyinResult;
import com.mayabot.nlp.module.pinyin.Pinyins;
import com.mayabot.nlp.module.pinyin.split.PinyinSplits;
import com.mayabot.nlp.module.summary.KeywordSummary;
import com.mayabot.nlp.module.summary.SentenceSummary;
import com.mayabot.nlp.module.trans.Simplified2Traditional;
import com.mayabot.nlp.module.trans.Traditional2Simplified;
import com.mayabot.nlp.module.trans.TransformService;
import com.mayabot.nlp.segment.FluentLexerBuilder;
import com.mayabot.nlp.segment.Lexer;
import com.mayabot.nlp.segment.Lexers;
import com.mayabot.nlp.segment.Sentence;
import com.mayabot.nlp.segment.plugins.customwords.CustomDictionaryPlugin;
import com.mayabot.nlp.segment.plugins.customwords.MemCustomDictionary;

import java.io.File;
import java.util.Arrays;
import java.util.List;

public class MyNLP {
    public static void main(String[] args) {
        System.out.println("你好！");

//        CORE分词器
        Lexer lexer = Lexers.coreBuilder() // CORE分词器构建器
                .withPos()  // 开启词性标注功能
                .withPersonName()  // 开启人名识别功能
                .build();

        Sentence sentence = lexer.scan("mynlp是mayabot开源的中文NLP工具包。");

        System.out.println(sentence.toList());

        // 感知机分词器：
        Lexer lexer2 = Lexers
                .perceptronBuilder() // 感知机分词器
                .withPos()
                .withPersonName()  // 开启命名实体识别
                .withNer()
                .build();

        System.out.println(lexer2.scan("2001年，他还在纽约医学院工作时，在英国学术刊物《自然》上发表一篇论文"));

        // Lexer自定义扩展插件示例
         MemCustomDictionary dictionary = new MemCustomDictionary(); // 	一个自定义词典的实现
         dictionary.addWord("逛吃行动");
        dictionary.addWord("高尚湾");
         dictionary.rebuild(); // 词典需要rebuild生效

         FluentLexerBuilder builder = Lexers.coreBuilder()
                 .withPos()
                 .withPersonName();

         builder.with(new CustomDictionaryPlugin(dictionary));  // 配置CustomDictionaryPlugin插件

        Lexer lexer3 = builder.build();

         System.out.println(lexer3.scan("逛吃行动小组在高尚湾成立"));

//        转换中文到对应的拼音
        PinyinResult result = Pinyins.convert("招商银行001, 推出朝朝盈理财产品€");

        System.out.println(result.asString());
        System.out.println(result.asHeadString(","));

        result.fuzzy(true);
        System.out.println(result.fuzzy(true).asString());

        result.keepPunctuation(true);
        //result.keepAlpha(true);
        result.keepNum(false);
        result.keepOthers(true);

        System.out.println(result.asString());

//        拼音流切分是指，将连续的拼音字母切分为一个一个原子单位。
//        拼音流切分
        System.out.println(PinyinSplits.split("nizhidaowozaishuoshenmema"));

//        mynlp采用fasttext算法提供文本分类功能，你可以训练、评估自己的分类模型。
//        训练数据是个纯文本文件，每一行一条数据，词之间使用空格分开，每一行必须包含至少一个label标签。默认 情况下，是一个带`label`前缀的字符串。
//        __label__tag1  saints rally to beat 49ers the new orleans saints survived it all hurricane ivan
//        __label__积极  这个 商品 很 好 用 。
//        所以你的训练语料需要提前进行分词预处理。
        // 训练参数
        InputArgs trainArgs = new InputArgs();
        trainArgs.setLoss(LossName.hs);
        trainArgs.setEpoch(10);  // 迭代次数
        trainArgs.setDim(100);  // dim 向量维度
        trainArgs.setLr(0.2);  // 学习率

        // 训练测试文件生成见： https://github.com/mayabot/mynlp/blob/V3.2.0/mynlp-example/src/main/java/classification/HotelCommentExampleTrain.java
        File trainFile = new File("example.data/hotel-train-seg.txt");
        File testFile = new File("example.data/hotel-test-seg.txt");
        File modelFile = new File("example.data/hotel.model");

//        FastText fastText = FastText.trainSupervised(trainFile, trainArgs);  // 训练一个分类模型
//        try {
//            fastText.saveModel("example.data/hotel.model");  // 保存模型文件
//        } catch (Exception e) {
//            e.printStackTrace();
//        }

        FastText fastText = FastText.loadModel(modelFile, true); // 加载已经训练好的模型
//        预测分类
        Lexer lexer4 = Lexers.coreBuilder().build();
        Sentence words = lexer4.scan("这个酒店的服务态度好差呀，不建议大家去住宿");
        System.out.println(String.format("%s", words));
        List<ScoreLabelPair> label1 = fastText.predict(words.toWordList(), 5,0);
        List<ScoreLabelPair> label2 = fastText.predict(Arrays.asList("洗漱台 很 干净 ， 早餐 供应 充足 ， 晚上 睡觉 很 安静".split(" ")), 5,0);
        System.out.println(String.format("预测结果：%s, %s", label1, label2));
//        FastText qFastText = fastText.quantize();  // 使用乘积量化压缩模型

//        fastText.test(testFile,1,0.0f,true);  // 使用测试数据评估模型
//        System.out.println("--------------");
//        qFastText.test(testFile,1,0.0f,true);

//        简繁转换
        Simplified2Traditional s2t = TransformService.simplified2Traditional();
        System.out.println(s2t.transform("台湾是中华人民共和国不可分割的一部分！"));

        Traditional2Simplified t2s = TransformService.traditional2Simplified();
        System.out.println(t2s.transform("軟件和體育的藝術"));

//        关键字摘要
        KeywordSummary keywordSummary = new KeywordSummary();
        List<String> result1 = keywordSummary.keyword("7月30日，@南京发布 微博发布致歉信：因我组工作疏忽，在7月29日下发的《关于做好湖南省张家界市来宁人员信息核查和健康管理的通知》中，误将湖南省张家界市写成湖北省。在此，向湖北省、湖南省和广大网民致歉。我们将认真吸取教训，切实改正错误，防止类似问题再次发生。",10);
        System.out.println(String.format("生成的关键字摘要：%s", result1));

//        句子摘要
        SentenceSummary sentenceSummary = new SentenceSummary();
        String document = "7月30日，@南京发布 微博发布致歉信：因我组工作疏忽，在7月29日下发的《关于做好湖南省张家界市来宁人员信息核查和健康管理的通知》中，误将湖南省张家界市写成湖北省。在此，向湖北省、湖南省和广大网民致歉。我们将认真吸取教训，切实改正错误，防止类似问题再次发生。";
        List<String> result2 = sentenceSummary.summarySentences(document, 10);
        System.out.println(String.format("生成的句子摘要：%s", result2));

        // 词向量训练
        //支持cow和Skipgram两种模型
        // FastText.trainCow(file,inputArgs)
        // //Or
        // FastText.trainSkipgram(file,inputArgs)

//        词向量近邻
        List<ScoreLabelPair> result3 = fastText.nearestNeighbor("酒店",5);
        System.out.println(String.format("近邻词向量：%s", result3));

        // 类比， A-B+C
        List<ScoreLabelPair> result4 = fastText.analogies("国王","皇后","男",5);
        System.out.println(String.format("类比：%s", result4));
    }
}

//资料来源： https://mynlp.mayabot.com/
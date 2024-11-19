package similarity;

import similarity.util.tokenizerimpl.BertTokenizer;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.compress.utils.IOUtils;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Session.Runner;
import org.tensorflow.Tensor;
import org.tensorflow.framework.ConfigProto;

import java.io.InputStream;
import java.util.List;

@Slf4j
public class SimBertCompute {

    private String file = "/similarity/SimBERT-tiny.pb";
    private Session session;
    private BertTokenizer bertTokenizer;

    private SimBertCompute() {
        log.info("construt SimBertCompute");
        Graph graph = new Graph();
        try (InputStream inputStream = SimBertCompute.class.getResourceAsStream(file)) {
            graph.importGraphDef(IOUtils.toByteArray(inputStream));
            ConfigProto config = ConfigProto.newBuilder().setAllowSoftPlacement(true).putDeviceCount("CPU", 2).build();
            session = new Session(graph, config.toByteArray());
            bertTokenizer = new BertTokenizer();
        } catch (Exception e) {
            log.error("初始化相似度模型失败:", e);
            throw new RuntimeException("初始化相似度模型失败：", e);
        }
    }

    private static class SimBertComputeHolder {
        private static SimBertCompute simBertCompute = new SimBertCompute();
    }

    public static SimBertCompute getInstance() {
        return SimBertComputeHolder.simBertCompute;
    }

    public double simCompute(String text1, String text2) {
        log.info("begin sim compute,{},{}",text1,text2);
        double totalScore = 0;
        try {
            float[][] input1 = new float[2][1];
            List<Integer> ids1=bertTokenizer.convertTokensToIds(bertTokenizer.tokenize(text1));
            input1[0]= new float[ids1.size()];
            for(int i=0;i< ids1.size();i++){
                input1[0][i]=ids1.get(i);
            }
            List<Integer> ids2=bertTokenizer.convertTokensToIds(bertTokenizer.tokenize(text2));
            input1[1]=new float[ids2.size()];
            for(int i=0;i< ids2.size();i++){
                input1[1][i]=ids2.get(i);
            }
            float[][] input2 =new float[2][1];
            input2[0]=new float[input1[0].length];
            input2[1]=new float[input1[0].length];
            Tensor wordIdsTensor = Tensor.create(input1);
            Tensor continueValueTensor = Tensor.create(input2);
            Runner runner = session.runner();
            Tensor out = runner.feed("Input-Token", wordIdsTensor).feed("Input-Segment", continueValueTensor).fetch("lambda_2/strided_slice").run().get(0);
            float[][] r = new float[out.numDimensions()][out.numElements()/out.numDimensions()];
            out.copyTo(r);
            float[] vector1 = r[0];
            float[] vector2 = r[1];
            double sum = 0;
            double sq1 = 0;
            double sq2 = 0;

            for (int i = 0; i < vector1.length; i++) {
                sum += vector1[i] * vector2[i];
                sq1 += vector1[i] * vector1[i];
                sq2 += vector2[i] * vector2[i];
            }
            totalScore = sum / (Math.sqrt(sq1) * Math.sqrt(sq2));
            totalScore= Double.isNaN(totalScore) ? 0 : totalScore;
            totalScore=(Math.abs(totalScore)-0.78)/0.22;
            if(totalScore<0){
                totalScore=0;
            }
            if(totalScore>1){
                totalScore=1;
            }
            out.close();
            log.info("sim compute finish,result:{}",totalScore);
        } catch (Exception e) {
            log.error("计算相似度出错:",e);
            throw new RuntimeException(e);
        }
        return totalScore;
    }
}


FROM java:latest

RUN /usr/bin/wget -O /tmp/pdfbox-app-2.0.12.jar http://mirrors.shu.edu.cn/apache/pdfbox/2.0.12/pdfbox-app-2.0.12.jar

WORKDIR /tmp

ENTRYPOINT ["java", "-jar", "/tmp/pdfbox-app-2.0.12.jar"]

CMD [""]

# docker build -t pdfbox-app:2.0.12 -f Dockerfile_pdfbox  .

# docker run -v $PWD:/mnt --rm -it pdfbox-app:2.0.12 ExtractText /mnt/information_extraction_qa_2018-11-22_140504_1542865819085117848.pdf
# docker run -v /home/core/pdfbox:/mnt pdfbox-app:2.0.12 PDFSplit /mnt/TheDockerBook_sample.pdf

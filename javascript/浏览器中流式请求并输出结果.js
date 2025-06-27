
const queryBody = {
    "searchSentence": "请详细介绍中国各个省会城市？"
};

const headers = {
    "Content-Type": "application/json",
    "Accept-Encoding": "identity"  // 明确拒绝压缩
};

const url = "http://10.119.110.120:80890/api/stream";

async function fetchStream() {
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(queryBody),
            timeout: 600000
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let answer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            answer += chunk;

            // 处理每一行
            const lines = answer.split('\n');
            for (let i = 0; i < lines.length - 1; i++) {
                console.log("请求结果：", lines[i]);
            }

            // 保留最后一行未处理的部分
            answer = lines[lines.length - 1];
        }

        // 处理最后一行
        if (answer.length > 0) {
            console.log("请求结果：", answer);
        }

    } catch (error) {
        console.error('请求失败:', error);
    }
}

fetchStream();

// 需要在http页面运行
// npm install node-fetch
// node script.js

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 为了保证流式输出，还需要中间nginx等代理层也支持，比如，不能启用压缩，也不能用缓存等；
// nginx的配置示例， conf/nginx.conf
//http {
//    ...
//    server {
//        listen       8000;
//        server_name  localhost;
//
//        location /api/stream {
//            proxy_pass http://10.119.110.120:80890/api/stream;
//            proxy_set_header Connection "";
//            proxy_http_version 1.1;
//            chunked_transfer_encoding off;
//        }


//      // config/index.js中针对流式接口的特殊配置示例
//      '/api/stream': {
//        target: 'http://10.119.110.120:8090',
//        changeOrigin: true,
//        selfHandleResponse: true, // 关键配置：手动处理响应
//        agent: new http.Agent({
//          keepAlive: true,
//          maxSockets: Infinity, // 允许无限并发连接
//          timeout: 600000 // 设置连接超时时间为600秒
//        }),
//        proxyTimeout: 600 * 1000,
//        // 新增核心配置
//        proxyOptions: {
//          buffer: false // 禁用代理缓冲
//        },
//        onProxyReq: (proxyReq, req, res) => {
//          proxyReq.setHeader('Connection', 'keep-alive');
//          proxyReq.setHeader('Accept', 'text/event-stream');
//          proxyReq.setHeader('Accept-Encoding', 'identity'); // 禁止压缩
//          console.log('代理请求头:', proxyReq.getHeaders());
//
//          // 强制使用 HTTP/1.1 的替代方案
//          if (proxyReq instanceof http.ClientRequest) {
//            proxyReq.shouldKeepAlive = true;
//          }
//        },
//        onProxyRes: (proxyRes, req, res) => {
//          proxyRes.headers['connection'] = 'keep-alive';
//          proxyRes.headers['cache-control'] = 'no-cache';
//          console.log('实际使用的HTTP版本:', proxyRes.httpVersion); // 应显示 1.1
//          console.log('代理响应头:', proxyRes.headers);
//          // 管道传输响应流
//          proxyRes.pipe(res);
//        }
//      },

# aiohttp 为支持异步编程的http请求库
import aiohttp
import asyncio

async def fetch(session, url):
  print("发送请求：", url)
  async with session.get(url, verify_ssl=False) as response:
      content = await response.content.read()
      file_name = url.rsplit('_')[-1]
      with open(file_name, mode='wb') as file_object:
          file_object.write(content)

async def main():
  async with aiohttp.ClientSession() as session:
      url_list = [
          'https://www.1.jpg',
          'https://www.2.jpg',
          'https://www.3.jpg'
      ]
      tasks = [asyncio.create_task(fetch(session, url)) for url in url_list]
      await asyncio.wait(tasks)


if __name__ == '__main__':
  asyncio.run(main())
  # 注意：这里不能使用main()
  # 因为，执行协程函数只会创建协程对象，函数内部代码不会执行。如果想要运行协程函数内部代码，必须要将协程对象交给事件循环来处理。

# 在编写程序时候可以通过如下代码来获取和创建事件循环。

# 方式一：
# import asyncio
# 生成或获取一个事件循环
# loop = asyncio.get_event_loop()
# 将任务添加到事件循环中
# loop.run_until_complete(任务)

# 方式二（python3.7及以上版本支持）：
# asyncio.run( 任务 )


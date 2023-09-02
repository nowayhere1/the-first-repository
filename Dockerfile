#继承官方python版本
FROM python:3.7
#拷贝文件
COPY . /app
#指定工作目录
WORKDIR /app
#执行指令
RUN apt-get update &&  apt-get install ffmpeg libsm6 libxext6  -y &&  pip install -r requirements.txt
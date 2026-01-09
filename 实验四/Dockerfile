FROM python:3.10-slim

# 配置 pip 使用阿里云镜像源
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 设置工作目录
WORKDIR /app

# 复制 requirements.txt（如果有）
RUN pip install --no-cache-dir \
    huggingface_hub==0.15.1 \
    sentence-transformers==2.2.2 \
    transformers==4.30.2 \
    accelerate==0.21.0 \
    streamlit==1.29.0 \
    pymilvus[milvus_lite]==2.3.4

# 复制项目文件
COPY . .

# 暴露端口（根据你的应用修改）
EXPOSE 8501

# 运行应用（根据你的应用修改）

# 启动命令
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]





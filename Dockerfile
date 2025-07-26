FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    fonts-wqy-zenhei \
    fonts-wqy-microhei \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app/ .

# 设置执行权限
RUN chmod +x run.sh

# 清除matplotlib字体缓存
RUN rm -rf ~/.cache/matplotlib

# 创建输出目录
RUN mkdir -p /app/outputs

# 设置环境变量
ENV PYTHONUNBUFFERED=1

CMD ["bash", "run.sh"]

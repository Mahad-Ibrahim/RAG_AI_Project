# /bot/Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Jump straight to pip using the high-speed Asian mirror
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple --default-timeout=100 -r requirements.txt

COPY . .

RUN chmod +x start.sh

EXPOSE 8000

CMD ["/app/start.sh"]



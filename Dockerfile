FROM python:3.11-slim

# install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# install eslint globally for JS/TS support
RUN npm install -g eslint

# set working directory
WORKDIR /app

# copy requirements first (Docker layer caching)
COPY requirements.txt .

# install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy all project files
COPY . .

# expose port
EXPOSE 8000

# start the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
from app.main import create_app
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
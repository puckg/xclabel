## xclabel
* 作者：北小菜
* 作者主页：https://www.yuturuishi.com
* gitee开源地址：https://gitee.com/Vanishi/xclabel
* github开源地址：https://github.com/beixiaocai/xclabel

### 软件介绍
xclabel是一款开源图像标注与模型训练工具，采用Python+Flask开发，跨平台支持Windows/Linux/Mac。

**核心功能：**
- 多种标注类型（矩形、多边形等），支持图片、视频、LabelMe数据集导入
- AI自动标注，支持大模型（LMStudio、vLLM、ollama、阿里云）对图片和视频自动标注
- YOLO模型训练全流程：数据集上传、模型训练、断点恢复、模型测试、参数查看与下载
- YOLO格式数据集导出，可自定义训练/验证/测试比例
- 内置文件管理系统，支持文件浏览、上传、下载
- 全部静态资源本地化，支持离线部署

### 使用说明

1. **安装依赖**：
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   
   # 如需训练功能
   
   # 安装yolo11的ultralytics依赖库
   pip install ultralytics==8.3.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
   pip install numpy==1.26.4 -i https://pypi.tuna.tsinghua.edu.cn/simple 
   
   # 安装cpu版torch依赖库
   pip install torch==2.1.2 torchvision==0.16.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
   
   # 安装cuda版torch依赖库
   pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
   

   ```

2. **启动服务**：
   ```bash
   python app.py --host 0.0.0.0 --port 9924
   ```

3. **访问服务**：浏览器打开 http://127.0.0.1:9924

4. **模型训练**：访问 http://127.0.0.1:9924/training ，上传数据集→选择模型→开始训练→测试/下载模型

### 项目结构
```
xclabel/
├── app.py                    # 主应用文件
├── AiUtils.py                # AI自动标注工具类
├── requirements.txt          # 依赖列表
├── static/                   # 静态资源（图标、样式、脚本、Socket.IO本地库）
├── templates/                # 页面模板
│   ├── index.html            # 标注主页
│   ├── training.html         # 训练面板
│   ├── ai_config.html        # AI配置
│   └── file_manager.html     # 文件管理
├── pre_models/               # 预训练模型（.pt文件）
├── uploads/                  # 上传存储（运行时自动创建）
│   ├── annotations/          # 标注数据
│   ├── config/               # 配置文件
│   ├── samples/              # 标注图片
│   └── training_datasets/    # 训练数据集
├── runs/                     # 训练输出（运行时自动创建）
└── tmp/                      # 训练临时文件（运行时自动创建）
```

### 快捷键
- **Ctrl+S**：保存标注
- **Ctrl+Shift+D**：清除标注

### 技术栈
Flask + Flask-SocketIO | HTML/CSS/JS | OpenCV/PIL | Ultralytics YOLO11 | Socket.IO

### 版本历史
查看完整更新记录：[CHANGELOG.md](CHANGELOG.md)

### 授权协议
本项目自有代码使用MIT协议，保留版权信息即可自由使用。使用第三方库请遵循其各自授权协议。

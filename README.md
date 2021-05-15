# This is zyliu's Reinforcement Learning Course Procedure

## 开发环境
1. 安装anaconda
2. 在conda创建一个新的环境rl
3. 安装Git
4. 安装模块:
    * pytorch
    * swig（不知道有什么用）
    * jupyter
    * git clone https://github.com/openai/gym.git
    * 把克隆的gym里的setup.py和MuJoCo相关的两行代码注释掉
    * 到克隆的gym文件夹执行shell：`pip install e .[all]`
5. VSCode直接运行jupyter可能会被拒绝，使用命令：`jupyter notebook`
    * 注意，conda自带jupyter，不需要去环境里安装jupyter，去环境里也无法使用命令。
    * 在我进行了一波操作之后，VSCode又好了，很奇怪
6. 配置jupyter的工作目录：
    * 使用命令得到配置文件地址：`jupyter notebook --generate-config`
    * 默认应该是在：`C:\Users\Zeaon.L\.jupyter\jupyter_notebook_config.py`
    * 把 `#c.NotebookApp.notebook_dir = ''` 改为我们想要的目录，记得去掉注释，字符串也r去转义
    * 可能会在jupyter找不到环境，是因为我们创造的环境需要 `conda install ipykernel` ，然后从jupyter的Kernel切换

## 部署环境
> 我们配置好了开发环境的时候，去执行gym的测试代码，看看小车爬山等能否正常执行
## 0.方案一：pycharm 直接连接

- 工具=》启动ssh会话=》选择对应服务器，进入会话，2之后操作

## 1.安装screen

以下任意一条命令安装screen
 `yum install screen`
`sudo apt-get install screen`
`conda install screen`

## 2.新建会话窗口

方法一：创建名为name的窗口并进入（推荐）`screen -S sessionname`
方法二：创建无名窗口并进入`screen`

## 3.列出窗口列表，可以看到新建的窗口

`screen -ls`

## 4.在新建窗口执行python命令

- ### 进入需要运行的conda环境

  ```yaml
  source activate conda_env_name
  # 到指定代码目录下运行python文件 ：主函数
  python main.py
  ```

- ### 进入目标线程，恢复会话窗口

  ```bash
  screen -r sessionname
  ```

- ### 切换回主窗口：Ctrl A + D

- ### 杀死窗口

  ```bash
  # 杀死线程号为threadnum的窗口
    `kill -9 threadnum`
  # 清除杀死的窗口， 杀死后的窗口不清除，仍会占用资源
    `screen -wipe`
  ```

  
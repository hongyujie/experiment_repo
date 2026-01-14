# Linux学习笔记
## 一、一些shell命令
- ls: 列出当前目录下的文件和子目录
  - ls -a: 列出所有文件，包括隐藏文件，例如以.开头的文件
  - ls -l: 以长格式列出文件和目录，包括权限、所有者、大小、修改时间等信息
  - ls path: 列出path目录下的文件和子目录
- cd: 切换当前目录
- pwd: 显示当前所在目录的路径
- clear: 清除屏幕上的内容，使命令行界面更加清晰
- uname: 显示系统信息，例如内核版本、操作系统名称等
- cat: 查看文件内容
  - cat filename: 查看filename文件的内容
  - cat path/filename: 查看path目录下filename文件的内容
- sudo: 以超级用户权限执行命令
  - sudo apt-get install package: 安装package软件包
  - sudo su: 切换到超级用户（root）权限     
  - exit: 退出root权限
- ctrl+c：中断终端命令，在执行命令的过程中，如果需要中断命令的执行，可以使用Ctrl+C组合键。
- touch: 创建一个空文件
  - touch filename: 创建一个空文件filename
- cp: 复制文件
  - cp filename1 filename2: 复制文件filename1到filename2
- rm: 删除文件
  - rm filename: 删除文件filename
- mkdir: 创建目录(创建文件夹)
  - mkdir dirname: 创建目录dirname
  - rmdir dirname: 删除空目录dirname
  - rm -r dirname: 删除非空目录dirname    
- mv: 移动文件，文件重命名       
  - mv filename1 filename2: 移动文件filename1到filename2，相当于是将filename1重命名为filename2
  - mv dirname1 dirname2: 移动目录dirname1到dirname2
  - mv filename dirname: 移动文件filename到目录dirname
- ifconfig: 显示网络接口信息
- reboot: 重启系统
- poweroff: 关闭系统
- sync: 同步内存中的数据到硬盘
- find: 查找文件或目录
  - find -name filename: 在当前目录下查找文件名(filename)，这里是省略了查找起始路径，默认用当前目录 .
  - find path -name filename: 在path目录下查找文件名(filename)
- ./:是 .（当前目录）和 /（Linux 路径分隔符）的结合，核心作用是明确指定「当前目录下的某个文件 / 程序」
- .:. 是 Linux 中对「当前你所在的目录」的简写符号，是最基础、最常用的路径标识，没有任何额外操作，单纯指代目录本身。
- du: 显示目录或文件的磁盘占用空间
  - du -h: 以人类可读的格式显示磁盘占用空间，例如1K、2M、3G等
  - du -h filename: 显示filename文件的磁盘占用空间，-h 表示人类可读的格式
  - du -sh dirname: 显示dirname目录的磁盘占用空间
    - -s:只显示目录占用空间的大小，不要显示其下子目录和文件占用的大小。
- gedit: 打开文本编辑器，用于编辑文本文件
  - gedit filename: 打开filename文件，用于编辑
- top: 显示系统资源占用情况，包括CPU、内存、进程等
  - 按q键退出top界面
- file: 显示文件类型
  - file filename: 显示filename文件的类型
## 二、安装软件的方法
- sudo apt-get install package(sudo apt install package): 安装package软件包
- 下载.deb软件包
  - 下载.deb文件到本地，例如package.deb
  - sudo dpkg -i package.deb: 安装package.deb软件包
## 三、vim编辑器的使用
- vim filename（vi filename）: 新建一个文件filename，进入vim编辑器。
一般刚刚进入vim编辑器时，默认是以只读模式打开文档，要编辑的话得切换到输入模式。
  - i:在当前光标所在字符的前面，转为插入模式
  - a:在当前光标所在字符的后面，转为插入模式，也是常用的插入模式。
  - o: 在当前光标所在行的下面，插入一个新行，转为插入模式
- esc: 退出插入模式，返回正常模式(只读模式)
- 退出vim编辑器时，需要先输入esc键退出插入模式，然后输入:wq键保存并退出，或者输入:q!键不保存并退出。如果没有编辑文字，也可也直接esc+:q退出。
下面的操作都是在普通模式（esc）下使用：
- dd：删除当前光标所在行
- u: 撤销上一次操作
- .: 重复上一次操作
- yy: 复制当前光标所在行
- p: 粘贴复制的内容
- 一般复制粘贴可以这样，先复制，复制的方法有yy、yw、v+y，再配合上p进行粘贴
  - yy:复制当前光标所在行
  - yw:复制当前光标所在单词   
  - v+y: 可视化选择复制，先按v进入可视化模式，然后用方向键选中要复制的内容，最后按y复制。
- v+d: 可视化选择删除，先按v进入可视化模式，然后用方向键选中要删除的内容，最后按d删除。
- 还可以用数字表示重复操作
  - 2yy: 复制当前光标所在行下面的2行
  - 3dd: 删除当前光标所在行下面的3行
- 一些光标移动常用方法：
  - 0: 移动到当前行的开头
  - $: 移动到当前行的结尾
  - gg: 移动到文件的开头
  - G: 移动到文件的结尾
  - nG: 移动到文件的第n行
## 四、Ubantu文件系统
在linux中“/”是根目录，所有的文件和目录都在根目录下，根目录下有很多子目录，每个子目录都有自己的作用。
- /bin: 存放系统最基本的二进制可执行文件，包括一些常用的命令，如ls、cp、mv等。
- /boot: 存放系统启动时需要的文件，如内核文件、引导加载程序等。
- /cdrom: 存放挂载的CD-ROM设备的挂载点。
- /dev: 存放设备驱动文件，如鼠标、键盘、硬盘等。
- /etc: 存放系统配置文件，如网络配置、用户配置例如用户账号密码等。
- /home: 存放用户的个人目录，每个用户都有一个对应的目录，用于存储个人文件和配置。
- /lib: 存放系统库文件，如动态链接库等。
- /media: 存放可移动媒体设备的挂载点，如USB闪存盘、CD-ROM等。
- /mnt: 存放临时挂载点，用于挂载其他文件系统。
- /opt: 存放可选的应用程序软件包，如第三方软件等。
- /proc: 存放系统运行时的进程信息和统计数据。
- /root: 存放root用户的个人目录，与/home目录不同，/root目录是系统管理员的个人目录。
- /sbin: 存放系统管理员使用的可执行文件，如系统管理命令等。
- /srv: 存放服务相关的文件和目录，如Web服务器的网站文件等。
- /sys: 存放系统设备和驱动程序的信息。
- /tmp: 存放临时文件，系统会定期清理该目录下的文件。
- /usr: usr不是user的缩写，而是UNIX System Resource的缩写，存放用户级别的应用程序和文件，如用户安装的软件包等。
- /var: 存放可变的文件，如日志文件、邮件队列等。
## 五、Ubantu下压缩和解压缩
常用的压缩扩展名有.tar、.tar.gz、.tar.bz2等。
### gzip压缩工具
gzip工具负责压缩.gz格式的文件
### bzip2压缩工具
bzip2工具负责压缩.bz2格式的文件

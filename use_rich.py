from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich.tree import Tree
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.status import Status
from rich.prompt import Prompt, IntPrompt
import time

# 创建控制台对象
console = Console()

# 1. 基本彩色文本输出
console.print("\n[bold magenta]===== 1. 基本彩色文本输出 =====[/bold magenta]")
console.print("[red]红色文本[/red]")
console.print("[green]绿色文本[/green]")
console.print("[blue]蓝色文本[/blue]")
console.print("[yellow]黄色文本[/yellow]")
console.print("[bold]粗体文本[/bold]")
console.print("[italic]斜体文本[/italic]")
console.print("[underline]下划线文本[/underline]")

# 组合样式
console.print("[bold red]粗体红色文本[/bold red]")
console.print("[italic blue]斜体蓝色文本[/italic blue]")
console.print("[underline green]下划线绿色文本[/underline green]")

# 2. 表格示例
console.print("\n[bold magenta]===== 2. 表格示例 =====[/bold magenta]")
table = Table(title="学生成绩表")
table.add_column("姓名", justify="center", style="cyan")
table.add_column("科目", justify="center", style="magenta")
table.add_column("成绩", justify="center", style="green")
table.add_column("等级", justify="center", style="yellow")

table.add_row("张三", "数学", "95", "A+")
table.add_row("李四", "英语", "88", "A")
table.add_row("王五", "物理", "76", "B+")
console.print(table)

# 3. 进度条示例
console.print("\n[bold magenta]===== 3. 进度条示例 =====[/bold magenta]")
for i in track(range(10), description="处理中..."):
    time.sleep(0.1)  # 模拟处理时间

# 4. 树状结构示例
console.print("\n[bold magenta]===== 4. 树状结构示例 =====[/bold magenta]")
tree = Tree("[bold green]文件系统[/bold green]")
tree.add("[cyan]Documents")
tree.add("[cyan]Downloads")

pictures = tree.add("[cyan]Pictures")
pictures.add("[magenta]Summer.jpg")
pictures.add("[magenta]Winter.jpg")

music = tree.add("[cyan]Music")
music.add("[blue]Rock")
music.add("[blue]Pop")
console.print(tree)

# 5. 面板示例
console.print("\n[bold magenta]===== 5. 面板示例 =====[/bold magenta]")
panel = Panel(
    "这是Rich库的面板示例\n可以包含多行文本\n非常适合展示重要信息",
    title="[bold blue]Rich面板[/bold blue]",
    border_style="green"
)
console.print(panel)

# 6. 语法高亮示例
console.print("\n[bold magenta]===== 6. 语法高亮示例 =====[/bold magenta]")
python_code = '''
def hello_rich():
    """Rich库示例函数"""
    console = Console()
    console.print("[bold green]Hello, Rich!")
    return True

hello_rich()
'''
syntax = Syntax(python_code, "python", theme="monokai", line_numbers=True)
console.print(syntax)

# 7. Markdown渲染示例
console.print("\n[bold magenta]===== 7. Markdown渲染示例 =====[/bold magenta]")
markdown_text = '''
# Rich库

**Rich**是一个功能强大的Python库，用于在终端中创建富文本和精美界面。

## 主要特性
- 彩色文本
- 表格
- 进度条
- 树状结构
- 语法高亮

## 安装方法
```bash
pip install rich
```
'''
md = Markdown(markdown_text)
console.print(md)

# 8. 状态指示器示例
console.print("\n[bold magenta]===== 8. 状态指示器示例 =====[/bold magenta]")
with Status("[bold blue]正在执行任务...") as status:
    time.sleep(2)
    status.update("[bold green]任务完成！")
    time.sleep(1)

# 9. 用户输入示例
console.print("\n[bold magenta]===== 9. 用户输入示例 =====[/bold magenta]")
name = Prompt.ask("请输入您的姓名")
age = IntPrompt.ask("请输入您的年龄", default=18)
console.print(f"\n[bold cyan]您好，{name}！您今年{age}岁。[/bold cyan]")

# 10. 自定义彩色输出
console.print("\n[bold magenta]===== 10. 自定义彩色输出 =====[/bold magenta]")
console.print("[color(123)]使用256色模式[/color(123)]")
console.print("[#FF5733]使用十六进制颜色[/#FF5733]")
console.print("[rgb(255, 100, 100)]使用RGB颜色[/rgb(255, 100, 100)]")

console.print("\n[bold green]Rich库演示完成！[/bold green]")

# 11. 清除控制台
console.print("\n[bold magenta]===== 11. 清除控制台 =====[/bold magenta]")
console.clear()


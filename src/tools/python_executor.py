import os
import uuid
import shutil
import subprocess
import tempfile
from langchain_core.tools import tool

PLOTS_DIR = os.path.join("data", "plots")


@tool
def python_executor(code: str) -> str:
    """在沙箱中执行 Python 代码进行数据分析和画图。

    支持 matplotlib、seaborn、numpy、pandas、scipy。
    画图时使用 plt.savefig('文件名.png') 保存图片，图片会自动被收集并返回可访问的 URL 路径。
    不能用网络请求和文件系统操作（除了在临时目录内写文件）。

    示例：
    - 画柱状图:
      import matplotlib.pyplot as plt
      plt.bar([1,2,3], [4,5,6])
      plt.savefig('bar.png')

    - 统计分析:
      import scipy.stats as stats
      result = stats.ttest_ind([1,2,3], [4,5,6])
      print(result)
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    wrapper = f"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
_cwd = os.getcwd()

{code}

# 扫描生成的图片
import glob
_plots = glob.glob(os.path.join(_cwd, '*.png')) + \\
         glob.glob(os.path.join(_cwd, '*.jpg')) + \\
         glob.glob(os.path.join(_cwd, '*.svg'))
for _p in _plots:
    print(f'__PLOT__:{{os.path.basename(_p)}}')
"""

    run_dir = tempfile.mkdtemp(prefix="pyexec_")
    try:
        result = subprocess.run(
            ["python", "-c", wrapper],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=run_dir,
        )

        output_parts = []
        plots = []

        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                if line.startswith("__PLOT__:"):
                    filename = line.split("__PLOT__:", 1)[1]
                    src = os.path.join(run_dir, filename)
                    if os.path.isfile(src):
                        base, ext = os.path.splitext(filename)
                        unique_name = f"{base}_{uuid.uuid4().hex[:8]}{ext}"
                        dst = os.path.join(PLOTS_DIR, unique_name)
                        shutil.move(src, dst)
                        plots.append(f"/plots/{unique_name}")
                else:
                    output_parts.append(line)

        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr}")

        text_output = "\n".join(output_parts).strip()

        if plots:
            plot_imgs = "\n".join(f"![图表]({url})" for url in plots)
            if text_output:
                return f"{text_output}\n\n## 生成的图表\n{plot_imgs}"
            return f"## 生成的图表\n{plot_imgs}"

        return text_output or "(无输出)"

    except subprocess.TimeoutExpired:
        return "执行超时（60秒），请简化代码或减小数据规模"
    except Exception as e:
        return f"执行错误: {e}"
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

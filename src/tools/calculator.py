import ast
import math
import operator
from langchain_core.tools import tool

# 白名单运算符
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

# 白名单函数和常量
_SAFE_NAMES = {
    "abs": abs,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "ceil": math.ceil,
    "floor": math.floor,
    "round": round,
    "factorial": math.factorial,
    "degrees": math.degrees,
    "radians": math.radians,
}


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        return _SAFE_OPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp):
        return _SAFE_OPS[type(node.op)](_eval_ast(node.operand))
    if isinstance(node, ast.Call):
        name = node.func.id  # type: ignore[attr-defined]
        if name not in _SAFE_NAMES:
            raise ValueError(f"不允许的函数: {name}")
        args = [_eval_ast(a) for a in node.args]
        return _SAFE_NAMES[name](*args)
    if isinstance(node, ast.Name):
        if node.id not in _SAFE_NAMES:
            raise ValueError(f"不允许的标识符: {node.id}")
        return _SAFE_NAMES[node.id]
    raise ValueError(f"不支持的表达式")


@tool
def calculator(expression: str) -> str:
    """安全地计算数学表达式。支持 +、-、*、/、**（幂）、sqrt、sin、cos、tan、log、exp、pi、e 等。

    示例：
    - "2 + 3 * 4"
    - "sqrt(16)"
    - "sin(pi / 2)"
    - "log(e ** 3)"
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_ast(tree.body)
        # 格式化结果，避免浮点噪音
        if isinstance(result, float) and result == int(result):
            return str(int(result))
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"

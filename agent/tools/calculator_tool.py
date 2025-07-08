from langchain.tools import Tool

def evaluate_expression(expression: str) -> str:
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

calculator_tool = Tool(
    name="Calculator",
    description="Use this tool to evaluate math expressions like '2 + 2 * 3'.",
    func=evaluate_expression
)

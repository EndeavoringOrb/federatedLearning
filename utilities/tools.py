def apply_operator(op, val1, val2):
    if op == "+":
        return val1 + val2
    elif op == "-":
        return val1 - val2
    elif op == "*":
        return val1 * val2
    elif op == "/":
        return val1 / val2
    return 0


def get_token_type(token):
    if token in "()":
        return "parentheses"
    elif token in ("+", "-", "*", "/"):
        return "op"
    else:
        return "value"


def expr_text(ops, vals):
    text = ""
    if len(vals) > 1:
        text += f"{vals[-2]}"
    if len(ops) > 0:
        text += f" {ops[-1]} "
    if len(vals) > 0:
        text += f"{vals[-1]}"
    return f"({text})"


def calculate(expr: str):
    # Stacks for operators and values
    ops = []
    vals = []
    in_val = False
    val = ""

    # Tokenize the expression by splitting it into components (numbers and operators)
    tokens = expr.replace(" ", "")

    for token in tokens:
        token_type = get_token_type(token)
        in_val = token_type == "value"
        if in_val:
            val += token
        else:
            if val:
                vals.append(float(val))
                val = ""

        if token == "(":
            continue
        elif token_type == "op":
            # It's an operator, push it onto the stack
            ops.append(token)
        elif token == ")":
            print(expr_text(ops, vals), end="")
            # Pop operator and two operands, compute and push result back
            op = ops.pop()
            val2 = vals.pop()
            val1 = vals.pop()
            result = apply_operator(op, val1, val2)
            vals.append(result)
            print(f" -> {result}")

    return str(vals.pop())


def test_calculate():
    assert calculate("(2+2)") == "4.0"
    assert calculate("(5-3)") == "2.0"
    assert calculate("(5*2)") == "10.0"
    assert calculate("(2/4)") == "0.5"
    assert calculate("((((3+(3/4))*5)/(8-2))-2)") == "1.125"


if __name__ == "__main__":
    while True:
        text = input("Enter expression: ")
        result = calculate(text)
        print(f"Result: {result}")

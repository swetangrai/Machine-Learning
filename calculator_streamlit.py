import streamlit as st

from calculator import add, subtract, multiply, divide


def main():
    st.title("Simple Calculator")
    st.write("Use this app to perform basic arithmetic operations.")

    num1 = st.number_input("First number", value=0.0, format="%f")
    num2 = st.number_input("Second number", value=0.0, format="%f")

    operation = st.radio(
        "Choose an operation",
        ("Add", "Subtract", "Multiply", "Divide"),
    )

    if st.button("Calculate"):
        try:
            if operation == "Add":
                result = add(num1, num2)
                symbol = "+"
            elif operation == "Subtract":
                result = subtract(num1, num2)
                symbol = "-"
            elif operation == "Multiply":
                result = multiply(num1, num2)
                symbol = "*"
            else:
                result = divide(num1, num2)
                symbol = "/"

            st.success(f"{num1} {symbol} {num2} = {result}")
        except ValueError as err:
            st.error(str(err))


if __name__ == "__main__":
    main()

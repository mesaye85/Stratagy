import pandas as pd

# Examine Book3.xlsx
print("Examining Book3.xlsx:")
try:
    book3 = pd.read_excel("Book3.xlsx", sheet_name=None)
    for sheet_name, sheet_data in book3.items():
        print(f"Sheet: {sheet_name}")
        print(sheet_data.head())
        print(f"Columns: {sheet_data.columns.tolist()}")
        print("\n")
except Exception as e:
    print(f"Error reading Book3.xlsx: {e}")

# Examine Book1.xlsx
print("\nExamining Book1.xlsx:")
try:
    book1 = pd.read_excel("Book1.xlsx", sheet_name=None)
    for sheet_name, sheet_data in book1.items():
        print(f"Sheet: {sheet_name}")
        print(sheet_data.head())
        print(f"Columns: {sheet_data.columns.tolist()}")
        print("\n")
except Exception as e:
    print(f"Error reading Book1.xlsx: {e}")
import nbformat
from nbclient import NotebookClient

print("Reading NFL.ipynb...")
with open("e:/Kaggle/NFL.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

print("Executing NFL.ipynb...")
client = NotebookClient(nb, timeout=600, kernel_name="python3")
client.execute()

print("Saving executed NFL.ipynb...")
with open("e:/Kaggle/NFL.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Notebook executed and saved successfully with all cell outputs!")

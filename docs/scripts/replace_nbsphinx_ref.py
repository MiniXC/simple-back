from glob import glob
import re
import json

hrefs = re.compile('href="#([^"]+)"')

refs = []

with open("../_build/html/api/simple_back.html") as api:
    refs += hrefs.findall(api.read())

refs = set(refs)


def find_ref(x):
    for r in refs:
        if r.endswith(x):
            return r


link = re.compile("\[`([^\]`]*)`\][^(]")

for file in glob("../../docs/**/*.ipynb", recursive=True):
    if "_build" not in file:
        print(file)
        with open(file) as api:
            j_file_content = json.loads(api.read())
            for i, cell in enumerate(j_file_content["cells"]):
                if cell["cell_type"] == "markdown":
                    for j, line in enumerate(cell["source"]):
                        for match in link.findall(line):
                            level = file.split("docs/")[1].count("/")
                            base = "".join(["../"] * level)
                            base += "api/simple_back.html"
                            line = line.replace(
                                f"[`{match}`]", f"[`{match}`]({base}#{find_ref(match)})"
                            )
                            j_file_content["cells"][i]["source"][j] = line
            with open(file, "w") as outfile:
                json.dump(j_file_content, outfile)

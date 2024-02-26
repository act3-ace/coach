# %%
import networkx as nx
import matplotlib.pyplot as plt

s = ""
with open("../app.py", "r") as f:
    in_callback = False
    callbacks = []
    s = ""

    for line in f:
        if in_callback:
            s += line

            if line.startswith("def"):
                in_callback = False
                callbacks.append((line , s))
                

        if line.startswith("@callback"):
            s = ""
            in_callback = True

        

# %%
info = dict()


for d, c in callbacks:
    Outputs = []
    Input = []
    State = []

    c = c.strip()
    cs = c.split(",")
    i = 0
    while i < len(cs):
        st = cs[i].strip()

        if st.startswith("Output"):
            target = st.split("(")[1].replace("'","").replace('"',"")
            atr = cs[i+1].replace("'","").replace('"',"").replace(")","")
            Outputs.append([target,atr])
            i+=1

        if st.startswith("Input"):
            target = st.split("(")[1].replace("'","").replace('"',"")
            atr = cs[i+1].replace("'","").replace('"',"").replace(")","")
            Input.append([target,atr])
            i+=1

        if st.startswith("State"):
            target = st.split("(")[1].replace("'","").replace('"',"")
            atr = cs[i+1].replace("'","").replace('"',"").replace(")","")
            State.append([target,atr])
            i+=1

        i+=1

    name = d.split()[1].split("(")[0]
    info[name] = {
                    "Input": Input, 
                    "Output": Outputs, 
                    "State": State
                }

info
# %%
edges = []
functions = []
inputs = []
outputs = []
states = []
for name, s in info.items():
    functions.append(name)

    for i in s["Input"]:
        inputs.append(i[0])
        edges.append( (i[0], name) )

    for i in s["Output"]:
        outputs.append(i[0])
        edges.append( (name, i[0]) )

    for i in s["State"]:
        states.append(i[0])
        edges.append( (i[0], name) )

edges
# %%
G = nx.Graph().to_directed()

# %%
color_map = []
for n in G:
    n = str(n)
    if n in functions:
        color_map.append("blue")
    elif n in inputs:
        color_map.append("green")
    elif n in outputs:
        color_map.append("red")
    elif n in states:
        color_map.append("orange")
    else:
        color_map.append("black")

color_map

# %%
for e in edges:
    G.add_edge(e[0], e[1])
# G = nx.from_edgelist(edges).to_directed()

f = plt.figure(figsize=[12,12])
ax = f.gca()
pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
nx.draw(
    G, 
    pos=pos, 
    ax=ax, 
    with_labels=False,
    node_color=color_map
)
text = nx.draw_networkx_labels(G, pos)
for _, t in text.items():
    t.set_rotation(20) 

print(text)
# %%

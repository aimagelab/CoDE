import pickle
from shutil import copy
import os

l = []
l.append("./debug/1275001022128/1275001022128_gen0.jpg")
l.append("./debug/1275001022128/1275001022128_gen1.jpg")
l.append("./debug/1275001022128/1275001022128_gen2.jpg")
l.append("./debug/1275001022128/1275001022128_gen3.jpg")
l.append("./debug/1275001022128/1275001022128_url.txt")
l.append("./debug/1275001022128/1275001022128_prompt.txt")
l.append("./debug/1275001022128/1275001022128_real.jpg")

d = {"1275001022128": l}
with open('./debug/example.pkl', 'wb') as f:
    pickle.dump(d, f)
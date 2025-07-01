import random
import os
import glob

list_shards = list(glob.glob(os.path.join('/work/horizon_ria_elsa/Elsa_datasetv2/webdatasets_small_elsa_v2/image_transf',"*.tar")))
list_shards = [element.split("/")[-1] for element in list_shards]
random.seed(42)
random.shuffle(list_shards)
shards_eval = list_shards[:24]
shards_test = list_shards[24:48]
shards_train = list_shards[48:96]

shards_eval_no_transf = [el.split("_transf.tar")[0] + ".tar" for el in shards_eval]
shards_test_no_transf = [el.split("_transf.tar")[0] + ".tar" for el in shards_test]

with open("../dataset/shards/elsa_v2_eval_transf.shards", "w") as f:
    f.write("\n".join(shards_eval))
with open("../dataset/shards/elsa_v2_test_transf.shards", "w") as f:
    f.write("\n".join(shards_test))
with open("../dataset/shards/elsa_v2_train_transf.shards", "w") as f:
    f.write("\n".join(shards_train))
with open("../dataset/shards/elsa_v2_eval.shards", "w") as f:
    f.write("\n".join(shards_eval_no_transf))
with open("../dataset/shards/elsa_v2_test.shards", "w") as f:
    f.write("\n".join(shards_test_no_transf))
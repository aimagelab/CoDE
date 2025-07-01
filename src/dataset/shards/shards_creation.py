# file to generate the shards

import os
import random

tars_path_transf='/work/horizon_ria_elsa/Elsa_datasetv2_fix/wds_test_small/transf'
tars_path_no_transf='/work/horizon_ria_elsa/Elsa_datasetv2_fix/wds_test_small/no_transf'

shards_output_path='/homes/fcocchi/contrastive-fake/dataset/shards'
shard_file_name_no_transf= 'elsav2-test-no_transf-fix.shards'
shard_file_name_transf= 'elsav2-test-transf-fix.shards'

num= 24
os.listdir(tars_path_transf)
os.listdir(tars_path_no_transf)

id_considered= random.sample(range(1, len(os.listdir(tars_path_transf))), num)
assert len(id_considered) == len(set(id_considered))

# write no transf shards
with open(os.path.join(shards_output_path, shard_file_name_no_transf), 'w') as f:
    for el in id_considered:
        # write the shards
        val= sorted(os.listdir(tars_path_no_transf))[el]
        # modify the name
        f.write(str(val) + "\n")

# write transf shards
with open(os.path.join(shards_output_path, shard_file_name_transf), 'w') as f:
    for el in id_considered:
        # write the shards
        val= sorted(os.listdir(tars_path_transf))[el]
        # modify the name
        f.write(str(val) + "\n")

print('Done')

import pickle
import zlib
pickle_file = open('ex_list', 'rb')
ex_list = pickle.load(pickle_file)
pickle_file = open('ex_list_new', 'wb')
instance_all = []
print("len(ex_list):", len(ex_list))
for i in range(0, len(ex_list)):
    print("i:", i, "/", len(ex_list))
    data_compress = ex_list[i]
    instance_all.append(data_compress)
    instance = pickle.loads(zlib.decompress(data_compress))
    instance['file_name']="convert_" + instance['file_name']
    #instance['start_time']  = instance['start_time']
    #instance['city_name']  = instance['city_name']
    #convert x
    instance['cent_x'] = - instance['cent_x']
    #instance['cent_y'] = instance['cent_y']
    #instance['agent_pred_index'] = instance['agent_pred_index']
    #instance['two_seconds'] = instance['two_seconds']
    #instance['angle'] = instance['angle']
    #instance['trajs'] = instance['trajs']
    for i in range(0, len(instance['agents'])):
        for j in range(0, len(instance['agents'][i])):
            instance['agents'][i][j][0] = -instance['agents'][i][j][0]
    #instance['map_start_polyline_idx'] = instance['map_start_polyline_idx']

    for i in range(0, len(instance['polygons'])):
        for j in range(0, len(instance['polygons'][i])):
            instance['polygons'][i][j][0] = -instance['polygons'][i][j][0]

    for i in range(0, len(instance['goals_2D'])):
        instance['goals_2D'][i][0] = -instance['goals_2D'][i][0]
    #instance['goals_2D_labels'] = instance['goals_2D_labels']
    #instance['stage_one_label'] = instance['stage_one_label']
    #instance['matrix'] = TBD

    for i in range(0, len(instance['labels'])):
        instance['labels'][i][0] = -instance['labels'][i][0]
    instance['polyline_spans'] = instance['polyline_spans']
    instance['labels_is_valid'] = instance['labels_is_valid']
    instance['eval_time'] = instance['eval_time']

    agent_num = instance['polyline_spans'][instance['map_start_polyline_idx']].start
    for i in range(0, agent_num):
        instance['matrix'][i][0] = - instance['matrix'][i][0]
        instance['matrix'][i][2] = - instance['matrix'][i][2]
    for i in range(agent_num, instance['matrix'].shape[0]):
        instance['matrix'][i][-1] = -instance['matrix'][i][-1]
        instance['matrix'][i][-3] = -instance['matrix'][i][-3]
        instance['matrix'][i][-17] = -instance['matrix'][i][-17]
    data_compress = zlib.compress(pickle.dumps(instance))
    instance_all.append(data_compress)
pickle.dump(instance_all, pickle_file)
pickle_file.close()

import json
import numpy as np
import os
# json_data = open('accomplishment.json').read()
# gg = json.loads(json_data)

# print(len(gg['data']))

# print(gg['data'][0])


def transfer(file_name):
    json_data = json.loads(open(file_name).read())

    word_list = json_data['word']
    x_list = [point['pos'][0] for point in json_data['data']]
    y_list = [point['pos'][1] for point in json_data['data']]
    time_stamp = [point['time'] for point in json_data['data']]
    first_time = time_stamp[0]
    tag_list = [point['tag'] for point in json_data['data']]
    tag_set = set(tag_list)
    tar_set = list(tag_set)
    tar_set = sorted(tar_set)
    print(tar_set)
    time_stamp[:] = [ x - first_time for x in time_stamp] 
    #### calculate cos and sin
    sin_list = []
    cos_list = []
    x_sp_list = []
    y_sp_list = []
    pen_up_list = []
    writing_sin = []
    writing_cos = []
    angle_stroke = []
    for tag in tag_set:
        ##collect 
        x_point = []
        y_point = []
        time_stroke = []
        angle_stroke = []
        w_sin_stroke = []
        w_cos_stroke = []
        pen_up_stroke = []
        for id_point, point in enumerate(x_list):
            if tag_list[id_point] == tag:
                x_point.append(point)
                y_point.append(y_list[id_point])
                time_stroke.append(time_stamp[id_point])


        if len(x_point) < 3 :
            sin_stroke = []
            cos_stroke = []
            for _ in range(len(x_point)):
                
                sin_stroke += [0.0]
                cos_stroke += [1.0]
        else:
            for idx in range(1, len(x_point) - 1):
                x_prev = x_point[idx - 1]
                y_prev = y_point[idx - 1]
                x_next = x_point[idx + 1]
                y_next = y_point[idx + 1]
                x_now = x_point[idx]
                y_now = y_point[idx]
                
                p0 = [x_prev, y_prev]
                p1 = [x_now, y_now]
                p2 = [x_next, y_next]
                v0 = np.array(p0) - np.array(p1)
                v1 = np.array(p2) - np.array(p1)
                angle = np.math.atan2(
                    np.linalg.det([v0, v1]), np.dot(v0, v1))
                angle_stroke.append(angle)
            new_angle_stroke = [0] + angle_stroke + [0]
            sin_stroke = np.sin(new_angle_stroke).tolist()
            cos_stroke = np.cos(new_angle_stroke).tolist()
        sin_list += sin_stroke
        cos_list += cos_stroke
        ####calculate spped
        if len(x_point) <2:
            for _ in range(len(x_point)):
                x_sp_list += [0]
                y_sp_list += [0]
            x_sp = [0]
            y_sp = [0]
        else:
            time_list = np.asarray(time_stroke, dtype=np.float32)
            time_list_moved = np.array(time_list)[1:]
            time_diff = np.subtract(time_list_moved, time_list[:-1])
            for idx, v in enumerate(time_diff):
                if v == 0:
                    time_diff[idx] = 0.001
            x_point_moved = np.array(x_point)[1:]
            y_point_moved = np.array(y_point)[1:]
            x_diff = np.subtract(x_point_moved, x_point[:-1])
            y_diff = np.subtract(y_point_moved, y_point[:-1])
            x_sp = np.divide(x_diff, time_diff).tolist()
            y_sp = np.divide(y_diff, time_diff).tolist()
            x_sp = [0] + x_sp
            y_sp = [0] + y_sp
            x_sp_list += x_sp
            y_sp_list += y_sp
        #####pen up and down 
        pen_up_stroke = [1] * (len(x_point) - 1 ) + [0]
        pen_up_list += pen_up_stroke
        for idx, x_v in enumerate(x_sp):
            y_v = y_sp[idx]
            slope = np.sqrt(x_v * x_v + y_v * y_v)
            if slope != 0:
                w_sin_stroke.append(y_v / slope)
                w_cos_stroke.append(x_v / slope)
            else:
                w_sin_stroke.append(0)
                w_cos_stroke.append(1)
        writing_sin += w_sin_stroke
        writing_cos += w_cos_stroke
    x_cor = np.asarray(x_list, dtype=np.float32)
    y_cor = np.asarray(y_list, dtype=np.float32)
    time_stamp = np.asarray(time_stamp, dtype=np.float32)
    sin_list = np.asarray(sin_list, dtype=np.float32)
    cos_list = np.asarray(cos_list, dtype=np.float32)
    x_sp_list = np.asarray(x_sp_list, dtype=np.float32)
    y_sp_list = np.asarray(y_sp_list, dtype=np.float32)
    pen_up_list = np.asarray(pen_up_list, dtype=np.float32)
    writing_cos = np.asarray(writing_cos, dtype=np.float32)
    writing_sin = np.asarray(writing_sin, dtype=np.float32)
    k = (x_sp_list, y_sp_list, x_cor, y_cor, sin_list, cos_list, writing_sin, writing_cos, pen_up_list, time_stamp)
    for each in k:
        print(each.shape)
    text_line_data = np.stack(
        (x_sp_list, y_sp_list, x_cor, y_cor, sin_list, cos_list, writing_sin, writing_cos, pen_up_list, time_stamp), axis=1)
    return text_line_data, word_list

    # print(len(x_list))
    # print(x_list[0])
    # print(time_stamp[0])
    # print(time_stamp[1])

def Multifolders():
    folder_list = os.listdir('normalized_voc')
    text_line_data_all = []
    word_all = []
    for folder in folder_list:
        file_list = os.listdir(os.path.join('normalized_voc',folder))
        for each in file_list:
            temp, word = transfer(os.path.join('normalized_voc', folder, each))
            text_line_data_all.append(temp)
            word_all.append(word)
    text_line_data_all = np.array(text_line_data_all)
    word_all = np.array(word_all)
    np.save("VRdataAll",text_line_data_all)
    np.save("VRlabelAll", word_all)

def main():
    file_list = os.listdir('2')
    text_line_data_all = []
    word_all = []
    for each in file_list:
        temp, word = transfer(os.path.join('2', each))
        text_line_data_all.append(temp)
        word_all.append(word)
    text_line_data_all = np.array(text_line_data_all)
    word_all = np.array(word_all)
    np.save("VRdataTag",text_line_data_all)
    np.save("VRlabelTag", word_all)
if __name__ == '__main__':
    Multifolders()
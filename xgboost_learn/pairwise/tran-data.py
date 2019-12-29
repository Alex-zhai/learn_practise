

def save_data(group_data, output_feature, output_group):
    if len(group_data) == 0:
        return
    output_group.write(str(len(group_data)) + "\n")
    for data in group_data:
        feats = [p for p in data[2:] if float(p.split(':')[1]) != 0.0]
        output_feature.write(data[0] + " " + " ".join(feats) + "\n")

if __name__ == "__main__":

    fi = open("raw_data")
    output_feature = open("mq2008.train", "w")
    output_group = open("mq2008.train.group", "w")

    group_data = []
    group = ""
    for line in fi:
        if not line:
            break
        if "#" in line:
            line = line[:line.index("#")]
        splits = line.strip().split(" ")
        print(splits)
        if splits[1] != group:
            save_data(group_data, output_feature, output_group)
            group_data = []
        group = splits[1]
        group_data.append(splits)
    save_data(group_data, output_feature, output_group)
    fi.close()
    output_feature.close()
    output_group.close()


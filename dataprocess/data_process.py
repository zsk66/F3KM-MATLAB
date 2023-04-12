from collections import defaultdict
import pandas as pd
import configparser
import sys
from sklearn.preprocessing import StandardScaler

def clean_data(df, config, dataset):
    # CLEAN data -- only keep columns as specified by the config file
    selected_columns = config[dataset].getlist("columns")
    variables_of_interest = config[dataset].getlist("variable_of_interest")

    # Bucketize text data
    text_columns = config[dataset].getlist("text_columns", [])
    for col in text_columns:
        # Cat codes is the 'category code'. Aka it creates integer buckets automatically.
        df[col] = df[col].astype('category').cat.codes

    # Remove the unnecessary columns. Save the variable of interest column, in case
    # it is not used for clustering.
    variable_columns = [df[var] for var in variables_of_interest]
    # df = df[[col for col in selected_columns]]

    # Convert to float, otherwise JSON cannot serialize int64
    for col in df:
        if col in text_columns or col not in selected_columns: continue
        df[col] = df[col].astype(float)

    if config["DEFAULT"].getboolean("describe_selected"):
        print(df.describe())

    return df, variable_columns

def scale_data(df):
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df[df.columns]), columns=df.columns)
    return df

def subsample_data(df, N):
    return df.sample(n=N).reset_index(drop=True)

def take_by_key(dic, seq):
    return {k : v for k, v in dic.items() if k in seq}

def read_list(config_string, delimiter=','):
    config_list = config_string.replace("\n", "").split(delimiter)
    return [s.strip() for s in config_list]

if __name__ == '__main__':

    config_file = "config/example_config.ini"
    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(config_file)

    # Create your own entry in `example_config.ini` and change this str to run
    # your own trial
    config_str = "bank" if len(sys.argv) == 1 else sys.argv[1]

    print("Using config_str = {}".format(config_str))
    data_dir = config[config_str].get("data_dir")
    dataset = config[config_str].get("dataset")
    clustering_config_file = config[config_str].get("config_file")
    num_clusters = list(map(int, config[config_str].getlist("num_clusters")))
    delta = config[config_str].getfloat("deltas")
    max_points = config[config_str].getint("max_points")
    violating = config["DEFAULT"].getboolean("violating")
    violation = config["DEFAULT"].getfloat("violation")

    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(clustering_config_file)
    # Read data in from a given csv_file found in config
    csv_file = config[dataset]["csv_file"]
    df = pd.read_csv(csv_file, sep=config[dataset]["separator"])

    # Subsample data if needed
    if max_points and len(df) > max_points:
        df = subsample_data(df, max_points)

    df.fillna(df.median(), inplace=True)

    # Clean the data (bucketize text data)
    df, _ = clean_data(df, config, dataset)
    # variable_of_interest (list[str]) : variables that we would like to collect statistics for
    variable_of_interest = config[dataset].getlist("variable_of_interest")

    # Assign each data point to a color, based on config file
    # attributes (dict[str -> defaultdict[int -> list[int]]]) : holds indices of points for each color class
    # color_flag (dict[str -> list[int]]) : holds map from point to color class it belongs to (reverse of `attributes`)
    attributes, color_flag = {}, {}
    for variable in variable_of_interest:
        colors = defaultdict(list)
        this_color_flag = [0] * len(df)

        condition_str = variable + "_conditions"
        bucket_conditions = config[dataset].getlist(condition_str)

        # For each row, if the row passes the bucket condition,
        # then the row is added to that color class
        for i, row in df.iterrows():
            for bucket_idx, bucket in enumerate(bucket_conditions):
                if eval(bucket)(row[variable]):
                    colors[bucket_idx].append(i)
                    this_color_flag[i] = bucket_idx

        attributes[variable] = colors
        color_flag[variable] = this_color_flag

    # representation (dict[str -> dict[int -> float]]) : representation of each color compared to the whole dataset
    representation = {}
    for var, bucket_dict in attributes.items():
        representation[var] = {k: (len(bucket_dict[k]) / len(df)) for k in bucket_dict.keys()}

    # Select only the desired columns
    selected_columns = config[dataset].getlist("columns")
    df = df[[col for col in selected_columns]]

    # Scale data if desired
    scaling = config["DEFAULT"].getboolean("scaling")
    if scaling:
        df = scale_data(df)

    fairness_vars = config[dataset].getlist("fairness_variable")


    alpha, beta = {}, {}
    a_val, b_val = 1 / (1 - delta), 1 - delta
    for var, bucket_dict in attributes.items():
        alpha[var] = {k: a_val * representation[var][k] for k in bucket_dict.keys()}
        beta[var] = {k: b_val * representation[var][k] for k in bucket_dict.keys()}

    # Only include the entries for the variables we want to perform fairness on
    # (in `fairness_vars`). The others are kept for statistics.
    fp_color_flag, fp_alpha, fp_beta = (take_by_key(color_flag, fairness_vars),
                                        take_by_key(alpha, fairness_vars),
                                        take_by_key(beta, fairness_vars))
    # save data
    df.to_csv('G:/F3KM/F3KM-MATLAB/'+dataset+'/'+dataset+'.csv', encoding="utf-8")
    pd.DataFrame(color_flag).to_csv('G:/F3KM/F3KM-MATLAB/'+dataset+'/'+dataset+"_color.csv", encoding="utf-8")



